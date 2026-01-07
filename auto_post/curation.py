"""
Automated Source Curation Module

Monitors scraper health, removes failing sources, and discovers AI-powered
replacements with auto-generated scraper code.
"""

import json
import os
import re
import ast
import shutil
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any
import time

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import google.generativeai as genai

from .config import NEWS_SOURCES, REQUEST_HEADERS

# Configure logging
logger = logging.getLogger('curation')
handler = logging.FileHandler('/Users/rshao/AutoBlogger/AutoBlogger/curation.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Paths
BASE_DIR = '/Users/rshao/AutoBlogger/AutoBlogger'
HEALTH_FILE = os.path.join(BASE_DIR, 'source_health.json')
CONFIG_FILE = os.path.join(BASE_DIR, 'auto_post', 'config.py')
SCRAPERS_FILE = os.path.join(BASE_DIR, 'auto_post', 'scrapers.py')
FAILED_SCRAPERS_DIR = os.path.join(BASE_DIR, 'failed_scrapers')

# Constants
FAILURE_THRESHOLD = 3
MAX_REPLACEMENTS_PER_RUN = 3
MAX_CODE_GENERATION_ATTEMPTS = 2
MAX_DISCOVERY_ATTEMPTS = 3


# ============================================================================
# HEALTH TRACKING FUNCTIONS
# ============================================================================

def load_source_health() -> Dict:
    """Load source health data from JSON file."""
    if not os.path.exists(HEALTH_FILE):
        # Initialize with empty structure
        return {
            'sources': {},
            'replacement_queue': [],
            'manual_review': []
        }

    try:
        with open(HEALTH_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading source health: {e}")
        return {
            'sources': {},
            'replacement_queue': [],
            'manual_review': []
        }


def save_source_health(data: Dict) -> None:
    """Save source health data to JSON file."""
    try:
        with open(HEALTH_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving source health: {e}")


def record_success(scraper_name: str) -> None:
    """Record successful scrape - resets consecutive failures."""
    health = load_source_health()

    if scraper_name not in health['sources']:
        health['sources'][scraper_name] = {
            'consecutive_failures': 0,
            'last_success': None,
            'last_failure': None,
            'total_successes': 0,
            'total_failures': 0
        }

    source = health['sources'][scraper_name]
    source['consecutive_failures'] = 0
    source['last_success'] = datetime.now(timezone.utc).isoformat()
    source['total_successes'] = source.get('total_successes', 0) + 1

    save_source_health(health)
    logger.info(f"Success recorded for {scraper_name}")


def record_failure(scraper_name: str, error: str) -> None:
    """Record failed scrape - increments consecutive failures."""
    health = load_source_health()

    if scraper_name not in health['sources']:
        health['sources'][scraper_name] = {
            'consecutive_failures': 0,
            'last_success': None,
            'last_failure': None,
            'total_successes': 0,
            'total_failures': 0
        }

    source = health['sources'][scraper_name]
    source['consecutive_failures'] = source.get('consecutive_failures', 0) + 1
    source['last_failure'] = datetime.now(timezone.utc).isoformat()
    source['last_failure_reason'] = error
    source['total_failures'] = source.get('total_failures', 0) + 1

    save_source_health(health)
    logger.warning(f"Failure recorded for {scraper_name}: {error} (consecutive: {source['consecutive_failures']})")


# ============================================================================
# CODE VALIDATION FUNCTIONS
# ============================================================================

def validate_scraper_code(code_string: str) -> Tuple[bool, str]:
    """
    Multi-layer validation of AI-generated scraper code.

    Returns:
        (is_valid, error_message)
    """
    # 1. Syntax validation
    try:
        ast.parse(code_string)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    # 2. Security checks - dangerous patterns
    dangerous_patterns = [
        (r'\bexec\s*\(', 'exec() call'),
        (r'\beval\s*\(', 'eval() call'),
        (r'\b__import__\s*\(', '__import__() call'),
        (r'\bos\.system\s*\(', 'os.system() call'),
        (r'\bsubprocess\.', 'subprocess usage'),
        (r'\bopen\s*\([^)]*["\']w', 'file write operation'),
    ]

    for pattern, desc in dangerous_patterns:
        if re.search(pattern, code_string):
            return False, f"Security risk: {desc} found"

    # 3. Function signature validation
    if not re.search(r'def scrape_\w+\(url\):', code_string):
        return False, "Invalid function signature (must be: def scrape_X(url):)"

    # 4. Return statement validation
    if 'return news_items' not in code_string and 'return []' not in code_string:
        return False, "Missing proper return statement"

    # 5. Required imports check (should use these modules)
    if 'requests' not in code_string or 'BeautifulSoup' not in code_string:
        return False, "Missing required imports (requests, BeautifulSoup)"

    return True, "Validation passed"


def test_scraper_execution(scraper_func, url: str, expected_source: str, expected_category: str) -> Tuple[bool, Any]:
    """
    Execute generated scraper and validate output.

    Returns:
        (success, error_or_data)
    """
    try:
        # Execute scraper with timeout
        items = scraper_func(url)

        # Validate returns list
        if not isinstance(items, list):
            return False, f"Scraper returned {type(items)} instead of list"

        # Check minimum items
        if len(items) == 0:
            return False, "No items scraped"

        # Validate each item
        for i, item in enumerate(items):
            # Check required fields
            required_fields = ['title', 'url', 'source', 'category']
            missing = [f for f in required_fields if f not in item]
            if missing:
                return False, f"Item {i} missing fields: {missing}"

            # Check reasonable title length
            title_len = len(item['title'])
            if title_len < 10 or title_len > 500:
                return False, f"Suspicious title length: {title_len}"

            # Check valid URL
            if not item['url'].startswith('http'):
                return False, f"Invalid URL: {item['url']}"

            # Check source name matches
            if item['source'] != expected_source:
                return False, f"Source mismatch: expected {expected_source}, got {item['source']}"

            # Check category matches
            if item['category'] != expected_category:
                return False, f"Category mismatch: expected {expected_category}, got {item['category']}"

        return True, items

    except Exception as e:
        return False, f"Execution failed: {str(e)}"


# ============================================================================
# AI-POWERED SOURCE DISCOVERY
# ============================================================================

def find_replacement_source(category: str, original_name: str, original_url: str) -> Optional[Dict]:
    """
    Use Gemini with web search to find replacement source.

    Returns:
        Dict with keys: name, url, reasoning, scraping_strategy
        Or None if no suitable replacement found
    """
    try:
        # Configure Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment")
            return None

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')

        # Build discovery prompt
        prompt = f"""You are a research assistant for a legal news aggregation system.

TASK: Find a high-quality replacement news source for the {category} category.

ORIGINAL SOURCE (now failing): {original_name} - {original_url}

REQUIREMENTS:
1. Must cover {category} topics (legal news related to this area)
2. Publicly accessible (no paywall or login required)
3. Regular publication schedule (at least weekly updates)
4. Reputable journalism (known publications or official agencies)
5. HTML structure suitable for web scraping (not heavy JavaScript)
6. Provides article titles, links, and optionally summaries

SEARCH STRATEGY:
- Search for: "{category} legal news sources", "best {category} news sites"
- Consider: industry publications, government agencies, legal journals
- Prioritize: RSS-friendly sites, clear HTML structure

ANALYSIS:
Visit the candidate source and analyze:
- HTML structure (what selectors would extract articles?)
- Update frequency (are articles dated? how recent?)
- Content quality (relevant to legal topics?)

OUTPUT (JSON):
{{
  "name": "Source Display Name",
  "url": "https://full-url-to-news-feed",
  "reasoning": "2-3 sentences on why this is a good replacement",
  "scraping_strategy": {{
    "article_selector": "CSS selector for article containers",
    "title_selector": "CSS selector for titles within articles",
    "link_selector": "CSS selector for links",
    "summary_selector": "CSS selector for summaries (if available)"
  }},
  "confidence": "high/medium/low"
}}

If you cannot find a suitable replacement, return {{"error": "explanation"}}.

IMPORTANT: Return ONLY valid JSON, no markdown formatting or code blocks."""

        logger.info(f"Searching for replacement for {original_name} in category {category}")

        # Call Gemini with web search
        response = model.generate_content(prompt)
        result_text = response.text.strip()

        # Clean up response (remove markdown code blocks if present)
        result_text = re.sub(r'^```json\s*', '', result_text)
        result_text = re.sub(r'\s*```$', '', result_text)

        # Parse JSON response
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            logger.error(f"Response was: {result_text[:500]}")
            return None

        # Check for error
        if 'error' in result:
            logger.warning(f"Gemini could not find replacement: {result['error']}")
            return None

        # Validate required fields
        required = ['name', 'url', 'reasoning', 'scraping_strategy']
        if not all(k in result for k in required):
            logger.error(f"Gemini response missing required fields: {result}")
            return None

        # Validate URL is accessible
        try:
            test_response = requests.get(result['url'], headers=REQUEST_HEADERS, timeout=10)
            test_response.raise_for_status()
            logger.info(f"Successfully validated replacement source: {result['name']} - {result['url']}")
        except Exception as e:
            logger.warning(f"Discovered source not accessible: {result['url']} - {e}")
            return None

        return result

    except Exception as e:
        logger.error(f"Error in source discovery: {e}")
        return None


# ============================================================================
# AI-POWERED SCRAPER CODE GENERATION
# ============================================================================

def get_example_scrapers() -> str:
    """Get 2-3 example scrapers from scrapers.py to show pattern."""
    try:
        with open(SCRAPERS_FILE, 'r') as f:
            content = f.read()

        # Extract aboutlawsuits scraper (lines ~25-50)
        aboutlawsuits_match = re.search(
            r'def scrape_aboutlawsuits\(url\):.*?(?=\ndef\s|\nSCRAPERS\s|\Z)',
            content,
            re.DOTALL
        )

        # Extract fda scraper
        fda_match = re.search(
            r'def scrape_fda\(url\):.*?(?=\ndef\s|\nSCRAPERS\s|\Z)',
            content,
            re.DOTALL
        )

        examples = []
        if aboutlawsuits_match:
            examples.append(aboutlawsuits_match.group(0))
        if fda_match:
            examples.append(fda_match.group(0))

        return '\n\n'.join(examples)
    except Exception as e:
        logger.error(f"Error reading example scrapers: {e}")
        return ""


def generate_scraper_code(source_name: str, source_url: str, category: str, scraping_strategy: Dict) -> Optional[str]:
    """
    Use Gemini to generate scraper function code.

    Returns:
        Complete function code as string, or None if generation fails
    """
    try:
        # Configure Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment")
            return None

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')

        # Get example scrapers
        examples = get_example_scrapers()

        # Sanitize name for function
        func_name = re.sub(r'[^a-z0-9]', '', source_name.lower())

        # Build code generation prompt
        prompt = f"""You are an expert Python web scraper developer.

TASK: Write a news scraper function for {source_name}.

SOURCE DETAILS:
- Name: {source_name}
- URL: {source_url}
- Category: {category}

HTML STRUCTURE ANALYSIS:
{json.dumps(scraping_strategy, indent=2)}

PATTERN TO FOLLOW (examples from existing scrapers):

{examples}

REQUIREMENTS:
1. Function name: scrape_{func_name}(url)
2. Returns: List of dictionaries with keys: title, url, summary, source, category
3. Limit to maximum 5 items
4. Include try-except with proper error handling (print error, return [])
5. Use REQUEST_HEADERS from config module for requests
6. Set timeout of 30 seconds for requests
7. Handle relative URLs using urljoin
8. Deduplicate by URL if needed
9. Source name in each item should be: "{source_name}"
10. Category in each item should be: "{category}"
11. Summary can be empty string if not available
12. Use BeautifulSoup for HTML parsing

CRITICAL REQUIREMENTS:
- Follow the EXACT pattern shown in the examples
- Include proper imports at the top: import requests, from bs4 import BeautifulSoup, from urllib.parse import urljoin
- Use the CSS selectors from the HTML structure analysis above
- Handle edge cases (missing elements, empty text, etc.)
- Return empty list on any error

OUTPUT: Return ONLY the complete Python function code with imports, no explanations, no markdown formatting."""

        logger.info(f"Generating scraper code for {source_name}")

        # Call Gemini
        response = model.generate_content(prompt)
        code = response.text.strip()

        # Clean up response (remove markdown code blocks if present)
        code = re.sub(r'^```python\s*', '', code)
        code = re.sub(r'^```\s*', '', code)
        code = re.sub(r'\s*```$', '', code)

        logger.info(f"Generated code for {source_name} ({len(code)} characters)")

        return code

    except Exception as e:
        logger.error(f"Error generating scraper code: {e}")
        return None


# ============================================================================
# CONFIG FILE MANIPULATION
# ============================================================================

def disable_source_in_config(scraper_name: str) -> bool:
    """
    Set enabled=False for source in config.py.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Backup config first
        backup_path = CONFIG_FILE + '.backup'
        shutil.copy(CONFIG_FILE, backup_path)

        # Read config
        with open(CONFIG_FILE, 'r') as f:
            content = f.read()

        # Find source dict and update enabled field
        # Pattern: 'scraper': 'name' ... 'enabled': True/False
        pattern = rf"('scraper':\s*'{scraper_name}'.*?'enabled':\s*)(True|False)"

        if not re.search(pattern, content, re.DOTALL):
            logger.error(f"Could not find scraper '{scraper_name}' in config")
            return False

        updated_content = re.sub(pattern, r'\1False', content, flags=re.DOTALL)

        # Validate syntax
        try:
            ast.parse(updated_content)
        except SyntaxError as e:
            logger.error(f"Generated invalid Python in config: {e}")
            return False

        # Write updated config
        with open(CONFIG_FILE, 'w') as f:
            f.write(updated_content)

        # Remove backup after success
        os.remove(backup_path)

        logger.info(f"Disabled source in config: {scraper_name}")
        return True

    except Exception as e:
        logger.error(f"Error disabling source in config: {e}")
        # Rollback
        if os.path.exists(backup_path):
            shutil.copy(backup_path, CONFIG_FILE)
            os.remove(backup_path)
        return False


def add_source_to_config(source_data: Dict) -> bool:
    """
    Append new source to NEWS_SOURCES in config.py.

    Args:
        source_data: Dict with keys: name, url, category, scraper

    Returns:
        True if successful, False otherwise
    """
    try:
        # Backup config first
        backup_path = CONFIG_FILE + '.backup'
        shutil.copy(CONFIG_FILE, backup_path)

        # Read config
        with open(CONFIG_FILE, 'r') as f:
            content = f.read()

        # Format new source dict
        new_source = f"""    {{
        'name': '{source_data['name']}',
        'url': '{source_data['url']}',
        'category': '{source_data['category']}',
        'enabled': True,
        'scraper': '{source_data['scraper']}'
    }}"""

        # Find end of NEWS_SOURCES list (closing bracket)
        # Insert new source before the closing bracket
        # Look for the pattern: closing brace, closing bracket on NEWS_SOURCES
        pattern = r'(    \}\s*\n)(\])'

        if not re.search(pattern, content):
            logger.error("Could not find NEWS_SOURCES closing bracket")
            return False

        updated_content = re.sub(pattern, r'\1,\n' + new_source + r'\n\2', content, count=1)

        # Validate syntax
        try:
            ast.parse(updated_content)
        except SyntaxError as e:
            logger.error(f"Generated invalid Python in config: {e}")
            return False

        # Write updated config
        with open(CONFIG_FILE, 'w') as f:
            f.write(updated_content)

        # Remove backup after success
        os.remove(backup_path)

        logger.info(f"Added source to config: {source_data['name']}")
        return True

    except Exception as e:
        logger.error(f"Error adding source to config: {e}")
        # Rollback
        if os.path.exists(backup_path):
            shutil.copy(backup_path, CONFIG_FILE)
            os.remove(backup_path)
        return False


def add_scraper_to_module(scraper_name: str, scraper_code: str) -> bool:
    """
    Add new scraper function to scrapers.py.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Backup scrapers first
        backup_path = SCRAPERS_FILE + '.backup'
        shutil.copy(SCRAPERS_FILE, backup_path)

        # Read scrapers
        with open(SCRAPERS_FILE, 'r') as f:
            content = f.read()

        # Find SCRAPERS dictionary location
        scrapers_dict_match = re.search(r'(# Map scraper names to functions\s*\n)?SCRAPERS\s*=\s*\{', content)
        if not scrapers_dict_match:
            logger.error("Could not find SCRAPERS dictionary")
            return False

        insertion_point = scrapers_dict_match.start()

        # Insert function before SCRAPERS dictionary
        new_content = (
            content[:insertion_point] +
            f"\n\n{scraper_code}\n\n" +
            content[insertion_point:]
        )

        # Update SCRAPERS dictionary
        # Find closing brace of SCRAPERS dict
        scrapers_end = new_content.find('}', new_content.find('SCRAPERS = {'))
        if scrapers_end == -1:
            logger.error("Could not find SCRAPERS dictionary closing brace")
            return False

        new_entry = f"    '{scraper_name}': scrape_{scraper_name},\n"
        new_content = (
            new_content[:scrapers_end] +
            new_entry +
            new_content[scrapers_end:]
        )

        # Validate syntax
        try:
            ast.parse(new_content)
        except SyntaxError as e:
            logger.error(f"Generated invalid Python in scrapers: {e}")
            return False

        # Write updated scrapers
        with open(SCRAPERS_FILE, 'w') as f:
            f.write(new_content)

        # Remove backup after success
        os.remove(backup_path)

        logger.info(f"Added scraper to module: {scraper_name}")
        return True

    except Exception as e:
        logger.error(f"Error adding scraper to module: {e}")
        # Rollback
        if os.path.exists(backup_path):
            shutil.copy(backup_path, SCRAPERS_FILE)
            os.remove(backup_path)
        return False


# ============================================================================
# MAIN CURATION ORCHESTRATION
# ============================================================================

def run_source_curation() -> Dict:
    """
    Main entry point - run after scraping to manage source health.

    Returns:
        Dict with summary: sources_disabled, sources_added, pending_replacements
    """
    logger.info("=== Starting source curation ===")

    result = {
        'sources_disabled': [],
        'sources_added': [],
        'pending_replacements': [],
        'errors': []
    }

    # Load health data
    health = load_source_health()

    # Phase 1: Identify failing sources (>= 3 consecutive failures)
    failing_sources = []
    for scraper_name, data in health['sources'].items():
        if data.get('consecutive_failures', 0) >= FAILURE_THRESHOLD:
            # Find corresponding source in NEWS_SOURCES
            source_info = next((s for s in NEWS_SOURCES if s['scraper'] == scraper_name), None)
            if source_info and source_info.get('enabled', False):
                failing_sources.append({
                    'scraper': scraper_name,
                    'name': source_info['name'],
                    'url': source_info['url'],
                    'category': source_info['category'],
                    'failures': data['consecutive_failures'],
                    'last_error': data.get('last_failure_reason', 'Unknown')
                })
                logger.info(f"Identified failing source: {source_info['name']} ({data['consecutive_failures']} failures)")

    # Phase 2: Disable failing sources
    for source in failing_sources:
        if disable_source_in_config(source['scraper']):
            result['sources_disabled'].append(source['name'])

            # Update health data
            health['sources'][source['scraper']]['disabled_at'] = datetime.now(timezone.utc).isoformat()
            health['sources'][source['scraper']]['disabled_reason'] = f"{source['failures']} consecutive failures"

            # Add to replacement queue if not already there
            already_queued = any(
                item['scraper'] == source['scraper']
                for item in health['replacement_queue']
            )
            if not already_queued:
                health['replacement_queue'].append({
                    'scraper': source['scraper'],
                    'original_name': source['name'],
                    'original_url': source['url'],
                    'category': source['category'],
                    'queued_at': datetime.now(timezone.utc).isoformat(),
                    'attempts': 0
                })
                logger.info(f"Added {source['name']} to replacement queue")

    save_source_health(health)

    # Phase 3-5: Process replacement queue (limit to MAX_REPLACEMENTS_PER_RUN)
    replacements_processed = 0

    for item in health['replacement_queue'][:]:  # Copy list to allow modification during iteration
        if replacements_processed >= MAX_REPLACEMENTS_PER_RUN:
            logger.info(f"Reached max replacements per run ({MAX_REPLACEMENTS_PER_RUN})")
            break

        replacements_processed += 1

        # Check attempt count
        if item['attempts'] >= MAX_DISCOVERY_ATTEMPTS:
            logger.warning(f"Max discovery attempts reached for {item['original_name']}, moving to manual review")
            health['replacement_queue'].remove(item)
            health['manual_review'].append({
                **item,
                'reason': 'Max discovery attempts exceeded',
                'moved_at': datetime.now(timezone.utc).isoformat()
            })
            save_source_health(health)
            continue

        item['attempts'] += 1
        save_source_health(health)

        logger.info(f"Processing replacement for {item['original_name']} (attempt {item['attempts']})")

        # Phase 3: Discover replacement source
        discovered_source = find_replacement_source(
            item['category'],
            item['original_name'],
            item['original_url']
        )

        if not discovered_source:
            logger.warning(f"Could not discover replacement for {item['original_name']}")
            result['pending_replacements'].append(item['original_name'])
            continue

        # Phase 4: Generate scraper code
        func_name = re.sub(r'[^a-z0-9]', '', discovered_source['name'].lower())

        code_generated = False
        generated_code = None

        for attempt in range(MAX_CODE_GENERATION_ATTEMPTS):
            generated_code = generate_scraper_code(
                discovered_source['name'],
                discovered_source['url'],
                item['category'],
                discovered_source['scraping_strategy']
            )

            if not generated_code:
                logger.warning(f"Code generation failed for {discovered_source['name']} (attempt {attempt + 1})")
                continue

            # Validate code
            is_valid, error_msg = validate_scraper_code(generated_code)
            if not is_valid:
                logger.warning(f"Code validation failed: {error_msg} (attempt {attempt + 1})")
                continue

            code_generated = True
            break

        if not code_generated:
            logger.error(f"Failed to generate valid code for {discovered_source['name']}")

            # Save failed attempt for debugging
            os.makedirs(FAILED_SCRAPERS_DIR, exist_ok=True)
            failed_file = os.path.join(FAILED_SCRAPERS_DIR, f"{func_name}_{int(time.time())}.py")
            with open(failed_file, 'w') as f:
                f.write(f"# Failed scraper generation for {discovered_source['name']}\n")
                f.write(f"# Category: {item['category']}\n")
                f.write(f"# URL: {discovered_source['url']}\n")
                f.write(f"# Validation error: {error_msg}\n\n")
                if generated_code:
                    f.write(generated_code)

            result['pending_replacements'].append(item['original_name'])
            continue

        # Phase 5: Test scraper execution
        # Import and test the generated function
        try:
            # Create a temporary module to test the code
            exec_globals = {
                'requests': requests,
                'BeautifulSoup': BeautifulSoup,
                'urljoin': urljoin,
                'REQUEST_HEADERS': REQUEST_HEADERS
            }
            exec(generated_code, exec_globals)

            scraper_func = exec_globals.get(f'scrape_{func_name}')
            if not scraper_func:
                logger.error(f"Could not find function scrape_{func_name} in generated code")
                result['pending_replacements'].append(item['original_name'])
                continue

            # Test execution
            test_success, test_result = test_scraper_execution(
                scraper_func,
                discovered_source['url'],
                discovered_source['name'],
                item['category']
            )

            if not test_success:
                logger.error(f"Scraper test failed: {test_result}")

                # Save failed attempt
                os.makedirs(FAILED_SCRAPERS_DIR, exist_ok=True)
                failed_file = os.path.join(FAILED_SCRAPERS_DIR, f"{func_name}_{int(time.time())}.py")
                with open(failed_file, 'w') as f:
                    f.write(f"# Failed scraper test for {discovered_source['name']}\n")
                    f.write(f"# Category: {item['category']}\n")
                    f.write(f"# URL: {discovered_source['url']}\n")
                    f.write(f"# Test error: {test_result}\n\n")
                    f.write(generated_code)

                result['pending_replacements'].append(item['original_name'])
                continue

            logger.info(f"Scraper test passed for {discovered_source['name']}")

        except Exception as e:
            logger.error(f"Error testing scraper: {e}")
            result['pending_replacements'].append(item['original_name'])
            continue

        # Deploy: Add to scrapers.py and config.py
        if not add_scraper_to_module(func_name, generated_code):
            logger.error(f"Failed to add scraper to module")
            result['errors'].append(f"Failed to add {discovered_source['name']} to scrapers.py")
            continue

        if not add_source_to_config({
            'name': discovered_source['name'],
            'url': discovered_source['url'],
            'category': item['category'],
            'scraper': func_name
        }):
            logger.error(f"Failed to add source to config")
            result['errors'].append(f"Failed to add {discovered_source['name']} to config.py")
            continue

        # Success! Initialize health tracking and remove from queue
        health['sources'][func_name] = {
            'consecutive_failures': 0,
            'last_success': None,
            'last_failure': None,
            'total_successes': 0,
            'total_failures': 0,
            'added_at': datetime.now(timezone.utc).isoformat(),
            'replaced': item['original_name']
        }

        health['replacement_queue'].remove(item)
        save_source_health(health)

        result['sources_added'].append(discovered_source['name'])
        logger.info(f"Successfully deployed new source: {discovered_source['name']}")

    logger.info("=== Curation complete ===")
    return result
