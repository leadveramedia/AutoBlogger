#!/usr/bin/env python3
"""
Automated Blog Post Generator for Legal News
Scrapes multiple news sources, generates AI content with Gemini, and posts to Sanity.io
Focused on: Personal Injury Law, Employment Law, Mass Torts
"""

import os
import re
import json
import uuid
import time
from datetime import datetime
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from google import genai

# --- CONFIGURATION ---
# Sanity API configuration
SANITY_PROJECT_ID = os.environ.get('SANITY_PROJECT_ID', '')
SANITY_DATASET = os.environ.get('SANITY_DATASET', 'production')
SANITY_TOKEN = os.environ.get('SANITY_TOKEN', '')

# Default author for auto-generated posts
DEFAULT_AUTHOR = os.environ.get('DEFAULT_AUTHOR', 'Case Value Expert')

SANITY_BASE_URL = f"https://{SANITY_PROJECT_ID}.api.sanity.io/v2021-06-07/data/mutate/{SANITY_DATASET}"
SANITY_QUERY_URL = f"https://{SANITY_PROJECT_ID}.api.sanity.io/v2021-06-07/data/query/{SANITY_DATASET}"

HEADERS = {
    'Authorization': f"Bearer {SANITY_TOKEN}",
    'Content-Type': 'application/json'
}

# Initialize Gemini Client
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

# User agent for scraping
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# Sanity Assets URL for image uploads
SANITY_ASSETS_URL = f"https://{SANITY_PROJECT_ID}.api.sanity.io/v1/assets/images/{SANITY_DATASET}"


def generate_image_with_gemini(alt_text):
    """
    Generate an image using Imagen 4 model.
    Returns the image bytes or None if generation fails.
    """
    print(f"Generating image with prompt: {alt_text[:60]}...")

    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not set")
        return None

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        # Create a professional, legal-themed image prompt
        image_prompt = f"""Generate a professional stock photo for a law firm blog article.

Subject: {alt_text}

Requirements:
- Professional, clean, modern aesthetic
- Trustworthy and empathetic tone
- No text, watermarks, or logos
- Suitable for a personal injury law firm website
- High quality, photorealistic style"""

        # Use Imagen 4 for image generation
        response = client.models.generate_images(
            model='imagen-4.0-generate-001',
            prompt=image_prompt,
            config={
                'number_of_images': 1,
            }
        )

        # Check if we got an image in the response
        if response.generated_images and len(response.generated_images) > 0:
            image = response.generated_images[0]
            if hasattr(image, 'image') and hasattr(image.image, 'image_bytes'):
                image_data = image.image.image_bytes
                print(f"Image generated successfully ({len(image_data)} bytes)")
                return image_data

        print("No image was generated in response")
        return None

    except Exception as e:
        print(f"Error generating image: {e}")
        # If image generation fails, that's okay - we continue without image
        return None


def upload_image_to_sanity(image_bytes, filename="blog-image.png"):
    """
    Upload an image to Sanity's asset pipeline.
    Returns the asset reference or None if upload fails.
    """
    print(f"Uploading image to Sanity...")

    if not all([SANITY_PROJECT_ID, SANITY_TOKEN, SANITY_DATASET]):
        print("Error: Missing Sanity configuration")
        return None

    if not image_bytes:
        print("Error: No image data to upload")
        return None

    try:
        headers = {
            'Authorization': f"Bearer {SANITY_TOKEN}",
            'Content-Type': 'image/png'
        }

        response = requests.post(
            f"{SANITY_ASSETS_URL}?filename={filename}",
            headers=headers,
            data=image_bytes,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            asset_id = result.get('document', {}).get('_id')
            print(f"Image uploaded successfully. Asset ID: {asset_id}")
            return {
                "_type": "image",
                "asset": {
                    "_type": "reference",
                    "_ref": asset_id
                }
            }
        else:
            print(f"Failed to upload image: {response.status_code}")
            print(response.text)
            return None

    except Exception as e:
        print(f"Error uploading image: {e}")
        return None


# --- NEWS SOURCES CONFIGURATION ---
# Each source has its own scraping configuration
NEWS_SOURCES = [
    {
        'name': 'AboutLawsuits',
        'url': 'https://www.aboutlawsuits.com',
        'category': 'mass_torts',
        'enabled': True,
        'scraper': 'aboutlawsuits'
    },
    {
        'name': 'FDA Recalls',
        'url': 'https://www.fda.gov/safety/recalls-market-withdrawals-safety-alerts',
        'category': 'mass_torts',
        'enabled': True,
        'scraper': 'fda'
    },
    {
        'name': 'EEOC News',
        'url': 'https://www.eeoc.gov/newsroom',
        'category': 'employment_law',
        'enabled': True,
        'scraper': 'eeoc'
    },
    {
        'name': 'OSHA News',
        'url': 'https://www.osha.gov/news/newsreleases',
        'category': 'personal_injury',
        'enabled': True,
        'scraper': 'osha'
    },
    {
        'name': 'Courthouse News',
        'url': 'https://www.courthousenews.com',
        'category': 'general_legal',
        'enabled': True,
        'scraper': 'courthousenews'
    },
    {
        'name': 'ConsumerSafety',
        'url': 'https://www.consumersafety.org/news/',
        'category': 'mass_torts',
        'enabled': True,
        'scraper': 'consumersafety'
    },
    {
        'name': 'Bloomberg Law Daily Labor',
        'url': 'https://news.bloomberglaw.com/daily-labor-report',
        'category': 'employment_law',
        'enabled': True,
        'scraper': 'bloomberg'
    },
    {
        'name': 'AP News',
        'url': 'https://apnews.com/',
        'category': 'general_legal',
        'enabled': True,
        'scraper': 'apnews'
    },
    {
        'name': 'CNN US',
        'url': 'https://www.cnn.com/us',
        'category': 'general_legal',
        'enabled': True,
        'scraper': 'cnn'
    },
    {
        'name': 'NY Times US',
        'url': 'https://www.nytimes.com/section/us',
        'category': 'general_legal',
        'enabled': True,
        'scraper': 'nytimes'
    },
    {
        'name': 'ProPublica',
        'url': 'https://www.propublica.org/',
        'category': 'general_legal',
        'enabled': True,
        'scraper': 'propublica'
    },
    {
        'name': 'OnScene TV',
        'url': 'https://onscene.tv/',
        'category': 'personal_injury',
        'enabled': True,
        'scraper': 'onscenetv'
    }
]


def generate_key():
    """Generate a unique key for Portable Text blocks."""
    return uuid.uuid4().hex[:12]


def convert_markdown_to_portable_text(markdown_content):
    """
    Convert Markdown content to Sanity Portable Text format.
    Handles paragraphs, headings, bold, italic, links, and lists.
    """
    blocks = []
    lines = markdown_content.split('\n')
    current_list_type = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            current_list_type = None
            i += 1
            continue

        # Check for headings
        if line.startswith('### '):
            current_list_type = None
            blocks.append({
                "_type": "block",
                "_key": generate_key(),
                "style": "h3",
                "markDefs": [],
                "children": parse_inline_content(line[4:])
            })
            i += 1
            continue

        if line.startswith('## '):
            current_list_type = None
            blocks.append({
                "_type": "block",
                "_key": generate_key(),
                "style": "h2",
                "markDefs": [],
                "children": parse_inline_content(line[3:])
            })
            i += 1
            continue

        # Check for bullet lists
        if line.startswith('- ') or line.startswith('* '):
            current_list_type = 'bullet'
            list_item_content = line[2:]
            children, mark_defs = parse_inline_with_links(list_item_content)
            blocks.append({
                "_type": "block",
                "_key": generate_key(),
                "style": "normal",
                "listItem": "bullet",
                "level": 1,
                "markDefs": mark_defs,
                "children": children
            })
            i += 1
            continue

        # Check for numbered lists
        numbered_match = re.match(r'^(\d+)\.\s+(.+)$', line)
        if numbered_match:
            current_list_type = 'number'
            list_item_content = numbered_match.group(2)
            children, mark_defs = parse_inline_with_links(list_item_content)
            blocks.append({
                "_type": "block",
                "_key": generate_key(),
                "style": "normal",
                "listItem": "number",
                "level": 1,
                "markDefs": mark_defs,
                "children": children
            })
            i += 1
            continue

        # Regular paragraph
        current_list_type = None
        children, mark_defs = parse_inline_with_links(line)
        blocks.append({
            "_type": "block",
            "_key": generate_key(),
            "style": "normal",
            "markDefs": mark_defs,
            "children": children
        })
        i += 1

    return blocks


def parse_inline_content(text):
    """Parse inline content (bold, italic) without links."""
    children = []
    parts = re.findall(r'\*\*(.+?)\*\*|\*([^*]+?)\*|([^*]+)', text)

    for bold, italic, normal in parts:
        if bold:
            children.append({
                "_type": "span",
                "_key": generate_key(),
                "text": bold,
                "marks": ["strong"]
            })
        elif italic:
            children.append({
                "_type": "span",
                "_key": generate_key(),
                "text": italic,
                "marks": ["em"]
            })
        elif normal:
            children.append({
                "_type": "span",
                "_key": generate_key(),
                "text": normal,
                "marks": []
            })

    if not children:
        children.append({
            "_type": "span",
            "_key": generate_key(),
            "text": text,
            "marks": []
        })

    return children


def parse_inline_with_links(text):
    """Parse inline content including links, bold, and italic."""
    children = []
    mark_defs = []

    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    last_end = 0

    for match in re.finditer(link_pattern, text):
        if match.start() > last_end:
            before_text = text[last_end:match.start()]
            children.extend(parse_inline_content(before_text))

        link_key = generate_key()
        link_text = match.group(1)
        link_url = match.group(2)

        mark_defs.append({
            "_type": "link",
            "_key": link_key,
            "href": link_url
        })

        children.append({
            "_type": "span",
            "_key": generate_key(),
            "text": link_text,
            "marks": [link_key]
        })

        last_end = match.end()

    if last_end < len(text):
        remaining_text = text[last_end:]
        children.extend(parse_inline_content(remaining_text))

    if not children:
        children.append({
            "_type": "span",
            "_key": generate_key(),
            "text": text,
            "marks": []
        })

    return children, mark_defs


# --- SCRAPER FUNCTIONS FOR EACH SOURCE ---

def scrape_aboutlawsuits(url):
    """Scrape AboutLawsuits.com for mass tort and class action news."""
    print(f"  Scraping AboutLawsuits...")
    news_items = []

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find article links - they use WordPress block structure
        articles = soup.find_all(['article', 'div'], class_=lambda x: x and 'wp-block' in str(x), limit=10)

        # Also try finding by h4 tags which contain headlines
        headlines = soup.find_all('h4', limit=10)

        for headline in headlines:
            link = headline.find('a') or headline.find_parent('a')
            if link and link.get('href'):
                title = headline.get_text(strip=True)
                href = link.get('href')
                if title and '/about/' not in href.lower():  # Skip about pages
                    news_items.append({
                        'title': title,
                        'url': href if href.startswith('http') else urljoin(url, href),
                        'summary': '',
                        'source': 'AboutLawsuits',
                        'category': 'mass_torts'
                    })

        # Deduplicate by URL
        seen_urls = set()
        unique_items = []
        for item in news_items:
            if item['url'] not in seen_urls:
                seen_urls.add(item['url'])
                unique_items.append(item)

        return unique_items[:5]

    except Exception as e:
        print(f"  Error scraping AboutLawsuits: {e}")
        return []


def scrape_fda(url):
    """Scrape FDA recalls and safety alerts."""
    print(f"  Scraping FDA Recalls...")
    news_items = []

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # FDA uses a datatable structure
        rows = soup.select('table tbody tr, .views-row')

        for row in rows[:10]:
            # Try to find brand name and link
            brand_cell = row.select_one('.views-field-brand-name a, td a')
            if brand_cell:
                title = brand_cell.get_text(strip=True)
                href = brand_cell.get('href', '')

                # Get description if available
                desc_cell = row.select_one('.views-field-field-product-description-1, td:nth-child(2)')
                summary = desc_cell.get_text(strip=True) if desc_cell else ''

                if title:
                    news_items.append({
                        'title': f"FDA Recall: {title}",
                        'url': href if href.startswith('http') else urljoin('https://www.fda.gov', href),
                        'summary': summary[:200],
                        'source': 'FDA',
                        'category': 'mass_torts'
                    })

        return news_items[:5]

    except Exception as e:
        print(f"  Error scraping FDA: {e}")
        return []


def scrape_eeoc(url):
    """Scrape EEOC newsroom for employment discrimination cases."""
    print(f"  Scraping EEOC News...")
    news_items = []

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # EEOC uses Drupal views-row structure
        rows = soup.select('.views-row, .news-item, article')

        for row in rows[:10]:
            title_tag = row.select_one('h3 a, h2 a, .title a')
            if title_tag:
                title = title_tag.get_text(strip=True)
                href = title_tag.get('href', '')

                # Get summary
                summary_tag = row.select_one('p, .summary, .description')
                summary = summary_tag.get_text(strip=True)[:200] if summary_tag else ''

                if title:
                    news_items.append({
                        'title': title,
                        'url': href if href.startswith('http') else urljoin('https://www.eeoc.gov', href),
                        'summary': summary,
                        'source': 'EEOC',
                        'category': 'employment_law'
                    })

        return news_items[:5]

    except Exception as e:
        print(f"  Error scraping EEOC: {e}")
        return []


def scrape_osha(url):
    """Scrape OSHA news releases for workplace safety violations."""
    print(f"  Scraping OSHA News...")
    news_items = []

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # OSHA uses h5 for titles
        headlines = soup.select('h5, .views-row h3, .news-release-title')

        for headline in headlines[:10]:
            link = headline.find('a') or headline.find_parent('a')
            if not link:
                # Try finding link in parent container
                parent = headline.find_parent(['div', 'article', 'li'])
                if parent:
                    link = parent.find('a')

            if link and link.get('href'):
                title = headline.get_text(strip=True)
                href = link.get('href', '')

                # Get summary from following p tag
                summary_tag = headline.find_next('p')
                summary = summary_tag.get_text(strip=True)[:200] if summary_tag else ''

                if title:
                    news_items.append({
                        'title': title,
                        'url': href if href.startswith('http') else urljoin('https://www.osha.gov', href),
                        'summary': summary,
                        'source': 'OSHA',
                        'category': 'personal_injury'
                    })

        return news_items[:5]

    except Exception as e:
        print(f"  Error scraping OSHA: {e}")
        return []


def scrape_courthousenews(url):
    """Scrape Courthouse News for federal/state court rulings."""
    print(f"  Scraping Courthouse News...")
    news_items = []

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find article headlines
        articles = soup.select('article, .post, .story')

        for article in articles[:10]:
            title_tag = article.select_one('h2 a, h3 a, .headline a, .title a')
            if title_tag:
                title = title_tag.get_text(strip=True)
                href = title_tag.get('href', '')

                summary_tag = article.select_one('p, .excerpt, .summary')
                summary = summary_tag.get_text(strip=True)[:200] if summary_tag else ''

                if title:
                    news_items.append({
                        'title': title,
                        'url': href if href.startswith('http') else urljoin(url, href),
                        'summary': summary,
                        'source': 'Courthouse News',
                        'category': 'general_legal'
                    })

        return news_items[:5]

    except Exception as e:
        print(f"  Error scraping Courthouse News: {e}")
        return []


def scrape_consumersafety(url):
    """Scrape ConsumerSafety.org for product liability news."""
    print(f"  Scraping ConsumerSafety...")
    news_items = []

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find article headlines
        articles = soup.select('article, .post, .news-item')
        headlines = soup.select('h2 a, h3 a')

        for headline in headlines[:10]:
            title = headline.get_text(strip=True)
            href = headline.get('href', '')

            if title and href:
                news_items.append({
                    'title': title,
                    'url': href if href.startswith('http') else urljoin(url, href),
                    'summary': '',
                    'source': 'ConsumerSafety',
                    'category': 'mass_torts'
                })

        return news_items[:5]

    except Exception as e:
        print(f"  Error scraping ConsumerSafety: {e}")
        return []


def scrape_bloomberg(url):
    """Scrape Bloomberg Law Daily Labor Report."""
    print(f"  Scraping Bloomberg Law...")
    news_items = []

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        headlines = soup.select('article a, h2 a, h3 a, [class*="headline"] a')

        for headline in headlines[:10]:
            title = headline.get_text(strip=True)
            href = headline.get('href', '')

            if title and href and len(title) > 20:
                news_items.append({
                    'title': title,
                    'url': href if href.startswith('http') else urljoin('https://news.bloomberglaw.com', href),
                    'summary': '',
                    'source': 'Bloomberg Law',
                    'category': 'employment_law'
                })

        return news_items[:5]

    except Exception as e:
        print(f"  Error scraping Bloomberg: {e}")
        return []


def scrape_apnews(url):
    """Scrape AP News."""
    print(f"  Scraping AP News...")
    news_items = []

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        headlines = soup.select('a[class*="Link"], h2 a, h3 a, [data-key="card-headline"] a')

        for headline in headlines[:15]:
            title = headline.get_text(strip=True)
            href = headline.get('href', '')

            if title and href and len(title) > 20:
                news_items.append({
                    'title': title,
                    'url': href if href.startswith('http') else urljoin('https://apnews.com', href),
                    'summary': '',
                    'source': 'AP News',
                    'category': 'general_legal'
                })

        return news_items[:5]

    except Exception as e:
        print(f"  Error scraping AP News: {e}")
        return []


def scrape_cnn(url):
    """Scrape CNN US section."""
    print(f"  Scraping CNN US...")
    news_items = []

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        headlines = soup.select('a[data-link-type="article"], span.container__headline-text, h3 a')

        for headline in headlines[:10]:
            if headline.name == 'span':
                parent_link = headline.find_parent('a')
                if parent_link:
                    title = headline.get_text(strip=True)
                    href = parent_link.get('href', '')
                else:
                    continue
            else:
                title = headline.get_text(strip=True)
                href = headline.get('href', '')

            if title and href and len(title) > 15:
                news_items.append({
                    'title': title,
                    'url': href if href.startswith('http') else urljoin('https://www.cnn.com', href),
                    'summary': '',
                    'source': 'CNN',
                    'category': 'general_legal'
                })

        return news_items[:5]

    except Exception as e:
        print(f"  Error scraping CNN: {e}")
        return []


def scrape_nytimes(url):
    """Scrape NY Times US section."""
    print(f"  Scraping NY Times US...")
    news_items = []

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        headlines = soup.select('article h2 a, article h3 a, [class*="story"] a')

        for headline in headlines[:10]:
            title = headline.get_text(strip=True)
            href = headline.get('href', '')

            if title and href and len(title) > 15:
                news_items.append({
                    'title': title,
                    'url': href if href.startswith('http') else urljoin('https://www.nytimes.com', href),
                    'summary': '',
                    'source': 'NY Times',
                    'category': 'general_legal'
                })

        return news_items[:5]

    except Exception as e:
        print(f"  Error scraping NY Times: {e}")
        return []


def scrape_propublica(url):
    """Scrape ProPublica."""
    print(f"  Scraping ProPublica...")
    news_items = []

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        headlines = soup.select('article h2 a, article h3 a, [class*="hed"] a, .story-title a')

        for headline in headlines[:10]:
            title = headline.get_text(strip=True)
            href = headline.get('href', '')

            if title and href and len(title) > 15:
                news_items.append({
                    'title': title,
                    'url': href if href.startswith('http') else urljoin(url, href),
                    'summary': '',
                    'source': 'ProPublica',
                    'category': 'general_legal'
                })

        return news_items[:5]

    except Exception as e:
        print(f"  Error scraping ProPublica: {e}")
        return []


def scrape_onscenetv(url):
    """Scrape OnScene TV for breaking news and incidents."""
    print(f"  Scraping OnScene TV...")
    news_items = []

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # OnScene TV typically has article/post structures
        headlines = soup.select('article h2 a, article h3 a, .post-title a, .entry-title a, h2 a, h3 a')

        for headline in headlines[:10]:
            title = headline.get_text(strip=True)
            href = headline.get('href', '')

            if title and href and len(title) > 15:
                news_items.append({
                    'title': title,
                    'url': href if href.startswith('http') else urljoin(url, href),
                    'summary': '',
                    'source': 'OnScene TV',
                    'category': 'personal_injury'
                })

        return news_items[:5]

    except Exception as e:
        print(f"  Error scraping OnScene TV: {e}")
        return []


# Map scraper names to functions
SCRAPERS = {
    'aboutlawsuits': scrape_aboutlawsuits,
    'fda': scrape_fda,
    'eeoc': scrape_eeoc,
    'osha': scrape_osha,
    'courthousenews': scrape_courthousenews,
    'consumersafety': scrape_consumersafety,
    'bloomberg': scrape_bloomberg,
    'apnews': scrape_apnews,
    'cnn': scrape_cnn,
    'nytimes': scrape_nytimes,
    'propublica': scrape_propublica,
    'onscenetv': scrape_onscenetv
}


def scrape_all_sources():
    """Scrape all enabled news sources and return combined results."""
    print("Starting multi-source news scraping...")
    all_news = []

    for source in NEWS_SOURCES:
        if not source.get('enabled', False):
            continue

        scraper_func = SCRAPERS.get(source['scraper'])
        if scraper_func:
            try:
                items = scraper_func(source['url'])
                all_news.extend(items)
                print(f"  Found {len(items)} items from {source['name']}")

                # Be polite - small delay between requests
                time.sleep(1)

            except Exception as e:
                print(f"  Error with {source['name']}: {e}")

    print(f"\nTotal scraped: {len(all_news)} news items from all sources")
    return all_news


def get_existing_posts():
    """Fetch existing blog posts from Sanity for internal linking."""
    print("Fetching existing blog posts for internal linking...")

    if not SANITY_PROJECT_ID:
        print("Warning: SANITY_PROJECT_ID not set")
        return "No existing posts found."

    query = '*[_type == "blogPost"] | order(publishedAt desc) [0...20] {title, "slug": slug.current, excerpt}'
    encoded_query = requests.utils.quote(query)

    try:
        response = requests.get(
            f"{SANITY_QUERY_URL}?query={encoded_query}",
            headers={'Authorization': f"Bearer {SANITY_TOKEN}"} if SANITY_TOKEN else {},
            timeout=30
        )

        if response.status_code == 200:
            posts = response.json().get('result', [])
            if not posts:
                return "No existing posts found."

            link_database = "\n".join([
                f"- Title: {p.get('title', 'Untitled')}, Slug: {p.get('slug', '')}, Summary: {p.get('excerpt', 'No summary.')}"
                for p in posts
            ])
            print(f"Found {len(posts)} existing posts for internal linking")
            return link_database
        else:
            print(f"Error fetching posts: {response.status_code} - {response.text}")
            return "No existing posts found."
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return "No existing posts found."


def select_best_articles(news_items, num_articles=2):
    """
    Use Gemini AI to select the best articles for blog generation.
    Criteria: Balance of high visibility in today's news AND connection to
    casevalue.law's mission (personal injury, employment, mass tort law education).
    Returns a list of selected articles.
    """
    if not news_items:
        return []

    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY not set, using fallback selection")
        return news_items[:num_articles] if news_items else []

    print(f"Using Gemini to analyze {len(news_items)} articles for best {num_articles} selections...")

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Format all articles for Gemini analysis
    articles_list = []
    for i, item in enumerate(news_items):
        articles_list.append({
            'index': i,
            'title': item.get('title', ''),
            'source': item.get('source', ''),
            'category': item.get('category', ''),
            'summary': item.get('summary', '')[:200] if item.get('summary') else ''
        })

    articles_json = json.dumps(articles_list, indent=2)
    today_date = datetime.utcnow().strftime('%Y-%m-%d')

    prompt = f"""You are an editorial assistant for casevalue.law, a legal case evaluation website focused on educating potential clients about personal injury law, employment law, and mass tort litigation.

Today's date: {today_date}

Your task is to select the {num_articles} BEST articles from today's news to generate educational blog posts.

**MANDATORY FILTER - MUST BE FROM TODAY:**
Only consider articles that appear to be published TODAY ({today_date}). Reject articles that:
- Reference events from previous days/weeks/months
- Appear to be evergreen/undated content
- Mention specific past dates that are not today
If an article's headline suggests it's breaking news or a current development, it's likely from today.

**SELECTION CRITERIA (both must be considered):**

1. **News Visibility/Impact**: Prioritize stories that are making headlines today, have broad public interest, or involve significant developments that many people are talking about.

2. **Mission Alignment**: The article must have a clear connection to at least one of these areas:
   - Personal injury (accidents, product defects, medical injuries, workplace injuries)
   - Employment law (discrimination, wrongful termination, wage disputes, workplace rights)
   - Mass torts (class actions, product liability affecting many people, pharmaceutical injuries, environmental harm)

3. **Diversity**: Try to select articles from different topics/categories to provide variety in blog content.

**ARTICLES TO ANALYZE:**
{articles_json}

**OUTPUT REQUIREMENTS:**
Return ONLY a JSON object with:
- "selected_indices": An array of {num_articles} index numbers of the best articles (integers), ordered by preference. Use an empty array [] if no suitable articles from today were found.
- "reasoning": Brief explanation (1-2 sentences) of why these articles were selected

Example output:
{{"selected_indices": [3, 7], "reasoning": "Article 3 is an FDA recall with high visibility; Article 7 covers a major employment discrimination case."}}

IMPORTANT: Return ONLY the JSON object, no additional text. Return empty array for selected_indices if no articles appear to be from today or meet the criteria."""

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={
                'response_mime_type': 'application/json'
            }
        )

        result = json.loads(response.text.strip())
        selected_indices = result.get('selected_indices', [])
        reasoning = result.get('reasoning', 'No reasoning provided')

        if not selected_indices:
            print("Gemini found no suitable articles from today matching the criteria.")
            print(f"Reasoning: {reasoning}")
            return []

        selected_articles = []
        for idx in selected_indices:
            if 0 <= idx < len(news_items):
                selected = news_items[idx]
                print(f"Gemini selected article #{idx}: {selected.get('title', '')[:50]}...")
                selected_articles.append(selected)
            else:
                print(f"Invalid index {idx}, skipping")

        print(f"Reasoning: {reasoning}")
        return selected_articles

    except Exception as e:
        print(f"Error in Gemini selection: {e}")
        print("No articles selected due to error")
        return []


def generate_article(news_item, all_news, link_database):
    """Generate a blog article using Gemini AI."""
    print(f"Generating article based on: {news_item['title'][:60]}...")

    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not set")
        return None

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Format the primary article
    primary_article = json.dumps(news_item, indent=2)

    # Format related articles for context
    related_articles = [item for item in all_news if item['url'] != news_item['url']][:5]
    related_context = json.dumps(related_articles, indent=2)

    prompt = f"""
You are a senior SEO content writer for a case evaluation website with a purpose of driving people to use the in website calculator for their legal matter.

Your task is to write an informative, engaging blog post based on the following news and a list of existing articles for internal linking.

**--- 1. PRIMARY NEWS ARTICLE ---**
{primary_article}

**--- 2. RELATED NEWS (for additional context) ---**
{related_context}

**--- 3. INTERNAL LINK DATABASE (ONLY use slugs from this list for links) ---**
{link_database}

**--- 4. OUTPUT REQUIREMENTS ---**

1. **slug**: URL-friendly version of the title in kebab-case (e.g., "understanding-texas-statute-of-limitations")

2. **excerpt**: Short summary for post listings and SEO. MAXIMUM 160 characters. Do NOT exceed 160.

3. **body_markdown**: The main article content with these requirements:
   - Exactly 4-5 paragraphs
   - Each paragraph MUST have its own H2 (##) title/heading
   - Optimized for SEO for a case evaluation website
   - Professional, informative tone empathetic to potential injury victims
   - Subtly position readers to consider legal help/case evaluation
   - Include relevant keywords naturally throughout
   - INTERNAL LINKS: ONLY link to slugs listed in the INTERNAL LINK DATABASE above. Use format: [Anchor Text](https://casevalue.law/blog/exact-slug-from-database). The anchor text should NOT be the full title of the linked article - instead use natural, contextual phrases that flow with the surrounding sentence. Example: "Learn more about [documenting your losses](https://casevalue.law/blog/lost-wages-article-slug) for your claim." Do NOT invent URLs. Do NOT use external links. If no relevant slug exists in the database, include NO links at all.
   - End with a call-to-action about getting a free case evaluation

4. **meta_title**: Title optimized for search engines. MAXIMUM 60 characters. Do NOT exceed 60.

5. **alt_text**: Alternative text for a featured image. Important for SEO and accessibility. Describe what an appropriate image would show related to the article topic.

6. **meta_description**: Description for search engines. MAXIMUM 160 characters. Do NOT exceed 160.

7. **keywords**: Array of 5-7 SEO keywords relevant to the article (e.g., "texas personal injury", "statute of limitations", "personal injury lawyer")

8. **categories**: Array of 1-3 categories from this list ONLY:
   - "personal-injury"
   - "medical-malpractice"
   - "motor-vehicle"
   - "wrongful-death"
   - "dog-bites"
   - "premises-liability"
   - "product-liability"
   - "employment-law"
   - "civil-rights"
   - "texas-law"
   - "legal-tips"
   - "case-studies"

9. **title**: A compelling, SEO-friendly headline for the blog post.

**--- JSON SCHEMA ---**
{{
    "title": "Compelling SEO-friendly headline for the blog post",
    "slug": "url-friendly-slug-here",
    "excerpt": "Max 160 chars. Count before submitting.",
    "body_markdown": "## First Paragraph Title\\n\\nFirst paragraph content...\\n\\n## Second Paragraph Title\\n\\nSecond paragraph content...",
    "meta_title": "Max 60 chars. Count before submitting.",
    "alt_text": "Descriptive alt text for featured image",
    "meta_description": "Max 160 chars. Count before submitting.",
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
    "categories": ["personal-injury", "texas-law"]
}}

CRITICAL RULES:
- Return ONLY the JSON object, no additional text or markdown code fences.
- HARD CHARACTER LIMITS - NEVER EXCEED: excerpt (max 160), meta_title (max 60), meta_description (max 160). Count each character before output. If over limit, shorten the text.
- LINKS: ONLY use slugs from the INTERNAL LINK DATABASE. If the database shows "Slug: example-slug", use [text](https://casevalue.law/blog/example-slug). NO invented links. If no matching slug exists, use NO links.
- Body must have exactly 4-5 paragraphs, each with its own ## heading.
"""

    max_retries = 2
    for attempt in range(max_retries):
        try:
            # Use JSON response mode for reliable parsing
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config={
                    'response_mime_type': 'application/json'
                }
            )

            json_string = response.text.strip()

            # Remove markdown code fences if present (shouldn't be needed with JSON mode)
            if json_string.startswith('```'):
                json_string = re.sub(r'^```(?:json)?\s*\n?', '', json_string)
                json_string = re.sub(r'\n?\s*```\s*$', '', json_string)

            article_data = json.loads(json_string)

            # Enforce character limits programmatically (Gemini doesn't always respect them)
            if article_data.get('excerpt') and len(article_data['excerpt']) > 160:
                article_data['excerpt'] = article_data['excerpt'][:157] + '...'
                print(f"  Truncated excerpt to 160 chars")
            if article_data.get('meta_title') and len(article_data['meta_title']) > 60:
                article_data['meta_title'] = article_data['meta_title'][:57] + '...'
                print(f"  Truncated meta_title to 60 chars")
            if article_data.get('meta_description') and len(article_data['meta_description']) > 160:
                article_data['meta_description'] = article_data['meta_description'][:157] + '...'
                print(f"  Truncated meta_description to 160 chars")

            print(f"Generated article: {article_data.get('title', 'Untitled')}")
            return article_data

        except json.JSONDecodeError as e:
            print(f"Attempt {attempt + 1}: JSON decode error: {e}")
            if attempt < max_retries - 1:
                print("Retrying article generation...")
                time.sleep(2)
            else:
                print(f"Raw AI output: {response.text[:500]}...")
                return None
        except Exception as e:
            print(f"Error generating content: {e}")
            return None

    return None


def post_to_sanity(article_data):
    """Post the generated article to Sanity.io CMS."""
    print(f"Posting article to Sanity: {article_data.get('title', 'Untitled')}")

    if not all([SANITY_PROJECT_ID, SANITY_TOKEN, SANITY_DATASET]):
        print("Error: Missing Sanity configuration (PROJECT_ID, TOKEN, or DATASET)")
        return False

    portable_body = convert_markdown_to_portable_text(article_data.get('body_markdown', ''))

    # Image generation using Gemini 2.5 Flash Image model
    # To disable: set ENABLE_IMAGE_GENERATION=false in environment
    alt_text = article_data.get('alt_text', '')
    main_image = None
    enable_image_gen = os.environ.get('ENABLE_IMAGE_GENERATION', 'true').lower() == 'true'

    if alt_text and enable_image_gen:
        print("\n--- Generating Featured Image ---")
        image_bytes = generate_image_with_gemini(alt_text)

        if image_bytes:
            slug = article_data.get('slug', 'blog-image')
            filename = f"{slug}.png"
            image_asset = upload_image_to_sanity(image_bytes, filename)

            if image_asset:
                main_image = image_asset
                main_image['alt'] = alt_text
                print(f"Featured image ready with alt text")
            else:
                print("Warning: Image upload failed, continuing without image")
        else:
            print("Warning: Image generation failed, continuing without image")

    # Document structure matching blogPost schema exactly
    document = {
        "_type": "blogPost",
        "title": article_data.get('title', ''),
        "slug": {
            "_type": "slug",
            "current": article_data.get('slug', '')
        },
        "author": DEFAULT_AUTHOR,
        "publishedAt": datetime.utcnow().isoformat() + "Z",
        "excerpt": article_data.get('excerpt', ''),
        "categories": article_data.get('categories', ['personal-injury']),
        "body": portable_body,
        "seo": {
            "metaTitle": article_data.get('meta_title', ''),
            "metaDescription": article_data.get('meta_description', ''),
            "keywords": article_data.get('keywords', [])
        },
        "featured": False
    }

    # Add mainImage if we have one
    if main_image:
        document["mainImage"] = main_image

    payload = {
        "mutations": [
            {"create": document}
        ]
    }

    try:
        response = requests.post(
            SANITY_BASE_URL,
            headers=HEADERS,
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            print("SUCCESS: Article published to Sanity.io!")
            print(f"Document ID: {result.get('results', [{}])[0].get('id', 'unknown')}")
            return True
        else:
            print(f"FAILURE: Sanity API Error: {response.status_code}")
            print(response.text)
            return False

    except requests.RequestException as e:
        print(f"Request error: {e}")
        return False


def main():
    """Main execution flow."""
    print("=" * 60)
    print("  Legal News Blog Post Generator")
    print("  Personal Injury | Employment Law | Mass Torts")
    print(f"  Started at: {datetime.utcnow().isoformat()}")
    print("=" * 60)

    # Validate configuration
    missing_config = []
    if not GEMINI_API_KEY:
        missing_config.append("GEMINI_API_KEY")
    if not SANITY_PROJECT_ID:
        missing_config.append("SANITY_PROJECT_ID")
    if not SANITY_TOKEN:
        missing_config.append("SANITY_TOKEN")

    if missing_config:
        print(f"\nWarning: Missing environment variables: {', '.join(missing_config)}")
        print("Some features may not work correctly.")

    # Step 1: Scrape All News Sources
    print("\n--- Step 1: Scraping News Sources ---")
    all_news = scrape_all_sources()

    if not all_news:
        print("No articles scraped from any source. Exiting.")
        return

    # Step 2: Select Best Articles (must be from today and match mission criteria)
    print("\n--- Step 2: Selecting Articles (filtering for today's news) ---")
    selected_articles = select_best_articles(all_news, num_articles=2)

    if not selected_articles:
        print("No suitable articles from today found that match our criteria. Exiting.")
        print("The script will try again tomorrow when fresh news is available.")
        return

    print(f"\nSelected {len(selected_articles)} articles for blog generation:")
    for i, article in enumerate(selected_articles, 1):
        print(f"  {i}. {article['title'][:60]}...")
        print(f"     Source: {article['source']} | Category: {article['category']}")

    # Step 3: Get Internal Links
    print("\n--- Step 3: Fetching Internal Links ---")
    link_database = get_existing_posts()

    # Step 4 & 5: Generate and Post Each Article
    success_count = 0
    fail_count = 0

    for i, selected_article in enumerate(selected_articles, 1):
        print(f"\n{'='*60}")
        print(f"  Processing Article {i} of {len(selected_articles)}")
        print(f"{'='*60}")

        # Step 4: Generate Article
        print(f"\n--- Step 4.{i}: Generating Article ---")
        generated_article = generate_article(selected_article, all_news, link_database)

        if not generated_article:
            print(f"Article {i} generation failed. Skipping.")
            fail_count += 1
            continue

        # Display generated content summary
        print(f"\n  Generated Content:")
        print(f"  - Title: {generated_article.get('title', 'N/A')}")
        print(f"  - Slug: {generated_article.get('slug', 'N/A')}")
        print(f"  - Meta Title ({len(generated_article.get('meta_title', ''))} chars): {generated_article.get('meta_title', 'N/A')}")
        print(f"  - Excerpt ({len(generated_article.get('excerpt', ''))} chars): {generated_article.get('excerpt', 'N/A')[:80]}...")
        print(f"  - Categories: {', '.join(generated_article.get('categories', []))}")
        print(f"  - Keywords: {', '.join(generated_article.get('keywords', []))}")
        print(f"  - Alt Text: {generated_article.get('alt_text', 'N/A')[:60]}...")

        # Step 5: Post to Sanity
        print(f"\n--- Step 5.{i}: Posting to Sanity ---")
        success = post_to_sanity(generated_article)

        if success:
            success_count += 1
        else:
            fail_count += 1

        # Small delay between posts
        if i < len(selected_articles):
            time.sleep(2)

    print("\n" + "=" * 60)
    print(f"  COMPLETE: {success_count} blog post(s) published successfully!")
    if fail_count > 0:
        print(f"  FAILED: {fail_count} blog post(s) could not be published.")
    print("=" * 60)


if __name__ == "__main__":
    main()
