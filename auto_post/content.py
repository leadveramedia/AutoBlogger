"""
Content generation using Gemini AI.
Includes article generation and image generation.
"""

import re
import json
import time
from datetime import datetime, timezone

from google import genai

from .config import GEMINI_API_KEY


def sanitize_json_control_chars(s):
    """
    Fix control characters inside JSON string values.
    Gemini sometimes outputs literal newlines/tabs inside JSON strings
    which breaks json.loads(). This escapes them properly.
    """
    result = []
    in_string = False
    escape_next = False
    for char in s:
        if escape_next:
            result.append(char)
            escape_next = False
            continue
        if char == '\\':
            result.append(char)
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            result.append(char)
            continue
        if in_string:
            if char == '\n':
                result.append('\\n')
            elif char == '\r':
                result.append('\\r')
            elif char == '\t':
                result.append('\\t')
            elif ord(char) < 32:
                result.append(f'\\u{ord(char):04x}')
            else:
                result.append(char)
        else:
            result.append(char)
    return ''.join(result)


def extract_json_fields_fallback(raw_text):
    """
    Fallback JSON extraction using regex when json.loads fails.
    Extracts key fields from malformed JSON response.
    """
    import re

    result = {}

    # Extract title
    title_match = re.search(r'"title"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', raw_text)
    if title_match:
        result['title'] = title_match.group(1).replace('\\"', '"').replace('\\n', '\n')

    # Extract slug
    slug_match = re.search(r'"slug"\s*:\s*"([^"]*)"', raw_text)
    if slug_match:
        result['slug'] = slug_match.group(1)

    # Extract excerpt
    excerpt_match = re.search(r'"excerpt"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', raw_text)
    if excerpt_match:
        result['excerpt'] = excerpt_match.group(1).replace('\\"', '"').replace('\\n', ' ')[:160]

    # Extract body_markdown - find from "body_markdown": " to the next top-level key
    body_match = re.search(r'"body_markdown"\s*:\s*"(.*?)(?:"\s*,\s*"(?:meta_title|alt_text|meta_description|keywords|categories)")', raw_text, re.DOTALL)
    if body_match:
        body = body_match.group(1)
        # Unescape common sequences
        body = body.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
        result['body_markdown'] = body

    # Extract meta_title
    meta_title_match = re.search(r'"meta_title"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', raw_text)
    if meta_title_match:
        result['meta_title'] = meta_title_match.group(1).replace('\\"', '"')[:60]

    # Extract alt_text
    alt_match = re.search(r'"alt_text"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', raw_text)
    if alt_match:
        result['alt_text'] = alt_match.group(1).replace('\\"', '"')

    # Extract meta_description
    meta_desc_match = re.search(r'"meta_description"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', raw_text)
    if meta_desc_match:
        result['meta_description'] = meta_desc_match.group(1).replace('\\"', '"')[:160]

    # Extract keywords array
    keywords_match = re.search(r'"keywords"\s*:\s*\[([^\]]*)\]', raw_text)
    if keywords_match:
        keywords_str = keywords_match.group(1)
        keywords = re.findall(r'"([^"]*)"', keywords_str)
        result['keywords'] = keywords if keywords else ['personal injury', 'legal news']

    # Extract categories array
    categories_match = re.search(r'"categories"\s*:\s*\[([^\]]*)\]', raw_text)
    if categories_match:
        categories_str = categories_match.group(1)
        categories = re.findall(r'"([^"]*)"', categories_str)
        result['categories'] = categories if categories else ['legal-tips']

    return result if result.get('title') and result.get('body_markdown') else None


def validate_article_data(article_data):
    """
    Validate and fix article data - enforce limits and add fallbacks.
    """
    # Enforce character limits
    if article_data.get('excerpt') and len(article_data['excerpt']) > 160:
        article_data['excerpt'] = article_data['excerpt'][:157] + '...'
        print(f"  Truncated excerpt to 160 chars")
    if article_data.get('meta_title') and len(article_data['meta_title']) > 60:
        article_data['meta_title'] = article_data['meta_title'][:57] + '...'
        print(f"  Truncated meta_title to 60 chars")
    if article_data.get('meta_description') and len(article_data['meta_description']) > 160:
        article_data['meta_description'] = article_data['meta_description'][:157] + '...'
        print(f"  Truncated meta_description to 160 chars")

    # Fallback for missing required fields
    if not article_data.get('alt_text'):
        title = article_data.get('title', 'Legal news article')
        article_data['alt_text'] = f"Professional image representing {title[:80]}"
        print(f"  Generated fallback alt_text")
    if not article_data.get('meta_title'):
        article_data['meta_title'] = article_data.get('title', 'Legal News')[:57] + '...'
        print(f"  Generated fallback meta_title")
    if not article_data.get('keywords'):
        article_data['keywords'] = ['personal injury', 'legal news', 'case evaluation']
        print(f"  Generated fallback keywords")
    if not article_data.get('categories'):
        article_data['categories'] = ['legal-tips']
        print(f"  Generated fallback categories")

    return article_data


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

        image_prompt = f"""Generate a professional stock photo for a law firm blog article.

Subject: {alt_text}

Requirements:
- Professional, clean, modern aesthetic
- Trustworthy and empathetic tone
- No text, watermarks, or logos
- Suitable for a personal injury law firm website
- High quality, photorealistic style"""

        response = client.models.generate_images(
            model='imagen-4.0-generate-001',
            prompt=image_prompt,
            config={
                'number_of_images': 1,
            }
        )

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
        return None


def select_best_articles(news_items, num_articles=2, used_topics=None):
    """
    Use Gemini AI to select the best articles for blog generation.
    Filters out articles about topics we've already covered.
    """
    if not news_items:
        return []

    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY not set, using fallback selection")
        return news_items[:num_articles] if news_items else []

    print(f"Using Gemini to analyze {len(news_items)} articles for best {num_articles} selections...")

    client = genai.Client(api_key=GEMINI_API_KEY)

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
    today_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    # Format used topics for the prompt
    used_topics_list = used_topics if used_topics else []
    used_topics_text = ""
    if used_topics_list:
        used_topics_text = f"""
**CRITICAL - TOPICS ALREADY COVERED (DO NOT SELECT):**
We have already published blog posts about these topics. Do NOT select any article that covers the same story, event, case, or subject matter - even if from a different source or with a different angle:

{chr(10).join('- ' + topic for topic in used_topics_list[-100:])}

For example, if we already covered "Columbia University $21M antisemitism settlement", reject ANY article about that settlement regardless of the source or headline.
"""

    prompt = f"""You are an editorial assistant for casevalue.law, a legal case evaluation website focused on educating potential clients about personal injury law, employment law, and mass tort litigation.

Today's date: {today_date}

Your task is to select the {num_articles} BEST articles from today's news to generate educational blog posts.
{used_topics_text}
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

4. **Novelty**: Do NOT select articles about topics we've already covered (see list above if provided).

**ARTICLES TO ANALYZE:**
{articles_json}

**OUTPUT REQUIREMENTS:**
Return ONLY a JSON object with:
- "selected_indices": An array of {num_articles} index numbers of the best articles (integers), ordered by preference. Use an empty array [] if no suitable NEW articles from today were found.
- "topic_summaries": An array of brief topic descriptions (10-15 words each) for each selected article, describing the specific case/event/story. Example: "Columbia University $21M antisemitism settlement EEOC case"
- "reasoning": Brief explanation (1-2 sentences) of why these articles were selected

Example output:
{{"selected_indices": [3, 7], "topic_summaries": ["FDA recall of contaminated infant formula by Abbott", "Amazon warehouse OSHA violations $1.5M fine"], "reasoning": "Article 3 is an FDA recall with high visibility; Article 7 covers a major workplace safety case."}}

IMPORTANT: Return ONLY the JSON object, no additional text. Return empty array for selected_indices if no articles appear to be from today, meet the criteria, or if all suitable articles cover topics we've already blogged about."""

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
        topic_summaries = result.get('topic_summaries', [])
        reasoning = result.get('reasoning', 'No reasoning provided')

        if not selected_indices:
            print("Gemini found no suitable NEW articles from today matching the criteria.")
            print(f"Reasoning: {reasoning}")
            return []

        selected_articles = []
        for i, idx in enumerate(selected_indices):
            if 0 <= idx < len(news_items):
                selected = news_items[idx]
                # Attach topic summary to the article for later tracking
                if i < len(topic_summaries):
                    selected['topic_summary'] = topic_summaries[i]
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

    primary_article = json.dumps(news_item, indent=2)
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

3. **body_markdown**: The main article content with EXACTLY 6 sections. Each section MUST have:
   - A DYNAMIC, ENGAGING H2 (##) heading that describes the content (NOT generic titles like "News Summary" or "Liability Analysis")
   - A FULL PARAGRAPH (4-6 sentences minimum)

   **SECTION 1 - What Happened:** (one full paragraph)
   Create an engaging H2 title that captures the news event (e.g., "## Florida Files Discrimination Suit Against Starbucks" or "## Gas Explosion Rocks Bay Area Neighborhood").
   Summarize what happened with brief editorialization and context. Include ONE natural link to the original news source using anchor text like "according to [recent reports](source-url)".

   **SECTION 2 - Legal Liability:** (one full paragraph)
   Create an H2 title about fault/liability (e.g., "## Who Bears Responsibility for the Explosion?" or "## Breaking Down the Discrimination Claims").
   Provide the lawyer's point of view on liabilities. Discuss who may be at fault and why, and explain legal theories that could apply.

   **SECTION 3 - Steps to Take:** (one full paragraph)
   Create an H2 title about taking action (e.g., "## Protecting Your Rights After a Similar Incident" or "## What to Do If You Face Workplace Discrimination").
   Practical steps someone should take if they experience this type of event. If relevant, link to existing articles using format [Anchor Text](https://casevalue.law/blog/exact-slug-from-database).

   **SECTION 4 - Compensation:** (one full paragraph)
   Create an H2 title about recovery (e.g., "## What Victims Could Recover in Damages" or "## Understanding Settlement Ranges").
   Discuss common recovery and settlement ranges, factors affecting amounts, and types of damages recoverable.

   **SECTION 5 - Legal Framework:** (one full paragraph)
   Create an H2 title about applicable laws (e.g., "## Federal Laws Protecting Workers from Discrimination" or "## Premises Liability Laws in California").
   Cover relevant laws, regulations, statutes of limitations, and state-specific considerations.

   **SECTION 6 - Get Help:** (one full paragraph)
   Create an H2 title encouraging action (e.g., "## Find Out What Your Case Is Worth" or "## Take the First Step Toward Justice").
   Strong call-to-action directing readers to use our free case evaluator.

   **Additional Requirements:**
   - H2 headings must be DYNAMIC and SPECIFIC to the article content - never use generic titles
   - Each section MUST be a FULL PARAGRAPH (4-6 sentences minimum)
   - Optimized for SEO for a case evaluation website
   - Professional, informative tone empathetic to potential injury victims
   - Include relevant keywords naturally throughout

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
    "body_markdown": "## Florida Files Landmark Discrimination Suit\\n\\nFull paragraph about what happened...\\n\\n## Breaking Down the Legal Claims Against Starbucks\\n\\nFull paragraph on liability...\\n\\n## Protecting Your Rights in the Workplace\\n\\nFull paragraph on steps to take...\\n\\n## What Discrimination Victims Could Recover\\n\\nFull paragraph on settlements...\\n\\n## Federal and State Employment Laws at Play\\n\\nFull paragraph on laws...\\n\\n## Find Out What Your Case Is Worth\\n\\nFull paragraph call to action...",
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
- Body must have EXACTLY 6 sections, each with a DYNAMIC ## heading specific to the article content. DO NOT use generic headings like "News Summary" or "Liability Analysis".
"""

    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config={
                    'response_mime_type': 'application/json'
                }
            )

            json_string = response.text.strip()

            if json_string.startswith('```'):
                json_string = re.sub(r'^```(?:json)?\s*\n?', '', json_string)
                json_string = re.sub(r'\n?\s*```\s*$', '', json_string)

            # Sanitize control characters in JSON string values
            json_string = sanitize_json_control_chars(json_string)
            article_data = json.loads(json_string)

            # Validate and add fallbacks
            article_data = validate_article_data(article_data)

            print(f"Generated article: {article_data.get('title', 'Untitled')}")
            return article_data

        except json.JSONDecodeError as e:
            print(f"Attempt {attempt + 1}: JSON decode error: {e}")
            if attempt < max_retries - 1:
                print("Retrying article generation...")
                time.sleep(2)
            else:
                # Try fallback regex extraction before giving up
                print(f"Attempting fallback JSON extraction...")
                fallback_data = extract_json_fields_fallback(response.text)
                if fallback_data:
                    print(f"Fallback extraction successful!")
                    # Validate and add fallbacks
                    fallback_data = validate_article_data(fallback_data)
                    return fallback_data
                print(f"Fallback failed. Raw AI output: {response.text[:500]}...")
                return None
        except Exception as e:
            print(f"Error generating content: {e}")
            return None

    return None


def generate_article_from_title(title, link_database):
    """Generate a comprehensive blog article from a pre-defined title."""
    print(f"Generating article from title: {title[:60]}...")

    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not set")
        return None

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""
You are a senior SEO content writer for casevalue.law, a case evaluation website that helps people understand the value of their legal claims.

Your task is to write a comprehensive, authoritative blog post for the following title:

**TITLE:** {title}

**--- INTERNAL LINK DATABASE (ONLY use slugs from this list for links) ---**
{link_database}

**--- OUTPUT REQUIREMENTS ---**

1. **slug**: URL-friendly version of the title in kebab-case (e.g., "car-accident-settlements-guide-2025")

2. **excerpt**: Short summary for post listings and SEO. MAXIMUM 160 characters. Do NOT exceed 160.

3. **body_markdown**: The main article content with these requirements:
   - MINIMUM 2500 WORDS - This is a comprehensive, authoritative guide
   - 10-15 major sections, each with its own H2 (##) heading
   - Use H3 (###) for subsections within each major section
   - Include detailed explanations, examples, and specific information
   - Use bullet points, numbered lists, and tables where helpful
   - Cover the topic THOROUGHLY - anticipate and answer all reader questions
   - Include statistics, typical ranges, and real-world examples where applicable
   - Professional, informative tone empathetic to potential injury victims
   - Include specific, actionable information that demonstrates legal expertise
   - Explain legal concepts clearly for non-lawyers while maintaining accuracy
   - Address common misconceptions and frequently asked questions
   - Discuss state-by-state variations where relevant
   - Include practical tips and step-by-step guidance
   - Naturally position readers to consider getting a free case evaluation
   - Include relevant keywords naturally throughout
   - INTERNAL LINKS: ONLY link to slugs listed in the INTERNAL LINK DATABASE above. Use format: [Anchor Text](https://casevalue.law/blog/exact-slug-from-database). The anchor text should NOT be the full title of the linked article - instead use natural, contextual phrases that flow with the surrounding sentence. Do NOT invent URLs. If no relevant slug exists in the database, include NO internal links.
   - End with a strong call-to-action about getting a free case evaluation

4. **meta_title**: Title optimized for search engines. MAXIMUM 60 characters. Do NOT exceed 60.

5. **alt_text**: Alternative text for a featured image. Important for SEO and accessibility. Describe what an appropriate image would show related to the article topic.

6. **meta_description**: Description for search engines. MAXIMUM 160 characters. Do NOT exceed 160.

7. **keywords**: Array of 5-7 SEO keywords relevant to the article

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

9. **title**: Use the exact title provided above.

**--- JSON SCHEMA ---**
{{
    "title": "{title}",
    "slug": "url-friendly-slug-here",
    "excerpt": "Max 160 chars. Count before submitting.",
    "body_markdown": "## First Section Title\\n\\nFirst section content...\\n\\n## Second Section Title\\n\\nSecond section content...",
    "meta_title": "Max 60 chars. Count before submitting.",
    "alt_text": "Descriptive alt text for featured image",
    "meta_description": "Max 160 chars. Count before submitting.",
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
    "categories": ["personal-injury", "legal-tips"]
}}

CRITICAL RULES:
- Return ONLY the JSON object, no additional text or markdown code fences.
- HARD CHARACTER LIMITS - NEVER EXCEED: excerpt (max 160), meta_title (max 60), meta_description (max 160). Count each character before output.
- MINIMUM WORD COUNT: body_markdown MUST contain at least 2500 words. This is a comprehensive legal guide.
- This is an EVERGREEN educational guide - do not reference current events or dates unless essential.
- Body must have 10-15 sections with ## headings for an authoritative, comprehensive, in-depth guide.
- Each section should be detailed and informative - aim for 150-250 words per section minimum.
"""

    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config={
                    'response_mime_type': 'application/json'
                }
            )

            json_string = response.text.strip()

            if json_string.startswith('```'):
                json_string = re.sub(r'^```(?:json)?\s*\n?', '', json_string)
                json_string = re.sub(r'\n?\s*```\s*$', '', json_string)

            # Sanitize control characters in JSON string values
            json_string = sanitize_json_control_chars(json_string)
            article_data = json.loads(json_string)

            # Validate and add fallbacks
            article_data = validate_article_data(article_data)

            print(f"Generated article: {article_data.get('title', 'Untitled')}")
            return article_data

        except json.JSONDecodeError as e:
            print(f"Attempt {attempt + 1}: JSON decode error: {e}")
            if attempt < max_retries - 1:
                print("Retrying article generation...")
                time.sleep(2)
            else:
                # Try fallback regex extraction before giving up
                print(f"Attempting fallback JSON extraction...")
                fallback_data = extract_json_fields_fallback(response.text)
                if fallback_data:
                    print(f"Fallback extraction successful!")
                    # Validate and add fallbacks
                    fallback_data = validate_article_data(fallback_data)
                    return fallback_data
                print(f"Fallback failed. Raw AI output: {response.text[:500]}...")
                return None
        except Exception as e:
            print(f"Error generating content: {e}")
            return None

    return None
