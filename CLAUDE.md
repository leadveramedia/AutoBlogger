# AutoBlogger - Automated Legal News Blog Generator

## Project Overview

AutoBlogger is an automated content generation system for casevalue.law, a legal case evaluation website. It scrapes legal news from multiple sources, uses AI to select relevant articles, generates SEO-optimized blog posts, and publishes them to Sanity CMS.

**Target Practice Areas:**
- Personal Injury, Medical Malpractice, Motor Vehicle Accidents
- Wrongful Death, Dog Bites, Premises Liability
- Product Liability, Employment Law, Civil Rights
- Worker's Compensation, Social Security Disability
- Intellectual Property, Professional Malpractice, Class Action

## Architecture

```
run.py                 # Entry point - orchestrates the full pipeline
auto_post/
├── __init__.py        # Package exports
├── config.py          # Configuration, API keys, news sources, keywords
├── scrapers.py        # News scraper functions for each source
├── content.py         # AI content generation (Gemini) and image generation (Imagen)
├── sanity.py          # Sanity CMS API integration
├── curation.py        # Source health tracking and auto-curation
└── utils.py           # Utility functions (title list, topic tracking)
```

## Key Files

### `run.py`
Main execution flow:
1. Scrape news from all enabled sources
2. Run source health curation (disable failing sources)
3. Select best articles using AI (filters for today's news, avoids duplicates)
4. Generate blog posts with AI
5. Post to Sanity CMS
6. Generate one article from pre-defined title list

### `auto_post/scrapers.py`
Contains individual scraper functions for each news source:
- Specialized sources: AboutLawsuits, FDA, EEOC, OSHA, ConsumerSafety, Bloomberg, NHTSA, DOL, Insurance Journal
- General news (filtered by keywords): AP News, CNN, NY Times, ProPublica, Courthouse News

**Important:** General news sources are filtered using `PRACTICE_AREA_KEYWORDS` to only keep relevant articles.

### `auto_post/content.py`
AI-powered content generation:
- `select_best_articles()` - Uses Gemini to pick best articles for blog posts
- `generate_article()` - Generates full blog post from news article
- `generate_article_from_title()` - Generates evergreen content from pre-defined titles
- `generate_image_with_gemini()` - Generates featured images using Imagen 4
- `detect_text_in_image()` - Validates generated images have no text (auto-retries if text detected)

### `auto_post/config.py`
Configuration constants:
- `NEWS_SOURCES` - List of news sources with URLs and scraper mappings
- `PRACTICE_AREA_KEYWORDS` - Keywords for filtering general news
- `VALID_CATEGORIES` - Blog post categories

### `auto_post/sanity.py`
Sanity CMS integration:
- `get_existing_posts()` - Fetches existing posts for internal linking
- `post_to_sanity()` - Creates new blog posts with images

## Environment Variables

**Required:**
```
GEMINI_API_KEY        # Google Gemini API key (for AI content generation)
SANITY_PROJECT_ID     # Sanity.io project ID
SANITY_TOKEN          # Sanity.io API token
```

**Optional:**
```
SANITY_DATASET        # Sanity dataset (default: production)
DEFAULT_AUTHOR        # Author name (default: Case Value Expert)
ENABLE_IMAGE_GENERATION  # Set to 'false' to disable images
```

## Data Files

- `titles.json` - Queue of pre-defined article titles to generate
- `used_topics.json` - Tracks published topics to avoid duplicates
- `source_health.json` - Tracks scraper success/failure counts
- `curation.log` - Log of source curation actions

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GEMINI_API_KEY="your-key"
export SANITY_PROJECT_ID="your-project"
export SANITY_TOKEN="your-token"

# Run the generator
python run.py
```

## Common Patterns

### Adding a New News Source
1. Add scraper function in `scrapers.py` (e.g., `scrape_newsite()`)
2. Add to `SCRAPERS` dict in `scrapers.py`
3. Add source config to `NEWS_SOURCES` in `config.py`
4. If general news, add scraper name to `GENERAL_NEWS_SCRAPERS` for keyword filtering

### Image Generation
Images are generated with Imagen 4 and validated to ensure no text appears:
- Prompt explicitly forbids text, signs, documents, etc.
- `detect_text_in_image()` uses Gemini vision to check for text
- Auto-retries up to 3 times if text is detected

### Article Generation Prompt Structure
Articles have 6 required sections:
1. What Happened (news summary with source link)
2. Legal Liability (lawyer's perspective)
3. Steps to Take (actionable advice)
4. Compensation (settlement ranges)
5. Legal Framework (applicable laws)
6. Get Help (call-to-action)

## Important Notes

- The system filters for **today's news only** to keep content timely
- `used_topics.json` prevents duplicate articles about the same news story
- Source health tracking auto-disables scrapers that fail repeatedly
- Internal links are pulled from existing Sanity posts for SEO
