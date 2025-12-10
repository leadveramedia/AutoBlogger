"""
Configuration and constants for the auto_post package.
"""

import os

# --- SANITY CONFIGURATION ---
SANITY_PROJECT_ID = os.environ.get('SANITY_PROJECT_ID', '')
SANITY_DATASET = os.environ.get('SANITY_DATASET', 'production')
SANITY_TOKEN = os.environ.get('SANITY_TOKEN', '')
DEFAULT_AUTHOR = os.environ.get('DEFAULT_AUTHOR', 'Case Value Expert')

SANITY_BASE_URL = f"https://{SANITY_PROJECT_ID}.api.sanity.io/v2021-06-07/data/mutate/{SANITY_DATASET}"
SANITY_QUERY_URL = f"https://{SANITY_PROJECT_ID}.api.sanity.io/v2021-06-07/data/query/{SANITY_DATASET}"
SANITY_ASSETS_URL = f"https://{SANITY_PROJECT_ID}.api.sanity.io/v1/assets/images/{SANITY_DATASET}"

SANITY_HEADERS = {
    'Authorization': f"Bearer {SANITY_TOKEN}",
    'Content-Type': 'application/json'
}

# --- GEMINI CONFIGURATION ---
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

# --- SCRAPING CONFIGURATION ---
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# --- PRE-DEFINED TITLES ---
TITLES_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'titles.json')

# --- NEWS SOURCES ---
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

# --- VALID CATEGORIES ---
VALID_CATEGORIES = [
    "personal-injury",
    "medical-malpractice",
    "motor-vehicle",
    "wrongful-death",
    "dog-bites",
    "premises-liability",
    "product-liability",
    "employment-law",
    "civil-rights",
    "texas-law",
    "legal-tips",
    "case-studies"
]
