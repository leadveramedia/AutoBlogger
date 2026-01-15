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

# --- USED TOPICS TRACKING (prevents duplicate blog posts about same news story) ---
USED_TOPICS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'used_topics.json')

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
    },
    {
        'name': 'NHTSA Press Releases',
        'url': 'https://www.nhtsa.gov/press-releases',
        'category': 'motor_vehicle',
        'enabled': True,
        'scraper': 'nhtsa'
    },
    {
        'name': 'DOL Newsroom',
        'url': 'https://www.dol.gov/newsroom/releases',
        'category': 'workers_comp',
        'enabled': True,
        'scraper': 'dol'
    },
    {
        'name': 'Insurance Journal National',
        'url': 'https://www.insurancejournal.com/news/national/',
        'category': 'personal_injury',
        'enabled': True,
        'scraper': 'insurancejournal'
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

# --- PRACTICE AREA KEYWORDS (for filtering general news sources) ---
# Articles from general news must contain at least one keyword to be included
PRACTICE_AREA_KEYWORDS = [
    # Personal Injury
    'personal injury', 'injured', 'injury lawsuit', 'injury claim', 'accident victim',
    'negligence', 'liability', 'damages awarded', 'settlement', 'compensation',

    # Medical Malpractice
    'medical malpractice', 'medical negligence', 'surgical error', 'misdiagnosis',
    'hospital negligence', 'doctor sued', 'patient death', 'medical error',
    'birth injury', 'anesthesia error', 'nursing home abuse', 'elder abuse',

    # Motor Vehicle Accidents
    'car accident', 'car crash', 'auto accident', 'vehicle accident', 'truck accident',
    'motorcycle accident', 'pedestrian hit', 'drunk driver', 'dui crash', 'fatal crash',
    'hit and run', 'multi-vehicle', 'rollover', 'head-on collision', 'rear-end',
    'uber accident', 'lyft accident', 'rideshare accident', 'bus accident',

    # Wrongful Death
    'wrongful death', 'fatal accident', 'death lawsuit', 'family sues', 'killed',
    'fatality', 'deceased', 'survivor lawsuit',

    # Dog Bites
    'dog bite', 'dog attack', 'animal attack', 'pit bull attack', 'mauled',
    'dog owner liable', 'dangerous dog',

    # Premises Liability
    'slip and fall', 'trip and fall', 'premises liability', 'property owner liable',
    'unsafe conditions', 'negligent security', 'swimming pool accident',
    'amusement park injury', 'store injury', 'parking lot assault',

    # Product Liability
    'product recall', 'defective product', 'product liability', 'consumer safety',
    'fda recall', 'cpsc recall', 'product defect', 'manufacturer liable',
    'toxic exposure', 'contaminated', 'dangerous product',

    # Employment Law
    'employment discrimination', 'workplace discrimination', 'wrongful termination',
    'sexual harassment', 'wage theft', 'unpaid overtime', 'retaliation',
    'hostile work environment', 'eeoc', 'ada violation', 'fmla violation',
    'whistleblower', 'class action employment',

    # Civil Rights
    'civil rights', 'police brutality', 'excessive force', 'false arrest',
    'wrongful imprisonment', 'civil liberties', 'constitutional violation',
    'section 1983', 'prisoner rights', 'inmate abuse',

    # Worker's Compensation
    'workers compensation', 'workplace injury', 'on the job injury', 'osha violation',
    'osha fine', 'workplace safety', 'occupational hazard', 'work accident',
    'construction accident', 'industrial accident', 'warehouse injury',

    # Social Security Disability
    'social security disability', 'ssdi', 'ssi benefits', 'disability benefits',
    'disability claim', 'disability denied',

    # Intellectual Property
    'patent infringement', 'trademark infringement', 'copyright infringement',
    'intellectual property lawsuit', 'trade secret',

    # Professional Malpractice
    'legal malpractice', 'accounting malpractice', 'professional negligence',
    'fiduciary duty', 'breach of duty',

    # Class Action
    'class action', 'mass tort', 'multidistrict litigation', 'mdl',
    'class certification', 'bellwether trial'
]
