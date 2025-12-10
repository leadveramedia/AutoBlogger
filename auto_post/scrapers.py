"""
News scraper functions for multiple sources.
"""

import time
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

from .config import NEWS_SOURCES, REQUEST_HEADERS


def scrape_aboutlawsuits(url):
    """Scrape AboutLawsuits.com for mass tort and class action news."""
    print(f"  Scraping AboutLawsuits...")
    news_items = []

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        headlines = soup.find_all('h4', limit=10)

        for headline in headlines:
            link = headline.find('a') or headline.find_parent('a')
            if link and link.get('href'):
                title = headline.get_text(strip=True)
                href = link.get('href')
                if title and '/about/' not in href.lower():
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

        rows = soup.select('table tbody tr, .views-row')

        for row in rows[:10]:
            brand_cell = row.select_one('.views-field-brand-name a, td a')
            if brand_cell:
                title = brand_cell.get_text(strip=True)
                href = brand_cell.get('href', '')

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

        rows = soup.select('.views-row, .news-item, article')

        for row in rows[:10]:
            title_tag = row.select_one('h3 a, h2 a, .title a')
            if title_tag:
                title = title_tag.get_text(strip=True)
                href = title_tag.get('href', '')

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

        headlines = soup.select('h5, .views-row h3, .news-release-title')

        for headline in headlines[:10]:
            link = headline.find('a') or headline.find_parent('a')
            if not link:
                parent = headline.find_parent(['div', 'article', 'li'])
                if parent:
                    link = parent.find('a')

            if link and link.get('href'):
                title = headline.get_text(strip=True)
                href = link.get('href', '')

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
