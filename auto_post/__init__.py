"""
Auto Post - Automated Legal News Blog Generator

Scrapes multiple news sources, generates AI content with Gemini,
and posts to Sanity.io CMS.

Focused on: Personal Injury Law, Employment Law, Mass Torts
"""

__version__ = "1.0.0"

from .scrapers import scrape_all_sources
from .content import select_best_articles, generate_article, generate_article_from_title
from .sanity import get_existing_posts, post_to_sanity
from .utils import load_title_list, save_title_list, load_used_topics, add_used_topic

__all__ = [
    'scrape_all_sources',
    'select_best_articles',
    'generate_article',
    'generate_article_from_title',
    'get_existing_posts',
    'post_to_sanity',
    'load_title_list',
    'save_title_list',
    'load_used_topics',
    'add_used_topic',
]
