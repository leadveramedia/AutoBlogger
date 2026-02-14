#!/usr/bin/env python3
"""
Test script to generate a TikTok video from an existing Sanity article.
Usage: python test_video.py
"""

import os
import sys
import requests
from pathlib import Path

# Load .env file
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, _, value = line.partition('=')
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

from auto_post.config import SANITY_PROJECT_ID, SANITY_TOKEN, SANITY_DATASET
from auto_post.video import generate_tiktok_video


def fetch_recent_article():
    """Fetch the most recent article from Sanity with full details."""
    query = '*[_type == "blogPost"] | order(publishedAt desc) [0] {title, "slug": slug.current, excerpt, "body_markdown": body, "meta_title": seo.metaTitle, "meta_description": seo.metaDescription, "keywords": seo.keywords, categories}'
    query_url = f"https://{SANITY_PROJECT_ID}.api.sanity.io/v2021-06-07/data/query/{SANITY_DATASET}"
    encoded_query = requests.utils.quote(query)

    response = requests.get(
        f"{query_url}?query={encoded_query}",
        headers={'Authorization': f"Bearer {SANITY_TOKEN}"},
        timeout=30
    )

    if response.status_code != 200:
        print(f"Error fetching article: {response.status_code}")
        return None

    result = response.json().get('result')
    if not result:
        print("No articles found in Sanity")
        return None

    return result


def main():
    print("=" * 60)
    print("  TikTok Video Generation Test")
    print("=" * 60)
    print(f"  VIDEO_SEED_MODE: {os.environ.get('VIDEO_SEED_MODE', 'i2v')}")
    sys.stdout.flush()

    # Verify env vars
    if not SANITY_PROJECT_ID or not SANITY_TOKEN:
        print("Error: SANITY_PROJECT_ID and SANITY_TOKEN must be set")
        return

    if not os.environ.get('GEMINI_API_KEY'):
        print("Error: GEMINI_API_KEY must be set")
        return

    # Fetch a recent article
    print("\nFetching most recent article from Sanity...")
    article = fetch_recent_article()
    if not article:
        return

    print(f"\nUsing article:")
    print(f"  Title: {article.get('title', 'N/A')}")
    print(f"  Slug: {article.get('slug', 'N/A')}")
    print(f"  Excerpt: {article.get('excerpt', 'N/A')[:80]}...")
    print(f"  Categories: {article.get('categories', [])}")

    # The body from Sanity is Portable Text (not markdown), so use excerpt + title as content
    # For a proper test, we'll construct a simplified article_data
    article_data = {
        'title': article.get('title', ''),
        'slug': article.get('slug', 'test-video'),
        'excerpt': article.get('excerpt', ''),
        'body_markdown': article.get('excerpt', '') * 5,  # Repeat excerpt as body proxy
        'keywords': article.get('keywords', []),
        'categories': article.get('categories', []),
    }

    # Generate the video
    print("\n" + "=" * 60)
    sys.stdout.flush()
    video_path = generate_tiktok_video(article_data)

    print("\n" + "=" * 60)
    if video_path:
        file_size = os.path.getsize(video_path)
        print(f"  SUCCESS: Video saved to {video_path} ({file_size / 1024 / 1024:.1f} MB)")
    else:
        print(f"  Video generation did not produce a file")
    print("=" * 60)


if __name__ == '__main__':
    main()
