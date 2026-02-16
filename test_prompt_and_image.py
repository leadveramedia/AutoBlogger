#!/usr/bin/env python3
"""
Test script that generates ONLY the video prompt (script/dialogue) and scene image.
Does NOT generate any video clips — just the two pieces needed for review.
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
from auto_post.video import (
    generate_video_prompt,
    _flow_upload_reference_images,
    _flow_generate_scene_image,
)


def fetch_recent_article():
    """Fetch the most recent article from Sanity."""
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
    print("  Prompt + Scene Image Test (no video generation)")
    print("=" * 60)
    sys.stdout.flush()

    # Fetch article
    print("\nFetching most recent article from Sanity...")
    article = fetch_recent_article()
    if not article:
        return

    article_data = {
        'title': article.get('title', ''),
        'slug': article.get('slug', 'test'),
        'excerpt': article.get('excerpt', ''),
        'body_markdown': article.get('excerpt', '') * 5,
        'keywords': article.get('keywords', []),
        'categories': article.get('categories', []),
    }
    print(f"  Title: {article_data['title']}")
    print(f"  Slug: {article_data['slug']}")

    # Step 1: Generate video prompt (script + dialogue)
    print("\n" + "=" * 60)
    print("  STEP 1: Generating video prompt via Gemini...")
    print("=" * 60)
    sys.stdout.flush()

    video_prompt = generate_video_prompt(article_data)
    if not video_prompt:
        print("FAILED: Could not generate video prompt")
        return

    print("\n--- SCRIPT ---")
    print(video_prompt.get('script', '(no script)'))
    print("\n--- APPEARANCE ---")
    print(video_prompt.get('appearance', '(no appearance)'))
    print("\n--- SETTING ---")
    print(video_prompt.get('setting', '(no setting)'))
    print("\n--- INITIAL PROMPT ---")
    print(video_prompt.get('initial_prompt', '(none)'))
    for i, ext in enumerate(video_prompt.get('extension_prompts', []), 1):
        print(f"\n--- EXTENSION {i} ---")
        print(ext)

    # Step 2: Generate scene image
    print("\n" + "=" * 60)
    print("  STEP 2: Generating scene image (face-matched)...")
    print("=" * 60)
    sys.stdout.flush()

    ref_ids = _flow_upload_reference_images()
    if not ref_ids:
        print("FAILED: Could not upload reference images")
        return

    print(f"  Uploaded {len(ref_ids)} reference image(s)")

    appearance_brief = video_prompt.get('appearance', 'casual athletic wear')
    setting_brief = video_prompt.get('setting', '')
    scene_id, scene_url = _flow_generate_scene_image(appearance_brief, ref_ids, setting=setting_brief)

    if not scene_id:
        print("FAILED: Could not generate scene image")
        return

    # Download scene image for review
    if scene_url:
        print(f"\n  Scene image URL: {scene_url}")
        try:
            img_resp = requests.get(scene_url, timeout=30)
            if img_resp.status_code == 200:
                out_path = Path(__file__).parent / 'test_scene_image.png'
                out_path.write_bytes(img_resp.content)
                print(f"  Scene image saved to: {out_path}")
                print(f"  Size: {len(img_resp.content) / 1024:.1f} KB")
            else:
                print(f"  Could not download scene image: {img_resp.status_code}")
        except Exception as e:
            print(f"  Error downloading scene image: {e}")
    else:
        print("  No scene image URL returned (only got mediaGenerationId)")

    print("\n" + "=" * 60)
    print("  DONE — Review script above and scene image file")
    print("=" * 60)


if __name__ == '__main__':
    main()
