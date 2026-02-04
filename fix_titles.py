#!/usr/bin/env python3
"""
Fix long blog post titles in Sanity CMS.

This script queries Sanity for blog posts with titles exceeding 60 characters,
uses Gemini AI to generate shorter versions, and updates them via the Sanity API.

Usage:
    python fix_titles.py              # Dry run (preview changes only)
    python fix_titles.py --apply      # Apply changes to Sanity
"""

import argparse
import sys
import time

import requests
from google import genai

from auto_post.config import (
    SANITY_PROJECT_ID, SANITY_TOKEN, SANITY_DATASET,
    SANITY_BASE_URL, SANITY_QUERY_URL, SANITY_HEADERS,
    GEMINI_API_KEY
)

MAX_TITLE_LENGTH = 60


def get_all_posts():
    """Fetch all blog posts from Sanity with _id and title."""
    print("Fetching all blog posts from Sanity...")

    if not SANITY_PROJECT_ID:
        print("Error: SANITY_PROJECT_ID not set")
        return []

    # Query for all posts with _id and title
    query = '*[_type == "blogPost"] | order(publishedAt desc) {_id, title, "slug": slug.current}'
    encoded_query = requests.utils.quote(query)

    try:
        response = requests.get(
            f"{SANITY_QUERY_URL}?query={encoded_query}",
            headers={'Authorization': f"Bearer {SANITY_TOKEN}"} if SANITY_TOKEN else {},
            timeout=30
        )

        if response.status_code == 200:
            posts = response.json().get('result', [])
            print(f"Found {len(posts)} total posts")
            return posts
        else:
            print(f"Error fetching posts: {response.status_code} - {response.text}")
            return []
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return []


def get_posts_with_long_titles(posts):
    """Filter posts with titles exceeding MAX_TITLE_LENGTH."""
    long_titles = [p for p in posts if len(p.get('title', '')) > MAX_TITLE_LENGTH]
    print(f"Found {len(long_titles)} posts with titles > {MAX_TITLE_LENGTH} characters")
    return long_titles


def shorten_title(original_title):
    """Use Gemini AI to generate a shorter version of the title."""
    if not GEMINI_API_KEY:
        print("  Warning: GEMINI_API_KEY not set, truncating instead")
        return original_title[:MAX_TITLE_LENGTH - 3] + '...'

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        prompt = f"""Rewrite this blog post title to be {MAX_TITLE_LENGTH} characters or less while keeping the key message and SEO value.

Original title ({len(original_title)} chars): {original_title}

Requirements:
- MUST be {MAX_TITLE_LENGTH} characters or less (including spaces)
- Keep the main topic and important keywords
- Make it compelling and click-worthy
- Do not use quotes around the title
- Do not add any explanation, just return the new title

Return ONLY the new title, nothing else."""

        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )

        new_title = response.text.strip().strip('"\'')

        # Verify length
        if len(new_title) <= MAX_TITLE_LENGTH:
            return new_title
        else:
            # Try once more with stricter instructions
            prompt2 = f"""Shorten this title to UNDER {MAX_TITLE_LENGTH} characters. Be concise.
Current ({len(new_title)} chars): {new_title}
Return ONLY the shortened title."""

            response2 = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt2
            )
            final_title = response2.text.strip().strip('"\'')

            if len(final_title) <= MAX_TITLE_LENGTH:
                return final_title
            else:
                # Last resort: truncate
                return final_title[:MAX_TITLE_LENGTH - 3] + '...'

    except Exception as e:
        print(f"  Error with AI shortening: {e}")
        return original_title[:MAX_TITLE_LENGTH - 3] + '...'


def update_post_title(post_id, new_title):
    """Update the title of a post in Sanity using patch mutation."""
    if not all([SANITY_PROJECT_ID, SANITY_TOKEN, SANITY_DATASET]):
        print("  Error: Missing Sanity configuration")
        return False

    payload = {
        "mutations": [
            {
                "patch": {
                    "id": post_id,
                    "set": {
                        "title": new_title
                    }
                }
            }
        ]
    }

    try:
        response = requests.post(
            SANITY_BASE_URL,
            headers=SANITY_HEADERS,
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            return True
        else:
            print(f"  Error updating: {response.status_code} - {response.text}")
            return False

    except requests.RequestException as e:
        print(f"  Request error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Fix long blog post titles in Sanity CMS')
    parser.add_argument('--apply', action='store_true', help='Apply changes (default is dry-run)')
    args = parser.parse_args()

    dry_run = not args.apply

    if dry_run:
        print("=" * 60)
        print("DRY RUN MODE - No changes will be made")
        print("Run with --apply to actually update titles")
        print("=" * 60)
    else:
        print("=" * 60)
        print("APPLY MODE - Changes will be made to Sanity")
        print("=" * 60)

    print()

    # Get all posts
    all_posts = get_all_posts()
    if not all_posts:
        print("No posts found. Exiting.")
        return

    # Filter for long titles
    long_title_posts = get_posts_with_long_titles(all_posts)
    if not long_title_posts:
        print("No posts with long titles found. Nothing to fix!")
        return

    print()
    print("=" * 60)
    print("PROPOSED TITLE CHANGES")
    print("=" * 60)

    changes = []
    for i, post in enumerate(long_title_posts, 1):
        post_id = post.get('_id')
        original_title = post.get('title', '')
        slug = post.get('slug', '')

        print(f"\n[{i}/{len(long_title_posts)}] {slug}")
        print(f"  Original ({len(original_title)} chars): {original_title}")

        new_title = shorten_title(original_title)
        print(f"  New      ({len(new_title)} chars): {new_title}")

        changes.append({
            'id': post_id,
            'slug': slug,
            'original': original_title,
            'new': new_title
        })

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    print()
    print("=" * 60)
    print(f"SUMMARY: {len(changes)} titles to update")
    print("=" * 60)

    if dry_run:
        print("\nThis was a dry run. To apply these changes, run:")
        print("  python fix_titles.py --apply")
        return

    # Apply changes
    print("\nApplying changes...")
    success_count = 0
    fail_count = 0

    for change in changes:
        print(f"  Updating: {change['slug'][:40]}...", end=" ")
        if update_post_title(change['id'], change['new']):
            print("OK")
            success_count += 1
        else:
            print("FAILED")
            fail_count += 1
        time.sleep(0.5)  # Rate limiting

    print()
    print("=" * 60)
    print(f"COMPLETE: {success_count} updated, {fail_count} failed")
    print("=" * 60)


if __name__ == '__main__':
    main()
