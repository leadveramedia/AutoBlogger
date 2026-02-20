#!/usr/bin/env python3
"""
Generate TikTok videos from a custom script.
Interactive: paste your script, optionally provide slug and topic context.
Generates 3 format variants (static, walk-and-talk, location-tour).

Usage: python generate_custom_video.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Load .env file
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, _, value = line.partition('=')
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

from auto_post.video import generate_three_videos


def get_multiline_input():
    """Read multi-line input until user enters a blank line."""
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == '':
            break
        lines.append(line)
    return ' '.join(lines)


def main():
    print("=" * 60)
    print("  Custom Script Video Generator")
    print("=" * 60)

    if not os.environ.get('GEMINI_API_KEY'):
        print("Error: GEMINI_API_KEY must be set")
        sys.exit(1)
    if not os.environ.get('USEAPI_TOKEN'):
        print("Error: USEAPI_TOKEN must be set")
        sys.exit(1)

    # Get script
    print("\nPaste your script below (press Enter on a blank line when done):")
    script = get_multiline_input()
    if not script.strip():
        print("Error: No script provided")
        sys.exit(1)

    word_count = len(script.split())
    print(f"\n  Script: {word_count} words")
    if word_count < 30:
        print("  Warning: Script seems short (target ~45-55 words for 18-20s video)")
    elif word_count > 70:
        print("  Warning: Script seems long (target ~45-55 words for 18-20s video)")

    # Get slug
    default_slug = f"custom-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    slug_input = input(f"\nSlug/filename [{default_slug}]: ").strip()
    slug = slug_input if slug_input else default_slug

    # Get topic context
    topic = input("Topic context (helps pick setting/appearance, Enter to skip): ").strip()

    # Build article_data
    article_data = {
        'title': topic or '',
        'slug': slug,
        'excerpt': topic or '',
        'body_markdown': script,
        'categories': [],
        'keywords': [k.strip() for k in topic.split(',') if k.strip()] if topic else [],
    }

    print(f"\n  Slug: {slug}")
    print(f"  Topic: {topic or '(inferred from script)'}")
    print(f"  Script: {script[:80]}...")
    print("\nGenerating 3 video variants (static, walk-and-talk, location-tour)...")
    print("=" * 60)

    results = generate_three_videos(article_data, custom_script=script)

    # Summary
    print("\n" + "=" * 60)
    if results:
        print(f"  Generated {len(results)} video(s):")
        for path in results:
            size = os.path.getsize(path) / 1024 / 1024
            print(f"    {path} ({size:.1f} MB)")
    else:
        print("  No videos were generated successfully")
    print("=" * 60)


if __name__ == '__main__':
    main()
