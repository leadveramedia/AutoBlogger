#!/usr/bin/env python3
"""
Generate TikTok videos from a custom script with setting and movements.
Interactive: paste your script, setting, and movements — all required.
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

    # Get script (required)
    print("\nPaste your script below (press Enter on a blank line when done):")
    script = get_multiline_input()
    if not script.strip():
        print("Error: No script provided")
        sys.exit(1)

    word_count = len(script.split())
    print(f"\n  Script: {word_count} words")
    if word_count < 30:
        print("  Warning: Script seems short (target ~40-50 words for ~22s video)")
    elif word_count > 60:
        print("  Warning: Script seems long (target ~40-50 words for ~22s video)")

    # Get setting (required)
    print("\nSetting/location (press Enter on a blank line when done):")
    setting = get_multiline_input()
    if not setting.strip():
        print("Error: No setting provided")
        sys.exit(1)

    # Get movements/actions (required)
    print("\nMovements/actions (press Enter on a blank line when done):")
    actions = get_multiline_input()
    if not actions.strip():
        print("Error: No movements/actions provided")
        sys.exit(1)

    # Get slug (optional, has default)
    default_slug = f"custom-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    slug_input = input(f"\nSlug/filename [{default_slug}]: ").strip()
    slug = slug_input if slug_input else default_slug

    # Build article_data
    article_data = {
        'title': '',
        'slug': slug,
        'excerpt': '',
        'body_markdown': script,
        'categories': [],
        'keywords': [],
    }

    print(f"\n  Slug: {slug}")
    print(f"  Script: {script[:80]}...")
    print(f"  Setting: {setting[:80]}...")
    print(f"  Actions: {actions[:80]}...")
    print("\nGenerating 3 video variants (static, walk-and-talk, location-tour)...")
    print("=" * 60)

    results = generate_three_videos(
        article_data,
        custom_script=script,
        custom_setting=setting,
        custom_actions=actions,
    )

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
