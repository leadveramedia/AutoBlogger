#!/usr/bin/env python3
"""
Backfill existing Sanity blog posts with landing page (calculator/state) links.

Usage:
    python backfill_landing_links.py          # Dry run - preview changes
    python backfill_landing_links.py --apply  # Actually update Sanity
"""

import os
import sys
import json
import time
import uuid
import re
from pathlib import Path

# Load .env file before importing auto_post (which reads os.environ at import time)
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

import requests
from google import genai

from auto_post.config import (
    GEMINI_API_KEY, SANITY_PROJECT_ID, SANITY_TOKEN, SANITY_DATASET,
    SANITY_QUERY_URL, SANITY_HEADERS, CALCULATOR_SLUGS, STATE_SLUGS,
)
from auto_post.content import build_landing_page_database


def generate_key():
    return uuid.uuid4().hex[:12]


def fetch_all_posts():
    """Fetch all blog posts from Sanity with body content."""
    query = '*[_type == "blogPost"] | order(publishedAt desc) {_id, title, "slug": slug.current, categories, excerpt, body}'
    encoded = requests.utils.quote(query)
    resp = requests.get(
        f"{SANITY_QUERY_URL}?query={encoded}",
        headers={'Authorization': f"Bearer {SANITY_TOKEN}"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get('result', [])


def portable_text_to_plain(body):
    """Extract readable plain text from Portable Text blocks."""
    if not body:
        return ""
    lines = []
    for block in body:
        if block.get('_type') != 'block':
            continue
        text_parts = []
        for child in block.get('children', []):
            if child.get('_type') == 'span':
                text_parts.append(child.get('text', ''))
        line = ''.join(text_parts).strip()
        if line:
            lines.append(line)
    return '\n\n'.join(lines)


def body_already_has_calculator_link(body):
    """Check if body already contains a casevalue.law calculator/state link."""
    if not body:
        return False
    for block in body:
        if block.get('_type') != 'block':
            continue
        for md in block.get('markDefs', []):
            href = md.get('href', '')
            if 'casevalue.law' in href and '/calculator/' in href:
                return True
            if 'casevalue.law' in href and '-calculator' in href:
                return True
    return False


def ask_gemini_for_links(client, plain_text, title, categories, landing_db):
    """Ask Gemini to identify 1-2 calculator links to add to an existing article."""
    prompt = f"""You are an SEO specialist. Given an existing blog post, identify 1-2 places to naturally insert links to calculator landing pages.

**Article Title:** {title}
**Categories:** {', '.join(categories or [])}

**Article Text:**
{plain_text[:4000]}

**Landing Page Database:**
{landing_db}

**Instructions:**
1. Identify the most relevant practice area calculator for this article
2. If a specific U.S. state is prominently discussed, use the state-specific calculator URL
3. Find 1-2 SHORT phrases (2-5 words) already in the article text that would make natural anchor text for the calculator link
4. The anchor text MUST be an EXACT substring of the article text above

Return a JSON array with 1-2 objects:
{{
  "links": [
    {{
      "anchor_text": "exact phrase from article",
      "url": "https://casevalue.law/calculator/practice-area or https://casevalue.law/state/practice-area-calculator"
    }}
  ]
}}

CRITICAL: anchor_text must be an EXACT match to text already in the article. Return ONLY the JSON, nothing else."""

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config={'response_mime_type': 'application/json'},
    )
    text = response.text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?\s*```\s*$', '', text)
    data = json.loads(text)
    return data.get('links', [])


def insert_link_in_body(body, anchor_text, url):
    """Insert a link into Portable Text body by finding anchor_text in spans and splitting."""
    for block in body:
        if block.get('_type') != 'block':
            continue
        children = block.get('children', [])
        for idx, child in enumerate(children):
            if child.get('_type') != 'span':
                continue
            span_text = child.get('text', '')
            pos = span_text.find(anchor_text)
            if pos == -1:
                continue

            # Found the anchor text in this span - split it
            existing_marks = child.get('marks', [])
            link_key = generate_key()

            new_children = []
            # Before text
            before = span_text[:pos]
            if before:
                new_children.append({
                    "_type": "span",
                    "_key": generate_key(),
                    "text": before,
                    "marks": list(existing_marks),
                })
            # Linked text
            new_children.append({
                "_type": "span",
                "_key": generate_key(),
                "text": anchor_text,
                "marks": list(existing_marks) + [link_key],
            })
            # After text
            after = span_text[pos + len(anchor_text):]
            if after:
                new_children.append({
                    "_type": "span",
                    "_key": generate_key(),
                    "text": after,
                    "marks": list(existing_marks),
                })

            # Replace the original span with new spans
            children[idx:idx + 1] = new_children

            # Add link markDef to block
            mark_defs = block.get('markDefs', [])
            mark_defs.append({
                "_type": "link",
                "_key": link_key,
                "href": url,
            })
            block['markDefs'] = mark_defs

            return True  # Successfully inserted

    return False


def patch_sanity_body(doc_id, body):
    """PATCH a Sanity document's body field."""
    url = f"https://{SANITY_PROJECT_ID}.api.sanity.io/v2021-06-07/data/mutate/{SANITY_DATASET}"
    payload = {
        "mutations": [{
            "patch": {
                "id": doc_id,
                "set": {"body": body}
            }
        }]
    }
    resp = requests.post(url, headers=SANITY_HEADERS, json=payload, timeout=30)
    resp.raise_for_status()
    return True


def backfill_post(post, landing_db, client, dry_run=True):
    """Process one post: determine links, insert them, optionally patch Sanity."""
    title = post.get('title', 'Untitled')
    doc_id = post.get('_id', '')
    body = post.get('body', [])

    if not body:
        print(f"  SKIP (no body): {title[:60]}")
        return False

    if body_already_has_calculator_link(body):
        print(f"  SKIP (already has calculator link): {title[:60]}")
        return False

    plain_text = portable_text_to_plain(body)
    if len(plain_text) < 100:
        print(f"  SKIP (too short): {title[:60]}")
        return False

    # Ask Gemini for link suggestions
    try:
        links = ask_gemini_for_links(
            client, plain_text, title,
            post.get('categories', []), landing_db,
        )
    except Exception as e:
        print(f"  ERROR (Gemini): {title[:60]} - {e}")
        return False

    if not links:
        print(f"  SKIP (no links suggested): {title[:60]}")
        return False

    # Make a deep copy of body for modification
    import copy
    modified_body = copy.deepcopy(body)

    inserted = 0
    for link in links[:2]:
        anchor = link.get('anchor_text', '')
        url = link.get('url', '')
        if not anchor or not url:
            continue
        if insert_link_in_body(modified_body, anchor, url):
            inserted += 1
            print(f"  + [{anchor}]({url})")
        else:
            print(f"  ~ anchor not found: \"{anchor}\"")

    if inserted == 0:
        print(f"  SKIP (no anchors matched): {title[:60]}")
        return False

    if dry_run:
        print(f"  DRY RUN: Would update {title[:60]} ({inserted} links)")
        return True
    else:
        try:
            patch_sanity_body(doc_id, modified_body)
            print(f"  UPDATED: {title[:60]} ({inserted} links)")
            return True
        except Exception as e:
            print(f"  ERROR (patch): {title[:60]} - {e}")
            return False


def main():
    dry_run = '--apply' not in sys.argv
    if dry_run:
        print("DRY RUN MODE - no changes will be made. Use --apply to update Sanity.\n")
    else:
        print("APPLY MODE - changes WILL be written to Sanity.\n")

    if not all([GEMINI_API_KEY, SANITY_PROJECT_ID, SANITY_TOKEN]):
        print("Error: Missing required env vars (GEMINI_API_KEY, SANITY_PROJECT_ID, SANITY_TOKEN)")
        sys.exit(1)

    client = genai.Client(api_key=GEMINI_API_KEY)
    landing_db = build_landing_page_database()

    print("Fetching all blog posts from Sanity...")
    posts = fetch_all_posts()
    print(f"Found {len(posts)} posts.\n")

    updated = 0
    skipped = 0
    errors = 0

    for i, post in enumerate(posts, 1):
        print(f"[{i}/{len(posts)}] {post.get('title', 'Untitled')[:60]}")
        result = backfill_post(post, landing_db, client, dry_run=dry_run)
        if result:
            updated += 1
        elif result is False:
            skipped += 1

        # Rate limit Gemini calls
        time.sleep(1)

    print(f"\nDone. Updated: {updated}, Skipped: {skipped}")
    if dry_run:
        print("(Dry run - no actual changes made. Use --apply to write changes.)")


if __name__ == '__main__':
    main()
