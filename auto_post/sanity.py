"""
Sanity.io CMS integration functions.
"""

import os
from datetime import datetime, timezone

import requests

from .config import (
    SANITY_PROJECT_ID, SANITY_TOKEN, SANITY_DATASET,
    SANITY_BASE_URL, SANITY_QUERY_URL, SANITY_ASSETS_URL,
    SANITY_HEADERS, DEFAULT_AUTHOR
)
from .utils import convert_markdown_to_portable_text
from .content import generate_image_with_gemini


def upload_image_to_sanity(image_bytes, filename="blog-image.png"):
    """
    Upload an image to Sanity's asset pipeline.
    Returns the asset reference or None if upload fails.
    """
    print(f"Uploading image to Sanity...")

    if not all([SANITY_PROJECT_ID, SANITY_TOKEN, SANITY_DATASET]):
        print("Error: Missing Sanity configuration")
        return None

    if not image_bytes:
        print("Error: No image data to upload")
        return None

    try:
        headers = {
            'Authorization': f"Bearer {SANITY_TOKEN}",
            'Content-Type': 'image/png'
        }

        response = requests.post(
            f"{SANITY_ASSETS_URL}?filename={filename}",
            headers=headers,
            data=image_bytes,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            asset_id = result.get('document', {}).get('_id')
            print(f"Image uploaded successfully. Asset ID: {asset_id}")
            return {
                "_type": "image",
                "asset": {
                    "_type": "reference",
                    "_ref": asset_id
                }
            }
        else:
            print(f"Failed to upload image: {response.status_code}")
            print(response.text)
            return None

    except Exception as e:
        print(f"Error uploading image: {e}")
        return None


def get_existing_posts():
    """Fetch existing blog posts from Sanity for internal linking."""
    print("Fetching existing blog posts for internal linking...")

    if not SANITY_PROJECT_ID:
        print("Warning: SANITY_PROJECT_ID not set")
        return "No existing posts found."

    query = '*[_type == "blogPost"] | order(publishedAt desc) [0...20] {title, "slug": slug.current, excerpt}'
    encoded_query = requests.utils.quote(query)

    try:
        response = requests.get(
            f"{SANITY_QUERY_URL}?query={encoded_query}",
            headers={'Authorization': f"Bearer {SANITY_TOKEN}"} if SANITY_TOKEN else {},
            timeout=30
        )

        if response.status_code == 200:
            posts = response.json().get('result', [])
            if not posts:
                return "No existing posts found."

            link_database = "\n".join([
                f"- Title: {p.get('title', 'Untitled')}, Slug: {p.get('slug', '')}, Summary: {p.get('excerpt', 'No summary.')}"
                for p in posts
            ])
            print(f"Found {len(posts)} existing posts for internal linking")
            return link_database
        else:
            print(f"Error fetching posts: {response.status_code} - {response.text}")
            return "No existing posts found."
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return "No existing posts found."


def post_to_sanity(article_data):
    """Post the generated article to Sanity.io CMS."""
    print(f"Posting article to Sanity: {article_data.get('title', 'Untitled')}")

    if not all([SANITY_PROJECT_ID, SANITY_TOKEN, SANITY_DATASET]):
        print("Error: Missing Sanity configuration (PROJECT_ID, TOKEN, or DATASET)")
        return False

    portable_body = convert_markdown_to_portable_text(article_data.get('body_markdown', ''))

    # Image generation
    alt_text = article_data.get('alt_text', '')
    main_image = None
    enable_image_gen = os.environ.get('ENABLE_IMAGE_GENERATION', 'true').lower() == 'true'

    if alt_text and enable_image_gen:
        print("\n--- Generating Featured Image ---")
        image_bytes = generate_image_with_gemini(alt_text)

        if image_bytes:
            slug = article_data.get('slug', 'blog-image')
            filename = f"{slug}.png"
            image_asset = upload_image_to_sanity(image_bytes, filename)

            if image_asset:
                main_image = image_asset
                main_image['alt'] = alt_text
                print(f"Featured image ready with alt text")
            else:
                print("Warning: Image upload failed, continuing without image")
        else:
            print("Warning: Image generation failed, continuing without image")

    # Document structure matching blogPost schema
    document = {
        "_type": "blogPost",
        "title": article_data.get('title', ''),
        "slug": {
            "_type": "slug",
            "current": article_data.get('slug', '')
        },
        "author": DEFAULT_AUTHOR,
        "publishedAt": datetime.now(timezone.utc).isoformat(),
        "excerpt": article_data.get('excerpt', ''),
        "categories": article_data.get('categories', ['personal-injury']),
        "body": portable_body,
        "seo": {
            "metaTitle": article_data.get('meta_title', ''),
            "metaDescription": article_data.get('meta_description', ''),
            "keywords": article_data.get('keywords', [])
        },
        "featured": False
    }

    # Add mainImage if we have one
    if main_image:
        document["mainImage"] = main_image

    payload = {
        "mutations": [
            {"create": document}
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
            result = response.json()
            print("SUCCESS: Article published to Sanity.io!")
            print(f"Document ID: {result.get('results', [{}])[0].get('id', 'unknown')}")
            return True
        else:
            print(f"FAILURE: Sanity API Error: {response.status_code}")
            print(response.text)
            return False

    except requests.RequestException as e:
        print(f"Request error: {e}")
        return False
