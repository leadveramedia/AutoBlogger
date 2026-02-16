#!/usr/bin/env python3
"""
One-off script to generate 9 reference images from a single source image
using useapi.net Flow (nano-banana-pro model).
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

from auto_post.video import (
    _flow_headers,
    _flow_post_image_with_retry,
)
from auto_post.config import USEAPI_BASE_URL, USEAPI_GOOGLE_EMAIL, SPOKESPERSON_IMAGES_DIR

SOURCE_IMAGE = os.path.join(SPOKESPERSON_IMAGES_DIR,
    '20260215_2145_Mixed Heritage Beauty_simple_compose_01khjfmkgvehzbc8psxmaqae6b.png')

# Reference image specs: (filename, prompt_suffix)
REF_SPECS = [
    ('ref_front.png',
     'Front-facing portrait, looking directly at camera. Neutral confident expression, slight closed-mouth smile. '
     'Head straight, shoulders square to camera.'),
    ('ref_3q_left_lo.png',
     'Three-quarter view from the left side, slight low angle looking up. Confident expression. '
     'Head turned slightly to her right, camera positioned below eye level.'),
    ('ref_3q_right.png',
     'Three-quarter view from the right side, at eye level. Natural relaxed expression. '
     'Head turned slightly to her left, camera at eye level.'),
    ('ref_3q_right_lo.png',
     'Three-quarter view from the right side, slight low angle looking up. Confident expression. '
     'Head turned slightly to her left, camera positioned below eye level.'),
    ('ref_eyes_closed.png',
     'Front-facing portrait, eyes gently closed. Peaceful, relaxed expression. '
     'Head straight, as if taking a calm breath.'),
    ('ref_profile_right.png',
     'Right profile view at eye level. Face turned 90 degrees to her left, showing full right profile. '
     'Neutral expression, jaw and cheekbone clearly visible.'),
    ('ref_profile_right_lo.png',
     'Right profile view, slight low angle. Face turned 90 degrees to her left, showing full right profile. '
     'Camera positioned slightly below eye level.'),
    ('ref_smile.png',
     'Front-facing portrait, warm genuine smile with teeth showing. '
     'Head straight, eyes bright and engaged, looking at camera.'),
]

BASE_PROMPT = """Generate a photorealistic face portrait of the EXACT person shown in the reference image.

FACE MATCHING (CRITICAL):
- Face, bone structure, skin texture, freckles, hair color, and features must be IDENTICAL to the reference image
- Same red-auburn long wavy hair, same green eyes, same freckle pattern, same face shape

FRAMING: Tight face close-up (face and neck only), vertical 9:16 portrait
LIGHTING: Natural, soft, well-lit face
BACKGROUND: Plain neutral gray

POSE: {pose}

RULES:
- Single person, face clearly visible
- Photorealistic, professional headshot style
- NO text, NO signs, NO watermarks
- NO sunglasses, NO accessories covering face
- Wearing a simple crew-neck gray t-shirt"""


def upload_source_image():
    """Upload the source image as a Flow asset and return its mediaGenerationId."""
    print(f"Uploading source image: {os.path.basename(SOURCE_IMAGE)}")
    with open(SOURCE_IMAGE, 'rb') as f:
        image_data = f.read()

    resp = requests.post(
        f'{USEAPI_BASE_URL}/assets/{USEAPI_GOOGLE_EMAIL}',
        headers=_flow_headers(content_type='image/png'),
        data=image_data,
        timeout=60,
    )

    if resp.status_code != 200:
        print(f"  Upload failed: {resp.status_code} {resp.text[:300]}")
        return None

    result = resp.json()
    media_id = result.get('mediaGenerationId', {}).get('mediaGenerationId')
    if media_id:
        print(f"  Uploaded -> {media_id[:40]}...")
        return media_id
    else:
        print(f"  Upload response missing mediaGenerationId: {result}")
        return None


def generate_ref_image(ref_id, filename, pose_prompt):
    """Generate a single reference image and save it."""
    prompt = BASE_PROMPT.format(pose=pose_prompt)

    payload = {
        'prompt': prompt,
        'model': 'nano-banana-pro',
        'aspectRatio': 'portrait',
        'count': 4,
        'reference_1': ref_id,
    }

    if USEAPI_GOOGLE_EMAIL:
        payload['email'] = USEAPI_GOOGLE_EMAIL

    print(f"\n  Generating {filename}...")
    candidates = _flow_post_image_with_retry(
        f'{USEAPI_BASE_URL}/images', payload, filename, return_all=True
    )

    if not candidates:
        print(f"  FAILED: No candidates for {filename}")
        return False

    # Use the first candidate (or best if we had scoring, but source is our ref)
    img_id, img_url = candidates[0]
    if not img_url:
        print(f"  FAILED: No URL for {filename}")
        return False

    # Download and save
    resp = requests.get(img_url, timeout=30)
    if resp.status_code != 200:
        print(f"  FAILED: Download error {resp.status_code}")
        return False

    out_path = os.path.join(SPOKESPERSON_IMAGES_DIR, filename)
    with open(out_path, 'wb') as f:
        f.write(resp.content)
    print(f"  Saved {filename} ({len(resp.content) / 1024:.0f} KB)")
    return True


def main():
    print("=" * 60)
    print("  Generate Reference Images from Source")
    print("=" * 60)

    if not os.path.exists(SOURCE_IMAGE):
        print(f"Source image not found: {SOURCE_IMAGE}")
        return

    # Upload source as reference
    ref_id = upload_source_image()
    if not ref_id:
        print("Failed to upload source image")
        return

    # Delete old reference images (but not the source)
    source_basename = os.path.basename(SOURCE_IMAGE)
    print(f"\nDeleting old reference images...")
    for f in os.listdir(SPOKESPERSON_IMAGES_DIR):
        if f == source_basename:
            continue
        if f.endswith(('.png', '.jpg', '.jpeg')):
            os.remove(os.path.join(SPOKESPERSON_IMAGES_DIR, f))
            print(f"  Deleted {f}")

    # Copy source as the full-body reference
    import shutil
    full_body_name = 'ref_full_body.png'
    shutil.copy2(SOURCE_IMAGE, os.path.join(SPOKESPERSON_IMAGES_DIR, full_body_name))
    print(f"\nCopied source as {full_body_name}")

    # Generate each reference image
    success = 0
    failed = []
    for filename, pose in REF_SPECS:
        if generate_ref_image(ref_id, filename, pose):
            success += 1
        else:
            failed.append(filename)

    print(f"\n{'=' * 60}")
    print(f"  Generated {success}/{len(REF_SPECS)} reference images")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print("=" * 60)

    # List final assets
    print(f"\nFinal contents of {SPOKESPERSON_IMAGES_DIR}:")
    for f in sorted(os.listdir(SPOKESPERSON_IMAGES_DIR)):
        if f.endswith(('.png', '.jpg', '.jpeg')):
            size = os.path.getsize(os.path.join(SPOKESPERSON_IMAGES_DIR, f))
            print(f"  {f} ({size / 1024:.0f} KB)")


if __name__ == '__main__':
    main()
