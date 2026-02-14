"""
TikTok video generation via useapi.net Google Flow (Veo 3.1 Fast).
"""

import os
import re
import json
import time
import random
import base64
import requests

from google import genai

from .config import (GEMINI_API_KEY, VIDEOS_DIR, SPOKESPERSON_IMAGES_DIR,
                     USEAPI_TOKEN, USEAPI_GOOGLE_EMAIL, USEAPI_BASE_URL,
                     VIDEO_SEED_MODE)
from .content import sanitize_json_control_chars
from .outfit_tracking import load_outfit_history, save_outfit

# --- Flow (useapi.net) Constants ---
FLOW_POLL_INTERVAL = 15      # seconds between polling
FLOW_MAX_POLLS = 50          # ~12.5 min max per operation
FLOW_MODEL = 'veo-3.1-fast'  # supports R2V, cheapest

# Supported image extensions (for Veo reference images)
_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
_MIME_TYPES = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}


# ============================================================
#  FLOW VIDEO GENERATION (useapi.net + Google Flow / Veo 3.1)
# ============================================================

def _flow_headers(content_type='application/json'):
    """Return headers for useapi.net API requests."""
    return {
        'Authorization': f'Bearer {USEAPI_TOKEN}',
        'Content-Type': content_type,
    }


def _flow_poll_job(job_id):
    """Poll a useapi.net job until completed/failed/timeout.
    Returns the completed job dict, or None on failure."""
    for poll in range(FLOW_MAX_POLLS):
        time.sleep(FLOW_POLL_INTERVAL)
        try:
            resp = requests.get(
                f'{USEAPI_BASE_URL}/jobs/{job_id}',
                headers=_flow_headers(),
                timeout=30,
            )
        except requests.RequestException as e:
            print(f"    Poll error: {e}")
            continue

        if resp.status_code != 200:
            print(f"    Poll returned {resp.status_code}: {resp.text[:300]}")
            continue

        result = resp.json()
        status = result.get('status', 'unknown')
        elapsed = (poll + 1) * FLOW_POLL_INTERVAL
        print(f"    Polling... status={status} ({elapsed}s elapsed)")

        if status == 'completed':
            return result
        if status in ('failed', 'nsfw'):
            print(f"    Job {status}: {result}")
            return None

    print(f"    Timeout after {FLOW_MAX_POLLS * FLOW_POLL_INTERVAL}s")
    return None


def _flow_upload_reference_images():
    """Upload spokesperson reference images to useapi.net as Flow assets.
    Returns list of mediaGenerationId strings (max 3 for R2V)."""
    if not os.path.exists(SPOKESPERSON_IMAGES_DIR):
        print("    No assets directory found")
        return []

    ref_ids = []
    for filename in sorted(os.listdir(SPOKESPERSON_IMAGES_DIR)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ('.jpg', '.jpeg', '.png'):
            continue

        filepath = os.path.join(SPOKESPERSON_IMAGES_DIR, filename)
        content_type = 'image/jpeg' if ext in ('.jpg', '.jpeg') else 'image/png'

        try:
            with open(filepath, 'rb') as f:
                image_data = f.read()

            resp = requests.post(
                f'{USEAPI_BASE_URL}/assets/{USEAPI_GOOGLE_EMAIL}',
                headers=_flow_headers(content_type=content_type),
                data=image_data,
                timeout=60,
            )

            if resp.status_code != 200:
                print(f"    Upload failed for {filename}: {resp.status_code} {resp.text[:300]}")
                continue

            result = resp.json()
            media_id = result.get('mediaGenerationId', {}).get('mediaGenerationId')
            if media_id:
                print(f"    Uploaded {filename} -> {media_id[:40]}...")
                ref_ids.append(media_id)
            else:
                print(f"    Upload response missing mediaGenerationId: {result}")
        except Exception as e:
            print(f"    Error uploading {filename}: {e}")

        if len(ref_ids) >= 3:
            break

    print(f"    {len(ref_ids)} reference image(s) uploaded")
    return ref_ids


def _extract_video_from_response(result):
    """Extract (mediaGenerationId, video_url) from a useapi.net response.
    Handles both top-level and nested 'response' structures."""
    # Operations may be at result.response.operations or result.operations
    ops = result.get('response', {}).get('operations', [])
    if not ops:
        ops = result.get('operations', [])

    for op in ops:
        if op.get('status') == 'MEDIA_GENERATION_STATUS_SUCCESSFUL':
            video = op.get('operation', {}).get('metadata', {}).get('video', {})
            vid_id = video.get('mediaGenerationId')
            vid_url = video.get('fifeUrl')
            if vid_id:
                return vid_id, vid_url
    if ops:
        print(f"    Generation not successful: {ops[0].get('status', 'unknown')}")
    else:
        print(f"    No operations in response: {list(result.keys())}")
    return None, None


FLOW_CAPTCHA_RETRIES = 3  # retries on 403 captcha failure


def _flow_post_with_retry(url, payload, label="request"):
    """POST to a Flow endpoint with retries on 403 captcha failures.
    Returns (mediaGenerationId, video_url) or (None, None)."""
    for attempt in range(FLOW_CAPTCHA_RETRIES):
        try:
            resp = requests.post(url, headers=_flow_headers(), json=payload, timeout=60)
        except requests.RequestException as e:
            print(f"    Flow {label} error: {e}")
            return None, None

        if resp.status_code == 200:
            return _extract_video_from_response(resp.json())

        if resp.status_code == 201:
            result = resp.json()
            job_id = result.get('jobid') or result.get('jobId')
            if not job_id:
                print(f"    No jobid in async response: {list(result.keys())}")
                return None, None
            print(f"    Job queued: {job_id[:60]}...")
            completed = _flow_poll_job(job_id)
            if not completed:
                return None, None
            return _extract_video_from_response(completed)

        if resp.status_code == 403 and 'reCAPTCHA' in resp.text:
            if attempt < FLOW_CAPTCHA_RETRIES - 1:
                print(f"    Captcha failed (attempt {attempt + 1}/{FLOW_CAPTCHA_RETRIES}), retrying...")
                time.sleep(5)
                continue
            print(f"    Captcha failed after {FLOW_CAPTCHA_RETRIES} attempts")
            return None, None

        print(f"    Flow {label} failed: {resp.status_code} {resp.text[:500]}")
        return None, None

    return None, None


def _extract_image_from_response(result):
    """Extract (mediaGenerationId, fifeUrl) from a useapi.net image response.
    Response structure: result.media[0].image.generatedImage.{mediaGenerationId, fifeUrl}"""
    media = result.get('media', [])
    if not media:
        media = result.get('response', {}).get('media', [])
    if media and isinstance(media, list):
        # Nested under media[0].image.generatedImage
        gen_img = media[0].get('image', {}).get('generatedImage', {})
        if gen_img:
            img_id = gen_img.get('mediaGenerationId')
            img_url = gen_img.get('fifeUrl')
            if img_id:
                return img_id, img_url
        # Fallback: try flat structure
        img_id = media[0].get('mediaGenerationId')
        img_url = media[0].get('fifeUrl')
        if img_id:
            return img_id, img_url
        print(f"    Image response missing mediaGenerationId: {list(media[0].keys())}")
    else:
        print(f"    No media in image response: {list(result.keys())}")
    return None, None


def _flow_post_image_with_retry(url, payload, label="image"):
    """POST to a Flow image endpoint with retries on 403 captcha failures.
    Returns (mediaGenerationId, fifeUrl) or (None, None)."""
    for attempt in range(FLOW_CAPTCHA_RETRIES):
        try:
            resp = requests.post(url, headers=_flow_headers(), json=payload, timeout=60)
        except requests.RequestException as e:
            print(f"    Flow {label} error: {e}")
            return None, None

        if resp.status_code == 200:
            return _extract_image_from_response(resp.json())

        if resp.status_code == 201:
            result = resp.json()
            job_id = result.get('jobid') or result.get('jobId')
            if not job_id:
                print(f"    No jobid in async image response: {list(result.keys())}")
                return None, None
            print(f"    Image job queued: {job_id[:60]}...")
            completed = _flow_poll_job(job_id)
            if not completed:
                return None, None
            return _extract_image_from_response(completed)

        if resp.status_code == 403 and 'reCAPTCHA' in resp.text:
            if attempt < FLOW_CAPTCHA_RETRIES - 1:
                print(f"    Captcha failed (attempt {attempt + 1}/{FLOW_CAPTCHA_RETRIES}), retrying...")
                time.sleep(5)
                continue
            print(f"    Captcha failed after {FLOW_CAPTCHA_RETRIES} attempts")
            return None, None

        print(f"    Flow {label} failed: {resp.status_code} {resp.text[:500]}")
        return None, None

    return None, None


def _flow_generate_scene_image(appearance_brief, ref_ids):
    """Generate a face-matched image of Valentina using nano-banana-pro.
    Uses reference images for character consistency. Face matching is #1 priority.
    Returns mediaGenerationId or None."""
    if not ref_ids:
        print("    No reference IDs for scene image generation")
        return None

    prompt = f"""Generate a photorealistic image of the EXACT person shown in the reference images.

PRIORITY #1 — FACE MATCHING (above all else):
- Face, bone structure, skin texture, freckles, and features must be IDENTICAL to the reference images
- If face accuracy and other instructions conflict, ALWAYS prioritize face accuracy
- The person in this image must be immediately recognizable as the same individual in the references

FRAMING: Medium close-up (chest and above), vertical 9:16 portrait
EXPRESSION: Natural, confident, looking at camera, as if about to speak
LIGHTING: Natural, well-lit face, phone camera quality
BRIEF STYLING: {appearance_brief}

RULES:
- Single person, face clearly visible and well-lit
- Photorealistic — shot on a phone camera
- NO text, NO signs, NO watermarks
- NO sunglasses or anything covering the face"""

    payload = {
        'prompt': prompt,
        'model': 'nano-banana-pro',
        'aspectRatio': 'portrait',
        'count': 1,
    }

    if USEAPI_GOOGLE_EMAIL:
        payload['email'] = USEAPI_GOOGLE_EMAIL

    # Add reference images (nano-banana-pro supports up to 10)
    for i, ref_id in enumerate(ref_ids, start=1):
        payload[f'reference_{i}'] = ref_id

    img_id, img_url = _flow_post_image_with_retry(
        f'{USEAPI_BASE_URL}/images', payload, "scene-image"
    )

    if img_id:
        print(f"    Scene image generated: {img_id[:40]}...")
    else:
        print("    Scene image generation failed")

    return img_id, img_url


def _flow_generate_clip(prompt, ref_ids=None, start_image_id=None):
    """Generate a single video clip via Google Flow.
    Returns (mediaGenerationId, video_url) or (None, None)."""
    payload = {
        'prompt': prompt,
        'model': FLOW_MODEL,
        'aspectRatio': 'portrait',
        'async': True,
    }

    if USEAPI_GOOGLE_EMAIL:
        payload['email'] = USEAPI_GOOGLE_EMAIL

    # I2V mode (startImage) and R2V mode (referenceImage) are mutually exclusive
    if start_image_id:
        payload['startImage'] = start_image_id
    elif ref_ids:
        for i, ref_id in enumerate(ref_ids[:3], start=1):
            payload[f'referenceImage_{i}'] = ref_id

    return _flow_post_with_retry(f'{USEAPI_BASE_URL}/videos', payload, "generate")


def _flow_extend_clip(media_id, prompt):
    """Extend a clip by ~8s via Google Flow.
    Returns (mediaGenerationId, video_url) or (None, None)."""
    payload = {
        'mediaGenerationId': media_id,
        'prompt': prompt,
        'model': FLOW_MODEL,
        'async': True,
    }

    return _flow_post_with_retry(f'{USEAPI_BASE_URL}/videos/extend', payload, "extend")


def _flow_concatenate(media_ids):
    """Concatenate clips into a single video via Google Flow.
    Returns raw video bytes or None."""
    media = []
    for i, mid in enumerate(media_ids):
        entry = {'mediaGenerationId': mid}
        # Trim 1s from start of extensions (to remove overlap)
        if i > 0:
            entry['trimStart'] = 1
        media.append(entry)

    payload = {'media': media}

    try:
        resp = requests.post(
            f'{USEAPI_BASE_URL}/videos/concatenate',
            headers=_flow_headers(),
            json=payload,
            timeout=180,  # concatenation can take up to 3 min
        )
    except requests.RequestException as e:
        print(f"    Concatenate error: {e}")
        return None

    if resp.status_code != 200:
        print(f"    Concatenate failed: {resp.status_code} {resp.text[:500]}")
        return None

    result = resp.json()
    # Response contains base64-encoded video as 'encodedVideo' or 'video'
    video_b64 = result.get('encodedVideo') or result.get('video')
    if video_b64:
        try:
            video_bytes = base64.b64decode(video_b64)
            print(f"    Concatenated video: {len(video_bytes) / 1024 / 1024:.1f} MB")
            return video_bytes
        except Exception as e:
            print(f"    Error decoding concatenated video: {e}")
            return None

    # Try URL-based response as fallback
    video_url = result.get('videoUrl') or result.get('url')
    if video_url:
        try:
            dl = requests.get(video_url, timeout=120)
            dl.raise_for_status()
            return dl.content
        except Exception as e:
            print(f"    Download from concat URL failed: {e}")
            return None

    print(f"    No video data in concatenate response: {list(result.keys())}")
    return None


def generate_tiktok_video_flow(article_data):
    """Generate a TikTok video via useapi.net Google Flow (Veo 3.1 Fast).
    Returns file path to saved video, or None on failure."""
    slug = article_data.get('slug', 'untitled')

    if not USEAPI_TOKEN:
        print("  Error: USEAPI_TOKEN not set, skipping Flow video generation")
        return None

    # Step 1: Generate prompts (reuse existing Gemini prompt generator)
    print("  [Flow] Step 1: Generating video script and prompts...")
    video_prompt = generate_video_prompt(article_data)
    if not video_prompt:
        return None

    # Step 2: Upload reference images
    print("  [Flow] Step 2: Uploading reference images...")
    ref_ids = _flow_upload_reference_images()
    if ref_ids:
        print(f"    Got {len(ref_ids)} reference ID(s)")

    # Step 3: Generate scene image with nano-banana-pro (face-matched)
    scene_image_id = None
    if ref_ids:
        appearance_brief = video_prompt.get('appearance', 'casual athletic wear')
        print(f"  [Flow] Step 3: Generating face-matched scene image (nano-banana-pro, mode={VIDEO_SEED_MODE})...")
        scene_image_id, _scene_url = _flow_generate_scene_image(appearance_brief, ref_ids)
        if not scene_image_id:
            print("    Falling back to original reference images")

    # Step 4: Generate initial 8s clip
    print("  [Flow] Step 4: Generating initial 8s clip...")
    if VIDEO_SEED_MODE == 'i2v' and scene_image_id:
        # I2V mode: scene image becomes the first frame
        clip1_id, clip1_url = _flow_generate_clip(
            video_prompt['initial_prompt'], start_image_id=scene_image_id
        )
    else:
        # R2V mode: use scene image as extra reference if available
        all_refs = ref_ids[:]
        if scene_image_id:
            all_refs = ref_ids[:2] + [scene_image_id]
        clip1_id, clip1_url = _flow_generate_clip(
            video_prompt['initial_prompt'], ref_ids=all_refs
        )
    if not clip1_id:
        print("  Initial clip generation failed")
        return None
    print(f"    Initial clip ready: {clip1_id[:40]}...")

    # Step 5: Extend 2x with continuation prompts
    media_ids = [clip1_id]
    extension_prompts = video_prompt.get('extension_prompts', [])[:2]
    for i, ext_prompt in enumerate(extension_prompts):
        print(f"  [Flow] Step {5 + i}: Extending clip ({i + 1}/{len(extension_prompts)})...")
        ext_id, ext_url = _flow_extend_clip(media_ids[-1], ext_prompt)
        if not ext_id:
            print(f"    Extension {i + 1} failed, using partial video")
            break
        media_ids.append(ext_id)
        print(f"    Extension {i + 1} ready: {ext_id[:40]}...")

    # Step 7: Concatenate or download single clip
    if len(media_ids) >= 2:
        print(f"  [Flow] Step 7: Concatenating {len(media_ids)} clips...")
        video_data = _flow_concatenate(media_ids)
    else:
        # Only 1 clip — download directly from URL
        print("  [Flow] Step 7: Downloading single clip...")
        try:
            dl = requests.get(clip1_url, timeout=120)
            dl.raise_for_status()
            video_data = dl.content
        except Exception as e:
            print(f"    Download failed: {e}")
            video_data = None

    if not video_data:
        print("  Failed to get final video data")
        return None

    # Step 6: Save
    output_path = os.path.join(VIDEOS_DIR, f"{slug}.mp4")
    with open(output_path, 'wb') as f:
        f.write(video_data)

    file_size = os.path.getsize(output_path)
    print(f"  Video saved: {output_path} ({file_size / 1024 / 1024:.1f} MB)")

    # Save outfit to history for future variation
    try:
        outfit = video_prompt.get('appearance', '')
        if outfit:
            save_outfit(outfit)
    except Exception as e:
        print(f"  Warning: Could not save outfit to history: {e}")

    return output_path


def generate_video_prompt(article_data):
    """
    Use Gemini to generate a detailed video prompt from article content.
    Returns dict with script, appearance, actions, setting, and Veo prompts.
    """
    if not GEMINI_API_KEY:
        print("  Error: GEMINI_API_KEY not set")
        return None

    client = genai.Client(api_key=GEMINI_API_KEY)

    title = article_data.get('title', '')
    excerpt = article_data.get('excerpt', '')
    body = article_data.get('body_markdown', '')
    categories = ', '.join(article_data.get('categories', []))
    keywords = ', '.join(article_data.get('keywords', []))

    # Load recent outfits to avoid repetition
    previous_outfits = load_outfit_history()

    # Build previous outfits context
    outfit_context = ""
    if previous_outfits:
        outfit_context = "\n\nPREVIOUS OUTFITS TO AVOID:\n"
        for i, outfit in enumerate(previous_outfits, 1):
            outfit_context += f"{i}. {outfit}\n"
        outfit_context += "\nDo NOT repeat any of these looks. Create something completely different.\n"

    # Randomly select video format
    video_format = random.choice(['static', 'walk-and-talk', 'location-tour'])
    print(f"  Selected format: {video_format}")

    # Build format-specific instructions
    if video_format == 'walk-and-talk':
        format_context = """

VIDEO FORMAT: WALK AND TALK
This video features Valentina walking while being filmed by a friend following alongside her.

IMPORTANT: See PROSTHETIC LEG PLACEMENT section. Prosthetic is on Valentina's LEFT leg (her anatomical left). Apply in all Veo prompts regardless of camera angle.
IMPORTANT: Valentina must NOT be holding a phone or any device. Both hands are free for gesturing.

Setting options (pick ONE):
• Urban outdoor: sidewalk, city street, urban environment
• Indoor casual: walking through home, office, or indoor space
• Natural outdoor: park path, trail, outdoor setting

Camera style:
• Third-person filming — a friend is walking alongside or in front of Valentina, filming her
• NOT a selfie — Valentina is NOT holding the camera. She has both hands free.
• Natural handheld sway from the friend walking — NOT stabilized, NOT smooth
• Mix of angles: mostly front-facing, occasionally from the side or slightly behind as the friend repositions
• Dynamic framing: sometimes closer (face/upper body), sometimes wider (showing full body including legs)
• Vary framing throughout: tight shots for emphasis, wider shots to show her walking and prosthetic naturally

Prosthetic visibility:
• At least one or two wider shots naturally showing the prosthetic (on her left leg)
• Present it naturally — not hidden, not spotlighted, just part of who she is
• NEVER mention or reference the prosthetic in the dialogue

Tone for walk-and-talk:
• More casual and energetic than static format
• Conversational like talking to a friend while walking
• Natural gestures with both hands (pointing, gesturing, expressive hand movements)
• Slightly breathier/more dynamic delivery

Movement description for Veo prompts:
• Describe her walking naturally (steady pace, not rushing) with prosthetic visible on her left leg
• Friend filming walks alongside — camera has natural handheld sway from a walking person
• Valentina's hands are FREE — she uses both hands for natural gestures while talking
• Background changes/moves as she walks — include real-world activity (other people, traffic, ambient life)
• Occasional wider angles show the prosthetic on Valentina's left leg, natural biological right leg
• Include at least 1-2 prompts with wider framing: "Camera pulls back showing Valentina walking, below-knee prosthetic on her left leg, natural biological right leg"
• Environment should feel busy and real, not an empty path or sterile location"""
    elif video_format == 'static':
        format_context = """

VIDEO FORMAT: STATIC
This video features Valentina in a fixed position (seated or standing), filmed by a friend or with the camera resting on a surface nearby.

Camera style:
• Camera resting on a surface or held by a friend — slight micro-drift, NOT tripod-locked
• Occasional subtle wobble or minor shift (surface vibrating, friend's hand shifting)
• Consistent general framing but with natural imperfection
• She maintains same general position

Setting:
• Casual real environment: couch, kitchen counter, desk, bed, car seat — NOT a studio or set
• Natural room lighting (window light, overhead lights) — NOT studio-lit
• Visible everyday clutter in background (not styled or cleaned for the shot)

Tone for static:
• Casual and confident
• Like she's casually sharing something she found out
• Still playful but grounded"""
    elif video_format == 'location-tour':
        format_context = """

VIDEO FORMAT: LOCATION TOUR
This video features Valentina at a TOPIC-RELEVANT location, filmed by a friend (third-person camera). She walks through the space, interacts with the environment, and explains the topic while the setting reinforces the content.

IMPORTANT: See PROSTHETIC LEG PLACEMENT section. Prosthetic is on Valentina's LEFT leg (her anatomical left), regardless of camera angle. Apply in wider shots. NEVER mention in dialogue.

Camera style:
• Third-person filming — a friend is following and filming her
• NOT selfie — the camera is separate from Valentina, showing her in the space
• Mix of shots: medium (waist up), full body, and occasional closer framing for emphasis
• Camera follows her as she moves through the space — natural handheld sway from the person filming
• Camera can be behind, beside, or facing her — varies naturally as the friend repositions
• Occasional wider establishing shots showing Valentina in the full environment

Opening options (pick ONE):
• Walk-in: Camera behind Valentina as she walks INTO the location, she turns to address camera after a few seconds
• Already there: Valentina is already in the location, camera approaches her and she starts talking
• Exploring: Camera catches her already looking at something in the location, she turns to camera to explain

Setting:
• MUST be relevant to the article topic — the location IS part of the content
• Real, lived-in environments with authentic details (signs, objects, architecture, other people)
• Natural ambient lighting from the environment — indoor or outdoor depending on location
• Background should have visual interest and reinforce the topic

Physical interaction (REQUIRED):
• Valentina must physically interact with the environment at least 2-3 times during the video
• Examples: pointing at something relevant, gesturing toward a feature, leaning on a railing, touching a wall/surface, picking up a relevant object
• Interaction should feel natural and support what she's saying — not forced or random
• She talks WHILE interacting, not stopping to interact silently

Tone for location-tour:
• Knowledgeable and curious — like she went to this place to show viewers something interesting
• More documentary/educational energy than the other formats
• Still maintains her personality — witty, confident, slightly playful
• Like a friend giving you a tour of somewhere relevant and explaining what it means for your situation"""

    prompt = f"""Given this article, create a high-retention 22–26 second vertical short-form video optimized for TikTok and Instagram Reels. Every video must maximize watch time, replays, and comments.

ARTICLE TITLE: {title}
ARTICLE SUMMARY: {excerpt}
ARTICLE CONTENT (first 3000 chars): {body[:3000]}
CATEGORIES: {categories}
KEYWORDS: {keywords}
{outfit_context}
{format_context}

PHYSICAL IDENTITY RULE (ABSOLUTE — DO NOT VIOLATE)
Valentina's physical features MUST be consistent across every segment and every video.

PROSTHETIC LEG PLACEMENT (ABSOLUTE):
• Valentina has a below-knee prosthetic on her LEFT leg (her anatomical left)
• Her RIGHT leg is always a natural biological leg — no prosthetic, no metal, no artificial limb on her right leg
• In Veo prompts, use anatomical/character-relative language: "Valentina's left leg has a below-knee prosthetic" — this is camera-angle-agnostic
• NEVER use screen-relative language like "right side of frame" for prosthetic placement — camera angles change, anatomy does not
• Every Veo prompt must include BOTH: positive ("Valentina's left leg has a below-knee prosthetic") AND negative ("her right leg is natural, no prosthetic on her right leg")
• This applies regardless of camera angle — whether she faces toward, away from, or sideways to the camera, her left leg is always the prosthetic

IDENTIFYING FEATURES CONSISTENCY:
• All tattoos must remain in the same placement, size, and design across every segment
• Freckles and sunspots must stay in the same pattern and locations throughout
• Skin marks, moles, and scars must be consistent — never appear/disappear between segments
• These features should match the reference images provided
• NEVER add, remove, or relocate identifying features between segments or videos
• Nail polish, piercings, and other semi-permanent features must remain consistent within a single video

LOCATION MAPPING (IMPORTANT)
Choose a location that is RELEVANT to the article topic. The setting should reinforce and support the content — it's part of the storytelling, not just a backdrop.

Location guidance by topic:
• Personal injury → busy intersection, hospital exterior, physical therapy waiting room, pharmacy
• Medical malpractice → medical office hallway, hospital waiting room, pharmacy aisle, urgent care entrance
• Employment law → office building exterior, break room, parking garage, HR office hallway
• Product liability → store aisle with products, warehouse, kitchen with appliances, retail shelf display
• Premises liability → commercial building entrance, retail store, parking lot, apartment hallway
• Dog bites → neighborhood park, residential sidewalk, veterinary office, dog park
• Workers comp → construction site exterior, factory entrance, warehouse, loading dock
• Motor vehicle → busy street corner, parking lot, auto body shop, highway overpass
• Wrongful death → courthouse steps, memorial park bench, quiet residential street
• Civil rights → government building exterior, school entrance, public plaza
• Class action → corporate office exterior, consumer store, pharmacy counter
• Social Security disability → government office exterior, medical clinic waiting room
• Intellectual property → office space, tech store, creative studio
• Professional malpractice → law office exterior, financial district, courthouse hallway
• General → pick the most visually interesting and topic-relevant real-world location

This applies to ALL video formats. Even static and walk-and-talk videos should be in topic-relevant locations when possible.

ENVIRONMENT INTERACTION RULE
Valentina must physically interact with her surroundings — not just stand in front of them.

Required interactions (include at least 1-2 per video):
• Point at or gesture toward something relevant in the location
• Gesture toward or point at something relevant in the environment
• Touch, lean on, or interact with environmental elements (railing, wall, counter, display)
• Pick up or handle a prop related to the topic
• Walk toward or examine something in the space

Interaction must:
• Feel natural and support what she's saying at that moment
• Happen WHILE she's talking, not as a separate silent action
• Be specific in the Veo prompt (e.g., "Valentina gestures toward the pharmacy counter while saying...")

VIDEO GOAL
Create a direct-to-camera legal explainer where Valentina introduces herself and explains ONE key legal insight from the article.

Tone must feel:
• Smart
• Human
• Clear
• Confident
• Trustworthy
• Playful, flirty, and confident

HOOK (MANDATORY — FIRST 3 SECONDS MUST STOP THE SCROLL)
The hook is the MOST IMPORTANT part of the video. If the first 3 seconds don't grab attention, nothing else matters.

Before writing the script, select ONE hook style and build the first 3 seconds around it. Rotate styles across videos for variety.

STYLE 1: CONTROVERSIAL / SHOCKING
Bold claims, myth-busting, "everything you know is wrong" energy. Makes people stop to argue or agree.
Examples:
• "Your lawyer is probably lying to you about this."
• "Most injury settlements are a scam — here's why."
• "Nobody wants you to know this about your case."
• "Stop trusting your insurance company. Seriously."

STYLE 2: FLIRTY / PROVOCATIVE
Lead with Valentina's personality. The innuendo IS the hook. Suggestive, confident, makes people watch to see where it's going.
Examples:
• "Let me tell you about the most satisfying settlement I've ever seen..."
• "Some cases just need a more... hands-on approach."
• "You'd be surprised how many people don't know how to get properly compensated."
• "I've been thinking about this one all day..."

STYLE 3: DRAMATIC / URGENT
Breaking-news energy. Creates FOMO and fear of missing critical info.
Examples:
• "This new ruling just changed everything for injury victims."
• "If you got hurt at work this year, you need to hear this right now."
• "Something just happened that affects millions of people and nobody's talking about it."
• "I just found out about this and I'm honestly shocked."

HOOK EMOTION RULE
The hook MUST trigger at least ONE:
• Shock
• Curiosity
• Outrage
• Disbelief
• Intrigue
• FOMO
• Urgency
• Concern

If the hook doesn't trigger any of these → rewrite until it does.

VISUAL HOOK (CRITICAL — NOT JUST WORDS)
The first 3 seconds must be visually attention-grabbing too:
• Valentina doing something mid-action (walking toward camera, leaning in, gesturing dramatically)
• Unexpected framing (extreme close-up that pulls back, starting from behind then turning to camera)
• Environment that creates visual interest (busy street, interesting location, movement)
• The visual and verbal hook must work TOGETHER — not just words over a static shot
• Valentina's expression and body language should match the hook energy (shocked face, knowing smirk, urgent lean-in)

HOOK TEST: Would someone scrolling at full speed STOP for this? If not → make it bolder.

UNIVERSAL UNDERSTANDING RULE (CRITICAL)
Simplify the EXPLANATION, not the hook. The hook can be provocative and punchy — simplicity applies to the legal insight and information that follows.

For the explanation sections (not the hook):
• Rewrite legal concepts into plain everyday language
• Use short sentences
• Remove jargon
• Use real life comparisons when helpful
• A teenager and a stressed adult should both understand it after one listen

PERFORMANCE STRUCTURE
0–3 sec → Hook (scroll-stopping opener)
3–8 sec → Problem or myth
8–16 sec → Insight
16–22 sec → Why it matters
22–26 sec → Takeaway or closing thought

HUMOR + SEXUAL INNUENDO RULE (REQUIRED)

Valentina's personality is clever, witty, and unapologetically flirty. Sexual innuendo is a KEY part of her brand — she's bold and doesn't hold back.

INNUENDO REQUIREMENT:
• Every video MUST include at least ONE strong sexual innuendo or double meaning
• More is welcome if they land naturally — don't force extras
• Each innuendo must be UNIQUE — never reuse phrasing from previous videos

INNUENDO TYPES (vary across videos — don't always use the same type):
• Legal-term double meanings (e.g., "firm grip on your case", "go all the way")
• Physical/body humor tied to the topic or setting
• Situational innuendo — something about the specific scenario that sounds suggestive
• Innocent phrase delivered with a knowing look that makes it suggestive
• Callback innuendo — revisit something said earlier with a new suggestive context

Innuendo delivery:
• Each innuendo should be a MOMENT — slight pause, knowing look, let it land
• Delivered with confidence — she knows exactly what she's saying
• Viewers SHOULD catch the double meaning — these are entertainment, not hidden
• Should make viewers smile, laugh, or comment

Overall humor style:
• Clever and witty
• Confident and sexually bold
• The kind of humor that makes you rewatch to catch what she said

AUDIO RULE
Valentina dialogue is the ONLY audio.
ALL spoken dialogue inside Veo prompts MUST be inside "quotes".

Vocal delivery style:
• Conversational tone — like a voice memo to a friend, NOT a news broadcast or teleprompter read
• Vocal variety: pitch shifts for emphasis, occasional light laugh or amused exhale
• Natural breath sounds between phrases (not silent gaps)
• Speak like she's explaining something interesting she just learned, not presenting

SPEECH FLOW RULE (CRITICAL)
Valentina's dialogue must sound like natural, unscripted conversation.

Dialogue pacing:
• Pauses must be under 1 second — no silence longer than a natural breath. Dead air kills watch time.
• NEVER leave gaps between sentences. Dialogue must flow continuously with no extended silence.
• Speed variation: faster when excited about a point, slower when emphasizing something important
• Occasional conversational filler woven in naturally ("honestly", "like", "here's the thing", "I mean")
• Self-corrections and mid-thought pivots are good (e.g., "well actually—", "no wait, it's even worse—")
• Use connecting words that sound natural, not rehearsed ("So get this...", "And the wild part is...", "But here's what nobody tells you...")

Script structure:
• Write as natural speech, not polished prose — include the imperfections real people have
• Should sound like she's responding to someone who just asked "what's going on with this?"
• Energy should fluctuate naturally — more animated for surprising parts, calmer for explanations
• Occasional rhetorical questions to the viewer ("crazy, right?", "you'd think that's obvious, right?")

If the script sounds like it could be a news anchor reading → rewrite to sound like a friend texting you but out loud.

VISUAL RULES
Format MUST be 16:9 horizontal
Valentina mostly maintains eye contact with camera but occasionally glances away naturally (thinking, gesturing at something, looking down briefly)
Lighting must feel natural and ambient — window light, overhead room light, or outdoor daylight. Slight overexposure from windows or uneven lighting is realistic and desirable for phone footage.

Framing:
• Mix medium shots (waist up) and full body shots throughout the video — don't stay locked on one framing
• Full body shots for: establishing the location, showing environment interaction, walking moments, showing prosthetic naturally
• Closer shots for: emphasis moments, emotional delivery, hook lines, key insights
• Show enough of the environment to establish context — don't crop out the setting

CONTINUITY RULE (CRITICAL)
The video MUST feel like ONE CONTINUOUS TAKE from a single recording session.

Camera style:
• Real handheld phone footage feel — visible sway, not gimbal-stabilized
• Natural imperfections: slight drift, occasional minor reframing as she adjusts grip
• Occasional subtle focus hunting moments (phone autofocus behavior)
• NO tripod-locked static feel and NO gimbal-smooth feel — should look like a real person holding a phone
• General framing stays consistent but with natural hand movement throughout

For STATIC format:
• Valentina maintains same general position/pose between segments
• Camera height and angle stay consistent
• No sudden jumps in positioning

For WALK-AND-TALK format:
• Valentina continues walking at same pace — friend filming walks alongside
• Camera maintains similar distance/framing (sometimes closer/wider is okay)
• Background progresses naturally (she's moving forward)
• Natural handheld movement continues throughout — from the friend filming, NOT Valentina
• Valentina must NOT be holding a phone at any point — both hands free for gestures

For LOCATION-TOUR format:
• Camera (held by friend) can be behind, beside, or facing Valentina — varies naturally
• Camera position can shift as the friend repositions (natural movement, not jump cuts)
• Valentina progresses through the space — same location, different areas
• Maintain location continuity — same venue, progressing naturally through it
• Mix of medium and full body shots as friend adjusts distance
• Friend filming may occasionally shift angle (front to side, side to behind) — this is natural

ZERO TOLERANCE FOR VISIBLE CUTS (CRITICAL)
Each extension MUST begin at the EXACT same state as the previous segment ended:
• Valentina's body position, hand placement, and head angle must MATCH the end of the previous clip exactly
• Same lighting quality, direction, and intensity — no shifts at join points
• Same background elements — no sudden changes in what's behind her
• Same camera distance and framing — no jumps in zoom or angle
• Dialogue continues IMMEDIATELY with no pause or silence at the start of any extension

Extension prompts must:
• Describe the continuation state explicitly: "Valentina continues from the same position, same lighting, same background..."
• Never introduce new scene elements, lighting changes, or position shifts at the start
• Match the previous segment's ending momentum (if she was mid-gesture, continue that gesture)

Write extension prompts as: "Continuing seamlessly — Valentina [maintains her exact position/continues walking at the same pace] with the same lighting and framing, speaking without pause..."

If ANY extension would create a visible cut → rewrite the prompt to match the previous segment's ending state exactly.

ORGANIC REALISM RULE (CRITICAL)
The video must look like a real TikTok creator filmed it on her phone. NOT a production.

Scene authenticity:
• Environments must have everyday clutter — not styled or staged (coffee mug on table, shoes by door, jacket on chair)
• Vary between home/casual spaces and outdoor locations — never a studio or set
• Natural ambient lighting ONLY: window light casting real shadows, overhead room lights, outdoor daylight with natural variation
• Mixed color temperatures are realistic and good (warm lamp + cool window light)
• Background should include incidental real-world elements when outdoors (other people walking by, cars passing, ambient activity)

Environment as content:
• The location should reinforce the topic — the setting tells part of the story
• Include real environmental details relevant to the topic (signs, objects, architecture, displays)
• Valentina interacts with her surroundings, not just stands in front of them
• The viewer should be able to guess the topic partly from the setting alone

Positioning variety:
• Vary naturally: sometimes standing, sometimes sitting, sometimes in wheelchair
• Choose what fits the setting and topic best
• Positioning must feel natural and intentional (not random)

Imperfection details (IMPORTANT — these make it look real):
• Hair slightly out of place, wind-blown if outdoors
• Clothes with natural wrinkles, not freshly pressed
• Slight lens flare from bright light sources is okay and realistic
• Shadows falling naturally (not perfectly lit from all angles)
• Background slightly messy/lived-in — this is someone's real space

Movement authenticity:
• Natural micro-movements (weight shifts, breathing, blinking)
• Organic gestures that real people make when talking — not choreographed
• Facial expressions that react to what she's saying (surprise, amusement, concern)
• Occasional fidgeting: adjusting hair, touching face, shifting weight

REALISM TEST:
Would a viewer scrolling TikTok think this is just another creator making content? If it looks like a production or AI → revise prompts.
The goal: indistinguishable from a real person filming herself on her phone.

PHONE CAMERA AESTHETIC RULE
This is filmed on a smartphone, NOT a cinema camera. Always include:
• Smartphone wide-angle lens characteristics (slight barrel distortion at edges, mostly sharp with natural bokeh at distance)
• Auto-exposure behavior: slight brightness shifts when camera moves between light/shadow areas
• Natural phone camera colors — no cinematic color grading, no teal-and-orange, no film grain
• Auto-white-balance: slight color temperature shifts when panning between different light sources
• NEVER describe cinematic lenses (35mm, 50mm, anamorphic) — this is a phone

LENGTH RULE
Script must sound natural when spoken in 22–26 seconds.

FINAL QA + SPELLCHECK + VIDEO QUALITY VERIFICATION (MANDATORY)
Before returning final JSON output, perform ALL checks below.

TEXT QA CHECK
Verify:
• All spelling is correct
• No duplicated words
• No broken sentences
• Closing thought is strong and memorable
• Tone is playful and on-brand

If errors → fix before output.

SCRIPT CLARITY QA
Verify:
• Script is understandable on first listen
• No legal jargon remains
• Sentences are short and natural

If not → simplify.

AI VIDEO RENDER QUALITY QA (CRITICAL)
Ensure final generated video would pass these checks:

Human Realism
• Face proportions natural
• Eyes aligned and realistic
• Mouth sync matches speech
• Hands have correct finger count and structure
• Skin texture realistic (not plastic or melted)

Physical Identity Consistency (CRITICAL)
• Prosthetic is on Valentina's LEFT leg (her anatomical left) — if it appears on her right leg → FIX IMMEDIATELY
• All tattoos consistent in placement, size, and design across every segment
• Freckles, sunspots, moles consistent — never appearing/disappearing between segments
• All identifying features match reference images

Motion Quality
• No jittering
• No morphing between frames
• No limb warping
• Natural blinking and breathing motion

Visual Integrity
• No background melting or warping
• No clothing distortion
• No lighting flicker
• No logo distortions

NO TEXT RULE (ABSOLUTE)
The video must contain ZERO text of any kind:
• No subtitles or captions
• No text overlays or graphics
• No on-screen words or labels
• No signs or documents in background
• No text on clothing or props

If text appears in any form → regenerate immediately.

Camera Quality
• No unnatural zoom jumps
• No frame tearing
• No unnatural depth distortion

If ANY issue detected → regenerate scene description.

FINAL VIDEO QUALITY PASS RULE
If video would look:
AI generated
Uncanny
Distorted
Low quality
Unnatural human

→ Must revise prompts before output.

DIALOGUE SPLITTING RULE (CRITICAL)
The full script MUST be split across the initial_prompt and exactly 2 extension_prompts. Each prompt MUST contain its exact dialogue portion in "quotes". If any prompt has no quoted dialogue, the video will have silent dead air — this is a failure. Split the script roughly evenly (~20 words per segment for a 60-word script).

OUTPUT FORMAT
Return ONLY valid JSON.

{{
"script": "Full 22–26 second spoken script (~55–70 words)",
"appearance": "Describe Valentina's outfit and look for this video. STYLE GUIDE: Athleisure meets professional casual. Tops must be form-fitting with low necklines (V-necks, scoop necks, ribbed tanks, wrap tops) — bust accentuated. Bottoms: fitted leggings, joggers, or tailored jeans. Colors: neutrals and earth tones (black, white, cream, beige, olive, tan, charcoal, rust, sage). Hair: always long black wavy hair, styling can vary (down, ponytail, half-up, loose braid). Accessories: minimal — small earrings, simple chain necklace, or a watch only. Footwear: sneakers, ankle boots, or sandals. NEVER: costumes, corporate suits, glasses, hats, scarves. Must differ from previous outfits listed above — vary the specific top style, bottom, colors, and hair styling while staying within the style guide.",
"actions": "Highly specific gestures tied to exact words being spoken.",
"setting": "Visually interesting environment relevant to topic.",
"initial_prompt": "Veo prompt for first 8 seconds. MUST include the first ~20 words of dialogue in quotes. Include full scene, lighting, camera framing, and gestures tied to specific dialogue words.",
"extension_prompts": [
"Seconds 8–16: MUST include the next ~20 words of dialogue in exact quotes. Dialogue starts IMMEDIATELY — no pause at the start. Continue from the EXACT frame where the previous segment ended (same position, lighting, background, framing). Specific gestures tied to specific dialogue words.",
"Seconds 16–24: MUST include the final ~20 words of dialogue in exact quotes. Dialogue starts IMMEDIATELY — no pause at the start. Continue from the EXACT frame where the previous segment ended (same position, lighting, background, framing). Closing moment with emotional delivery and strong ending."
]
}}

NATURAL CLOSE (IMPORTANT)
The video should end with a strong closing moment, not trail off or feel abrupt.

Closing strategies (pick the best fit):
• Punchy one-liner that summarizes the key insight
• Rhetorical question that makes viewers think and drives comments
• Relatable "what would you do?" moment
• Callback to the hook (creates a loop effect that drives replays)
• Confident closing statement with a knowing look or slight smile
• Valentina can mention casevalue.law casually if it fits, but NEVER as a pitch

Energy during close:
• Slightly warmer and more personal
• Maintain same personality and conversational tone
• Like wrapping up a great story to a friend

Visual consistency:
• No camera changes or setup shifts
• Same position and framing throughout
• Natural body language (maybe a slight lean-in or knowing smile)

CREATIVE VARIATION RULE
Every video must vary:
• Outfit
• Location
• Lighting
• Camera distance
• Emotional tone
• Gesture style

REFERENCE IMAGE OVERRIDE RULE (CRITICAL)
The reference images show Valentina with one specific look, but you MUST create varied appearances within her style guide.

For the "appearance" field — follow Valentina's STYLE GUIDE:
• Athleisure meets professional casual — sporty, polished, approachable
• Tops: ALWAYS form-fitting with low necklines (V-necks, scoop necks, ribbed tanks, wrap tops) — bust accentuated
• Bottoms: fitted leggings, joggers, tailored jeans, fitted trousers
• Colors: neutrals and earth tones ONLY — black, white, cream, beige, olive, tan, charcoal, rust, sage, gray
• Hair: ALWAYS long black wavy hair — vary styling only (down, ponytail, half-up, loose braid, swept to one side)
• Accessories: minimal jewelry ONLY — small earrings, simple chain/necklace, maybe a watch. NO glasses, hats, or scarves
• Footwear: clean sneakers, ankle boots, or sandals depending on setting
• NEVER: costumes, themed outfits, corporate suits, heavy formal wear
• Vary the specific pieces within these constraints — different top style, different bottom, different color combo, different hair styling

The reference images are for facial features only — NOT for outfit/hair consistency.
Be creative within the style constraints above.
"""

    try:
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt,
            config={
                'response_mime_type': 'application/json'
            }
        )

        json_string = response.text.strip()

        if json_string.startswith('```'):
            json_string = re.sub(r'^```(?:json)?\s*\n?', '', json_string)
            json_string = re.sub(r'\n?\s*```\s*$', '', json_string)

        json_string = sanitize_json_control_chars(json_string)
        video_prompt = json.loads(json_string)

        # Validate required fields
        required = ['initial_prompt', 'extension_prompts']
        for field in required:
            if field not in video_prompt:
                print(f"  Error: Video prompt missing required field: {field}")
                return None

        if not isinstance(video_prompt.get('extension_prompts'), list):
            print(f"  Error: extension_prompts is not a list")
            return None

        # Validate dialogue presence in all prompt segments
        dialogue_re = re.compile(r'"[^"]{10,}"')
        all_prompts = [video_prompt.get('initial_prompt', '')] + video_prompt.get('extension_prompts', [])
        for idx, p in enumerate(all_prompts):
            if not dialogue_re.search(p):
                segment_name = "initial" if idx == 0 else f"extension {idx}"
                print(f"  Warning: {segment_name} prompt has no quoted dialogue — video may have silent sections")

        # Post-process: inject critical rules into every Veo prompt
        appearance_desc = video_prompt.get('appearance', '')

        # Condensed rules (no redundant suffix — single injection)
        veo_rules = (
            "RULES: Valentina's LEFT leg has below-knee prosthetic, RIGHT leg is natural. "
            "ZERO text/subtitles/captions/UI/phones in frame. Raw phone footage look. "
        )

        # Extension prefix adds face lock + continuity (since extensions can't use reference images)
        ext_prefix = (
            veo_rules
            + "FACE LOCK: Same exact person as previous segment — Latina woman, defined cheekbones, "
            "warm brown skin with freckles, brown eyes, long black wavy hair. "
            "Face MUST match previous clip exactly. "
            + "CONTINUITY: Continue from EXACT end frame — same position, lighting, background, angle. No visible cut. "
        )
        if appearance_desc:
            ext_prefix += f"APPEARANCE: {appearance_desc} "

        if video_prompt.get('initial_prompt'):
            initial_prefix = veo_rules
            if appearance_desc:
                initial_prefix += f"APPEARANCE: {appearance_desc} "
            video_prompt['initial_prompt'] = initial_prefix + video_prompt['initial_prompt']
        for idx, ext in enumerate(video_prompt.get('extension_prompts', [])):
            video_prompt['extension_prompts'][idx] = ext_prefix + ext

        print(f"  Video script generated ({len(video_prompt.get('script', ''))} chars)")
        return video_prompt

    except json.JSONDecodeError as e:
        print(f"  Error parsing video prompt JSON: {e}")
        return None
    except Exception as e:
        print(f"  Error generating video prompt: {e}")
        return None


# ============================================================
#  PUBLIC API — Router (Flow via useapi.net)
# ============================================================

def generate_tiktok_video(article_data):
    """
    Generate a TikTok video for an article.
    Uses Google Flow via useapi.net (Veo 3.1 Fast).
    """
    slug = article_data.get('slug', 'untitled')
    print(f"\n--- Generating TikTok Video for: {slug} ---")

    os.makedirs(VIDEOS_DIR, exist_ok=True)

    return generate_tiktok_video_flow(article_data)
