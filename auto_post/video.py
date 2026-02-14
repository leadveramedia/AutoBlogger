"""
TikTok video generation for evergreen articles.
Primary: Argil API (avatar-based). Fallback: Veo 3.1 (prompt-based).
"""

import os
import re
import json
import time
import random
import base64
import requests

from google import genai
from google.genai import types

from .config import (GEMINI_API_KEY, VIDEOS_DIR, SPOKESPERSON_IMAGES_DIR,
                     ARGIL_API_KEY, ARGIL_VOICE_ID, ARGIL_BASE_URL,
                     USEAPI_TOKEN, USEAPI_GOOGLE_EMAIL, USEAPI_BASE_URL)
from .content import sanitize_json_control_chars
from .outfit_tracking import load_outfit_history, save_outfit

# --- Argil Constants ---
ARGIL_POLL_INTERVAL = 15     # seconds between polling
ARGIL_MAX_POLLS = 40         # ~10 min max wait

# --- Veo Constants ---
POLL_INTERVAL = 20           # seconds between polling
MAX_POLLS = 20               # max polls per clip (~6.7 min)
INITIAL_DURATION = 8         # seconds for first clip
NUM_EXTENSIONS = 3           # number of extensions to reach ~29s
EXTENSION_PROCESS_DELAY = 30 # seconds to wait for server-side processing before extending
EXTENSION_MAX_RETRIES = 2    # max retries per extension on INVALID_ARGUMENT

# --- Flow (useapi.net) Constants ---
FLOW_POLL_INTERVAL = 15      # seconds between polling
FLOW_MAX_POLLS = 50          # ~12.5 min max per operation
FLOW_MODEL = 'veo-3.1-fast'  # supports R2V, cheapest

# Supported image extensions (for Veo reference images)
_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
_MIME_TYPES = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}


# ============================================================
#  ARGIL VIDEO GENERATION (PRIMARY)
# ============================================================

def _pick_video_style():
    """Randomly select a video style with layout for Argil videos."""
    style_type = random.choice(['talking-head', 'split-screen', 'dynamic'])

    if style_type == 'talking-head':
        return {
            'name': 'talking-head',
            'autoBrolls': None,
            'zoom_range': (1.0, 1.15),
        }
    elif style_type == 'split-screen':
        return {
            'name': 'split-screen',
            'autoBrolls': {
                'enable': True,
                'source': 'AVATAR_ACTION',
                'layout': random.choice(['SPLIT_AVATAR_LEFT', 'SPLIT_AVATAR_RIGHT']),
            },
            'zoom_range': (1.0, 1.2),
        }
    else:
        return {
            'name': 'dynamic',
            'autoBrolls': {
                'enable': True,
                'source': 'AVATAR_ACTION',
                'layout': random.choice([
                    'FULLSCREEN', 'BACKGROUND',
                    'AVATAR_BOTTOM_LEFT', 'AVATAR_BOTTOM_RIGHT',
                ]),
            },
            'zoom_range': (1.0, 1.4),
        }


def _argil_headers():
    """Return headers for Argil API requests."""
    return {'x-api-key': ARGIL_API_KEY, 'Content-Type': 'application/json'}


def _fetch_valentina_avatars():
    """Fetch all usable Valentina avatars from Argil API."""
    try:
        resp = requests.get(f'{ARGIL_BASE_URL}/avatars',
                           headers=_argil_headers(), timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  Error fetching Argil avatars: {e}")
        return []

    avatars = resp.json()
    # Filter: name OR actorName contains "valentina" (case-insensitive) AND status is IDLE
    valentinas = [
        a for a in avatars
        if ('valentina' in a.get('name', '').lower()
            or 'valentina' in a.get('actorName', '').lower())
        and a.get('status') == 'IDLE'
    ]
    print(f"  Found {len(valentinas)} usable Valentina avatar(s) out of {len(avatars)} total")
    for v in valentinas:
        print(f"    - {v['name']} (id: {v['id'][:12]}...)")
    return valentinas


def _generate_avatar_image(description):
    """Generate a new image of Valentina using reference images + description.
    Uses Gemini native image generation to maintain facial consistency.
    Returns image bytes (PNG) or None."""
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Load reference images from assets/
    ref_parts = []
    for filename in sorted(os.listdir(SPOKESPERSON_IMAGES_DIR)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in _IMAGE_EXTENSIONS:
            continue
        filepath = os.path.join(SPOKESPERSON_IMAGES_DIR, filename)
        with open(filepath, 'rb') as f:
            image_bytes = f.read()
        mime_type = _MIME_TYPES.get(ext, 'image/jpeg')
        ref_parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
        if len(ref_parts) >= 2:
            break

    if not ref_parts:
        print("    No reference images found in assets/")
        return None

    print(f"    Loaded {len(ref_parts)} reference image(s) for generation")

    prompt_text = f"""Generate a NEW image of this EXACT person (shown in the reference images).

CRITICAL: The person's face, features, and identity must be IDENTICAL to the reference images.

New image requirements:
- Aspect ratio: 9:16 (vertical/portrait, 1080x1920)
- FRAMING: Upper body / waist-up shot, facing the camera directly
- EXPRESSION: Natural, confident, slight smile — like she's about to speak to the camera
- EYE CONTACT: Looking directly at the camera
- The person is shown: {description}
- Photorealistic, natural lighting, shot on a phone camera
- Single person in frame, face clearly visible and well-lit
- This image will be used as a video avatar — the face must be sharp, centered, and unobstructed
- NO text, NO signs, NO labels, NO watermarks in the image
- NO sunglasses or anything covering the face

Keep the face identical. Only change the outfit, setting, and pose."""

    contents = ref_parts + [prompt_text]

    response = client.models.generate_content(
        model='gemini-2.5-flash-image',
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE'],
        )
    )

    # Extract image from response
    if not response.candidates or not response.candidates[0].content:
        reason = getattr(response.candidates[0], 'finish_reason', 'unknown') if response.candidates else 'no candidates'
        print(f"    Gemini returned no content (reason: {reason})")
        return None

    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.mime_type.startswith('image/'):
            print(f"    Avatar image generated ({len(part.inline_data.data)} bytes)")
            return part.inline_data.data

    print("    No image found in Gemini response")
    return None


def _compress_image_for_upload(image_bytes):
    """Resize and compress image for Argil avatar upload.
    Argil requires: 9:16 or 16:9 aspect ratio, 720p–4K resolution, max 10MB.
    Server JSON body limit is small, so we target ~50KB JPEG (≈70KB base64)."""
    from PIL import Image as PILImage
    import io

    img = PILImage.open(io.BytesIO(image_bytes))
    # Resize to exactly 720x1280 (9:16 portrait, minimum 720p)
    img = img.convert('RGB').resize((720, 1280), PILImage.LANCZOS)
    # Compress as JPEG — try decreasing quality until under 50KB
    for quality in (80, 65, 50, 35):
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        data = buf.getvalue()
        if len(data) <= 50_000:
            print(f"    Resized to 720x1280, compressed: {len(image_bytes)} -> {len(data)} bytes (JPEG q={quality})")
            return data, 'image/jpeg'
    print(f"    Resized to 720x1280, compressed: {len(image_bytes)} -> {len(data)} bytes (JPEG q={quality})")
    return data, 'image/jpeg'


def _create_argil_avatar(image_bytes, name):
    """Create a new Argil avatar from image bytes. Returns avatar dict or None."""
    image_bytes, mime_type = _compress_image_for_upload(image_bytes)
    b64 = base64.b64encode(image_bytes).decode('utf-8')

    payload = {
        'type': 'IMAGE',
        'name': name,
        'datasetImage': {'base64': f'data:{mime_type};base64,{b64}'},
    }
    if ARGIL_VOICE_ID:
        payload['voiceId'] = ARGIL_VOICE_ID

    try:
        resp = requests.post(f'{ARGIL_BASE_URL}/avatars',
                             headers=_argil_headers(), json=payload, timeout=60)
    except requests.RequestException as e:
        print(f"  Avatar creation request failed: {e}")
        return None

    if resp.status_code not in (200, 201):
        print(f"  Avatar creation failed: {resp.status_code} {resp.text[:500]}")
        return None

    avatar = resp.json()
    avatar_id = avatar['id']

    # If already IDLE (no training needed), return immediately
    if avatar.get('status') == 'IDLE':
        print(f"    Avatar created and ready (id: {avatar_id[:12]}...)")
        return avatar

    print(f"    Avatar created, training started (id: {avatar_id[:12]}...)")

    # Poll until IDLE (training complete, ~30s)
    for i in range(20):  # max 5 min
        time.sleep(15)
        try:
            resp = requests.get(f'{ARGIL_BASE_URL}/avatars/{avatar_id}',
                               headers=_argil_headers(), timeout=30)
            status = resp.json().get('status')
        except requests.RequestException as e:
            print(f"    Poll error: {e}")
            continue
        print(f"    Avatar training... status={status} ({(i+1)*15}s)")
        if status == 'IDLE':
            return resp.json()
        if status in ('TRAINING_FAILED', 'REFUSED'):
            print(f"  Avatar training failed: {status}")
            return None

    print(f"  Avatar training timeout")
    return None


def generate_argil_script(article_data):
    """
    Use Gemini to generate a script optimized for Argil's moments format.
    Returns dict with 'script', 'moments' (list of transcript strings ≤250 chars), 'appearance'.
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
    outfit_context = ""
    if previous_outfits:
        outfit_context = "\n\nPREVIOUS OUTFITS TO AVOID:\n"
        for i, outfit in enumerate(previous_outfits, 1):
            outfit_context += f"{i}. {outfit}\n"
        outfit_context += "\nDo NOT repeat any of these looks. Create something completely different.\n"

    prompt = f"""Given this article, create a high-retention 22–26 second vertical short-form video script optimized for TikTok and Instagram Reels. Every video must maximize watch time, replays, and comments.

ARTICLE TITLE: {title}
ARTICLE SUMMARY: {excerpt}
ARTICLE CONTENT (first 3000 chars): {body[:3000]}
CATEGORIES: {categories}
KEYWORDS: {keywords}
{outfit_context}

IMPORTANT: This script will be spoken by an AI avatar named Valentina. You are writing ONLY the spoken dialogue — the avatar system handles all visuals automatically. Focus 100% on making the SCRIPT compelling.

AVATAR IMAGE NOTE: The "appearance" field you provide will be used to GENERATE AN ACTUAL PHOTO of Valentina via AI image generation. This photo becomes her avatar for the video. Write it as an image generation prompt — be specific about outfit, setting/background, pose, and vibe.

VIDEO GOAL
Create a direct-to-camera legal explainer where Valentina explains ONE key legal insight from the article.

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

Valentina's personality is clever, witty, and subtly flirty. Sexual innuendo is a KEY part of her brand.

INNUENDO REQUIREMENT:
• Every video MUST include at least THREE sexual innuendos or double meanings
• Use wordplay related to legal terminology
• Deliver with natural confidence — these should CLEARLY land as double meanings
• Spread them throughout: one near the hook (0–8s), one in the middle (8–18s), one near the close (18–26s)
• Each innuendo must use DIFFERENT wordplay — never repeat the same type of double meaning

Examples of GOOD legal innuendo:
• "We'll handle your case personally and go deep into the details"
• "Building a strong position takes time and the right approach"
• "You need someone who knows how to work it from every angle"
• "Getting satisfied with your settlement"
• "We'll make sure you're fully covered"
• "Let us take the load off your shoulders"
• "Sometimes you need to go all the way to get what you deserve"
• "We know how to get you into a really good position"
• "It's all about knowing when to push harder"
• "Trust me, size matters when it comes to your settlement"
• "We like to get on top of these cases right away"
• "The longer you wait, the harder it gets"
• "We won't stop until you're completely satisfied with the outcome"
• "Let us show you what a firm grip on your case looks like"

Overall humor style:
• Clever and witty
• Confident and flirty
• Bold but fun

Guardrails:
• Keep it fun, not vulgar — no crude or explicit language
• No morbid, death, or injury jokes
• Innuendo should make viewers laugh, not cringe

CRITICAL: THREE innuendos per video is the MINIMUM. Each should clearly land as a double meaning. If you only wrote one or two, go back and add more. Do NOT reduce the count due to over-caution — this is Valentina's brand and viewers expect it.

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

LENGTH RULE
Script must sound natural when spoken in 22–26 seconds. Approximately 55–70 words total.

NATURAL CLOSE (IMPORTANT)
The video should end with a strong closing moment, not trail off or feel abrupt.

Closing strategies (pick the best fit):
• Punchy one-liner that summarizes the key insight
• Rhetorical question that makes viewers think and drives comments
• Relatable "what would you do?" moment
• Callback to the hook (creates a loop effect that drives replays)
• Confident closing statement
• Valentina can mention casevalue.law casually if it fits, but NEVER as a pitch

FINAL QA CHECK
Before returning final JSON output:
• All spelling is correct
• No duplicated words
• No broken sentences
• Closing thought is strong and memorable
• Script is understandable on first listen
• No legal jargon remains
• Sentences are short and natural
• At least THREE innuendos are present

OUTPUT FORMAT
Return ONLY valid JSON.

{{
"script": "Full 22–26 second spoken script (~55–70 words). This is the COMPLETE dialogue Valentina will speak.",
"moments": [
"First segment of dialogue (max 250 characters). This is the hook — the first 3 seconds.",
"Second segment continuing the dialogue (max 250 characters). Problem or myth.",
"Third segment (max 250 characters). The insight.",
"Fourth segment (max 250 characters). Why it matters + closing thought."
],
"appearance": "IMAGE GENERATION PROMPT: Describe Valentina in a specific outfit, setting, and pose for a 9:16 portrait photo. Must include: 1) Outfit details (style, colors, accessories), 2) Background/setting relevant to the article topic, 3) Pose: upper body facing camera with confident expression. Vary wildly from previous videos — different clothing style (streetwear/business/casual/athletic/edgy), different colors, different hairstyle, different setting. Be specific and creative. Example: 'Valentina in a fitted black blazer over a red silk camisole, hair in a sleek low ponytail, standing in front of a courthouse entrance with stone columns, warm afternoon light, confident slight smile, facing camera.'"
}}

MOMENT RULES:
• Split the script into 4-6 natural segments at sentence or clause boundaries
• Each moment MUST be ≤250 characters
• Together, all moments must form the complete script — no words missing or added
• Split at natural pause points (end of sentence, dramatic pause, topic shift)
• The "script" field must be the exact concatenation of all moments
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
        script_data = json.loads(json_string)

        # Validate required fields
        if 'moments' not in script_data or not isinstance(script_data['moments'], list):
            print("  Error: Script missing 'moments' array")
            return None

        if len(script_data['moments']) < 2:
            print("  Error: Need at least 2 moments")
            return None

        # Validate moment lengths
        for i, moment in enumerate(script_data['moments']):
            if len(moment) > 250:
                print(f"  Warning: Moment {i+1} is {len(moment)} chars (max 250), truncating")
                script_data['moments'][i] = moment[:250]

        print(f"  Argil script generated: {len(script_data['moments'])} moments, "
              f"{len(script_data.get('script', ''))} chars total")
        return script_data

    except json.JSONDecodeError as e:
        print(f"  Error parsing Argil script JSON: {e}")
        return None
    except Exception as e:
        print(f"  Error generating Argil script: {e}")
        return None


def _create_and_render_argil_video(slug, script_data, avatar_id, available_gestures, style):
    """Create an Argil video, render it, and poll until done.
    Returns the video download URL, or None on failure."""
    print(f"  Creating video on Argil (style: {style['name']})...")

    moments = []
    zoom_min, zoom_max = style['zoom_range']
    for i, transcript in enumerate(script_data['moments']):
        moment = {
            'transcript': transcript,
            'avatarId': avatar_id,
            'zoom': {
                'level': round(random.uniform(zoom_min, zoom_max), 2),
            },
        }
        if ARGIL_VOICE_ID:
            moment['voiceId'] = ARGIL_VOICE_ID
        if available_gestures and i > 0 and random.random() < 0.5:
            moment['gestureSlug'] = random.choice(available_gestures)
        moments.append(moment)

    payload = {
        'name': f"CaseValue - {slug}",
        'moments': moments,
        'aspectRatio': '9:16',
        'subtitles': {'enable': False},
    }

    if style['autoBrolls']:
        payload['autoBrolls'] = style['autoBrolls']

    try:
        resp = requests.post(f'{ARGIL_BASE_URL}/videos',
                             headers=_argil_headers(), json=payload, timeout=120)
    except requests.RequestException as e:
        print(f"    Argil API connection error: {e}")
        return None

    if resp.status_code not in (200, 201):
        print(f"    Argil create failed: {resp.status_code} {resp.text[:500]}")
        return None

    video_data = resp.json()
    video_id = video_data['id']
    print(f"    Video created: {video_id}")

    # Trigger render
    print("    Starting render...")
    try:
        resp = requests.post(f'{ARGIL_BASE_URL}/videos/{video_id}/render',
                             headers=_argil_headers(), timeout=30)
    except requests.RequestException as e:
        print(f"    Argil render connection error: {e}")
        return None

    if resp.status_code != 200:
        print(f"    Argil render failed: {resp.status_code} {resp.text[:500]}")
        return None

    render_status = resp.json().get('status', 'unknown')
    print(f"    Render started (status: {render_status})")
    if render_status == 'FAILED':
        print("    Render immediately failed")
        return None

    # Poll until DONE
    print("    Waiting for video generation...")
    for poll in range(ARGIL_MAX_POLLS):
        time.sleep(ARGIL_POLL_INTERVAL)
        try:
            resp = requests.get(f'{ARGIL_BASE_URL}/videos/{video_id}',
                               headers=_argil_headers(), timeout=30)
        except requests.RequestException as e:
            print(f"    Poll error: {e}")
            continue

        result = resp.json()
        status = result.get('status')
        elapsed = (poll + 1) * ARGIL_POLL_INTERVAL
        print(f"    Polling... status={status} ({elapsed}s elapsed)")

        if status == 'DONE':
            return result.get('videoUrl')
        if status == 'FAILED':
            print(f"    Video generation FAILED")
            return None

    print(f"    Timeout after {ARGIL_MAX_POLLS * ARGIL_POLL_INTERVAL}s")
    return None


def generate_tiktok_video_argil(article_data):
    """
    Generate a TikTok video via Argil API.
    Returns the file path to the saved video, or None on failure.
    """
    slug = article_data.get('slug', 'untitled')

    # Step 1: Generate script via Gemini
    print("  Step 1: Generating video script for Argil...")
    script_data = generate_argil_script(article_data)
    if not script_data:
        return None

    # Step 1.5: Generate a new avatar image using Gemini + reference images
    print("  Step 1.5: Generating avatar image...")
    appearance = script_data.get('appearance', '')
    avatar_image = None
    if appearance:
        try:
            avatar_image = _generate_avatar_image(appearance)
        except Exception as e:
            print(f"    Image generation failed: {e}")

    # Step 1.6: Create avatar on Argil (or fall back to existing)
    available_gestures = []
    if avatar_image:
        print("  Step 1.6: Creating new Argil avatar...")
        avatar_name = f"Auto-{slug[:30]}"
        avatar = _create_argil_avatar(avatar_image, avatar_name)
        if avatar:
            avatar_id = avatar['id']
            print(f"    New avatar created: {avatar_name} (id: {avatar_id[:12]}...)")
            available_gestures = [g['slug'] for g in avatar.get('gestures', []) if g.get('slug')]
        else:
            avatar_image = None  # fall through to existing avatars

    if not avatar_image:
        print("  Step 1.6: Falling back to existing Valentina avatars...")
        valentinas = _fetch_valentina_avatars()
        if not valentinas:
            print("  No usable avatars found")
            return None
        selected = random.choice(valentinas)
        avatar_id = selected['id']
        print(f"    Selected: {selected['name']} (id: {avatar_id[:12]}...)")
        # Fetch full avatar data to get gestures
        try:
            full_resp = requests.get(f'{ARGIL_BASE_URL}/avatars/{avatar_id}',
                                      headers=_argil_headers(), timeout=30)
            full_avatar = full_resp.json()
            available_gestures = [g['slug'] for g in full_avatar.get('gestures', []) if g.get('slug')]
        except Exception:
            available_gestures = []

    if available_gestures:
        print(f"    Available gestures: {available_gestures}")

    # Step 2: Create, render, and poll video on Argil
    style = _pick_video_style()
    video_url = _create_and_render_argil_video(
        slug, script_data, avatar_id, available_gestures, style
    )

    # If autoBrolls style failed, retry with talking-head
    if not video_url and style['autoBrolls']:
        print("  Retrying as talking-head (no autoBrolls)...")
        style = {'name': 'talking-head', 'autoBrolls': None, 'zoom_range': (1.0, 1.15)}
        video_url = _create_and_render_argil_video(
            slug, script_data, avatar_id, available_gestures, style
        )

    if not video_url:
        return None

    # Step 5: Download and save
    print("  Step 5: Downloading video...")
    output_path = os.path.join(VIDEOS_DIR, f"{slug}.mp4")
    try:
        dl_resp = requests.get(video_url, timeout=120)
        dl_resp.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(dl_resp.content)
    except Exception as e:
        print(f"  Error downloading video: {e}")
        return None

    file_size = os.path.getsize(output_path)
    print(f"  Video saved: {output_path} ({file_size / 1024 / 1024:.1f} MB)")

    # Save outfit to history for future variation
    try:
        outfit = script_data.get('appearance', '')
        if outfit:
            save_outfit(outfit)
    except Exception as e:
        print(f"  Warning: Could not save outfit to history: {e}")

    return output_path


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


def _flow_generate_clip(prompt, ref_ids=None):
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

    # Add reference images for R2V mode (max 3)
    if ref_ids:
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
        print(f"    Got {len(ref_ids)} reference ID(s) for R2V mode")

    # Step 3: Generate initial 8s clip with reference images
    print("  [Flow] Step 3: Generating initial 8s clip...")
    clip1_id, clip1_url = _flow_generate_clip(
        video_prompt['initial_prompt'], ref_ids=ref_ids
    )
    if not clip1_id:
        print("  Initial clip generation failed")
        return None
    print(f"    Initial clip ready: {clip1_id[:40]}...")

    # Step 4: Extend 2x with continuation prompts
    media_ids = [clip1_id]
    extension_prompts = video_prompt.get('extension_prompts', [])[:2]
    for i, ext_prompt in enumerate(extension_prompts):
        print(f"  [Flow] Step {4 + i}: Extending clip ({i + 1}/{len(extension_prompts)})...")
        ext_id, ext_url = _flow_extend_clip(media_ids[-1], ext_prompt)
        if not ext_id:
            print(f"    Extension {i + 1} failed, using partial video")
            break
        media_ids.append(ext_id)
        print(f"    Extension {i + 1} ready: {ext_id[:40]}...")

    # Step 5: Concatenate or download single clip
    if len(media_ids) >= 2:
        print(f"  [Flow] Step 6: Concatenating {len(media_ids)} clips...")
        video_data = _flow_concatenate(media_ids)
    else:
        # Only 1 clip — download directly from URL
        print("  [Flow] Step 6: Downloading single clip...")
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


# ============================================================
#  VEO VIDEO GENERATION (LEGACY — kept for reference)
# ============================================================

def _load_spokesperson_images():
    """Load all reference images from the assets directory (up to 3 for Veo)."""
    if not os.path.exists(SPOKESPERSON_IMAGES_DIR):
        print(f"  Warning: Assets directory not found at {SPOKESPERSON_IMAGES_DIR}")
        print(f"  Video will be generated without character reference")
        return []

    images = []
    for filename in sorted(os.listdir(SPOKESPERSON_IMAGES_DIR)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in _IMAGE_EXTENSIONS:
            continue
        filepath = os.path.join(SPOKESPERSON_IMAGES_DIR, filename)
        try:
            with open(filepath, 'rb') as f:
                image_bytes = f.read()
            mime_type = _MIME_TYPES.get(ext, 'image/jpeg')
            images.append(types.Image(image_bytes=image_bytes, mime_type=mime_type))
            print(f"  Loaded reference image: {filename}")
        except Exception as e:
            print(f"  Warning: Could not load {filename}: {e}")

        if len(images) >= 3:  # Veo 3.1 supports max 3 reference images
            break

    if not images:
        print(f"  Warning: No reference images found in {SPOKESPERSON_IMAGES_DIR}")
    else:
        print(f"  Loaded {len(images)} reference image(s)")

    return images


def _poll_video_operation(client, operation, label="video"):
    """Poll a video generation operation until done or timeout."""
    polls = 0
    while not operation.done:
        if polls >= MAX_POLLS:
            print(f"    Timeout waiting for {label} after {polls * POLL_INTERVAL}s")
            return None
        time.sleep(POLL_INTERVAL)
        operation = client.operations.get(operation)
        polls += 1
        elapsed = polls * POLL_INTERVAL
        print(f"    Polling {label}... ({elapsed}s elapsed)")

    if operation.error:
        print(f"    {label} generation failed: {operation.error}")
        return None

    if not operation.result or not operation.result.generated_videos:
        print(f"    {label} generation returned no video")
        return None

    return operation.result.generated_videos[0]


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

IMPORTANT: See PROSTHETIC LEG PLACEMENT section. Prosthetic is on the RIGHT side of the screen. Apply in all Veo prompts.
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
• At least one or two wider shots naturally showing the prosthetic (right side of frame)
• Present it naturally — not hidden, not spotlighted, just part of who she is
• NEVER mention or reference the prosthetic in the dialogue

Tone for walk-and-talk:
• More casual and energetic than static format
• Conversational like talking to a friend while walking
• Natural gestures with both hands (pointing, gesturing, expressive hand movements)
• Slightly breathier/more dynamic delivery

Movement description for Veo prompts:
• Describe her walking naturally (steady pace, not rushing) with prosthetic visible on right side of frame
• Friend filming walks alongside — camera has natural handheld sway from a walking person
• Valentina's hands are FREE — she uses both hands for natural gestures while talking
• Background changes/moves as she walks — include real-world activity (other people, traffic, ambient life)
• Occasional wider angles show prosthetic on right side of frame, natural biological leg on left side of frame
• Include at least 1-2 prompts with wider framing: "Camera pulls back showing Valentina walking, prosthetic visible on right side of frame, natural leg on left side of frame"
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

IMPORTANT: See PROSTHETIC LEG PLACEMENT section. Prosthetic on RIGHT side of screen when facing camera. Apply in wider shots. NEVER mention in dialogue.

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
• Valentina has a below-knee prosthetic leg
• The prosthetic is ALWAYS on the RIGHT side of the screen (viewer's right)
• The LEFT side of the screen ALWAYS shows a natural biological leg — no prosthetic, no metal, no artificial limb on the left side of frame
• In Veo prompts, ONLY use screen-relative language: "prosthetic on the right side of frame"
• NEVER write "left leg" or "right leg" in Veo prompts — use "right side of frame" and "left side of frame" only
• Every Veo prompt must include BOTH: positive ("prosthetic on right side of frame") AND negative ("natural leg on left side of frame, no prosthetic")

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

Valentina's personality is clever, witty, and subtly flirty. Sexual innuendo is a KEY part of her brand.

INNUENDO REQUIREMENT:
• Every video MUST include at least THREE sexual innuendos or double meanings
• Use wordplay related to legal terminology
• Deliver with natural confidence — these should CLEARLY land as double meanings
• Spread them throughout: one near the hook (0–8s), one in the middle (8–18s), one near the close (18–26s)
• Each innuendo must use DIFFERENT wordplay — never repeat the same type of double meaning
• Vary delivery across the three: knowing pause, playful smile, raised eyebrow, amused glance

Examples of GOOD legal innuendo:
• "We'll handle your case personally and go deep into the details"
• "Building a strong position takes time and the right approach"
• "You need someone who knows how to work it from every angle"
• "Getting satisfied with your settlement"
• "We'll make sure you're fully covered"
• "Let us take the load off your shoulders"
• "Sometimes you need to go all the way to get what you deserve"
• "We know how to get you into a really good position"
• "It's all about knowing when to push harder"
• "Trust me, size matters when it comes to your settlement"
• "We like to get on top of these cases right away"
• "The longer you wait, the harder it gets"
• "We won't stop until you're completely satisfied with the outcome"
• "Let us show you what a firm grip on your case looks like"

Innuendo delivery:
• Each innuendo should be a MOMENT — slight pause, knowing look, let it land
• Delivered with a clear knowing look, playful pause, or wink-wink tone
• Viewers SHOULD catch the suggestive level — these are entertainment, not hidden easter eggs
• Should make viewers smile and want to comment
• Three different innuendos = three different moments of humor — space them out so each one lands

Overall humor style:
• Clever and witty
• Confident and flirty
• Bold but fun

Guardrails:
• Keep it fun, not vulgar — no crude or explicit language
• No morbid, death, or injury jokes
• Innuendo should make viewers laugh, not cringe

CRITICAL: THREE innuendos per video is the MINIMUM. Each should clearly land as a double meaning. If you only wrote one or two, go back and add more. Do NOT reduce the count due to over-caution — this is Valentina's brand and viewers expect it.

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
• Prosthetic is on RIGHT side of screen — if it appears on left side of screen → FIX IMMEDIATELY
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

OUTPUT FORMAT
Return ONLY valid JSON.

{{
"script": "Full 22–26 second spoken script (~55–70 words)",
"appearance": "Completely unique outfit that is NOTICEABLY DIFFERENT from previous videos. Vary: clothing style (streetwear/business/casual/athletic/edgy), colors, accessories, hairstyle, makeup vibe. Be specific and creative. Reference images show one look but you MUST describe a different appearance for variety.",
"actions": "Highly specific gestures tied to exact words being spoken.",
"setting": "Visually interesting environment relevant to topic.",
"initial_prompt": "Veo prompt for first 6–8 seconds. Include full scene, lighting, camera framing, dialogue in quotes, and gestures tied to specific words.",
"extension_prompts": [
"Seconds 8–14: Dialogue starts IMMEDIATELY — no pause at the start. Continue from the EXACT frame where the previous segment ended (same position, lighting, background, framing). Valentina continues [same format], speaking naturally with dialogue in quotes. Gestures tied to specific words.",
"Seconds 14–20: Dialogue starts IMMEDIATELY — no pause at the start. Continue from the EXACT frame where the previous segment ended (same position, lighting, background, framing). Valentina maintains continuous presence [walking/stationary], emotional tone matching dialogue in quotes. Natural flow. [If walking: background progresses naturally. If static: no scene changes.]",
"Seconds 20–26: Dialogue starts IMMEDIATELY — no pause at the start. Continue from the EXACT frame where the previous segment ended (same position, lighting, background, framing). Valentina wraps up with a memorable closing moment — a punchy takeaway, a thought-provoking rhetorical question, or a callback to the hook. She can mention casevalue.law casually if it fits naturally, but NO sales pitch. Slightly warmer energy but same personality. End like she's finishing a great conversation with a friend."
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
The reference images show Valentina with one specific look, but you MUST create varied appearances.

For the "appearance" field:
• Describe a DIFFERENT outfit style than previous videos
• Specify different colors, different clothing types
• Vary hairstyle: down, up, ponytail, bun, curly, straight, braided
• Vary accessories: jewelry, glasses, hats, scarves
• Vary makeup: natural, bold, minimal, colorful

The reference images are for facial features only — NOT for outfit/hair consistency.
Be bold and creative with the appearance variation.
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

        # Post-process: sandwich critical rules into every Veo prompt
        # 1. Prosthetic placement
        # 2. No text/UI/phone screens
        # 3. Continuity between extensions
        # 4. Appearance consistency
        appearance_desc = video_prompt.get('appearance', '')

        veo_prefix = (
            "ABSOLUTE RULES: "
            "Prosthetic leg on RIGHT side of screen only. Natural leg on LEFT side of screen — no prosthetic on left side. "
            "NO text, NO subtitles, NO captions, NO phone screens, NO UI overlays, NO graphics of any kind in the video. "
            "NO phones visible in Valentina's hands or in the frame. "
        )
        veo_suffix = (
            " REMINDERS: Prosthetic on RIGHT side of screen only. No prosthetic on LEFT side. "
            "ZERO text, subtitles, captions, phone screens, or UI overlays. No phones in frame. "
            "This must look like raw camera footage with no post-production overlays."
        )

        # For extensions, add continuity + appearance
        ext_prefix = (
            veo_prefix
            + "CONTINUITY: Continue from the EXACT frame where the previous segment ended — same body position, same lighting, same background, same camera angle. No visible cut or transition. "
        )
        if appearance_desc:
            ext_prefix += f"APPEARANCE: {appearance_desc} "

        if video_prompt.get('initial_prompt'):
            if appearance_desc:
                video_prompt['initial_prompt'] = veo_prefix + f"APPEARANCE: {appearance_desc} " + video_prompt['initial_prompt'] + veo_suffix
            else:
                video_prompt['initial_prompt'] = veo_prefix + video_prompt['initial_prompt'] + veo_suffix
        for idx, ext in enumerate(video_prompt.get('extension_prompts', [])):
            video_prompt['extension_prompts'][idx] = ext_prefix + ext + veo_suffix

        print(f"  Video script generated ({len(video_prompt.get('script', ''))} chars)")
        return video_prompt

    except json.JSONDecodeError as e:
        print(f"  Error parsing video prompt JSON: {e}")
        return None
    except Exception as e:
        print(f"  Error generating video prompt: {e}")
        return None


def generate_tiktok_video_veo(article_data):
    """
    Generate a ~30 second TikTok spokesperson video via Veo 3.1.
    Returns the file path to the saved video, or None on failure.
    """
    slug = article_data.get('slug', 'untitled')

    if not GEMINI_API_KEY:
        print("  Error: GEMINI_API_KEY not set, skipping Veo video generation")
        return None

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Step 1: Generate video prompt via Gemini
    print("  [Veo] Step 1: Generating video script and prompts...")
    video_prompt = generate_video_prompt(article_data)
    if not video_prompt:
        return None

    # Step 2: Load reference images
    print("  [Veo] Step 2: Loading spokesperson reference images...")
    spokesperson_images = _load_spokesperson_images()
    reference_images = []
    for img in spokesperson_images:
        reference_images.append(
            types.VideoGenerationReferenceImage(
                image=img,
                reference_type="ASSET",
            )
        )

    # Step 3: Generate initial 8-second clip
    print(f"  [Veo] Step 3: Generating initial {INITIAL_DURATION}s clip...")
    config = types.GenerateVideosConfig(
        aspect_ratio="16:9",
        resolution="720p",
        duration_seconds=INITIAL_DURATION,
    )
    if reference_images:
        config.reference_images = reference_images

    operation = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt=video_prompt['initial_prompt'],
        config=config,
    )

    generated = _poll_video_operation(client, operation, "initial clip")
    if not generated:
        return None

    current_video = generated.video
    print(f"    Initial clip generated successfully")

    # Step 4: Extend 3 times (with processing delay and retry)
    extension_prompts = video_prompt.get('extension_prompts', [])[:NUM_EXTENSIONS]
    for i, ext_prompt in enumerate(extension_prompts):
        ext_num = i + 1
        print(f"  [Veo] Step {3 + ext_num}: Generating extension {ext_num}/{NUM_EXTENSIONS}...")

        # Wait for server-side video processing before extending
        print(f"    Waiting {EXTENSION_PROCESS_DELAY}s for video processing...")
        time.sleep(EXTENSION_PROCESS_DELAY)

        ext_success = False
        for retry in range(EXTENSION_MAX_RETRIES + 1):
            try:
                ext_config = types.GenerateVideosConfig(
                    resolution="720p",
                    number_of_videos=1,
                )

                operation = client.models.generate_videos(
                    model="veo-3.1-generate-preview",
                    prompt=ext_prompt,
                    video=current_video,
                    config=ext_config,
                )

                generated = _poll_video_operation(client, operation, f"extension {ext_num}")
                if not generated:
                    print(f"    Extension {ext_num} failed, saving partial video")
                    break

                current_video = generated.video
                print(f"    Extension {ext_num} completed")
                ext_success = True
                break
            except Exception as e:
                error_str = str(e)
                if 'INVALID_ARGUMENT' in error_str and retry < EXTENSION_MAX_RETRIES:
                    print(f"    Extension {ext_num} not ready (attempt {retry + 1}), waiting {EXTENSION_PROCESS_DELAY}s...")
                    time.sleep(EXTENSION_PROCESS_DELAY)
                else:
                    print(f"    Extension {ext_num} error: {e}")
                    print(f"    Saving partial video with {i} extension(s)")
                    break

        if not ext_success:
            break

    # Step 5: Save the video
    output_path = os.path.join(VIDEOS_DIR, f"{slug}.mp4")
    try:
        if not current_video.video_bytes:
            print(f"  Downloading video from remote...")
            client.files.download(file=current_video)
        current_video.save(output_path)
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
    except Exception as e:
        print(f"  Error saving video: {e}")
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
