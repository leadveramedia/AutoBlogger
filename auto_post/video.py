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
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types

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
    Returns dict with 'body', 'face', 'all' ref ID lists, plus 'primary_body' and 'primary_face'.
    Images with 'body' in the filename are categorized as body refs.
    Primary refs (always used for clip generation): ref_full_body.png (body), ref_smile.png (face)."""
    if not os.path.exists(SPOKESPERSON_IMAGES_DIR):
        print("    No assets directory found")
        return {'body': [], 'face': [], 'all': [], 'primary_body': None, 'primary_face': None}

    body_refs = []
    face_refs = []
    primary_body = None
    primary_face = None
    total = 0
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
                is_body = 'body' in filename.lower()
                category = "body" if is_body else "face"
                print(f"    Uploaded {filename} [{category}] -> {media_id[:40]}...")
                if is_body:
                    body_refs.append(media_id)
                else:
                    face_refs.append(media_id)
                total += 1

                # Track primary refs by filename
                if filename == 'ref_full_body.png':
                    primary_body = media_id
                elif filename == 'ref_smile.png':
                    primary_face = media_id
            else:
                print(f"    Upload response missing mediaGenerationId: {result}")
        except Exception as e:
            print(f"    Error uploading {filename}: {e}")

        if total >= 10:
            break

    # Fallback if primary files weren't found
    if not primary_body and body_refs:
        primary_body = body_refs[0]
    if not primary_face and face_refs:
        primary_face = face_refs[0]

    print(f"    {total} reference image(s) uploaded ({len(body_refs)} body, {len(face_refs)} face)")
    return {
        'body': body_refs, 'face': face_refs, 'all': body_refs + face_refs,
        'primary_body': primary_body, 'primary_face': primary_face,
    }


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


FLOW_GENERATION_RETRIES = 3  # retries on clip generation/extension failure


def _flow_post_with_retry(url, payload, label="request"):
    """POST to a Flow endpoint with unlimited retries on 403 captcha failures.
    Returns (mediaGenerationId, video_url) or (None, None)."""
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.post(url, headers=_flow_headers(), json=payload, timeout=120)
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
            delay = 15 if attempt % 5 == 0 else 5
            print(f"    Captcha failed (attempt {attempt}), retrying in {delay}s...")
            time.sleep(delay)
            continue

        print(f"    Flow {label} failed: {resp.status_code} {resp.text[:500]}")
        return None, None


def _extract_image_from_response(result):
    """Extract (mediaGenerationId, fifeUrl) from a useapi.net image response.
    Response structure: result.media[0].image.generatedImage.{mediaGenerationId, fifeUrl}"""
    images = _extract_all_images_from_response(result)
    if images:
        return images[0]
    return None, None


def _extract_all_images_from_response(result):
    """Extract ALL (mediaGenerationId, fifeUrl) pairs from a useapi.net image response.
    Returns list of (img_id, img_url) tuples."""
    media = result.get('media', [])
    if not media:
        media = result.get('response', {}).get('media', [])
    if not media or not isinstance(media, list):
        print(f"    No media in image response: {list(result.keys())}")
        return []

    images = []
    for item in media:
        # Nested under media[N].image.generatedImage
        gen_img = item.get('image', {}).get('generatedImage', {})
        if gen_img:
            img_id = gen_img.get('mediaGenerationId')
            img_url = gen_img.get('fifeUrl')
            if img_id:
                images.append((img_id, img_url))
                continue
        # Fallback: try flat structure
        img_id = item.get('mediaGenerationId')
        img_url = item.get('fifeUrl')
        if img_id:
            images.append((img_id, img_url))

    if not images:
        print(f"    Image response missing mediaGenerationId: {list(media[0].keys())}")
    return images


def _flow_post_image_with_retry(url, payload, label="image", return_all=False):
    """POST to a Flow image endpoint with retries on 403 captcha failures.
    If return_all=True, returns list of (id, url) tuples. Otherwise returns single (id, url)."""
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.post(url, headers=_flow_headers(), json=payload, timeout=120)
        except requests.RequestException as e:
            print(f"    Flow {label} error: {e}")
            return [] if return_all else (None, None)

        if resp.status_code == 200:
            if return_all:
                return _extract_all_images_from_response(resp.json())
            return _extract_image_from_response(resp.json())

        if resp.status_code == 201:
            result = resp.json()
            job_id = result.get('jobid') or result.get('jobId')
            if not job_id:
                print(f"    No jobid in async image response: {list(result.keys())}")
                return [] if return_all else (None, None)
            print(f"    Image job queued: {job_id[:60]}...")
            completed = _flow_poll_job(job_id)
            if not completed:
                return [] if return_all else (None, None)
            if return_all:
                return _extract_all_images_from_response(completed)
            return _extract_image_from_response(completed)

        if resp.status_code == 403 and 'reCAPTCHA' in resp.text:
            delay = 15 if attempt % 5 == 0 else 5
            print(f"    Captcha failed (attempt {attempt}), retrying in {delay}s...")
            time.sleep(delay)
            continue

        print(f"    Flow {label} failed: {resp.status_code} {resp.text[:500]}")
        return [] if return_all else (None, None)


def _score_face_similarity(candidate_url, ref_image_path):
    """Use Gemini vision to score how similar a candidate face is to the reference.
    Returns a score 1-10, or 0 on failure."""
    if not GEMINI_API_KEY or not candidate_url:
        return 0
    try:
        # Download candidate image
        resp = requests.get(candidate_url, timeout=30)
        if resp.status_code != 200:
            return 0
        candidate_bytes = resp.content

        # Load reference image
        with open(ref_image_path, 'rb') as f:
            ref_bytes = f.read()

        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                "Rate 1-10 how similar the face in Image A is to Image B. "
                "Consider: bone structure, eye shape/color, nose shape, lip shape, "
                "skin tone, freckle pattern, cheekbone definition, jawline, hair. "
                "10 = clearly the same person. 1 = completely different person. "
                "Respond with ONLY a single number.",
                types.Part.from_bytes(data=candidate_bytes, mime_type='image/png'),
                "Image A (candidate) above. Image B (reference) below.",
                types.Part.from_bytes(data=ref_bytes, mime_type='image/png'),
            ]
        )
        score = int(response.text.strip().split()[0])
        return max(1, min(10, score))
    except Exception as e:
        print(f"    Face scoring error: {e}")
        return 0


def _flow_generate_scene_image(appearance_brief, ref_ids, setting=""):
    """Generate face-matched images of Valentina using nano-banana-pro.
    Generates 4 candidates, scores each with Gemini vision, picks the best.
    Returns (mediaGenerationId, fifeUrl) or (None, None)."""
    if not ref_ids:
        print("    No reference IDs for scene image generation")
        return None, None

    setting_line = setting if setting else 'neutral, well-lit environment'

    prompt = f"""Generate a photorealistic image of the EXACT person shown in the reference images.

PRIORITY #1 — FACE MATCHING (above all else):
- Face, bone structure, skin texture, freckles, and features must be IDENTICAL to the reference images
- If face accuracy and other instructions conflict, ALWAYS prioritize face accuracy
- The person in this image must be immediately recognizable as the same individual in the references

FRAMING: Three-quarter body (head to mid-thigh), vertical 9:16 portrait
BODY TYPE: Size 8-10 (US), toned hourglass figure — fit and proportional, NOT overweight, NOT plus-size, NOT skinny. Must match the full-body reference image.
EXPRESSION: Natural, confident, looking at camera, as if about to speak
LIGHTING: Natural, well-lit face, natural casual lighting
BRIEF STYLING: {appearance_brief}
SETTING/BACKGROUND: {setting_line}

RULES:
- Single person, face clearly visible and well-lit
- Photorealistic — casual, natural style
- NO text, NO signs, NO watermarks
- NO sunglasses or anything covering the face"""

    payload = {
        'prompt': prompt,
        'model': 'nano-banana-pro',
        'aspectRatio': 'portrait',
        'count': 4,
    }

    if USEAPI_GOOGLE_EMAIL:
        payload['email'] = USEAPI_GOOGLE_EMAIL

    # Add reference images (nano-banana-pro supports up to 10)
    for i, ref_id in enumerate(ref_ids, start=1):
        payload[f'reference_{i}'] = ref_id

    candidates = _flow_post_image_with_retry(
        f'{USEAPI_BASE_URL}/images', payload, "scene-image", return_all=True
    )

    if not candidates:
        print("    Scene image generation failed")
        return None, None

    print(f"    Generated {len(candidates)} scene image candidate(s)")

    # If only 1 candidate, use it directly
    if len(candidates) == 1:
        img_id, img_url = candidates[0]
        print(f"    Scene image: {img_id[:40]}...")
        return img_id, img_url

    # Score each candidate against the front-facing reference
    ref_front_path = os.path.join(SPOKESPERSON_IMAGES_DIR, 'ref_front.png')
    if not os.path.exists(ref_front_path):
        # Fallback to first image in assets
        for f in sorted(os.listdir(SPOKESPERSON_IMAGES_DIR)):
            if f.endswith(('.png', '.jpg', '.jpeg')):
                ref_front_path = os.path.join(SPOKESPERSON_IMAGES_DIR, f)
                break

    best_id, best_url, best_score = None, None, 0
    for idx, (img_id, img_url) in enumerate(candidates):
        score = _score_face_similarity(img_url, ref_front_path)
        print(f"    Candidate {idx + 1}: score={score}/10")
        if score > best_score:
            best_id, best_url, best_score = img_id, img_url, score

    if best_id:
        print(f"    Best scene image: score={best_score}/10, id={best_id[:40]}...")
    else:
        print("    All candidates scored 0 — using first candidate")
        best_id, best_url = candidates[0]
    return best_id, best_url


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


def _flow_upscale_clip(media_id, resolution='1080p'):
    """Upscale a video clip to 1080p or 4K via Google Flow.
    Returns (mediaGenerationId, video_url) or (None, None)."""
    payload = {
        'mediaGenerationId': media_id,
        'resolution': resolution,
        'async': True,
    }

    return _flow_post_with_retry(f'{USEAPI_BASE_URL}/videos/upscale', payload, "upscale")


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


def _local_concatenate_with_crossfade(clip_urls, crossfade_duration=0.3, trim_extensions=1.0):
    """Download clips and concatenate locally with ffmpeg crossfade.
    Returns video bytes or None."""
    import subprocess
    import tempfile

    ffmpeg = _get_ffmpeg()
    if not ffmpeg or len(clip_urls) < 2:
        return None

    tmp_dir = tempfile.mkdtemp(prefix='autoblogger_')
    clip_paths = []

    try:
        # Download each clip
        for i, url in enumerate(clip_urls):
            clip_path = os.path.join(tmp_dir, f'clip_{i}.mp4')
            try:
                dl = requests.get(url, timeout=120)
                dl.raise_for_status()
                with open(clip_path, 'wb') as f:
                    f.write(dl.content)
                clip_paths.append(clip_path)
                print(f"    Downloaded clip {i + 1}: {len(dl.content) / 1024 / 1024:.1f} MB")
            except Exception as e:
                print(f"    Failed to download clip {i + 1}: {e}")
                return None

        # Trim extensions (skip first N seconds) and re-encode to ensure consistent format
        trimmed_paths = []
        for i, path in enumerate(clip_paths):
            if i == 0:
                trimmed_paths.append(path)
            else:
                trimmed = os.path.join(tmp_dir, f'trimmed_{i}.mp4')
                cmd = [
                    ffmpeg, '-i', path,
                    '-ss', str(trim_extensions),
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                    '-c:a', 'aac', '-b:a', '128k',
                    '-y', trimmed
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=60)
                if result.returncode != 0:
                    print(f"    Trim failed for clip {i + 1}: {result.stderr.decode()[:150]}")
                    return None
                trimmed_paths.append(trimmed)

        # Build xfade filter chain for video + acrossfade for audio
        output_path = os.path.join(tmp_dir, 'final.mp4')

        if len(trimmed_paths) == 2:
            # Simple 2-clip crossfade
            # Get duration of first clip
            dur1 = _get_video_duration(ffmpeg, trimmed_paths[0])
            offset = dur1 - crossfade_duration

            cmd = [
                ffmpeg,
                '-i', trimmed_paths[0],
                '-i', trimmed_paths[1],
                '-filter_complex',
                f"[0:v][1:v]xfade=transition=fade:duration={crossfade_duration}:offset={offset}[v];"
                f"[0:a][1:a]acrossfade=d={crossfade_duration}[a]",
                '-map', '[v]', '-map', '[a]',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                '-c:a', 'aac', '-b:a', '128k',
                '-y', output_path
            ]
        elif len(trimmed_paths) == 3:
            # 3-clip crossfade chain
            dur1 = _get_video_duration(ffmpeg, trimmed_paths[0])
            dur2 = _get_video_duration(ffmpeg, trimmed_paths[1])
            offset1 = dur1 - crossfade_duration
            offset2 = offset1 + dur2 - crossfade_duration

            cmd = [
                ffmpeg,
                '-i', trimmed_paths[0],
                '-i', trimmed_paths[1],
                '-i', trimmed_paths[2],
                '-filter_complex',
                f"[0:v][1:v]xfade=transition=fade:duration={crossfade_duration}:offset={offset1}[v01];"
                f"[v01][2:v]xfade=transition=fade:duration={crossfade_duration}:offset={offset2}[v];"
                f"[0:a][1:a]acrossfade=d={crossfade_duration}[a01];"
                f"[a01][2:a]acrossfade=d={crossfade_duration}[a]",
                '-map', '[v]', '-map', '[a]',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                '-c:a', 'aac', '-b:a', '128k',
                '-y', output_path
            ]
        else:
            print(f"    Unsupported clip count for local concat: {len(trimmed_paths)}")
            return None

        result = subprocess.run(cmd, capture_output=True, timeout=180)
        if result.returncode != 0:
            print(f"    Local crossfade concat failed: {result.stderr.decode()[:300]}")
            return None

        with open(output_path, 'rb') as f:
            video_bytes = f.read()
        print(f"    Crossfade concatenated: {len(video_bytes) / 1024 / 1024:.1f} MB")
        return video_bytes

    finally:
        # Clean up temp directory
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _get_video_duration(ffmpeg_path, video_path):
    """Get video duration in seconds using ffprobe."""
    import subprocess
    # Only replace the binary name, not directory parts like 'ffmpeg-full'
    dir_name = os.path.dirname(ffmpeg_path)
    ffprobe = os.path.join(dir_name, 'ffprobe')
    cmd = [
        ffprobe, '-v', 'quiet',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=10)
    try:
        return float(result.stdout.decode().strip())
    except (ValueError, AttributeError):
        return 8.0  # default assumption


# ffmpeg-full path (has drawtext, ass, subtitles filters)
FFMPEG_BIN = '/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg'


def _get_ffmpeg():
    """Return path to ffmpeg with text/subtitle support."""
    import shutil
    if os.path.isfile(FFMPEG_BIN):
        return FFMPEG_BIN
    # Fallback to system ffmpeg
    ffmpeg_path = shutil.which('ffmpeg')
    if not ffmpeg_path:
        print("  Warning: ffmpeg not found - captions and hook text will be skipped")
    return ffmpeg_path


def _overlay_hook_text(video_path, hook_text):
    """Burn bold hook text onto the first 3 seconds of the video using ffmpeg.
    Text scales in from small to full size over 0.5s (stepped), centered on screen."""
    import subprocess

    ffmpeg = _get_ffmpeg()
    if not hook_text:
        return video_path
    if not ffmpeg:
        print(f"    Hook text overlay skipped (ffmpeg not available)")
        return video_path

    def _escape_drawtext(text):
        """Escape special chars for ffmpeg drawtext filter."""
        t = text.replace("\\", "\\\\").replace("'", "'\\''")
        return t.replace(":", "\\:").replace("%", "%%")

    # Stepped scale-in: 5 sizes over 0.5s, then hold at full size until 3s
    SCALE_STEPS = [
        (0, 0.10, 20),
        (0.10, 0.20, 30),
        (0.20, 0.30, 40),
        (0.30, 0.40, 50),
        (0.40, 3.0, 56),
    ]

    def _build_filters_for_text(escaped_text, y_expr):
        """Build chained drawtext filters for one line of text with scale-in."""
        parts = []
        for t_start, t_end, fs in SCALE_STEPS:
            parts.append(
                f"drawtext=text='{escaped_text}'"
                f":fontfile={font}:fontsize={fs}:fontcolor=yellow"
                f":borderw=4:bordercolor=black"
                f":box=1:boxcolor=black@0.6:boxborderw=12"
                f":x=(w-text_w)/2:y={y_expr}"
                f":enable='between(t,{t_start},{t_end})'"
            )
        return ','.join(parts)

    tmp_path = video_path.replace('.mp4', '_tmp.mp4')

    # Find available font (cross-platform)
    font_candidates = [
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux (DejaVu Sans)
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",  # Linux (Liberation Sans)
        "Arial",  # Fallback to font name (if available on system)
    ]
    font = None
    for f in font_candidates:
        if os.path.isfile(f):
            font = f
            break
    if not font:
        font = "DejaVuSans-Bold"  # Last resort - hope it's installed

    # Auto-wrap: split into 2 lines if text is too long for 720px portrait
    if len(hook_text) > 18:
        mid = len(hook_text) // 2
        split_idx = hook_text.rfind(' ', 0, mid + 5)
        if split_idx == -1:
            split_idx = hook_text.find(' ', mid)
        if split_idx != -1:
            line1 = _escape_drawtext(hook_text[:split_idx])
            line2 = _escape_drawtext(hook_text[split_idx + 1:])
            # Center 2-line block: line1 above midpoint, line2 below
            filters1 = _build_filters_for_text(line1, '(h/2)-text_h-6')
            filters2 = _build_filters_for_text(line2, '(h/2)+6')
            filter_str = filters1 + ',' + filters2
        else:
            safe_text = _escape_drawtext(hook_text)
            filter_str = _build_filters_for_text(safe_text, '(h-text_h)/2')
    else:
        safe_text = _escape_drawtext(hook_text)
        filter_str = _build_filters_for_text(safe_text, '(h-text_h)/2')

    cmd = [
        ffmpeg, '-i', video_path,
        '-vf', filter_str,
        '-codec:a', 'copy',
        '-y', tmp_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode == 0:
            os.replace(tmp_path, video_path)
            print(f"    Hook text overlay applied")
            return video_path
        else:
            print(f"    ffmpeg overlay failed: {result.stderr.decode()[:200]}")
    except Exception as e:
        print(f"    ffmpeg overlay error: {e}")

    # Clean up temp file on failure
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return video_path




def generate_tiktok_video_flow(article_data, variant_suffix='', precomputed_prompt=None):
    """Generate a TikTok video via useapi.net Google Flow (Veo 3.1 Fast).

    Args:
        article_data: Article data dict (must have 'slug' key)
        variant_suffix: Optional suffix for output filename (e.g., '_v1', '_v2')
        precomputed_prompt: Optional pre-generated prompt dict (skips generation if provided)

    Returns file path to saved video, or None on failure."""
    slug = article_data.get('slug', 'untitled')

    if not USEAPI_TOKEN:
        print("  Error: USEAPI_TOKEN not set, skipping Flow video generation")
        return None

    # Step 1: Generate prompts (or use precomputed)
    if precomputed_prompt:
        print("  [Flow] Step 1: Using precomputed video script and prompts...")
        video_prompt = precomputed_prompt
    else:
        print("  [Flow] Step 1: Generating video script and prompts...")
        video_prompt = generate_video_prompt(article_data)
        if not video_prompt:
            return None

    # Step 2: Upload reference images
    print("  [Flow] Step 2: Uploading reference images...")
    ref_data = _flow_upload_reference_images()
    ref_ids = ref_data['all']  # flat list for scene image generation
    if ref_ids:
        print(f"    Got {len(ref_ids)} reference ID(s)")

    # Step 3: Generate scene image with nano-banana-pro (face-matched)
    scene_image_id = None
    if ref_ids:
        appearance_brief = video_prompt.get('appearance', 'casual athletic wear')
        setting_brief = video_prompt.get('setting', '')
        print(f"  [Flow] Step 3: Generating face-matched scene image (nano-banana-pro, mode={VIDEO_SEED_MODE})...")
        scene_image_id, scene_url = _flow_generate_scene_image(appearance_brief, ref_ids, setting=setting_brief)
        if scene_url:
            # Save scene image for debugging/review
            try:
                scene_resp = requests.get(scene_url, timeout=30)
                if scene_resp.status_code == 200:
                    scene_path = os.path.join(VIDEOS_DIR, f'{slug}_scene.png')
                    with open(scene_path, 'wb') as f:
                        f.write(scene_resp.content)
                    print(f"    Scene image saved: {scene_path}")
            except Exception as e:
                print(f"    Could not save scene image: {e}")
        if not scene_image_id:
            print("    Falling back to original reference images")

    # Step 4: Generate initial 8s clip (with retries)
    print("  [Flow] Step 4: Generating initial 8s clip...")
    clip1_id, clip1_url = None, None
    for retry in range(FLOW_GENERATION_RETRIES):
        if VIDEO_SEED_MODE == 'i2v' and scene_image_id:
            # I2V mode: scene image becomes the first frame
            clip1_id, clip1_url = _flow_generate_clip(
                video_prompt['initial_prompt'], start_image_id=scene_image_id
            )
        else:
            # R2V mode: fixed primary body ref + face ref + scene image
            clip_refs = [r for r in [ref_data['primary_body'], ref_data['primary_face'], scene_image_id] if r]
            if not clip_refs:
                clip_refs = ref_ids[:3]  # fallback to old behavior
            clip1_id, clip1_url = _flow_generate_clip(
                video_prompt['initial_prompt'], ref_ids=clip_refs
            )
        if clip1_id:
            break
        if retry < FLOW_GENERATION_RETRIES - 1:
            print(f"    Initial clip attempt {retry + 1}/{FLOW_GENERATION_RETRIES} failed, retrying in 5s...")
            time.sleep(5)
    if not clip1_id:
        print(f"  Initial clip generation failed after {FLOW_GENERATION_RETRIES} attempts")
        return None
    print(f"    Initial clip ready: {clip1_id[:40]}...")

    # Step 5: Extend 2x with continuation prompts (with retries)
    media_ids = [clip1_id]
    extension_prompts = video_prompt.get('extension_prompts', [])[:2]
    for i, ext_prompt in enumerate(extension_prompts):
        print(f"  [Flow] Step {5 + i}: Extending clip ({i + 1}/{len(extension_prompts)})...")
        ext_id, ext_url = None, None
        for retry in range(FLOW_GENERATION_RETRIES):
            ext_id, ext_url = _flow_extend_clip(media_ids[-1], ext_prompt)
            if ext_id:
                break
            if retry < FLOW_GENERATION_RETRIES - 1:
                print(f"    Extension {i + 1} attempt {retry + 1}/{FLOW_GENERATION_RETRIES} failed, retrying in 5s...")
                time.sleep(5)
        if not ext_id:
            print(f"    Extension {i + 1} failed after {FLOW_GENERATION_RETRIES} attempts, using partial video")
            break
        media_ids.append(ext_id)
        print(f"    Extension {i + 1} ready: {ext_id[:40]}...")

    # Step 7: Upscale all clips to 1080p (1080x1920 portrait)
    print(f"  [Flow] Step 7: Upscaling {len(media_ids)} clip(s) to 1080p...")
    upscaled_ids = []
    for i, mid in enumerate(media_ids):
        up_id, up_url = _flow_upscale_clip(mid, resolution='1080p')
        if up_id:
            upscaled_ids.append(up_id)
            print(f"    Clip {i + 1} upscaled to 1080p: {up_id[:40]}...")
        else:
            print(f"    Clip {i + 1} upscale failed, using original")
            upscaled_ids.append(mid)

    # Step 8: Concatenate clips (server-side, no fades)
    video_data = None
    if len(upscaled_ids) >= 2:
        print(f"  [Flow] Step 8: Concatenating {len(upscaled_ids)} clips (server)...")
        video_data = _flow_concatenate(upscaled_ids)
    else:
        # Only 1 clip — already upscaled, download from URL
        print("  [Flow] Step 8: Downloading single upscaled clip...")
        # Use the upscaled clip's URL if available
        up_id, up_url = _flow_upscale_clip(media_ids[0], resolution='1080p')
        if up_url:
            try:
                dl = requests.get(up_url, timeout=120)
                dl.raise_for_status()
                video_data = dl.content
            except Exception as e:
                print(f"    Download failed: {e}")
        if not video_data:
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

    # Step 9: Save
    output_path = os.path.join(VIDEOS_DIR, f"{slug}{variant_suffix}.mp4")
    with open(output_path, 'wb') as f:
        f.write(video_data)

    file_size = os.path.getsize(output_path)
    print(f"  Video saved: {output_path} ({file_size / 1024 / 1024:.1f} MB)")

    # Step 10: Overlay hook text
    hook_text = video_prompt.get('hook_text', '')
    if hook_text:
        print(f"  [Flow] Step 8: Overlaying hook text: {hook_text}")
        _overlay_hook_text(output_path, hook_text)

    # Save outfit to history for future variation
    try:
        outfit = video_prompt.get('appearance', '')
        if outfit:
            save_outfit(outfit)
    except Exception as e:
        print(f"  Warning: Could not save outfit to history: {e}")

    return output_path


def generate_video_prompt(article_data, video_format=None, custom_script=None,
                          custom_setting=None, custom_actions=None):
    """
    Use Gemini to generate a detailed video prompt from article content.

    Args:
        article_data: Article data dict
        video_format: Optional video format ('static', 'walk-and-talk', 'location-tour').
                     If None, randomly selects one.
        custom_script: Optional pre-written script text. When provided, Gemini only
                      generates visual elements and splits this exact script into Veo prompts.
        custom_setting: Optional custom setting/location description.
        custom_actions: Optional custom movements/actions description.

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

    # Select video format (or use provided one)
    if video_format is None:
        video_format = random.choice(['static', 'walk-and-talk', 'location-tour'])
    print(f"  Selected format: {video_format}")

    # Build format-specific instructions
    if video_format == 'walk-and-talk':
        format_context = """

VIDEO FORMAT: WALK AND TALK
This video features Valentina walking while being filmed by a friend following alongside her. She is ACTIVELY SPEAKING throughout — her mouth moves with every word, her expressions are animated, and she gestures with both hands.

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
• Vary framing throughout: tight shots for emphasis, wider shots to show her walking naturally

Tone for walk-and-talk:
• More casual and energetic than static format
• Conversational like talking to a friend while walking
• Natural gestures with both hands (pointing, gesturing, expressive hand movements)
• Slightly breathier/more dynamic delivery

Movement description for Veo prompts:
• Describe her walking naturally (steady pace, not rushing)
• Friend filming walks alongside — camera has natural handheld sway from a walking person
• Valentina's hands are FREE — she uses both hands for natural gestures while talking
• Background changes/moves as she walks — include real-world activity (other people, traffic, ambient life)
• Environment should feel busy and real, not an empty path or sterile location"""
    elif video_format == 'static':
        format_context = """

VIDEO FORMAT: STATIC
This video features Valentina in a fixed position (seated or standing), filmed by a friend or with the camera resting on a surface nearby. She is ACTIVELY SPEAKING throughout — her mouth moves, her expressions change, and she gestures naturally.

Camera style:
• Camera resting on a surface or held by a friend — slight micro-drift, NOT tripod-locked
• Occasional subtle wobble or minor shift (surface vibrating, friend's hand shifting)
• Consistent general framing but with natural imperfection
• She maintains same general position but is animated and expressive

Performance (CRITICAL — she must be visibly alive and moving):
• Mouth visibly moves with every word of dialogue — lips, jaw, tongue all animate naturally
• Facial expressions shift constantly: eyebrow raises, smirks, eye widening, knowing looks
• Hand gestures throughout: pointing, counting on fingers, open-palm emphasis, casual waves
• Natural body micro-movements: weight shifts, slight leans, head tilts, shoulder shrugs
• She is TALKING TO CAMERA like a real person — NOT a still photo with audio overlay

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
This video features Valentina at a TOPIC-RELEVANT location, filmed by a friend (third-person camera). She walks through the space, interacts with the environment, and explains the topic while the setting reinforces the content. She is ACTIVELY SPEAKING throughout — her mouth moves with every word, her expressions are animated, and she gestures naturally.

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

    # Build custom script override if provided
    custom_script_override = ""
    if custom_script:
        setting_override = ""
        if custom_setting:
            setting_override = f"""
CUSTOM SETTING (USE THIS EXACT SETTING — do NOT generate your own):
{custom_setting}
Use this setting in the "setting" JSON field verbatim, and incorporate it into all Veo prompts as the location/environment.
"""

        actions_override = ""
        if custom_actions:
            actions_override = f"""
CUSTOM ACTIONS/MOVEMENTS (USE THESE — do NOT generate your own):
{custom_actions}
These are high-level actions. Use them in the "actions" JSON field, and distribute them naturally across the 3 Veo prompt segments (initial_prompt, extension 1, extension 2) as visual descriptions.
"""

        custom_script_override = f"""

CUSTOM SCRIPT MODE (CRITICAL OVERRIDE — READ BEFORE ALL OTHER INSTRUCTIONS)
A pre-written script has been provided. You MUST:
1. Use this script EXACTLY as written in the "script" JSON field — copy it verbatim, do NOT rewrite, paraphrase, add, or remove any words
2. Split this exact script into 3 segments for initial_prompt + 2 extension_prompts (follow the DIALOGUE SPLITTING RULE below)
3. Generate appearance and hook_text that complement this script's content and tone
4. The spoken dialogue in each Veo prompt must use the exact words from the corresponding segment of this script
5. Ignore all instructions about writing hooks, innuendo, humor style, CTA, informative content, or script length — the script is already final and complete

PRE-WRITTEN SCRIPT (USE VERBATIM):
{custom_script}
{setting_override}{actions_override}
For hook_text: Extract the single most shocking or attention-grabbing number, fact, or claim from the script and create a 2-4 word ALL-CAPS overlay text.
"""

    prompt = f"""Given this article, create a high-retention ~22 second vertical short-form video optimized for TikTok and Instagram Reels. Every video must maximize watch time, replays, and comments.

ARTICLE TITLE: {title}
ARTICLE SUMMARY: {excerpt}
ARTICLE CONTENT (first 3000 chars): {body[:3000]}
CATEGORIES: {categories}
KEYWORDS: {keywords}
{outfit_context}
{format_context}
{custom_script_override}
PHYSICAL IDENTITY RULE (ABSOLUTE — DO NOT VIOLATE)
Valentina's physical features MUST be consistent across every segment and every video.

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

HOOK TEXT OVERLAY RULE
A bold text overlay will be burned onto the first 3 seconds of the video. You must provide a "hook_text" field with a SHORT, ALL-CAPS phrase (max 5 words) that:
• Highlights the most shocking number, fact, or claim from the story
• Works VISUALLY — someone scrolling with sound off should stop for this text alone
• Is DIFFERENT from the spoken hook (complementary, not redundant)
• Uses concrete specifics: dollar amounts, company names, shocking verbs
• Examples: "$334K SETTLEMENT", "FIRED FOR REPORTING ABUSE", "FDA RECALL ALERT"

UNIVERSAL UNDERSTANDING RULE (CRITICAL)
Simplify the EXPLANATION, not the hook. The hook can be provocative and punchy — simplicity applies to the legal insight and information that follows.

For the explanation sections (not the hook):
• Rewrite legal concepts into plain everyday language
• Use short sentences
• Remove jargon
• Use real life comparisons when helpful
• A teenager and a stressed adult should both understand it after one listen

INFORMATIVE CONTENT RULE (CRITICAL)
Every video must teach the viewer something SPECIFIC. Do NOT be vague.

REQUIRED in every script:
• At least ONE specific number (settlement amount, fine, number of victims, statute of limitations)
• The company/entity name or specific situation from the article
• ONE actionable insight the viewer can use ("if this happened to you, you have X months to file")

BANNED:
• Vague platitudes ("know your rights", "justice matters", "stay informed")
• Generic legal advice that could apply to anything
• Repeating the same insight across videos
• Talking about the law without connecting it to the specific story

PERFORMANCE STRUCTURE
0–3 sec → Hook (provocative scroll-stopping opener — bold claim, shocking fact, or dramatic question)
3–8 sec → The Story (what actually happened — specific names, numbers, consequences)
8–15 sec → Why You Should Care (authoritative expert — actionable legal insight in plain language)
15–22 sec → Soft CTA + Close (warm, natural sign-off mentioning casevalue.law)

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

VEO SAFETY RULE (CRITICAL — VIDEO GENERATION WILL FAIL WITHOUT THIS)
The VIDEO PROMPTS (initial_prompt, extension_prompts) are sent to Veo for video generation.
Veo has a strict content filter that REJECTS any sexually suggestive visual descriptions.

ALLOWED in video prompts:
• The spoken dialogue in "quotes" CAN contain witty innuendo and double meanings (audio only)
• Valentina can smile, laugh, or give a confident/knowing look

BANNED in video prompts (will cause generation failure):
• Words: "suggestive", "flirty", "seductive", "sultry", "sexual", "sensual", "provocative" (when describing visuals)
• Describing body movements as suggestive or sexual
• Directing Valentina to give "suggestive looks" or "flirty" expressions
• Any description of physical attractiveness or body-focused camera work
• References to "pull out", "going down", or other phrases that read as sexual in visual context

The innuendo lives in the WORDS she speaks, NOT in how the video prompt describes her actions.
Keep all visual action descriptions professional: "she smiles warmly", "confident expression", "gestures toward viewer".

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
• Full body shots for: establishing the location, showing environment interaction, walking moments
• Closer shots for: emphasis moments, emotional delivery, hook lines, key insights
• Show enough of the environment to establish context — don't crop out the setting

CONTINUITY RULE (CRITICAL)
The video MUST feel like ONE CONTINUOUS TAKE from a single recording session.

Camera style:
• Real handheld camera feel — visible sway, not gimbal-stabilized
• Natural imperfections: slight drift, occasional minor reframing as she adjusts grip
• Occasional subtle focus hunting moments (natural autofocus behavior)
• NO tripod-locked static feel and NO gimbal-smooth feel — should look like natural handheld footage
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
• Camera MUST always face Valentina from the FRONT or slight angle — NEVER from behind
• Valentina MUST always face the camera — she NEVER turns her back to it
• Camera position stays mostly consistent — friend walks alongside or in front
• Valentina can gesture at surroundings while FACING the camera (pointing, glancing sideways briefly)
• Maintain location continuity — same venue, same area, same props and objects throughout
• NO new objects, poles, furniture, or structural elements may appear between segments

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
• Vary naturally: sometimes standing, sometimes sitting
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
The goal: indistinguishable from a real creator making content.

CAMERA AESTHETIC RULE
This is filmed casually, NOT on a cinema camera. Always include:
• Smartphone wide-angle lens characteristics (slight barrel distortion at edges, mostly sharp with natural bokeh at distance)
• Auto-exposure behavior: slight brightness shifts when camera moves between light/shadow areas
• Natural phone camera colors — no cinematic color grading, no teal-and-orange, no film grain
• Auto-white-balance: slight color temperature shifts when panning between different light sources
• NEVER describe cinematic lenses (35mm, 50mm, anamorphic) — this is casual handheld

LENGTH RULE
Script must sound natural when spoken in ~22 seconds. Do NOT rush — leave natural pauses and breathing room between sentences.

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
The full script MUST be split across the initial_prompt and exactly 2 extension_prompts. Each prompt MUST contain its exact dialogue portion in "quotes". If any prompt has no quoted dialogue, the video will have silent dead air — this is a failure.

SPLIT AT SENTENCE BOUNDARIES: Always split the script between complete sentences — NEVER split mid-sentence. Segment word counts: initial prompt ~17-18 words, first extension ~13-15 words, FINAL extension ~12 words max. The last segment is shortest because dialogue must complete before the clip ends. Front-load the meatier content in the first two segments.

TRANSITION TIMING
The video is made of 3 clips joined together. Each extension clip's first ~1 second overlaps with the previous clip and gets trimmed.

START-OF-EXTENSION SILENCE: Each extension segment must begin with ~2 seconds of SILENT continuation (same pose, natural breathing, subtle expression shift, NO new words spoken) BEFORE the new dialogue starts. This provides buffer for the clip transition.

Dialogue in each clip should flow NATURALLY to the end — do NOT artificially cut off dialogue early. Let sentences finish naturally. The start-of-extension buffer handles the transition.

OUTPUT FORMAT
Return ONLY valid JSON.

{{
"hook_text": "SHORT ALL-CAPS TEXT for screen overlay during first 3 seconds. Ideally 2-3 words, max 4. Must fit on a narrow portrait screen. Bold, shocking, scroll-stopping. Must highlight the most shocking number, fact, or claim. Examples: '$200K PAYOUT', 'FIRED FOR SAFETY', 'YOUR DOCTOR LIED'. Must be different from the spoken hook.",
"script": "Full spoken script (~42–45 words across ~18 seconds of dialogue). Keep pacing conversational and unhurried — do NOT cram too many words in. Must include specific facts from the article (dollar amounts, company names, what happened). End with a natural casevalue.law mention.",
"appearance": "Describe Valentina's outfit and look for this video. BODY TYPE (MANDATORY — include verbatim in every appearance description): Valentina is 5'6, size 8-10 (US), with a toned hourglass figure — she is fit and proportional, NOT overweight, NOT plus-size, NOT heavy-set, NOT skinny. Think fit Instagram influencer body type. Always include 'size 8-10 toned hourglass figure' in the appearance description. IMPORTANT: Valentina has TWO natural biological legs — NEVER mention a prosthetic, artificial limb, or amputation. STYLE GUIDE — pick ONE category per video and rotate between them: (1) ATHLEISURE: sports bras, ribbed tanks, crop tops, leggings, joggers, sneakers. (2) SEXY/INFLUENCER: bodycon dresses, mini skirts, low-cut tops, off-shoulder tops, heels, thigh-high boots, fitted jeans. (3) SMART CASUAL: blazer over tee or tank, tailored trousers, button-down shirts, midi skirts, loafers, ankle boots. Colors: any — neutrals, earth tones, bold colors, pastels are all fine. Hair: always long red-auburn wavy hair, styling can vary (down, ponytail, half-up, loose braid, swept to one side). Accessories: minimal — small earrings, simple chain necklace, or a watch only. NEVER: costumes, glasses, hats, scarves, prosthetic legs. Must differ from previous outfits listed above — pick a DIFFERENT style category and vary specific pieces, colors, and hair styling.",
"actions": "Highly specific gestures tied to exact words being spoken.",
"setting": "Visually interesting environment relevant to topic.",
"initial_prompt": "Veo prompt for first 8 seconds. MUST include the first ~17-18 words of dialogue in quotes. Include full scene, lighting, camera framing, and gestures tied to specific dialogue words.",
"extension_prompts": [
"Seconds 8–15: First ~2 seconds are SILENT — same pose, natural breathing, subtle expression shift, NO new words spoken (overlap buffer from previous clip). Then at ~2 second mark, begin the next sentence of dialogue in exact quotes (~13-15 words). Continue from the EXACT frame where the previous segment ended (same position, lighting, background, framing). Specific gestures tied to specific dialogue words.",
"Seconds 15–22: First ~2 seconds are SILENT — same pose, natural breathing, subtle expression shift, NO new words spoken (overlap buffer from previous clip). Then at ~2 second mark, begin the final sentence(s) of dialogue in exact quotes (~12 words MAX — this is the LAST clip, dialogue MUST complete before clip ends). Continue from the EXACT frame where the previous segment ended (same position, lighting, background, framing). Keep it short and punchy with emotional delivery."
]
}}

CLOSING + CTA (REQUIRED — EVERY VIDEO)
Every video MUST end with a natural mention of casevalue.law. This is NOT a hard sell — it's a helpful recommendation from someone who cares.

CTA styles (rotate for variety):
• Helpful friend: "If you think this might apply to you, check out casevalue.law — they'll tell you what your case could be worth."
• Curious invitation: "Want to know what a case like this is actually worth? casevalue.law breaks it all down."
• Empowering: "Knowledge is power — head to casevalue.law to find out where you stand."
• Casual mention: "I'll leave a link to casevalue.law in the bio if you want to dig deeper."

CTA RULES:
• Must feel like a natural part of the conversation, NOT a commercial break
• NEVER use "call now", "act fast", "limited time", or other hard-sell language
• Valentina's tone should stay warm and helpful during the CTA
• The CTA should connect to the specific topic discussed (not generic)

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

For the "appearance" field — pick ONE style category per video and ROTATE between them:

CATEGORY 1 — ATHLEISURE: sports bras, ribbed tanks, crop tops, leggings, joggers, sneakers, athletic jackets
CATEGORY 2 — SEXY/INFLUENCER: bodycon dresses, mini skirts, low-cut tops, off-shoulder tops, crop tops with high-waisted jeans, heels, thigh-high boots, fitted dresses
CATEGORY 3 — SMART CASUAL: blazer over tee or tank, tailored trousers, button-down shirts (can be tied or unbuttoned), midi skirts, loafers, ankle boots, cardigans

• Colors: ANY — neutrals, earth tones, bold colors (red, cobalt, emerald), pastels, black, white — full range
• Hair: ALWAYS long red-auburn wavy hair — vary styling (down, ponytail, half-up, loose braid, swept to one side, messy waves)
• Accessories: minimal — small earrings, simple chain/necklace, maybe a watch. NO glasses, hats, or scarves
• Footwear: varies by category — sneakers, heels, boots, sandals, loafers
• NEVER: costumes, themed outfits, glasses, hats, scarves
• MUST pick a DIFFERENT category than previous outfits — maximize variety across videos

The reference images are for facial features only — NOT for outfit/hair consistency.
Be creative within the style constraints above.
"""

    max_retries = 3
    for attempt in range(1, max_retries + 1):
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
            missing_fields = [field for field in required if field not in video_prompt]
            if missing_fields:
                if attempt < max_retries:
                    print(f"  Retry {attempt}/{max_retries-1}: Missing field(s): {', '.join(missing_fields)}")
                    continue
                else:
                    print(f"  Error: Video prompt missing required field(s) after {max_retries} attempts: {', '.join(missing_fields)}")
                    return None

            if not isinstance(video_prompt.get('extension_prompts'), list):
                if attempt < max_retries:
                    print(f"  Retry {attempt}/{max_retries-1}: extension_prompts is not a list")
                    continue
                else:
                    print(f"  Error: extension_prompts is not a list after {max_retries} attempts")
                    return None

            # Validate dialogue presence in all prompt segments
            dialogue_re = re.compile(r'"[^"]{10,}"')
            all_prompts = [video_prompt.get('initial_prompt', '')] + video_prompt.get('extension_prompts', [])
            for idx, p in enumerate(all_prompts):
                if not dialogue_re.search(p):
                    segment_name = "initial" if idx == 0 else f"extension {idx}"
                    print(f"  Warning: {segment_name} prompt has no quoted dialogue — video may have silent sections")

            # Post-process: strip any prosthetic/amputation mentions Gemini may have added
            prosthetic_re = re.compile(
                r'[^.]*(?:prosthetic|artificial limb|amputation|amputee|carbon[- ]fiber leg|metal leg|below[- ]knee)[^.]*\.\s*',
                re.IGNORECASE
            )
            for field in ('appearance', 'initial_prompt', 'actions', 'setting'):
                if field in video_prompt and isinstance(video_prompt[field], str):
                    video_prompt[field] = prosthetic_re.sub('', video_prompt[field])
            if 'extension_prompts' in video_prompt:
                video_prompt['extension_prompts'] = [
                    prosthetic_re.sub('', p) for p in video_prompt['extension_prompts']
                ]

            # Post-process: strip subtitle/caption/text-overlay mentions from Veo prompts
            caption_re = re.compile(
                r'[^.]*(?:subtitle|caption|text overlay|lower third|title card|on-screen text|text on screen)[^.]*\.\s*',
                re.IGNORECASE
            )
            for field in ('initial_prompt', 'actions', 'setting'):
                if field in video_prompt and isinstance(video_prompt[field], str):
                    video_prompt[field] = caption_re.sub('', video_prompt[field])
            if 'extension_prompts' in video_prompt:
                video_prompt['extension_prompts'] = [
                    caption_re.sub('', p) for p in video_prompt['extension_prompts']
                ]

            # Post-process: inject critical rules into every Veo prompt
            appearance_desc = video_prompt.get('appearance', '')

            # Condensed rules (no redundant suffix — single injection)
            veo_rules = (
                "RULES: Valentina MUST be actively speaking — mouth visibly moving, facial expressions changing, "
                "natural hand gestures. She is a real person talking to camera, NOT a still image. Lip movements MUST precisely match spoken audio — every word's mouth shape syncs with the sound. "
                "FACE ON SCREEN AT ALL TIMES (CRITICAL): Valentina's face MUST be clearly visible in EVERY frame of the video. "
                "She MUST face the camera directly — NEVER turn her head away, NEVER look to the side for more than a split second, "
                "NEVER turn her back, NEVER walk away from camera. Her face fills the upper third of the frame at all times. "
                "The camera MUST stay in front of her — no profile shots, no behind shots, no over-shoulder angles. "
                "Valentina has TWO natural biological legs — NO prosthetic leg, NO artificial limb, NO metal leg, NO amputation. "
                "Both legs are completely natural and human. "
                "BODY TYPE: Valentina is size 8-10 (US), toned hourglass figure — fit and proportional. "
                "She is NOT overweight, NOT plus-size, NOT heavy-set, and NOT skinny. Think fit Instagram influencer. "
                "Her body MUST match the full-body reference image exactly. "
                "ABSOLUTELY NO TEXT IN VIDEO (CRITICAL — HIGHEST PRIORITY): "
                "The video MUST contain ZERO text of any kind rendered in the video frames. "
                "NO subtitles, NO captions, NO closed captions, NO text overlays, NO title cards, NO lower thirds, "
                "NO watermarks, NO labels, NO signs with readable text, NO text on clothing, NO UI elements, NO graphics with words. "
                "The ONLY content is Valentina speaking — audio dialogue only, NO visual text whatsoever. "
                "If the model wants to add captions or subtitles to match the dialogue — DO NOT. The audio speaks for itself. "
                "NO phones/cameras in frame. Natural handheld camera feel. "
            )

            # Extension prefix — minimal to let Veo naturally continue voice and visuals
            ext_prefix = (
                veo_rules
                + "CONTINUE from the previous clip's last frame — same person, same body type, same setting, same voice, same camera. "
                "The woman's body MUST remain the same size and proportions as the previous clip — NO weight changes between clips. "
                "First ~2 seconds: SILENT, same pose, natural breathing. New dialogue starts AFTER 2 seconds. "
            )
            if appearance_desc:
                ext_prefix += f"APPEARANCE: {appearance_desc} "

            # No-text reminder placed AFTER dialogue to reinforce the prohibition
            no_text_suffix = (
                " CRITICAL REMINDER: The dialogue above is SPOKEN AUDIO ONLY. "
                "Do NOT render any text, subtitles, or captions in the video frames. NO visual text whatsoever."
            )

            if video_prompt.get('initial_prompt'):
                initial_prefix = veo_rules
                if appearance_desc:
                    initial_prefix += f"APPEARANCE: {appearance_desc} "
                video_prompt['initial_prompt'] = initial_prefix + video_prompt['initial_prompt'] + no_text_suffix
            for idx, ext in enumerate(video_prompt.get('extension_prompts', [])):
                video_prompt['extension_prompts'][idx] = ext_prefix + ext + no_text_suffix

            print(f"  Video script generated ({len(video_prompt.get('script', ''))} chars)")
            return video_prompt

        except json.JSONDecodeError as e:
            if attempt < max_retries:
                print(f"  Retry {attempt}/{max_retries-1}: JSON parsing error: {e}")
                continue
            else:
                print(f"  Error parsing video prompt JSON after {max_retries} attempts: {e}")
                return None
        except Exception as e:
            if attempt < max_retries:
                print(f"  Retry {attempt}/{max_retries-1}: {e}")
                continue
            else:
                print(f"  Error generating video prompt after {max_retries} attempts: {e}")
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


def generate_three_videos(article_data, custom_script=None, custom_setting=None, custom_actions=None, formats=None):
    """
    Generate videos (static, walk-and-talk, location-tour) from the same article
    with different AI-generated prompts. Runs in parallel for efficiency.

    Args:
        article_data: Article data dict (must have 'slug' key)
        custom_script: Optional pre-written script. Passed to generate_video_prompt.
        custom_setting: Optional custom setting/location. Passed to generate_video_prompt.
        custom_actions: Optional custom movements/actions. Passed to generate_video_prompt.
        formats: Optional list of formats to generate. Defaults to all 3.

    Returns list of file paths (only successful videos), or empty list on complete failure.
    """
    slug = article_data.get('slug', 'untitled')

    all_formats = ['static', 'walk-and-talk', 'location-tour']
    all_suffixes = ['_v1', '_v2', '_v3']
    if formats is None:
        formats = all_formats
    pairs = [(s, f) for s, f in zip(all_suffixes, all_formats) if f in formats]
    suffixes = [s for s, _ in pairs]
    formats = [f for _, f in pairs]

    print(f"\n--- Generating {len(formats)} TikTok Video Variant(s) for: {slug} ---")

    os.makedirs(VIDEOS_DIR, exist_ok=True)

    # Pre-generate prompts sequentially to avoid outfit history race conditions
    print(f"\n  Pre-generating prompts for {len(formats)} format(s)...")
    prompts = []
    for fmt in formats:
        print(f"    Generating prompt for {fmt}...")
        prompt = generate_video_prompt(article_data, video_format=fmt, custom_script=custom_script,
                                              custom_setting=custom_setting, custom_actions=custom_actions)
        prompts.append(prompt)

    # Filter out None prompts
    valid_prompts = [(suffix, prompt) for suffix, prompt in zip(suffixes, prompts) if prompt is not None]
    if not valid_prompts:
        print("  Failed to generate any prompts, aborting video generation")
        return []

    # Generate videos in parallel using ThreadPoolExecutor
    print(f"\n  Generating {len(valid_prompts)} videos in parallel...")
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(
                generate_tiktok_video_flow,
                article_data,
                variant_suffix=suffix,
                precomputed_prompt=prompt
            ): suffix
            for suffix, prompt in valid_prompts
        }

        for future in as_completed(futures):
            suffix = futures[future]
            try:
                path = future.result()
                if path:
                    results.append(path)
                    print(f"  ✓ Video {suffix} completed: {path}")
                else:
                    print(f"  ✗ Video {suffix} failed (returned None)")
            except Exception as e:
                print(f"  ✗ Video {suffix} failed with exception: {e}")

    print(f"\n  Generated {len(results)} videos successfully")
    return results
