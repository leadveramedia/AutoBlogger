"""
TikTok video generation using Veo 3.1.
Generates ~30 second spokesperson videos for evergreen articles.
"""

import os
import re
import json
import time

from google import genai
from google.genai import types

from .config import GEMINI_API_KEY, VIDEOS_DIR, SPOKESPERSON_IMAGES_DIR
from .content import sanitize_json_control_chars
from .outfit_tracking import load_outfit_history, save_outfit

# Constants
POLL_INTERVAL = 20       # seconds between polling
MAX_POLLS = 20           # max polls per clip (~6.7 min)
INITIAL_DURATION = 8     # seconds for first clip
NUM_EXTENSIONS = 3       # number of extensions to reach ~29s

# Supported image extensions
_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
_MIME_TYPES = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}


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
    import random
    video_format = random.choice(['static', 'walk-and-talk'])
    print(f"  Selected format: {video_format}")

    # Build format-specific instructions
    if video_format == 'walk-and-talk':
        format_context = """

VIDEO FORMAT: WALK AND TALK
This video features Valentina filming herself with a front-facing phone camera while walking.

IMPORTANT: Valentina has a prosthetic leg (below-knee prosthetic on her LEFT leg). This should be naturally visible during walk-and-talk videos through varied camera angles and framing.

Setting options (pick ONE):
• Urban outdoor: sidewalk, city street, urban environment
• Indoor casual: walking through home, office, or indoor space
• Natural outdoor: park path, trail, outdoor setting

Camera style:
• Handheld selfie perspective (she's holding the phone with one hand while walking)
• Visible bounce and sway from walking — NOT stabilized, NOT smooth — real phone-in-hand footage
• Occasional arm fatigue moments: slight drift downward, quick reframe back up
• Dynamic angles: sometimes closer (face/upper body), sometimes wider (showing full body including legs)
• Vary framing throughout: tight shots for emphasis, wider shots to show her walking motion and prosthetic naturally
• Occasional off-center framing that she corrects — this is how real people film themselves
• Phone screen may occasionally catch glare or light shift when passing windows/light sources

Prosthetic visibility:
• At least one or two moments should include wider framing that shows her legs/prosthetic as she walks
• Present it naturally - not hidden, not spotlighted, just part of who she is
• The prosthetic should be visible through natural camera angle variation, not a deliberate "reveal"
• NEVER mention or reference the prosthetic in the dialogue - it's purely visual representation

Tone for walk-and-talk:
• More casual and energetic than static format
• Conversational like talking to a friend while walking
• Natural gestures with free hand (pointing, gesturing)
• Slightly breathier/more dynamic delivery

Movement description for Veo prompts:
• Describe her walking naturally (steady pace, not rushing) with left leg prosthetic visible
• Camera has real handheld bounce and sway — she's walking and holding a phone, not using a gimbal
• Background changes/moves as she walks — include real-world activity (other people, traffic, ambient life)
• Occasional wider angles capture her full body including her left leg prosthetic
• Include at least 1-2 prompts with wider framing: "Camera pulls back slightly showing Valentina walking confidently with her left leg prosthetic naturally visible as she continues speaking..."
• The prosthetic is just part of her appearance - treat it matter-of-factly in descriptions
• Environment should feel busy and real, not an empty path or sterile location"""
    else:  # static
        format_context = """

VIDEO FORMAT: STATIC
This video features Valentina in a fixed position (seated or standing) with her phone propped up or held by someone.

Camera style:
• Phone propped on a surface or held by a friend — slight micro-drift, NOT tripod-locked
• Occasional subtle wobble or minor shift (phone slipping slightly, surface vibrating)
• Consistent general framing but with natural imperfection
• She maintains same general position

Setting:
• Casual real environment: couch, kitchen counter, desk, bed, car seat — NOT a studio or set
• Natural room lighting (window light, overhead lights) — NOT studio-lit
• Visible everyday clutter in background (not styled or cleaned for the shot)

Tone for static:
• Casual and confident
• Like she just propped up her phone to share something she found out
• Still playful but grounded"""

    prompt = f"""Given this article, create a high-retention, conversion-focused 22–26 second vertical short-form video optimized for TikTok and Instagram Reels. Every video must maximize watch time, replays, comments, and clicks to casevalue.law.

ARTICLE TITLE: {title}
ARTICLE SUMMARY: {excerpt}
ARTICLE CONTENT (first 3000 chars): {body[:3000]}
CATEGORIES: {categories}
KEYWORDS: {keywords}
{outfit_context}
{format_context}

VIDEO GOAL
Create a direct-to-camera legal explainer where Valentina introduces herself and explains ONE key legal insight from the article.

Tone must feel:
• Smart
• Human
• Clear
• Confident
• Trustworthy
• Slightly playful but professional

Every video MUST end encouraging viewers to go to casevalue.law in a natural and organic way in relation to the context of the 

HOOK PSYCHOLOGY LAYER (MANDATORY)
Before writing the script, you MUST select ONE hook type and build the first 2 seconds around it.

Choose ONE:

Assumption Break
Challenge something people believe is true.

Risk Awareness
Suggest the viewer may be exposed to risk without realizing it.

Curiosity Gap
Tease missing information viewers need to know.

Authority Signal
State a calm insider truth most people don't know.

HOOK EMOTION RULE
The hook MUST trigger ONE:
• Curiosity
• Concern
• Relief
• Urgency

If no emotion → rewrite hook.

UNIVERSAL UNDERSTANDING RULE (CRITICAL)
You MUST break down complex legal or technical ideas so they are understandable to people of all ages, education levels, and backgrounds.

If article is college / legal level:
• Rewrite into plain everyday language
• Use short sentences
• Remove jargon
• Use real life comparisons when helpful
• Prioritize clarity over sounding smart

SIMPLICITY CHECK (MANDATORY)
Confirm before finalizing:
• Teenager can understand
• Stressed adult can understand quickly
• Clear after one listen

If not → simplify again.

PERFORMANCE STRUCTURE
0–2 sec → Hook
2–8 sec → Problem or myth
8–16 sec → Insight
16–22 sec → Why it matters
22–26 sec → CTA

HUMOR + SEXUAL INNUENDO RULE (REQUIRED)

Valentina's personality is clever, witty, and subtly flirty. Sexual innuendo is a KEY part of her brand.

INNUENDO REQUIREMENT:
• Every video MUST include at least one subtle sexual innuendo or double meaning
• Use wordplay related to legal terminology
• Deliver with natural confidence, not awkwardness

Examples of GOOD legal innuendo:
• "We'll handle your case personally and go deep into the details"
• "Building a strong position takes time and the right approach"
• "You need someone who knows how to work it from every angle"
• "Getting satisfied with your settlement"
• "We'll make sure you're fully covered"
• "Let us take the load off your shoulders"

Innuendo delivery:
• Woven naturally into the explanation (not forced or obvious)
• Delivered with a slight knowing smile or playful tone
• Works on two levels: innocent on surface, suggestive if you catch it
• Should make viewers smile, not cringe

Overall humor style:
• Clever and witty
• Confident and playful
• Natural (not try-hard)

Boundaries (but don't let these scare you away from good innuendo):
• No explicit sexual content or crude language
• No morbid, death, or injury jokes
• Innuendo should be subtle enough to fly under the radar

CRITICAL: Do NOT skip innuendo due to over-caution. Valentina is known for clever wordplay — it's expected and required.

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
• Natural micro-pauses for thinking beats (brief, not dead air)
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
• Valentina continues walking at same pace
• Camera maintains similar distance/framing (sometimes closer/wider is okay)
• Background progresses naturally (she's moving forward)
• Natural handheld movement continues throughout

Extension prompts must maintain:
• Same lighting quality and direction
• No scene cuts or hard transitions
• Natural flow of motion and dialogue (no jarring jumps)

Write extension prompts as: "Valentina continues [walking/speaking] from the same [camera angle/setup], maintaining eye contact..."

If extensions would create visible cuts → keep movements subtle and continuous.

ORGANIC REALISM RULE (CRITICAL)
The video must look like a real TikTok creator filmed it on her phone. NOT a production.

Scene authenticity:
• Environments must have everyday clutter — not styled or staged (coffee mug on table, shoes by door, jacket on chair)
• Vary between home/casual spaces and outdoor locations — never a studio or set
• Natural ambient lighting ONLY: window light casting real shadows, overhead room lights, outdoor daylight with natural variation
• Mixed color temperatures are realistic and good (warm lamp + cool window light)
• Background should include incidental real-world elements when outdoors (other people walking by, cars passing, ambient activity)

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
• CTA is present and exact
• Tone matches brand safety rules

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
"Seconds 8–14: Valentina continues [same format - walking if walk-and-talk, same position if static], speaking naturally with dialogue in quotes. Gestures tied to specific words. Camera and lighting consistent.",
"Seconds 14–20: Valentina maintains continuous presence [walking/stationary], emotional tone matching dialogue in quotes. Natural flow from previous segment. [If walking: background progresses naturally. If static: no scene changes.]",
"Seconds 20–26: Valentina naturally transitions from insight to personal relevance ('if this happened to you...') and mentions getting a free evaluation at casevalue.law conversationally. Slightly warmer energy but same personality. [Format-appropriate continuity]. Natural friendly close, not sales pitch."
]
}}

NATURAL CTA INTEGRATION (CRITICAL)

The call-to-action must feel like a natural conclusion to the conversation, not a sales pitch.

Integration style:
• Natural segue: insight → personal relevance → helpful resource
• Example flow: "If something like this happened to you..." → "you can get a free evaluation at casevalue.law"
• Blend the URL and "free evaluation" into conversational language
• NO abrupt shift to sales mode

Energy during CTA:
• Slightly more inviting/warm (subtle shift, not jarring)
• Maintain same personality and conversational tone
• Like a friend offering helpful advice, not selling

Visual consistency:
• No camera changes or setup shifts
• Same position and framing throughout
• Natural body language (maybe a slight lean-in or knowing smile, but subtle)

Language flexibility:
• Must mention: "free evaluation" or "free case evaluation" AND "casevalue.law"
• Can phrase naturally - doesn't need exact wording
• Work it into the sentence flow organically

Good example:
"So if this happened to you, you can actually get a free case evaluation at casevalue.law to see what your situation is worth."

Bad examples:
❌ "Alright, now for the important part - Get your free case evaluation at casevalue.law" (too abrupt)
❌ Suddenly shifts tone to formal/salesy
❌ Hard cut or visual change when mentioning CTA

Quality check: Does the CTA feel like Valentina genuinely helping, or like she's reading an ad? If ad → rewrite naturally.

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

        print(f"  Video script generated ({len(video_prompt.get('script', ''))} chars)")
        return video_prompt

    except json.JSONDecodeError as e:
        print(f"  Error parsing video prompt JSON: {e}")
        return None
    except Exception as e:
        print(f"  Error generating video prompt: {e}")
        return None


def generate_tiktok_video(article_data):
    """
    Generate a ~30 second TikTok spokesperson video for an article.
    Returns the file path to the saved video, or None on failure.
    """
    slug = article_data.get('slug', 'untitled')
    print(f"\n--- Generating TikTok Video for: {slug} ---")

    if not GEMINI_API_KEY:
        print("  Error: GEMINI_API_KEY not set, skipping video generation")
        return None

    os.makedirs(VIDEOS_DIR, exist_ok=True)

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Step 1: Generate video prompt via Gemini
    print("  Step 1: Generating video script and prompts...")
    video_prompt = generate_video_prompt(article_data)
    if not video_prompt:
        return None

    # Step 2: Load reference images
    print("  Step 2: Loading spokesperson reference images...")
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
    print(f"  Step 3: Generating initial {INITIAL_DURATION}s clip...")
    config = types.GenerateVideosConfig(
        aspect_ratio="16:9",
        resolution="720p",
        person_generation="allow_adult",
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

    # Step 4: Extend 3 times
    extension_prompts = video_prompt.get('extension_prompts', [])[:NUM_EXTENSIONS]
    for i, ext_prompt in enumerate(extension_prompts):
        ext_num = i + 1
        print(f"  Step {3 + ext_num}: Generating extension {ext_num}/{NUM_EXTENSIONS}...")

        try:
            ext_config = types.GenerateVideosConfig(
                resolution="720p",
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
        except Exception as e:
            print(f"    Extension {ext_num} error: {e}")
            print(f"    Saving partial video with {i} extension(s)")
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
