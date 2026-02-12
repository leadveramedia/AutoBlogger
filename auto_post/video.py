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
EXTENSION_PROCESS_DELAY = 30  # seconds to wait for server-side processing before extending
EXTENSION_MAX_RETRIES = 2     # max retries per extension on INVALID_ARGUMENT

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
• Third-person filming — a friend is walking alongside or in front of Valentina, holding the phone
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
    elif video_format == 'location-tour':
        format_context = """

VIDEO FORMAT: LOCATION TOUR
This video features Valentina at a TOPIC-RELEVANT location, filmed by a friend (third-person camera). She walks through the space, interacts with the environment, and explains the topic while the setting reinforces the content.

IMPORTANT: See PROSTHETIC LEG PLACEMENT section. Prosthetic on RIGHT side of screen when facing camera. Apply in wider shots. NEVER mention in dialogue.

Camera style:
• Third-person filming — a friend is holding the phone and following/filming her
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
• Examples: pointing at something relevant, gesturing toward a feature, leaning on a railing, touching a wall/surface, holding up her phone to show something, picking up a relevant object
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
• Hold up her phone to show something to camera
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

        # Post-process: sandwich prosthetic screen-direction into every Veo prompt (prefix + suffix)
        prosthetic_prefix = "ABSOLUTE RULE: Prosthetic leg on RIGHT side of screen only. Natural leg on LEFT side of screen — no prosthetic on left side. "
        prosthetic_suffix = " REMINDER: Prosthetic on RIGHT side of screen. No prosthetic on LEFT side of screen."
        if video_prompt.get('initial_prompt'):
            video_prompt['initial_prompt'] = prosthetic_prefix + video_prompt['initial_prompt'] + prosthetic_suffix
        for idx, ext in enumerate(video_prompt.get('extension_prompts', [])):
            video_prompt['extension_prompts'][idx] = prosthetic_prefix + ext + prosthetic_suffix

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
        print(f"  Step {3 + ext_num}: Generating extension {ext_num}/{NUM_EXTENSIONS}...")

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
