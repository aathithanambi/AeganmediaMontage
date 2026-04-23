"""STEP 2 — Scene Planning, Script Generation, Character & Location Extraction.

Responsible for:
  - Parsing production intent from raw user prompt
  - Reference video download / analysis / style extraction
  - Script generation (Gemini)
  - Character & location extraction with caching
  - Scene plan generation (standard and timeline-based with batching)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import textwrap
from pathlib import Path
from typing import Any

from webapp.workers.shared import (
    DEFAULT_STYLE,
    IMAGE_STYLE_PREFIX,
    SUPPORTED_LANGUAGES,
    _discover_tools,
    _env_truthy,
    _extract_keywords,
    _gemini_generate,
    _get_tool,
    _google_available,
    _log,
    _parse_json_response,
    _split_script_sections,
)

# ---------------------------------------------------------------------------
# Intent parsing (Gemini)
# ---------------------------------------------------------------------------

def _parse_production_intent(raw_prompt: str, ref_summary: str | None = None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "content_prompt": raw_prompt,
        "audio_language": "english",
        "subtitle_language": "",
        "target_duration": 60,
        "style_notes": "",
        "reference_driven": False,
    }

    if not _google_available():
        dur_match = re.search(r'(\d+)\s*(?:sec|second)', raw_prompt, re.IGNORECASE)
        if dur_match:
            defaults["target_duration"] = int(dur_match.group(1))
        for lang in SUPPORTED_LANGUAGES:
            if lang in raw_prompt.lower():
                if re.search(rf'audio\b.*\b{lang}|{lang}\b.*\baudio|voiceover\b.*\b{lang}', raw_prompt, re.IGNORECASE):
                    defaults["audio_language"] = lang
                if re.search(rf'subtitle\b.*\b{lang}|{lang}\b.*\bsubtitle|caption\b.*\b{lang}', raw_prompt, re.IGNORECASE):
                    defaults["subtitle_language"] = lang
        return defaults

    ref_context = ""
    if ref_summary:
        ref_context = f"\nReference video analysis:\n{ref_summary[:500]}\n"

    llm_prompt = f"""Analyze this video creation request. Separate the PRODUCTION INSTRUCTIONS from the CONTENT TOPIC.

User's raw prompt:
\"\"\"{raw_prompt}\"\"\"
{ref_context}
Return a JSON object with these keys:
- "content_prompt": The actual topic/story/subject to create a video about. Extract ONLY the creative content, NOT production instructions like duration, language, format. If the user mainly says "create something like the reference" without specifying a topic, set this to a description of what the reference video is about.
- "audio_language": Language for voiceover/narration (default "english"). Common: "tamil", "hindi", "telugu", "spanish", etc.
- "subtitle_language": Language for on-screen subtitles. Empty string "" if same as audio or not requested.
- "target_duration": Video length in seconds (integer). Default 60 if not specified.
- "style_notes": Any visual/editing style instructions (e.g., "cinematic", "fast-paced", "same editing as reference")
- "reference_driven": true if the user's main intent is to recreate/match a reference video's content, false if they have their own topic

IMPORTANT: Do NOT include phrases like "I want to create", "make a video about", "with subtitles" in content_prompt. Those are instructions, not content.

Respond ONLY with the JSON object."""

    _log("Parsing production intent via Gemini...")
    result = _gemini_generate(llm_prompt, max_tokens=1024)
    if result:
        try:
            parsed = _parse_json_response(result)
            if isinstance(parsed, dict):
                for key in defaults:
                    if key in parsed and parsed[key] is not None:
                        defaults[key] = parsed[key]
                if isinstance(defaults["target_duration"], str):
                    defaults["target_duration"] = int(re.sub(r'\D', '', defaults["target_duration"]) or "60")
                defaults["audio_language"] = defaults["audio_language"].lower().strip()
                defaults["subtitle_language"] = defaults["subtitle_language"].lower().strip()
                _log(f"Intent parsed: content='{defaults['content_prompt'][:100]}...', "
                     f"audio={defaults['audio_language']}, sub={defaults['subtitle_language']}, "
                     f"dur={defaults['target_duration']}s, ref_driven={defaults['reference_driven']}")
                return defaults
        except (json.JSONDecodeError, ValueError) as e:
            _log(f"Intent parse failed: {e}")

    dur_match = re.search(r'(\d+)\s*(?:sec|second)', raw_prompt, re.IGNORECASE)
    if dur_match:
        defaults["target_duration"] = int(dur_match.group(1))
    return defaults


# ---------------------------------------------------------------------------
# Reference video analysis
# ---------------------------------------------------------------------------

def step_download_reference(ref_url: str, project_dir: Path) -> str | None:
    if not ref_url:
        return None
    tool = _get_tool("video_downloader")
    if tool is None:
        _log("video_downloader unavailable, skipping reference download")
        return None
    out_dir = project_dir / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Downloading reference: {ref_url}")
    result = tool.execute({
        "url": ref_url,
        "output_dir": str(out_dir),
        "format": "video",
        "max_resolution": "720p",
        "max_duration_seconds": 1800,
    })
    if result.success and result.artifacts:
        _log(f"Reference downloaded: {result.artifacts[0]}")
        return result.artifacts[0]
    _log(f"Reference download failed: {result.error}")
    return None


def step_analyze_reference(ref_path: str, project_dir: Path) -> dict[str, Any]:
    analysis: dict[str, Any] = {}

    analyzer = _get_tool("video_analyzer")
    if analyzer:
        _log("Analyzing reference video...")
        result = analyzer.execute({
            "source": ref_path,
            "analysis_depth": "standard",
            "max_keyframes": 8,
            "output_dir": str(project_dir / "analysis"),
        })
        if result.success:
            brief_path = project_dir / "analysis" / "video_analysis_brief.json"
            if brief_path.exists():
                try:
                    analysis = json.loads(brief_path.read_text(encoding="utf-8"))
                    _log(f"Reference analysis complete: {len(analysis)} fields")
                except json.JSONDecodeError:
                    _log("Analysis brief was not valid JSON")
            if result.artifacts:
                analysis["keyframe_paths"] = result.artifacts
        else:
            _log(f"Video analyzer failed: {result.error}")

    return analysis


def _summarize_reference(analysis: dict[str, Any]) -> str:
    parts: list[str] = []
    if "duration" in analysis:
        parts.append(f"Duration: {analysis['duration']:.1f}s")
    if "scene_count" in analysis:
        parts.append(f"Scenes: {analysis['scene_count']}")
    transcript = analysis.get("transcript", {})
    if isinstance(transcript, dict):
        text = transcript.get("text", "")
        if text:
            parts.append(f"Transcript excerpt: {text[:500]}")
    elif isinstance(transcript, str):
        parts.append(f"Transcript excerpt: {transcript[:500]}")
    if "scenes" in analysis and isinstance(analysis["scenes"], list):
        for i, scene in enumerate(analysis["scenes"][:6]):
            if isinstance(scene, dict):
                desc = scene.get("description", scene.get("label", f"Scene {i+1}"))
                parts.append(f"Scene {i+1}: {desc}")
    if "energy_profile" in analysis:
        parts.append(f"Audio energy: {json.dumps(analysis['energy_profile'])[:200]}")
    return "\n".join(parts) if parts else "No analysis available."


def _analyze_reference_style(
    ref_summary: str,
    style_notes: str = "",
) -> dict[str, str]:
    if not _google_available() or not ref_summary:
        return dict(DEFAULT_STYLE)

    notes_block = ""
    if style_notes.strip():
        notes_block = (
            "\nUser style instructions from their prompt (honor these when consistent with the reference):\n"
            f"{style_notes.strip()[:100000]}\n"
        )

    llm_prompt = f"""Analyze this reference video and identify its visual style.

Reference video analysis:
{ref_summary[:800]}
{notes_block}
Return a JSON object:
- "art_style": Describe the visual art style (e.g. "oil painting", "watercolor illustration", "realistic photography", "anime/cartoon", "3D render", "vintage film", "minimalist flat design")
- "image_type": What type of images are used (e.g. "original photos", "AI-generated illustrations", "hand-drawn art", "stock footage stills", "mixed media")
- "editing_style": How scenes transition and are edited (e.g. "slow dissolves", "quick cuts", "smooth pan transitions", "ken burns with fade", "dynamic zoom")
- "color_palette": Dominant color mood (e.g. "warm earthy tones", "cool blue cinematic", "vibrant saturated", "muted pastel", "dark moody")
- "mood": Overall mood/tone (e.g. "dramatic", "cheerful", "nostalgic", "educational", "epic")

Respond ONLY with the JSON object."""

    _log("Analyzing reference visual style via Gemini...")
    result = _gemini_generate(llm_prompt, max_tokens=512)
    if result:
        try:
            parsed = _parse_json_response(result)
            if isinstance(parsed, dict):
                _log(f"Reference style: {parsed.get('art_style', '?')}, type={parsed.get('image_type', '?')}")
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    return dict(DEFAULT_STYLE)


# ---------------------------------------------------------------------------
# Character lock cache
# ---------------------------------------------------------------------------

def _character_lock_fingerprint(
    script_for_visuals: str,
    title: str,
    art_style: str,
    *,
    creator_topic: str = "",
    style_notes: str = "",
) -> str:
    blob = (
        f"{title}\n{art_style}\n{creator_topic[:4000]}\n{style_notes[:1500]}\n"
        f"{script_for_visuals[:12000]}"
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _character_lock_path(assets_dir: Path) -> Path:
    return assets_dir / "character_lock.json"


def _load_character_lock(assets_dir: Path, fingerprint: str) -> dict[str, Any] | None:
    path = _character_lock_path(assets_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("fingerprint") != fingerprint:
        return None
    payload = data.get("character_data")
    if isinstance(payload, dict):
        _log("Character lock cache HIT — skipping character extraction API call")
        return payload
    return None


def _save_character_lock(
    assets_dir: Path, fingerprint: str, character_data: dict[str, Any],
) -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)
    path = _character_lock_path(assets_dir)
    try:
        path.write_text(
            json.dumps(
                {"fingerprint": fingerprint, "character_data": character_data},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _log(f"Character lock saved ({path.name})")
    except OSError as e:
        _log(f"Character lock save failed: {e}")


def _normalize_character_entries(character_data: dict[str, Any]) -> dict[str, Any]:
    for ch in character_data.get("characters") or []:
        if not isinstance(ch, dict):
            continue
        if not ch.get("signature_outfit"):
            ch["signature_outfit"] = (
                ch.get("signature_garment")
                or ch.get("outfit")
                or ""
            ).strip()
        mc = ch.get("main_colors")
        if mc is None:
            mc = ch.get("palette") or ch.get("color_palette")
        if isinstance(mc, str):
            ch["main_colors"] = [c.strip() for c in mc.split(",") if c.strip()]
        elif isinstance(mc, list):
            ch["main_colors"] = [str(c).strip() for c in mc if str(c).strip()]
        else:
            ch["main_colors"] = []
        # Normalize face_lock — fall back to description if missing
        if not ch.get("face_lock"):
            ch["face_lock"] = (ch.get("description") or "").strip()[:300]
        # Normalize hair_color — fall back to extracting from description if missing
        if not ch.get("hair_color"):
            desc_lower = (ch.get("description") or "").lower()
            import re as _re
            hair_match = _re.search(r"([\w\s\-]+hair[\w\s\-]*)", desc_lower)
            ch["hair_color"] = hair_match.group(1).strip()[:120] if hair_match else ""
        # Normalize age_range
        if not ch.get("age_range"):
            ch["age_range"] = ""
    return character_data


# ---------------------------------------------------------------------------
# Character & scene extraction (Gemini)
# ---------------------------------------------------------------------------

def _extract_characters_and_scenes(
    transcript: str,
    title: str,
    ref_style: dict[str, str],
    *,
    creator_topic: str = "",
) -> dict[str, Any]:
    if not _google_available():
        return {"characters": [], "locations": [], "scenes": []}

    art_style = ref_style.get("art_style", "realistic")
    image_type = ref_style.get("image_type", "AI-generated")

    topic_block = ""
    if creator_topic.strip():
        topic_block = (
            f"\nCreator topic / intent (from the user's prompt — keep characters and story aligned with this):\n"
            f"{creator_topic.strip()[:100000]}\n"
        )

    llm_prompt = f"""Analyze this story/narration transcript and extract characters and scenes.

Title: {title}
Art style to use (from reference video when provided, else default): {art_style}
Image type: {image_type}
{topic_block}
Transcript:
{transcript[:8000]}

Return a JSON object with:
- "characters": Array of character objects, each with:
  - "name": character name
  - "description": detailed visual description (age, gender, face shape, hair style and color, skin tone/complexion, build, distinguishing marks)
  - "face_lock": ONE sentence describing IMMUTABLE facial features to repeat verbatim in every image prompt.
    Must include: face shape (e.g. "oval face"), eye shape and color (e.g. "dark brown almond-shaped eyes"), nose (e.g. "broad flat nose"), jaw (e.g. "strong squared jaw"), eyebrows (e.g. "thick arched eyebrows"), lips (e.g. "full lips").
    Example: "Oval face, dark brown almond-shaped eyes, broad nose, strong square jaw, thick arched black eyebrows, medium-full lips, short black hair swept back."
  - "skin_color": exact skin complexion (e.g. "warm medium brown", "fair olive", "deep ebony") — NEVER changes
  - "hair_color": exact hair color and style (e.g. "jet-black wavy hair shoulder-length", "salt-and-pepper short hair", "dark brown thick hair worn in a bun") — NEVER changes unless story says so
  - "age_range": approximate age as string (e.g. "mid-20s", "late 50s") — NEVER changes
  - "role": their role in the story
  - "signature_outfit": ONE concise sentence with EXACT garment types and MAIN colors (e.g. "sky-blue linen kurta, dark brown trousers, brown leather chappals")
  - "main_colors": array of 3-8 color names fixed for this character's clothes/accents
- "locations": Array of location objects, each with:
  - "name": location name
  - "description": detailed visual description (type, size, setting — village/city, colors, atmosphere, time of day, key props)
- "scenes": Array of scene objects (one per story beat), each with:
  - "sentence": the narration text for this scene
  - "characters_present": list of character names in this scene
  - "location": where this scene takes place
  - "action": what is happening
  - "mood": emotional tone of the scene
  - "camera_suggestion": suggested framing (e.g. "wide shot", "medium close-up", "extreme close-up of face")

Be EXTREMELY specific with face_lock and hair_color — these exact words will be copied into every AI image prompt to prevent the face and hair from changing between scenes.
The face_lock, skin_color, hair_color, age_range, signature_outfit and main_colors are CRITICAL for consistency.

Respond ONLY with the JSON object."""

    _log("Extracting characters, locations, and scenes via Gemini...")
    result = _gemini_generate(llm_prompt, max_tokens=4096)
    if result:
        try:
            parsed = _parse_json_response(result)
            if isinstance(parsed, dict):
                chars = parsed.get("characters", [])
                locs = parsed.get("locations", [])
                scenes = parsed.get("scenes", [])
                _log(f"Extracted: {len(chars)} characters, {len(locs)} locations, {len(scenes)} scenes")
                return _normalize_character_entries(parsed)
        except (json.JSONDecodeError, ValueError) as e:
            _log(f"Character extraction parse failed: {e}")

    return {"characters": [], "locations": [], "scenes": []}


# ---------------------------------------------------------------------------
# Script generation (Gemini)
# ---------------------------------------------------------------------------

def step_generate_script(
    prompt: str,
    title: str,
    ref_summary: str | None = None,
    target_duration: int = 60,
    language: str = "english",
    reference_driven: bool = False,
) -> str:
    if _google_available():
        ref_context = ""
        if ref_summary:
            ref_context = (
                f"\n\nReference video analysis (match this style, pacing, and content approach):"
                f"\n{ref_summary}\n"
            )
        lang_instruction = ""
        if language != "english":
            lang_instruction = (
                f"\nCRITICAL: Write the ENTIRE script in {language.upper()} language. "
                f"Every word of narration must be in {language}. "
                f"Do NOT write in English. The script will be read by a {language} TTS engine.\n"
            )
        ref_driven_instruction = ""
        if reference_driven and ref_summary:
            ref_driven_instruction = (
                "\nIMPORTANT: The user wants to recreate content similar to the reference video. "
                "Use the reference video's topic, storyline, and narrative approach as your primary guide. "
                "Create an original script that covers the same subject matter with the same style and tone.\n"
            )
        word_count = int(target_duration * 2.5) if language == "english" else int(target_duration * 2)

        llm_prompt = f"""You are a professional video scriptwriter. Write a narration script for a {target_duration}-second video.

Topic/Prompt: {prompt}
Title: {title}
{ref_context}{lang_instruction}{ref_driven_instruction}
Requirements:
- Write ONLY the narration text that will be spoken aloud
- Target approximately {target_duration} seconds of speech (~{word_count} words)
- Start with a compelling hook (first sentence should grab attention)
- Use clear, conversational language suitable for text-to-speech
- Include specific facts, examples, or details relevant to the topic
- End with a strong closing statement
- Do NOT include stage directions, timestamps, or [brackets]
- Do NOT include "Welcome to" or generic filler phrases
- Make every sentence informative and relevant

Write the script now:"""

        _log(f"Generating {language} script via Gemini...")
        result = _gemini_generate(llm_prompt, max_tokens=2048)
        if result and len(result.strip()) > 30:
            script = result.strip()
            script = re.sub(r'\[.*?\]', '', script)
            script = re.sub(r'\(.*?\)', '', script)
            script = re.sub(r'\n{3,}', '\n\n', script)
            _log(f"Script generated ({language}): {len(script)} chars")
            return script

    _log("Gemini unavailable — using template script")
    if len(prompt) > 300:
        return prompt
    keywords = _extract_keywords(prompt, max_words=6)
    topic = " ".join(keywords[:4]) if keywords else title
    return textwrap.dedent(f"""\
        {title}.

        In today's exploration, we dive into {topic}.
        {prompt}

        This is a fascinating area that continues to evolve and shape our understanding.
        From its origins to its modern applications, {topic} remains at the forefront of innovation.

        Whether you are a beginner or an expert, there is always something new to discover.

        Thank you for watching. This video was produced by AeganMedia Montage.
    """).strip()


# ---------------------------------------------------------------------------
# Scene plan with character-consistent image prompts (Gemini)
# ---------------------------------------------------------------------------

def step_generate_scene_plan(
    script: str,
    prompt: str,
    title: str,
    ref_summary: str | None = None,
    ref_style: dict[str, str] | None = None,
    character_data: dict[str, Any] | None = None,
    scene_count: int = 6,
    sentence_timings: list[dict[str, Any]] | None = None,
    *,
    style_notes: str = "",
) -> list[dict[str, Any]]:
    if _google_available():
        ref_context = ""
        if ref_summary:
            ref_context = (
                "\nReference video — match its visual style, pacing, and composition feel:\n"
                f"{ref_summary[:400]}\n"
            )

        style_context = ""
        if ref_style:
            style_context = f"""
Visual style (from reference video analysis when a URL was provided; otherwise defaults):
- Art style: {ref_style.get('art_style', 'realistic')}
- Image type: {ref_style.get('image_type', 'AI-generated')}
- Color palette: {ref_style.get('color_palette', 'natural')}
- Mood: {ref_style.get('mood', 'professional')}
"""
        if style_notes.strip():
            style_context += (
                f"\nUser style instructions from their prompt (honor alongside the above):\n"
                f"{style_notes.strip()[:100000]}\n"
            )

        char_context = ""
        if character_data and character_data.get("characters"):
            char_descs = []
            for ch in character_data["characters"]:
                if not isinstance(ch, dict):
                    continue
                nm = ch.get("name", "Unknown")
                outfit = (ch.get("signature_outfit") or "").strip()
                skin = (ch.get("skin_color") or "").strip()
                hair = (ch.get("hair_color") or "").strip()
                age = (ch.get("age_range") or "").strip()
                face_lock = (ch.get("face_lock") or ch.get("description") or "").strip()[:300]
                cols = ch.get("main_colors") or []
                col_txt = ", ".join(str(c) for c in cols[:8]) if cols else ""
                extra = ""
                if age:
                    extra += f" [LOCK age: {age}]"
                if skin:
                    extra += f" [LOCK skin: {skin}]"
                if hair:
                    extra += f" [LOCK hair: {hair}]"
                if col_txt:
                    extra += f" [LOCK colors: {col_txt}]"
                if outfit:
                    extra += f" [LOCK outfit: {outfit}]"
                if face_lock:
                    extra += f" [LOCK face: {face_lock}]"
                char_descs.append(f"  - {nm}:{extra}")
            char_context = (
                "\nCharacters — copy these EXACT features into every image_prompt for that character:\n"
                + "\n".join(char_descs) + "\n"
            )

            loc_descs = []
            for loc in character_data.get("locations", []):
                loc_descs.append(f"  - {loc.get('name', 'Unknown')}: {loc.get('description', 'No description')}")
            if loc_descs:
                char_context += "\nLocations:\n" + "\n".join(loc_descs) + "\n"

        timing_context = ""
        if sentence_timings:
            timing_limit = min(len(sentence_timings), 50)
            timing_context = "\nAudio sentence timings (match image duration to these):\n"
            for i, t in enumerate(sentence_timings[:timing_limit]):
                timing_context += f"  Sentence {i+1}: {t.get('start', 0):.1f}s - {t.get('end', 0):.1f}s ({t.get('duration', 0):.1f}s): \"{t.get('text', '')[:80]}\"\n"

        if scene_count > 20:
            return _generate_scene_plan_batched(
                script, prompt, title, scene_count,
                ref_context, style_context, char_context, timing_context, ref_style,
            )

        style_prefix_for_prompt = IMAGE_STYLE_PREFIX.rstrip(". ") + "."
        llm_prompt = f"""You are a video scene planner. Break this narration script into {scene_count} visual scenes.

Title: {title}
Topic / creator intent (user prompt): {prompt[:500]}
{ref_context}{style_context}{char_context}{timing_context}
Script:
{script[:2000]}

For each scene, provide a JSON array with exactly {scene_count} objects. Each object must have:
- "narration": the exact portion of the script for this scene (1-3 sentences)
- "image_prompt": A DETAILED prompt for AI image generation. CRITICAL RULES:
  * EVERY prompt MUST start with exactly: "{style_prefix_for_prompt}"
  * Then describe the SPECIFIC scene content AFTER the style prefix.
  * CAMERA ANGLE VARIETY — cycle through these shot types to keep visuals dynamic:
    - Close-up portrait: character's face filling the frame, blurred background, strong emotional expression
    - Medium shot: character(s) from waist-up, clear body language and interaction visible
    - Wide establishing shot: full environment visible, character(s) small within the scene
    - Low angle / dramatic: camera looking up at character for power/emotion
    - Over-the-shoulder: viewpoint behind one character looking at another
    Use the most dramatically appropriate angle for each scene's emotional beat.
  * CHARACTER CONSISTENCY: For EVERY character that appears, copy their EXACT description from the character list above — same skin color, same hair, same outfit, same body type. Never invent new colors or clothes.
  * OUTFIT CHANGES: Only change a character's outfit when the narration explicitly says so (e.g. "wore", "put on", "changed into"). State the new outfit explicitly and carry it forward.
  * BACKGROUND VARIETY: Even if the story stays in one location, vary the framing — indoors vs outdoors area, daytime vs golden hour, foreground props differ. Never reuse the exact same composition twice.
  * LIGHTING: Vary the mood through lighting — dramatic rim light, soft diffused window light, harsh midday sun, warm candle glow.
- "camera_angle": one of: "close-up", "medium", "wide", "low-angle", "over-shoulder", "aerial"
- "search_query": 2-4 word search query for the scene content
- "mood": one word describing the scene mood
- "duration": estimated seconds this scene should be shown (match audio timing if available)
- "transition": "dissolve" for emotional scenes, "fade" for scene changes, "zoom" for dramatic moments
- "characters_present": array of character names who appear visually in this scene

Respond ONLY with the JSON array."""

        _log("Generating scene plan via Gemini...")
        result = _gemini_generate(llm_prompt, max_tokens=4096)
        if result:
            try:
                scenes = _parse_json_response(result)
                if isinstance(scenes, list) and len(scenes) >= 2:
                    _log(f"Scene plan: {len(scenes)} scenes")
                    return scenes
            except (json.JSONDecodeError, ValueError) as e:
                _log(f"Scene plan JSON parse failed: {e}")

    _log("Gemini unavailable — using keyword-based scene plan")
    sections = _split_script_sections(script, target_count=scene_count)
    keywords = _extract_keywords(prompt)
    if not keywords:
        keywords = _extract_keywords(script)
    if not keywords:
        keywords = _extract_keywords(title) if title else []
    if not keywords:
        keywords = ["nature", "landscape", "abstract"]

    scenes: list[dict[str, Any]] = []
    for i, section in enumerate(sections):
        section_kw = _extract_keywords(section, max_words=3)
        query = " ".join(section_kw[:3]) if section_kw else keywords[i % len(keywords)]
        if len(query.strip()) < 3:
            query = keywords[i % len(keywords)]
        scenes.append({
            "narration": section,
            "image_prompt": f"{IMAGE_STYLE_PREFIX}Subject: {query}. Detailed, expressive characters.",
            "search_query": query,
            "mood": "cinematic",
            "duration": 6.0,
            "transition": "dissolve",
        })
    return scenes


def _generate_scene_plan_batched(
    script: str, prompt: str, title: str, total_scenes: int,
    ref_context: str, style_context: str, char_context: str,
    timing_context: str, ref_style: dict[str, str] | None,
) -> list[dict[str, Any]]:
    batch_size = 15
    all_scenes: list[dict[str, Any]] = []
    sentences = re.split(r'(?<=[.!?])\s+', script.strip())
    total_sentences = len(sentences)
    batches = (total_scenes + batch_size - 1) // batch_size

    for batch_idx in range(batches):
        start_scene = batch_idx * batch_size
        end_scene = min(start_scene + batch_size, total_scenes)
        count_this_batch = end_scene - start_scene

        sent_start = int(total_sentences * start_scene / total_scenes)
        sent_end = int(total_sentences * end_scene / total_scenes)
        script_chunk = " ".join(sentences[sent_start:sent_end])

        # Build a summary of the previous batch's last 3 scenes for cross-batch continuity
        prev_summary = ""
        if all_scenes:
            tail = all_scenes[-3:]
            prev_lines = [
                f"  Scene {start_scene - len(tail) + j + 1}: {s.get('narration', '')[:120]}"
                for j, s in enumerate(tail)
            ]
            prev_summary = (
                "\nPrevious batch (last scenes — maintain visual continuity):\n"
                + "\n".join(prev_lines) + "\n"
            )

        style_prefix_for_batch = IMAGE_STYLE_PREFIX.rstrip(". ") + "."
        llm_prompt = f"""You are a video scene planner. Create scenes {start_scene+1} to {end_scene} of a {total_scenes}-scene video.

Title: {title}
Topic: {prompt[:200]}
{ref_context}{style_context}{char_context}{prev_summary}
Script portion (scenes {start_scene+1}-{end_scene}):
{script_chunk[:3000]}

Create a JSON array with exactly {count_this_batch} scene objects. Each must have:
- "narration": portion of script for this scene
- "image_prompt": EVERY prompt MUST start with "{style_prefix_for_batch}" Then describe the specific scene.
  CAMERA ANGLE VARIETY — rotate through: close-up portrait (face fills frame), medium shot (waist-up), wide shot (full environment), low-angle dramatic, over-the-shoulder. Pick the most emotionally fitting angle.
  CHARACTER CONSISTENCY: copy the exact skin color, outfit, and colors from the character list above for every character who appears. Only change outfit if the narration says so.
  BACKGROUND VARIETY: vary framing and composition in every scene — different angles, props, lighting mood.
- "camera_angle": one of: "close-up", "medium", "wide", "low-angle", "over-shoulder", "aerial"
- "search_query": 2-4 word search query
- "mood": one word mood
- "duration": seconds (float)
- "transition": "dissolve" for emotional, "fade" for scene changes, "zoom" for dramatic
- "characters_present": array of character names visible in this scene

Respond ONLY with the JSON array."""

        _log(f"Generating scene plan batch {batch_idx+1}/{batches} ({count_this_batch} scenes)...")
        result = _gemini_generate(llm_prompt, max_tokens=4096)
        if result:
            try:
                scenes = _parse_json_response(result)
                if isinstance(scenes, list):
                    all_scenes.extend(scenes)
                    continue
            except (json.JSONDecodeError, ValueError) as e:
                _log(f"Scene plan batch {batch_idx+1} parse failed: {e}")

        for j in range(count_this_batch):
            idx = sent_start + int(j * (sent_end - sent_start) / count_this_batch)
            narr = sentences[idx] if idx < total_sentences else f"Scene {start_scene + j + 1}"
            all_scenes.append({
                "narration": narr,
                "image_prompt": f"{IMAGE_STYLE_PREFIX}{narr[:100]}",
                "search_query": " ".join(re.findall(r"[a-zA-Z]{3,}", narr)[:3]),
                "mood": "cinematic",
                "duration": 8.0,
                "transition": "fade",
            })

    _log(f"Batched scene plan: {len(all_scenes)} scenes from {batches} batch(es)")
    return all_scenes


# ---------------------------------------------------------------------------
# Timeline-based scene plan (option 2: custom audio with timestamps)
# ---------------------------------------------------------------------------

def step_generate_scene_plan_timeline(
    segments: list[dict[str, Any]],
    character_data: dict[str, Any],
    ref_style: dict[str, str],
    title: str,
    *,
    creator_topic: str = "",
    style_notes: str = "",
) -> list[dict[str, Any]]:
    if not segments:
        return []

    style_line = ref_style.get("art_style", DEFAULT_STYLE["art_style"])
    palette = ref_style.get("color_palette", DEFAULT_STYLE["color_palette"])
    mood = ref_style.get("mood", DEFAULT_STYLE["mood"])
    try:
        char_snip = json.dumps(character_data.get("characters", [])[:12], ensure_ascii=True)
    except (TypeError, ValueError):
        char_snip = "[]"
    char_snip = char_snip[:2800]

    def _fallback_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        fb: list[dict[str, Any]] = []
        for seg in batch:
            en = seg.get("text_en") or seg.get("text", "")
            fb.append({
                "narration": seg.get("text", ""),
                "image_prompt": f"{IMAGE_STYLE_PREFIX}{en}",
                "duration": float(seg.get("duration", 4.0)),
                "transition": "dissolve",
                "search_query": " ".join(re.findall(r"[a-zA-Z]{3,}", en)[:4]) or "story scene",
                "mood": "cinematic",
            })
        return fb

    def _one_batch(batch: list[dict[str, Any]], base_idx: int) -> list[dict[str, Any]]:
        lines: list[str] = []
        for j, seg in enumerate(batch):
            dur = float(seg.get("duration", 3.0))
            en = (seg.get("text_en") or seg.get("text", ""))[:450].replace("\n", " ")
            orig = (seg.get("text") or "")[:280].replace("\n", " ")
            lines.append(
                f"Scene {base_idx + j + 1} | duration_sec={dur:.3f} | EN: {en} | ORIG: {orig}"
            )
        block = "\n".join(lines)
        topic_ctx = ""
        if creator_topic.strip():
            topic_ctx = (
                f"\nCreator topic / intent (user prompt — use for overall subject, tone, and setting; "
                f"each scene's action must still match its EN line):\n{creator_topic.strip()[:100000]}\n"
            )
        notes_ctx = ""
        if style_notes.strip():
            notes_ctx = (
                f"\nUser style notes from prompt:\n{style_notes.strip()[:100000]}\n"
            )
        style_prefix_for_timeline = IMAGE_STYLE_PREFIX.rstrip(". ") + "."
        prompt = f"""You create still-image scenes for a narrated video. Return a JSON array of EXACTLY {len(batch)} objects, same order as the lines below.

Title: {title}
Visual style from reference video (match this look): art_style={style_line}; color_palette={palette}; mood={mood}.
Base rendering style (start EVERY image_prompt with this EXACT text): {style_prefix_for_timeline}
{topic_ctx}{notes_ctx}
Characters (keep face, signature_outfit, and main_colors IDENTICAL whenever the same person appears):
{char_snip}

Timed scenes (each line has duration_sec — copy it EXACTLY into the "duration" field as a float):
{block}

Rules:
- "narration": use the ORIG text for that scene (subtitle / spoken line).
- "image_prompt": MUST begin with the Base rendering style text above, then describe ONLY what the EN line says is happening.
  CAMERA ANGLE — for EACH scene pick the most emotionally fitting angle and state it clearly:
    • Close-up portrait: character face fills frame, bokeh background, intense expression
    • Medium shot (waist-up): body language and interaction visible
    • Wide establishing shot: full environment, characters small, sets the location clearly
    • Low-angle dramatic: camera below character, looks powerful/emotional
    • Over-the-shoulder: viewpoint behind one character facing another
  CHARACTER CONSISTENCY — for every character who appears:
    * Copy their EXACT skin_color, signature_outfit, and main_colors from the JSON above.
    * NEVER invent new clothing colors, change hairstyle, or alter skin tone between scenes.
    * Only change outfit if the narration EXPLICITLY says the character changed clothes (e.g. "wore", "put on", "changed into").
  BACKGROUND VARIETY: Even at the same location, vary framing — foreground props, lighting angle, time-of-day mood.
- "camera_angle": one of: "close-up", "medium", "wide", "low-angle", "over-shoulder", "aerial"
- "duration": must equal duration_sec from that line (float).
- "transition": "dissolve" or "fade".
- "search_query": 2-4 English keywords.
- "characters_present": array of character names who appear visually in this scene.

Respond ONLY with the JSON array."""

        _log(f"Timeline scene plan: batch {base_idx + 1}-{base_idx + len(batch)} ({len(batch)} scenes)...")
        result = _gemini_generate(prompt, max_tokens=8192)
        if result:
            try:
                scenes = _parse_json_response(result)
                if isinstance(scenes, list) and len(scenes) >= len(batch):
                    fixed: list[dict[str, Any]] = []
                    for j in range(len(batch)):
                        s = scenes[j] if j < len(scenes) else {}
                        if not isinstance(s, dict):
                            s = {}
                        dur = float(batch[j].get("duration", 4.0))
                        s["duration"] = dur
                        s["narration"] = s.get("narration") or batch[j].get("text", "")
                        ip = s.get("image_prompt") or ""
                        # Re-prefix if the model forgot to start with the style block
                        style_lower = IMAGE_STYLE_PREFIX[:30].lower()
                        if ip and style_lower[:15] not in ip.lower()[:60]:
                            ip = f"{IMAGE_STYLE_PREFIX}{batch[j].get('text_en', '')}"
                        s["image_prompt"] = ip
                        s.setdefault("transition", "dissolve")
                        s.setdefault("search_query", "story")
                        # Preserve characters_present so image_renderer can group scenes
                        if not s.get("characters_present"):
                            s["characters_present"] = []
                        fixed.append(s)
                    return fixed
            except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
                _log(f"Timeline scene JSON issue: {e}")
        return _fallback_batch(batch)

    n = len(segments)
    batch_n = 20
    if n <= batch_n:
        return _one_batch(segments, 0)
    all_out: list[dict[str, Any]] = []
    for i in range(0, n, batch_n):
        chunk = segments[i : i + batch_n]
        all_out.extend(_one_batch(chunk, i))
    return all_out
