"""AI-powered pipeline runner for web-submitted video production jobs.

Orchestrates OpenMontage tools with LLM intelligence:
  1. (optional) Download + analyze reference video
  2. Generate professional narration script via LLM
  3. Create scene-by-scene plan with image prompts + SFX hints
  4. Generate/fetch scene-matched images
  5. Generate narration audio (TTS)
  6. Fetch mood-matched background music
  6b. Generate AI sound effects per scene (ElevenLabs + Freesound)
  7. Compose final video with Ken Burns motion, crossfades, subtitles, SFX

Uses Google Gemini for script/scene intelligence when GOOGLE_API_KEY is set.
Falls back to template-based generation when no LLM is available.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import requests

PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))
os.chdir(str(PROJ_ROOT))

from tools.tool_registry import registry
from tools.base_tool import ToolStatus


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[pipeline-runner] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Tool helpers
# ---------------------------------------------------------------------------

def _discover_tools() -> None:
    registry.discover("tools")


def _get_tool(name: str) -> Any:
    tool = registry.get(name)
    if tool is None:
        return None
    if tool.get_status() != ToolStatus.AVAILABLE:
        return None
    return tool


# ---------------------------------------------------------------------------
# LLM — Google Gemini integration
# ---------------------------------------------------------------------------

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)


def _gemini_available() -> bool:
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))


def _gemini_generate(prompt: str, max_tokens: int = 4096, retries: int = 4) -> str | None:
    """Call Google Gemini with retry for rate limits. Returns text or None."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    models = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-pro"]

    for model in models:
        url = GEMINI_ENDPOINT.format(model=model) + f"?key={api_key}"
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.7,
            },
        }

        for attempt in range(retries):
            try:
                resp = requests.post(url, json=body, timeout=90)
                if resp.status_code == 429:
                    wait = min(5 * (attempt + 1), 60)
                    _log(f"Gemini rate-limited (429) on {model}, "
                         f"waiting {wait}s (attempt {attempt+1}/{retries})")
                    time.sleep(wait)
                    continue
                if resp.status_code == 404:
                    _log(f"Gemini model {model} not found (404), trying next model")
                    break
                resp.raise_for_status()
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except requests.exceptions.HTTPError as e:
                if "429" in str(e) and attempt < retries - 1:
                    wait = min(5 * (attempt + 1), 60)
                    _log(f"Gemini rate-limited on {model}, waiting {wait}s")
                    time.sleep(wait)
                    continue
                if "404" in str(e):
                    _log(f"Gemini model {model} not available, trying next")
                    break
                _log(f"Gemini API error ({model}): {e}")
                break
            except Exception as e:
                _log(f"Gemini API error ({model}): {e}")
                break

    return None


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _strip_urls(text: str) -> str:
    """Remove URLs and 'Reference:' prefixes from text."""
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'(?i)reference\s*:\s*', '', text)
    return text.strip()


def _extract_keywords(text: str, max_words: int = 4) -> list[str]:
    """Pull meaningful keywords from prompt text (URLs stripped first)."""
    text = _strip_urls(text)
    stop = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could", "about", "above",
        "after", "again", "all", "also", "am", "and", "any", "as", "at",
        "because", "before", "below", "between", "both", "but", "by", "came",
        "come", "could", "day", "each", "even", "find", "for", "from", "get",
        "give", "go", "going", "her", "here", "him", "his", "how", "i", "if",
        "in", "into", "it", "its", "just", "know", "like", "long", "look",
        "make", "many", "me", "most", "my", "new", "no", "not", "now", "of",
        "on", "one", "only", "or", "other", "our", "out", "over", "own",
        "people", "say", "see", "she", "so", "some", "take", "tell", "than",
        "that", "the", "their", "them", "then", "there", "these", "they",
        "thing", "think", "this", "those", "time", "to", "too", "two", "up",
        "us", "use", "very", "want", "way", "we", "well", "what", "when",
        "where", "which", "who", "why", "with", "work", "world", "year", "you",
        "your", "create", "video", "make", "please", "need", "want", "second",
        "minute", "seconds", "minutes", "same", "type", "based", "sharing",
        "link", "wan", "tthe", "referance", "http", "https", "www", "com",
        "youtube", "watch", "audio", "attached", "file", "shared", "sharing",
        "match", "matching", "sure", "tht", "images", "used", "using",
        "upload", "uploaded", "downloading", "download", "creating",
        "created", "generate", "generated", "prompt", "promt",
        "subtitle", "subtitles", "relavient", "relevant", "preparation",
        "check", "checking", "trying", "tried", "try",
    }
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    keywords = [w for w in words if w not in stop]
    seen: set[str] = set()
    unique: list[str] = []
    for w in keywords:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique[:max_words * 3]


SUPPORTED_LANGUAGES = {
    "english": "en", "tamil": "ta", "hindi": "hi", "telugu": "te",
    "kannada": "kn", "malayalam": "ml", "bengali": "bn", "marathi": "mr",
    "gujarati": "gu", "punjabi": "pa", "urdu": "ur", "arabic": "ar",
    "spanish": "es", "french": "fr", "german": "de", "italian": "it",
    "portuguese": "pt", "japanese": "ja", "korean": "ko", "chinese": "zh",
    "russian": "ru", "dutch": "nl", "polish": "pl", "turkish": "tr",
    "thai": "th", "vietnamese": "vi", "indonesian": "id", "malay": "ms",
    "swedish": "sv", "norwegian": "no", "danish": "da", "finnish": "fi",
}


def _parse_production_intent(raw_prompt: str, ref_summary: str | None = None) -> dict[str, Any]:
    """Use Gemini to separate production instructions from creative content.

    Returns a dict with keys:
      content_prompt  - the actual topic/story to create (NOT instructions)
      audio_language  - language for voiceover (default: english)
      subtitle_language - language for subtitles (default: same as audio)
      target_duration - video length in seconds (default: 60)
      style_notes     - editing/visual style instructions
      reference_driven - True if the user mainly relies on the reference video for content
    """
    defaults: dict[str, Any] = {
        "content_prompt": raw_prompt,
        "audio_language": "english",
        "subtitle_language": "",
        "target_duration": 60,
        "style_notes": "",
        "reference_driven": False,
    }

    if not _gemini_available():
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

Example: If user says "Create a 15 second video about space exploration in Tamil with English subtitles, cinematic style"
→ content_prompt: "space exploration — the wonders of the cosmos, rocket launches, planets, and humanity's quest to explore the stars"
→ audio_language: "tamil"
→ subtitle_language: "english"
→ target_duration: 15
→ style_notes: "cinematic"
→ reference_driven: false

Respond ONLY with the JSON object."""

    _log("Parsing production intent via Gemini...")
    result = _gemini_generate(llm_prompt, max_tokens=1024)
    if result:
        try:
            cleaned = result.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r'^```\w*\n?', '', cleaned)
                cleaned = re.sub(r'\n?```$', '', cleaned)
            parsed = json.loads(cleaned)
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


def _split_script_sections(text: str, target_count: int = 6) -> list[str]:
    """Split narration text into roughly equal sections for scene visuals."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= target_count:
        return sentences if sentences else [text]
    per_section = max(1, len(sentences) // target_count)
    sections: list[str] = []
    for i in range(0, len(sentences), per_section):
        chunk = " ".join(sentences[i:i + per_section])
        if chunk.strip():
            sections.append(chunk.strip())
    return sections[:target_count + 2]


# ---------------------------------------------------------------------------
# Step 0: Reference video analysis
# ---------------------------------------------------------------------------

def step_download_reference(ref_url: str, project_dir: Path) -> str | None:
    """Download reference video if URL provided. Returns local path or None."""
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
    """Analyze reference video for style, content, and structure."""
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

    transcriber = _get_tool("transcriber")
    if transcriber and "transcript" not in analysis:
        _log("Transcribing reference audio...")
        result = transcriber.execute({
            "input_path": ref_path,
            "model_size": "base",
            "output_dir": str(project_dir / "analysis"),
        })
        if result.success and result.artifacts:
            try:
                transcript_data = json.loads(
                    Path(result.artifacts[0]).read_text(encoding="utf-8")
                )
                analysis["transcript"] = transcript_data
                _log("Transcription complete")
            except (json.JSONDecodeError, IndexError):
                pass

    return analysis


def _summarize_reference(analysis: dict[str, Any]) -> str:
    """Build a text summary of reference video analysis for LLM context."""
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


# ---------------------------------------------------------------------------
# Step 1: LLM-powered script generation
# ---------------------------------------------------------------------------

def step_generate_script(
    prompt: str,
    title: str,
    ref_summary: str | None = None,
    target_duration: int = 60,
    language: str = "english",
    reference_driven: bool = False,
) -> str:
    """Generate a professional narration script using LLM or template fallback."""
    if _gemini_available():
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
            _log(f"LLM script generated ({language}): {len(script)} chars")
            return script

    _log("LLM unavailable — using template script")
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
# Step 2: LLM-powered scene plan
# ---------------------------------------------------------------------------

def step_generate_scene_plan(
    script: str,
    prompt: str,
    title: str,
    ref_summary: str | None = None,
    scene_count: int = 6,
) -> list[dict[str, str]]:
    """Generate scene-by-scene plan with image descriptions and search queries."""
    if _gemini_available():
        ref_context = ""
        if ref_summary:
            ref_context = (
                f"\nReference video style to match:\n{ref_summary[:400]}\n"
            )

        llm_prompt = f"""You are a video scene planner. Break this narration script into {scene_count} visual scenes.

Title: {title}
Topic: {prompt[:200]}
{ref_context}
Script:
{script[:2000]}

For each scene, provide a JSON array with exactly {scene_count} objects. Each object must have:
- "narration": the exact portion of the script for this scene (1-3 sentences)
- "image_prompt": a detailed prompt for AI image generation (describe the visual: subject, setting, lighting, mood, colors, camera angle). Be SPECIFIC to the topic, not generic.
- "search_query": 2-4 word search query for stock photo fallback, highly specific to the scene content
- "mood": one word describing the scene mood (e.g., inspiring, dramatic, calm, energetic)
- "sfx_prompt": (optional, only if the scene benefits from a sound effect) a short description of a sound effect that enhances this scene. Examples: "glass shattering", "crowd cheering in stadium", "rocket engine igniting", "gentle ocean waves", "thunder and lightning". Leave as "" if no SFX is needed for that scene. Do NOT put background music here, only specific sound effects tied to the visual content.

Respond ONLY with the JSON array, no other text. Example format:
[
  {{"narration": "...", "image_prompt": "...", "search_query": "...", "mood": "...", "sfx_prompt": "..."}},
  ...
]"""

        _log("Generating scene plan via Gemini...")
        result = _gemini_generate(llm_prompt, max_tokens=3000)
        if result:
            try:
                cleaned = result.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r'^```\w*\n?', '', cleaned)
                    cleaned = re.sub(r'\n?```$', '', cleaned)
                scenes = json.loads(cleaned)
                if isinstance(scenes, list) and len(scenes) >= 2:
                    _log(f"LLM scene plan: {len(scenes)} scenes")
                    return scenes
            except json.JSONDecodeError as e:
                _log(f"Scene plan JSON parse failed: {e}")

    _log("LLM unavailable — using keyword-based scene plan")
    sections = _split_script_sections(script, target_count=scene_count)
    keywords = _extract_keywords(prompt)
    if not keywords:
        keywords = _extract_keywords(script)
    if not keywords:
        keywords = _extract_keywords(title) if title else []
    if not keywords:
        keywords = ["nature", "landscape", "abstract"]

    scenes: list[dict[str, str]] = []
    for i, section in enumerate(sections):
        section_kw = _extract_keywords(section, max_words=3)
        query = " ".join(section_kw[:3]) if section_kw else keywords[i % len(keywords)]
        if len(query.strip()) < 3:
            query = keywords[i % len(keywords)]
        scenes.append({
            "narration": section,
            "image_prompt": (
                f"Professional cinematic photograph, 16:9, dramatic lighting, "
                f"shallow depth of field. Subject: {query}. "
                f"Photo-realistic, high detail."
            ),
            "search_query": query,
            "mood": "professional",
        })
    return scenes


# ---------------------------------------------------------------------------
# Step 3: TTS
# ---------------------------------------------------------------------------

def step_tts(text: str, output_path: Path, language: str = "english") -> str | None:
    """Generate narration audio in the specified language."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lang_code = SUPPORTED_LANGUAGES.get(language, "en")
    is_english = lang_code == "en"

    tts_tools = ["elevenlabs_tts", "google_tts", "openai_tts", "piper_tts"]
    if not is_english:
        tts_tools = ["elevenlabs_tts", "google_tts", "openai_tts"]

    for name in tts_tools:
        tool = _get_tool(name)
        if tool is None:
            continue
        _log(f"Generating {language} narration with {name}")
        params: dict[str, Any] = {"text": text, "output_path": str(output_path)}
        if name == "elevenlabs_tts":
            params["model_id"] = "eleven_multilingual_v2"
        elif name == "google_tts":
            voice_map = {
                "ta": "ta-IN-Standard-A", "hi": "hi-IN-Standard-A",
                "te": "te-IN-Standard-A", "kn": "kn-IN-Standard-A",
                "ml": "ml-IN-Standard-A", "bn": "bn-IN-Standard-A",
                "es": "es-ES-Studio-F", "fr": "fr-FR-Studio-A",
                "de": "de-DE-Studio-B", "ja": "ja-JP-Standard-A",
            }
            params["voice"] = voice_map.get(lang_code, "en-US-Studio-Q")
            params["speaking_rate"] = 0.95
        elif name == "piper_tts":
            if not is_english:
                continue
            params["model"] = "en_US-lessac-medium"
            params["download_dir"] = str(Path.home() / ".local" / "share" / "piper_models")
        result = tool.execute(params)
        if result.success and result.artifacts:
            _log(f"{language} narration generated: {result.artifacts[0]}")
            return result.artifacts[0]
        _log(f"{name} failed: {result.error}")
    _log("No TTS tool available — narration skipped")
    return None


# ---------------------------------------------------------------------------
# Step 3b: Subtitle translation
# ---------------------------------------------------------------------------

def step_translate_subtitles(
    narration_sections: list[str],
    source_language: str,
    target_language: str,
) -> list[str]:
    """Translate narration sections for subtitle display in a different language."""
    if source_language == target_language or not target_language:
        return narration_sections

    if not _gemini_available():
        _log("Gemini unavailable — cannot translate subtitles")
        return narration_sections

    _log(f"Translating subtitles from {source_language} to {target_language}...")
    joined = json.dumps(narration_sections, ensure_ascii=False)

    llm_prompt = f"""Translate each of these narration lines from {source_language} to {target_language}.
Return a JSON array of translated strings, maintaining the same array length and order.
Keep translations natural and concise (suitable for video subtitles — short, readable lines).

Input:
{joined}

Respond ONLY with the JSON array of translated strings."""

    result = _gemini_generate(llm_prompt, max_tokens=2048)
    if result:
        try:
            cleaned = result.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r'^```\w*\n?', '', cleaned)
                cleaned = re.sub(r'\n?```$', '', cleaned)
            translated = json.loads(cleaned)
            if isinstance(translated, list) and len(translated) == len(narration_sections):
                _log(f"Subtitles translated: {len(translated)} sections → {target_language}")
                return translated
        except (json.JSONDecodeError, ValueError) as e:
            _log(f"Subtitle translation parse failed: {e}")

    _log("Subtitle translation failed — using original narration")
    return narration_sections


# ---------------------------------------------------------------------------
# Step 3c: Audio transcription (for uploaded audio)
# ---------------------------------------------------------------------------

def step_transcribe_audio(audio_path: str, language: str = "english") -> str | None:
    """Transcribe uploaded audio to text using ElevenLabs Scribe or Gemini.

    This is critical when the user uploads audio — without transcription,
    the pipeline cannot know what the audio says and generates irrelevant images.
    """
    lang_code = SUPPORTED_LANGUAGES.get(language, "en")

    # Method 1: ElevenLabs Scribe API (best quality, supports many languages)
    el_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if el_key:
        _log(f"Transcribing audio via ElevenLabs Scribe ({language})...")
        try:
            with open(audio_path, "rb") as f:
                resp = requests.post(
                    "https://api.elevenlabs.io/v1/speech-to-text",
                    headers={"xi-api-key": el_key},
                    files={"file": (Path(audio_path).name, f, "audio/mpeg")},
                    data={
                        "model_id": "scribe_v1",
                        "language_code": lang_code,
                    },
                    timeout=120,
                )
            if resp.status_code == 200:
                data = resp.json()
                text = data.get("text", "")
                if text and len(text.strip()) > 10:
                    _log(f"Transcription complete: {len(text)} chars")
                    return text.strip()
                _log("Transcription returned empty text")
            else:
                _log(f"ElevenLabs Scribe failed: {resp.status_code} {resp.text[:200]}")
        except Exception as e:
            _log(f"ElevenLabs Scribe error: {e}")

    # Method 2: Google Cloud Speech-to-Text (via Gemini API)
    google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if google_key:
        _log(f"Transcribing audio via Google Gemini ({language})...")
        try:
            import base64
            with open(audio_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")

            ext = Path(audio_path).suffix.lower().lstrip(".")
            mime_map = {
                "mp3": "audio/mpeg", "m4a": "audio/mp4", "wav": "audio/wav",
                "ogg": "audio/ogg", "aac": "audio/aac", "flac": "audio/flac",
            }
            mime_type = mime_map.get(ext, "audio/mpeg")

            gemini_url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                "gemini-2.0-flash:generateContent"
                f"?key={google_key}"
            )
            body = {
                "contents": [{
                    "parts": [
                        {"inline_data": {"mime_type": mime_type, "data": audio_b64}},
                        {"text": f"Transcribe this audio to text. The audio is in {language}. "
                                 f"Return ONLY the transcribed text, nothing else."},
                    ]
                }],
                "generationConfig": {"maxOutputTokens": 4096, "temperature": 0.1},
            }
            resp = requests.post(gemini_url, json=body, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                if text and len(text.strip()) > 10:
                    _log(f"Gemini transcription complete: {len(text)} chars")
                    return text.strip()
            else:
                _log(f"Gemini transcription failed: {resp.status_code}")
        except Exception as e:
            _log(f"Gemini transcription error: {e}")

    # Method 3: Use the transcriber tool if available
    tool = _get_tool("transcriber")
    if tool is not None:
        _log("Transcribing audio via whisperx...")
        result = tool.execute({"audio_path": audio_path, "language": lang_code})
        if result.success and result.data:
            text = result.data.get("text", "")
            if text and len(text.strip()) > 10:
                _log(f"Whisperx transcription: {len(text)} chars")
                return text.strip()

    _log("Audio transcription unavailable — cannot determine audio content")
    return None


# ---------------------------------------------------------------------------
# Step 4: Image generation / fetching (scene-aware)
# ---------------------------------------------------------------------------

def step_fetch_images(
    scene_plan: list[dict[str, str]],
    output_dir: Path,
) -> list[str]:
    """Fetch images for each scene using AI generation or targeted stock search."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images: list[str] = []

    ai_tools = ["google_imagen", "flux_image", "openai_image", "grok_image"]
    ai_tool = None
    ai_tool_name = ""
    for name in ai_tools:
        t = _get_tool(name)
        if t is not None:
            ai_tool = t
            ai_tool_name = name
            _log(f"AI image generation available: {name}")
            break

    stock_tools = ["pexels_image", "pixabay_image"]
    stock_tool = None
    for name in stock_tools:
        t = _get_tool(name)
        if t is not None:
            stock_tool = t
            break

    for i, scene in enumerate(scene_plan):
        out = output_dir / f"scene_{i:02d}.png"
        image_prompt = scene.get("image_prompt", "")
        search_query = scene.get("search_query", "technology")

        if ai_tool and image_prompt:
            _log(f"Scene {i+1}/{len(scene_plan)}: AI generating ({ai_tool_name})")
            result = ai_tool.execute({
                "prompt": image_prompt,
                "aspect_ratio": "16:9",
                "output_path": str(out),
            })
            if result.success and result.artifacts:
                images.append(result.artifacts[0])
                continue
            _log(f"AI image failed for scene {i+1}: {result.error}")

        if stock_tool:
            out_stock = output_dir / f"scene_{i:02d}.jpg"
            _log(f"Scene {i+1}/{len(scene_plan)}: stock search '{search_query}'")
            result = stock_tool.execute({
                "query": search_query,
                "orientation": "landscape",
                "output_path": str(out_stock),
            })
            if result.success and result.artifacts:
                images.append(result.artifacts[0])
                continue
            _log(f"Stock image failed for scene {i+1}: {result.error}")

        _log(f"Scene {i+1}: no image obtained")

    return images


# ---------------------------------------------------------------------------
# Step 5: Background music
# ---------------------------------------------------------------------------

def step_fetch_music(
    scene_plan: list[dict[str, str]],
    keywords: list[str],
    output_dir: Path,
    target_duration: float = 60.0,
) -> str | None:
    """Fetch background music matched to the video mood."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "bgm.mp3"

    moods = [s.get("mood", "") for s in scene_plan if s.get("mood")]
    mood_str = moods[0] if moods else "cinematic"

    music_tools = ["music_gen", "suno_music", "pixabay_music", "freesound_music"]
    for name in music_tools:
        tool = _get_tool(name)
        if tool is None:
            continue
        _log(f"Fetching background music with {name} (mood: {mood_str})")
        topic = " ".join(keywords[:2]) if keywords else ""
        query = f"{mood_str} {topic} background instrumental".strip()

        if name in ("pixabay_music", "freesound_music"):
            result = tool.execute({
                "query": query,
                "min_duration": max(30, int(target_duration * 0.5)),
                "max_duration": int(target_duration * 2),
                "output_path": str(out),
            })
        elif name == "music_gen":
            result = tool.execute({
                "prompt": f"{mood_str} {topic} background music, instrumental, ambient",
                "duration_seconds": int(target_duration),
                "output_path": str(out),
            })
        elif name == "suno_music":
            result = tool.execute({
                "prompt": f"{mood_str} {topic} background music",
                "instrumental": True,
                "output_path": str(out),
            })
        else:
            continue
        if result.success and result.artifacts:
            _log(f"Background music: {result.artifacts[0]}")
            return result.artifacts[0]
        _log(f"{name} music failed: {result.error}")
    _log("No background music available")
    return None


# ---------------------------------------------------------------------------
# Step 5b: AI Sound Effects (ElevenLabs /v1/sound-generation + Freesound)
# ---------------------------------------------------------------------------

def _elevenlabs_sfx_available() -> bool:
    return bool(os.environ.get("ELEVENLABS_API_KEY"))


def _generate_sfx_elevenlabs(
    prompt: str, output_path: Path, duration: float = 5.0
) -> str | None:
    """Generate a sound effect using ElevenLabs Sound Generation API."""
    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not api_key:
        return None

    url = "https://api.elevenlabs.io/v1/sound-generation"
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    body: dict[str, Any] = {
        "text": prompt,
        "duration_seconds": min(max(duration, 0.5), 22.0),
        "prompt_influence": 0.5,
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=60)
            if resp.status_code == 429:
                wait = 2 ** attempt * 3
                _log(f"SFX rate-limited, retrying in {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(resp.content)
            if output_path.exists() and output_path.stat().st_size > 100:
                return str(output_path)
            return None
        except Exception as e:
            _log(f"ElevenLabs SFX attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
    return None


def _search_sfx_freesound(
    query: str, output_path: Path, max_duration: float = 10.0
) -> str | None:
    """Search Freesound for a matching sound effect clip."""
    tool = _get_tool("freesound_music")
    if tool is None:
        return None
    result = tool.execute({
        "query": query,
        "min_duration": 1,
        "max_duration": int(max_duration),
        "output_path": str(output_path),
    })
    if result.success and result.artifacts:
        return result.artifacts[0]
    return None


def step_generate_sfx(
    scene_plan: list[dict[str, str]],
    output_dir: Path,
    seconds_per_scene: float = 6.0,
) -> list[dict[str, Any]]:
    """Generate sound effects for scenes that request them.

    Returns a list of dicts with keys: path, scene_index, offset_seconds, duration.
    """
    sfx_entries: list[dict[str, Any]] = []
    sfx_scenes = [
        (i, s.get("sfx_prompt", "").strip())
        for i, s in enumerate(scene_plan)
        if s.get("sfx_prompt", "").strip()
    ]

    if not sfx_scenes:
        _log("No SFX requested for any scene")
        return sfx_entries

    _log(f"Generating sound effects for {len(sfx_scenes)} scene(s)")
    output_dir.mkdir(parents=True, exist_ok=True)

    has_elevenlabs = _elevenlabs_sfx_available()

    for scene_idx, sfx_prompt in sfx_scenes:
        out_file = output_dir / f"sfx_{scene_idx:02d}.mp3"
        sfx_duration = min(seconds_per_scene - 0.5, 8.0)

        sfx_path = None
        if has_elevenlabs:
            _log(f"  Scene {scene_idx+1} SFX via ElevenLabs: \"{sfx_prompt}\"")
            sfx_path = _generate_sfx_elevenlabs(sfx_prompt, out_file, sfx_duration)

        if not sfx_path:
            _log(f"  Scene {scene_idx+1} SFX via Freesound: \"{sfx_prompt}\"")
            sfx_path = _search_sfx_freesound(sfx_prompt, out_file, sfx_duration + 5)

        if sfx_path:
            actual_dur = _probe_duration(sfx_path)
            offset = scene_idx * seconds_per_scene
            sfx_entries.append({
                "path": sfx_path,
                "scene_index": scene_idx,
                "offset_seconds": offset,
                "duration": actual_dur,
            })
            _log(f"  Scene {scene_idx+1} SFX ready: {actual_dur:.1f}s at offset {offset:.1f}s")
        else:
            _log(f"  Scene {scene_idx+1} SFX generation failed — skipping")

    _log(f"Total SFX clips generated: {len(sfx_entries)}")
    return sfx_entries


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------

def _probe_duration(path: str) -> float:
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30,
        )
        return float(probe.stdout.strip())
    except (ValueError, subprocess.TimeoutExpired):
        return 0.0


def _escape_drawtext(text: str) -> str:
    for ch in ("\\", "'", ":", "%"):
        text = text.replace(ch, f"\\{ch}")
    return text


# ---------------------------------------------------------------------------
# Step 6: Compose video (FFmpeg slideshow with motion + transitions)
# ---------------------------------------------------------------------------

def step_compose_slideshow(
    images: list[str],
    audio_path: str | None,
    output_path: Path,
    seconds_per_image: float = 6.0,
    sections: list[str] | None = None,
    music_path: str | None = None,
    sfx_entries: list[dict[str, Any]] | None = None,
) -> str | None:
    """Compose a high-quality slideshow: Ken Burns motion, crossfades, subtitles, SFX."""
    if not images:
        _log("No images to compose")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    W, H, FPS = 1920, 1080, 30
    FADE_DUR = 1.0

    if audio_path and Path(audio_path).exists():
        total_dur = _probe_duration(audio_path)
        if total_dur > 0:
            seconds_per_image = max(4.0, total_dur / len(images))

    dur_frames = int(seconds_per_image * FPS)

    temp_dir = output_path.parent / ".runner_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    segments: list[Path] = []

    kb_patterns = [
        (1.0, 1.15, "center"),
        (1.15, 1.0, "center"),
        (1.0, 1.12, "left"),
        (1.12, 1.0, "right"),
    ]

    try:
        for i, img in enumerate(images):
            seg = temp_dir / f"seg_{i:04d}.mp4"
            z_start, z_end, anchor = kb_patterns[i % len(kb_patterns)]

            if anchor == "left":
                x_expr = "0"
            elif anchor == "right":
                x_expr = f"(iw*{z_end}-{W})"
            else:
                x_expr = f"(iw*zoom-{W})/2"
            y_expr = f"(ih*zoom-{H})/2"

            zoom_expr = f"{z_start}+({z_end}-{z_start})*(on/{dur_frames})"

            upscale_w = W * 2
            upscale_h = H * 2
            vf_parts = [
                f"scale={upscale_w}:{upscale_h}:force_original_aspect_ratio=increase",
                f"crop={upscale_w}:{upscale_h}",
                f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}'"
                f":d={dur_frames}:s={W}x{H}:fps={FPS}",
                "format=yuv420p",
            ]

            if sections and i < len(sections) and sections[i].strip():
                sub_text = _escape_drawtext(sections[i].strip()[:100])
                vf_parts.append(
                    f"drawtext=text='{sub_text}'"
                    f":fontsize=32:fontcolor=white"
                    f":borderw=2:bordercolor=black@0.8"
                    f":x=(w-tw)/2:y=h-70"
                    f":enable='between(t,0.5,{seconds_per_image - 0.5})'"
                )

            vf = ",".join(vf_parts)

            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", img,
                "-t", str(seconds_per_image),
                "-vf", vf,
                "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                "-r", str(FPS), "-pix_fmt", "yuv420p",
                seg,
            ]
            _log(f"Rendering scene {i+1}/{len(images)} (Ken Burns + subtitle)")
            try:
                subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=True)
                segments.append(seg)
            except subprocess.CalledProcessError as e:
                _log(f"Ken Burns failed for scene {i+1}, trying simple scale: "
                     f"{e.stderr[:150] if e.stderr else ''}")
                simple_cmd = [
                    "ffmpeg", "-y",
                    "-loop", "1", "-i", img,
                    "-t", str(seconds_per_image),
                    "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
                           f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p",
                    "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                    "-r", str(FPS), "-pix_fmt", "yuv420p",
                    seg,
                ]
                subprocess.run(simple_cmd, capture_output=True, text=True,
                               timeout=120, check=True)
                segments.append(seg)

        if len(segments) > 1:
            _log("Applying crossfade transitions")
            prev = segments[0]
            for i in range(1, len(segments)):
                xfade_out = temp_dir / f"xfade_{i:04d}.mp4"
                offset = seconds_per_image * i - FADE_DUR * i
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(prev), "-i", str(segments[i]),
                    "-filter_complex",
                    f"xfade=transition=fade:duration={FADE_DUR}:offset={offset:.2f},format=yuv420p",
                    "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                    str(xfade_out),
                ]
                try:
                    subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
                    prev = xfade_out
                except subprocess.CalledProcessError as e:
                    _log(f"Crossfade failed at segment {i}, falling back to concat: "
                         f"{e.stderr[:200] if e.stderr else ''}")
                    prev = None
                    break

            if prev and prev != segments[0]:
                video_track = prev
            else:
                concat_file = temp_dir / "concat.txt"
                with open(concat_file, "w") as f:
                    for seg in segments:
                        safe = str(seg.resolve()).replace("\\", "/")
                        f.write(f"file '{safe}'\n")
                concat_out = temp_dir / "concat_out.mp4"
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                     "-i", str(concat_file), "-c", "copy", str(concat_out)],
                    capture_output=True, text=True, timeout=120, check=True,
                )
                video_track = concat_out
        else:
            video_track = segments[0]

        has_narration = audio_path and Path(audio_path).exists()
        has_music = music_path and Path(music_path).exists()
        valid_sfx = [
            s for s in (sfx_entries or [])
            if s.get("path") and Path(s["path"]).exists()
        ]

        if has_narration or has_music or valid_sfx:
            audio_inputs: list[str] = []
            filter_parts: list[str] = []
            amix_labels: list[str] = []
            idx = 0

            if has_narration:
                audio_inputs.extend(["-i", audio_path])
                filter_parts.append(
                    f"[{idx}:a]aformat=sample_fmts=fltp:sample_rates=44100"
                    f":channel_layouts=stereo,volume=1.0[voice]"
                )
                amix_labels.append("[voice]")
                idx += 1

            if has_music:
                audio_inputs.extend(["-i", music_path])
                filter_parts.append(
                    f"[{idx}:a]aformat=sample_fmts=fltp:sample_rates=44100"
                    f":channel_layouts=stereo,volume=0.15[bgm]"
                )
                amix_labels.append("[bgm]")
                idx += 1

            if valid_sfx:
                total_video_dur = len(images) * seconds_per_image
                _log(f"Mixing {len(valid_sfx)} SFX clips into audio")
                for si, sfx in enumerate(valid_sfx):
                    audio_inputs.extend(["-i", sfx["path"]])
                    offset = sfx.get("offset_seconds", 0)
                    label = f"sfx{si}"
                    filter_parts.append(
                        f"[{idx}:a]aformat=sample_fmts=fltp:sample_rates=44100"
                        f":channel_layouts=stereo,volume=0.6,"
                        f"adelay={int(offset * 1000)}|{int(offset * 1000)},"
                        f"apad=whole_dur={total_video_dur}[{label}]"
                    )
                    amix_labels.append(f"[{label}]")
                    idx += 1

            if len(amix_labels) >= 2:
                _log(f"Mixing {len(amix_labels)} audio tracks (voice+BGM+SFX)")
                mixed_audio = temp_dir / "mixed_audio.aac"
                duration_mode = "first" if has_narration else "longest"
                filter_graph = (
                    ";".join(filter_parts)
                    + ";"
                    + "".join(amix_labels)
                    + f"amix=inputs={len(amix_labels)}"
                    + f":duration={duration_mode}:dropout_transition=3[out]"
                )
                cmd = (
                    ["ffmpeg", "-y"]
                    + audio_inputs
                    + ["-filter_complex", filter_graph,
                       "-map", "[out]", "-c:a", "aac", "-b:a", "192k",
                       str(mixed_audio)]
                )
                subprocess.run(cmd, capture_output=True, text=True,
                               timeout=180, check=True)
                audio_final = str(mixed_audio)
            elif has_narration:
                audio_final = audio_path
            elif has_music:
                audio_final = music_path
            elif valid_sfx:
                audio_final = valid_sfx[0]["path"]
            else:
                audio_final = None
        else:
            audio_final = None

        if audio_final:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_track),
                "-i", audio_final,
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v", "-map", "1:a",
                "-shortest",
                "-movflags", "+faststart",
                str(output_path),
            ]
            subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
        else:
            import shutil
            shutil.copy2(str(video_track), str(output_path))

        size_mb = output_path.stat().st_size / (1024 * 1024)
        _log(f"Video composed: {output_path} ({size_mb:.1f} MB)")
        return str(output_path)

    except subprocess.CalledProcessError as e:
        _log(f"FFmpeg composition failed: {e.stderr[:500] if e.stderr else e}")
        return None
    finally:
        import shutil as _shutil
        _shutil.rmtree(str(temp_dir), ignore_errors=True)


def step_compose_remotion(
    images: list[str],
    audio_path: str | None,
    output_path: Path,
    title: str | None = None,
    sections: list[str] | None = None,
    music_path: str | None = None,
    sfx_entries: list[dict[str, Any]] | None = None,
) -> str | None:
    """Attempt Remotion render. Falls back to FFmpeg slideshow."""
    composer_dir = PROJ_ROOT / "remotion-composer"
    if not (composer_dir / "node_modules").exists():
        _log("Remotion not installed, falling back to FFmpeg slideshow")
        return None

    import shutil
    if not shutil.which("npx"):
        _log("npx not found, falling back to FFmpeg slideshow")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    composition_data: dict[str, Any] = {
        "title": title or "Video",
        "scenes": [],
    }
    for i, img_path in enumerate(images):
        scene: dict[str, Any] = {
            "type": "image_scene",
            "imagePath": str(Path(img_path).resolve()),
            "durationInFrames": 180,
        }
        if sections and i < len(sections) and sections[i].strip():
            scene["subtitle"] = sections[i].strip()
        composition_data["scenes"].append(scene)

    if audio_path:
        composition_data["audioPath"] = str(Path(audio_path).resolve())
    if music_path and Path(music_path).exists():
        composition_data["musicPath"] = str(Path(music_path).resolve())
    if sfx_entries:
        composition_data["sfxTracks"] = [
            {
                "path": str(Path(s["path"]).resolve()),
                "sceneIndex": s["scene_index"],
                "offsetSeconds": s["offset_seconds"],
                "duration": s["duration"],
            }
            for s in sfx_entries
            if s.get("path") and Path(s["path"]).exists()
        ]

    props_file = (output_path.parent / "remotion_props.json").resolve()
    props_file.write_text(json.dumps(composition_data, indent=2), encoding="utf-8")

    total_frames = sum(s.get("durationInFrames", 180) for s in composition_data["scenes"])

    cmd = [
        "npx", "remotion", "render",
        "src/index.tsx", "Explainer",
        str(output_path.resolve()),
        "--props", str(props_file),
        "--frames", f"0-{total_frames - 1}",
    ]

    _log(f"Rendering with Remotion ({total_frames} frames)")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            cwd=str(composer_dir),
        )
        if proc.returncode == 0 and output_path.exists():
            _log(f"Remotion render complete: {output_path}")
            return str(output_path)
        _log(f"Remotion failed (exit {proc.returncode}): {proc.stderr[:500]}")
    except Exception as e:
        _log(f"Remotion error: {e}")

    return None


# ---------------------------------------------------------------------------
# Main pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(prompt_file: str) -> None:
    """Main pipeline execution — AI-powered video production."""
    _log(f"Starting pipeline runner (cwd={os.getcwd()})")
    _log(f"PROJ_ROOT={PROJ_ROOT}")
    _log(f"Python={sys.executable} {sys.version_info.major}.{sys.version_info.minor}")
    _log(f"Gemini LLM: {'available' if _gemini_available() else 'NOT available'}")

    _discover_tools()

    available = registry.get_available()
    unavailable = registry.get_unavailable()
    _log(f"Tools discovered: {len(available)} available, {len(unavailable)} unavailable")
    for t in available:
        _log(f"  OK   {t.name} ({t.provider})")
    for t in unavailable[:10]:
        _log(f"  MISS {t.name} ({t.provider})")

    prompt_path = Path(prompt_file)
    if not prompt_path.exists():
        _log(f"Prompt file not found: {prompt_file}")
        sys.exit(1)

    payload = json.loads(prompt_path.read_text(encoding="utf-8"))
    pipeline_name = payload.get("pipeline", "animated-explainer")
    project_id = payload.get("projectId", "web-project")
    title = payload.get("title", project_id)
    raw_prompt = payload.get("prompt", "")
    ref_url = payload.get("referenceUrl", "")
    uploaded_audio = payload.get("uploadedAudioPath", "")
    form_audio_lang = (payload.get("audioLanguage") or "").strip().lower()
    form_subtitle_lang = (payload.get("subtitleLanguage") or "").strip().lower()

    _log(f"Pipeline: {pipeline_name}")
    _log(f"Project:  {project_id}")
    _log(f"Title:    {title}")
    _log(f"Raw prompt: {raw_prompt[:300]}...")
    if uploaded_audio:
        _log(f"Custom audio: {uploaded_audio}")

    project_dir = Path("projects") / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = project_dir / "assets"
    renders_dir = project_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 0: Parse production intent ---
    ref_summary = None
    ref_path = step_download_reference(ref_url, project_dir) if ref_url else None
    if ref_path:
        _log(f"Reference available at: {ref_path}")
        analysis = step_analyze_reference(ref_path, project_dir)
        if analysis:
            ref_summary = _summarize_reference(analysis)
            _log(f"Reference summary:\n{ref_summary[:300]}...")

    intent = _parse_production_intent(raw_prompt, ref_summary)
    content_prompt = intent["content_prompt"]
    audio_lang = form_audio_lang or intent["audio_language"]
    subtitle_lang = form_subtitle_lang or intent["subtitle_language"]
    target_dur = intent["target_duration"]
    reference_driven = intent["reference_driven"]
    style_notes = intent.get("style_notes", "")

    if audio_lang not in SUPPORTED_LANGUAGES and audio_lang != "english":
        audio_lang = "english"
    if subtitle_lang and subtitle_lang not in SUPPORTED_LANGUAGES and subtitle_lang != "english":
        subtitle_lang = ""

    _log(f"Content prompt: {content_prompt[:200]}...")
    _log(f"Audio language: {audio_lang}")
    _log(f"Subtitle language: {subtitle_lang or '(same as audio)'}")
    _log(f"Target duration: {target_dur}s")
    _log(f"Reference-driven: {reference_driven}")

    # --- Phase 1b: Transcribe uploaded audio ---
    has_custom_audio = bool(uploaded_audio and Path(uploaded_audio).exists())
    audio_transcript = None
    if has_custom_audio:
        audio_dur = _probe_duration(uploaded_audio)
        _log(f"Custom audio duration: {audio_dur:.1f}s")
        target_dur = max(audio_dur, 15.0)
        audio_transcript = step_transcribe_audio(uploaded_audio, language=audio_lang)
        if audio_transcript:
            _log(f"Audio transcript ({len(audio_transcript)} chars):\n{audio_transcript[:300]}...")
        else:
            _log("WARNING: Could not transcribe audio — images may not match audio content")

    # --- Phase 2: Script generation (LLM-powered) ---
    if has_custom_audio:
        if audio_transcript:
            script = audio_transcript
        else:
            _log("WARNING: No transcript available for uploaded audio — "
                 "using title-based script (images may not match audio)")
            script = title if title and title != project_id else "Visual presentation"
    else:
        script = step_generate_script(
            content_prompt, title, ref_summary,
            target_duration=target_dur,
            language=audio_lang,
            reference_driven=reference_driven,
        )
    _log(f"Script ({len(script)} chars):\n{script[:300]}...")

    # --- Phase 3: Scene plan (LLM-powered) ---
    scene_count = max(2, min(10, int(target_dur / 5)))
    scene_plan = step_generate_scene_plan(
        script, content_prompt, title, ref_summary, scene_count=scene_count,
    )
    _log(f"Scene plan: {len(scene_plan)} scenes")
    for i, s in enumerate(scene_plan):
        sfx_tag = f" sfx=\"{s['sfx_prompt']}\"" if s.get("sfx_prompt") else ""
        _log(f"  Scene {i+1}: query='{s.get('search_query', '?')}' "
             f"mood={s.get('mood', '?')}{sfx_tag}")

    narration_sections = [s.get("narration", "") for s in scene_plan]

    # When using custom audio without a transcript, don't show
    # raw prompt or placeholder text as subtitles
    if has_custom_audio and not audio_transcript:
        _log("Skipping subtitles — no transcript available for uploaded audio")
        subtitle_sections = [""] * len(narration_sections)
    elif subtitle_lang and subtitle_lang != audio_lang:
        subtitle_sections = step_translate_subtitles(
            narration_sections, audio_lang, subtitle_lang,
        )
    else:
        subtitle_sections = narration_sections

    keywords = _extract_keywords(content_prompt)
    if not keywords:
        keywords = _extract_keywords(script)
    if not keywords:
        keywords = ["technology", "innovation", "digital"]

    # --- Phase 4: TTS (skipped if custom audio uploaded) ---
    if uploaded_audio and Path(uploaded_audio).exists():
        _log(f"Using uploaded audio: {uploaded_audio}")
        audio_path = uploaded_audio
    else:
        audio_path = step_tts(script, assets_dir / "narration.wav", language=audio_lang)

    # --- Phase 5: Scene-matched images ---
    images = step_fetch_images(scene_plan, assets_dir / "images")

    if not images:
        _log("No images fetched — generating gradient placeholders")
        img_dir = assets_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        colors = ["0x1a1a2e", "0x16213e", "0x0f3460", "0x533483", "0x2b2d42"]
        for ci in range(len(scene_plan)):
            placeholder = img_dir / f"placeholder_{ci:02d}.jpg"
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i",
                 f"color=c={colors[ci % len(colors)]}:s=1920x1080:d=1",
                 "-frames:v", "1", str(placeholder)],
                capture_output=True, text=True, timeout=30,
            )
            if placeholder.exists():
                images.append(str(placeholder))

    # --- Phase 6: Background music ---
    narration_dur = _probe_duration(audio_path) if audio_path else 0
    effective_dur = max(narration_dur, len(images) * (target_dur / max(len(images), 1)), target_dur)
    music_path = step_fetch_music(
        scene_plan, keywords, assets_dir / "music",
        target_duration=effective_dur,
    )

    # --- Phase 6b: AI Sound Effects ---
    spi = effective_dur / max(len(images), 1)
    sfx_entries = step_generate_sfx(scene_plan, assets_dir / "sfx", seconds_per_scene=spi)

    # --- Phase 7: Compose final video ---
    final_path = renders_dir / "final.mp4"

    video_path = step_compose_remotion(
        images, audio_path, final_path, title, subtitle_sections, music_path,
        sfx_entries=sfx_entries,
    )

    if not video_path:
        video_path = step_compose_slideshow(
            images, audio_path, final_path,
            sections=subtitle_sections, music_path=music_path,
            sfx_entries=sfx_entries,
        )

    if video_path and Path(video_path).exists():
        _log("Pipeline completed successfully")
        print(f"OUTPUT_VIDEO={video_path}")
        sys.exit(0)
    else:
        _log("Pipeline failed — no output video produced")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenMontage AI pipeline runner")
    parser.add_argument("--pipeline", required=True, help="Pipeline name")
    parser.add_argument("--project", required=True, help="Project ID")
    parser.add_argument("--prompt-file", required=True, help="Path to prompt JSON")
    args = parser.parse_args()
    run_pipeline(args.prompt_file)


if __name__ == "__main__":
    main()
