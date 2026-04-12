"""Google-only AI pipeline runner for web-submitted video production jobs.

All AI services use a single GOOGLE_API_KEY:
  - Gemini API: script writing, scene planning, intent parsing, audio transcription
  - Google Imagen: AI image generation
  - Google Cloud TTS: text-to-speech narration
  - FFmpeg: local video composition (Ken Burns, crossfades, subtitles)
  - yt-dlp: reference video download (local utility)
"""

from __future__ import annotations

import argparse
import base64
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
# Tool helpers (for yt-dlp and video_analyzer — local utilities)
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
# Google API key
# ---------------------------------------------------------------------------

def _google_api_key() -> str | None:
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


def _google_available() -> bool:
    return bool(_google_api_key())


# ---------------------------------------------------------------------------
# Gemini LLM
# ---------------------------------------------------------------------------

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)


def _gemini_generate(prompt: str, max_tokens: int = 4096, retries: int = 4) -> str | None:
    """Call Google Gemini with retry for rate limits. Returns text or None."""
    api_key = _google_api_key()
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
# Google Imagen — image generation
# ---------------------------------------------------------------------------

IMAGEN_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "imagen-3.0-generate-002:predict"
)


def _google_imagen_generate(
    prompt: str, output_path: Path, aspect_ratio: str = "16:9"
) -> str | None:
    """Generate an image via Google Imagen API. Returns path or None."""
    api_key = _google_api_key()
    if not api_key:
        return None

    url = f"{IMAGEN_ENDPOINT}?key={api_key}"
    body = {
        "instances": [{"prompt": prompt}],
        "parameters": {
            "aspectRatio": aspect_ratio,
            "sampleCount": 1,
        },
    }

    for attempt in range(3):
        try:
            resp = requests.post(
                url, json=body,
                headers={"Content-Type": "application/json"},
                timeout=90,
            )
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                _log(f"Imagen rate-limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()

            predictions = data.get("predictions", [])
            if not predictions:
                _log("Imagen returned no predictions")
                return None

            img_b64 = predictions[0].get("bytesBase64Encoded", "")
            if not img_b64:
                _log("Imagen returned empty image data")
                return None

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(base64.b64decode(img_b64))

            if output_path.exists() and output_path.stat().st_size > 500:
                return str(output_path)
            return None

        except requests.exceptions.HTTPError as e:
            _log(f"Imagen API error (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(3)
        except Exception as e:
            _log(f"Imagen error: {e}")
            break

    return None


# ---------------------------------------------------------------------------
# Google Cloud TTS
# ---------------------------------------------------------------------------

GOOGLE_TTS_ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"

GOOGLE_TTS_VOICE_MAP: dict[str, dict[str, str]] = {
    "en": {"languageCode": "en-US", "name": "en-US-Studio-Q"},
    "ta": {"languageCode": "ta-IN", "name": "ta-IN-Standard-A"},
    "hi": {"languageCode": "hi-IN", "name": "hi-IN-Standard-A"},
    "te": {"languageCode": "te-IN", "name": "te-IN-Standard-A"},
    "kn": {"languageCode": "kn-IN", "name": "kn-IN-Standard-A"},
    "ml": {"languageCode": "ml-IN", "name": "ml-IN-Standard-A"},
    "bn": {"languageCode": "bn-IN", "name": "bn-IN-Standard-A"},
    "mr": {"languageCode": "mr-IN", "name": "mr-IN-Standard-A"},
    "gu": {"languageCode": "gu-IN", "name": "gu-IN-Standard-A"},
    "pa": {"languageCode": "pa-IN", "name": "pa-IN-Standard-A"},
    "ur": {"languageCode": "ur-IN", "name": "ur-IN-Standard-A"},
    "ar": {"languageCode": "ar-XA", "name": "ar-XA-Standard-A"},
    "es": {"languageCode": "es-ES", "name": "es-ES-Studio-F"},
    "fr": {"languageCode": "fr-FR", "name": "fr-FR-Studio-A"},
    "de": {"languageCode": "de-DE", "name": "de-DE-Studio-B"},
    "it": {"languageCode": "it-IT", "name": "it-IT-Standard-A"},
    "pt": {"languageCode": "pt-BR", "name": "pt-BR-Standard-A"},
    "ja": {"languageCode": "ja-JP", "name": "ja-JP-Standard-A"},
    "ko": {"languageCode": "ko-KR", "name": "ko-KR-Standard-A"},
    "zh": {"languageCode": "cmn-CN", "name": "cmn-CN-Standard-A"},
    "ru": {"languageCode": "ru-RU", "name": "ru-RU-Standard-A"},
    "nl": {"languageCode": "nl-NL", "name": "nl-NL-Standard-A"},
    "pl": {"languageCode": "pl-PL", "name": "pl-PL-Standard-A"},
    "tr": {"languageCode": "tr-TR", "name": "tr-TR-Standard-A"},
    "th": {"languageCode": "th-TH", "name": "th-TH-Standard-A"},
    "vi": {"languageCode": "vi-VN", "name": "vi-VN-Standard-A"},
    "id": {"languageCode": "id-ID", "name": "id-ID-Standard-A"},
    "ms": {"languageCode": "ms-MY", "name": "ms-MY-Standard-A"},
    "sv": {"languageCode": "sv-SE", "name": "sv-SE-Standard-A"},
    "no": {"languageCode": "nb-NO", "name": "nb-NO-Standard-A"},
    "da": {"languageCode": "da-DK", "name": "da-DK-Standard-A"},
    "fi": {"languageCode": "fi-FI", "name": "fi-FI-Standard-A"},
}


def _google_tts(text: str, output_path: Path, language: str = "english") -> str | None:
    """Generate narration via Google Cloud Text-to-Speech. Returns path or None."""
    api_key = _google_api_key()
    if not api_key:
        _log("No GOOGLE_API_KEY — TTS unavailable")
        return None

    lang_code = SUPPORTED_LANGUAGES.get(language, "en")
    voice_cfg = GOOGLE_TTS_VOICE_MAP.get(lang_code, GOOGLE_TTS_VOICE_MAP["en"])

    url = f"{GOOGLE_TTS_ENDPOINT}?key={api_key}"
    body = {
        "input": {"text": text},
        "voice": {
            "languageCode": voice_cfg["languageCode"],
            "name": voice_cfg["name"],
        },
        "audioConfig": {
            "audioEncoding": "MP3",
            "speakingRate": 0.95,
            "pitch": 0.0,
        },
    }

    _log(f"Generating {language} narration via Google Cloud TTS...")
    try:
        resp = requests.post(
            url, json=body,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        audio_b64 = data.get("audioContent", "")
        if not audio_b64:
            _log("Google TTS returned empty audio")
            return None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(base64.b64decode(audio_b64))

        if output_path.exists() and output_path.stat().st_size > 500:
            _log(f"Narration generated ({language}): {output_path}")
            return str(output_path)
        _log("Google TTS output too small — may have failed")
        return None

    except requests.exceptions.HTTPError as e:
        _log(f"Google TTS API error: {e}")
        if "403" in str(e):
            _log("HINT: Enable 'Cloud Text-to-Speech API' in your Google Cloud project")
    except Exception as e:
        _log(f"Google TTS error: {e}")

    return None


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _strip_urls(text: str) -> str:
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'(?i)reference\s*:\s*', '', text)
    return text.strip()


def _extract_keywords(text: str, max_words: int = 4) -> list[str]:
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
        "youtube", "watch", "audio", "attached", "file", "shared",
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
    """Use Gemini to separate production instructions from creative content."""
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


# ---------------------------------------------------------------------------
# Step 1: Script generation (Gemini)
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
# Step 2: Scene plan (Gemini)
# ---------------------------------------------------------------------------

def step_generate_scene_plan(
    script: str,
    prompt: str,
    title: str,
    ref_summary: str | None = None,
    scene_count: int = 6,
) -> list[dict[str, str]]:
    if _google_available():
        ref_context = ""
        if ref_summary:
            ref_context = f"\nReference video style to match:\n{ref_summary[:400]}\n"

        llm_prompt = f"""You are a video scene planner. Break this narration script into {scene_count} visual scenes.

Title: {title}
Topic: {prompt[:200]}
{ref_context}
Script:
{script[:2000]}

For each scene, provide a JSON array with exactly {scene_count} objects. Each object must have:
- "narration": the exact portion of the script for this scene (1-3 sentences)
- "image_prompt": a detailed prompt for AI image generation (describe the visual: subject, setting, lighting, mood, colors, camera angle). Be SPECIFIC to the topic, not generic.
- "search_query": 2-4 word search query highly specific to the scene content
- "mood": one word describing the scene mood (e.g., inspiring, dramatic, calm, energetic)

Respond ONLY with the JSON array, no other text. Example format:
[
  {{"narration": "...", "image_prompt": "...", "search_query": "...", "mood": "..."}},
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
# Step 3: TTS (Google Cloud TTS only)
# ---------------------------------------------------------------------------

def step_tts(text: str, output_path: Path, language: str = "english") -> str | None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return _google_tts(text, output_path, language=language)


# ---------------------------------------------------------------------------
# Step 3b: Subtitle translation (Gemini)
# ---------------------------------------------------------------------------

def step_translate_subtitles(
    narration_sections: list[str],
    source_language: str,
    target_language: str,
) -> list[str]:
    if source_language == target_language or not target_language:
        return narration_sections
    if not _google_available():
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
                _log(f"Subtitles translated: {len(translated)} sections -> {target_language}")
                return translated
        except (json.JSONDecodeError, ValueError) as e:
            _log(f"Subtitle translation parse failed: {e}")

    _log("Subtitle translation failed — using original narration")
    return narration_sections


# ---------------------------------------------------------------------------
# Step 3c: Audio transcription (Gemini multimodal only)
# ---------------------------------------------------------------------------

def step_transcribe_audio(audio_path: str, language: str = "english") -> str | None:
    """Transcribe uploaded audio via Gemini multimodal."""
    api_key = _google_api_key()
    if not api_key:
        _log("No GOOGLE_API_KEY — cannot transcribe audio")
        return None

    _log(f"Transcribing audio via Gemini multimodal ({language})...")
    try:
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
            f"?key={api_key}"
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
                _log(f"Transcription complete: {len(text)} chars")
                return text.strip()
            _log("Transcription returned insufficient text")
        else:
            _log(f"Gemini transcription failed: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        _log(f"Gemini transcription error: {e}")

    _log("Audio transcription failed — cannot determine audio content")
    return None


# ---------------------------------------------------------------------------
# Step 4: Image generation (Google Imagen only)
# ---------------------------------------------------------------------------

def step_fetch_images(
    scene_plan: list[dict[str, str]],
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images: list[str] = []

    if not _google_available():
        _log("No GOOGLE_API_KEY — cannot generate images")
        return images

    _log("Generating images via Google Imagen...")

    for i, scene in enumerate(scene_plan):
        out = output_dir / f"scene_{i:02d}.png"
        image_prompt = scene.get("image_prompt", "")
        if not image_prompt:
            image_prompt = scene.get("search_query", "abstract background")

        _log(f"Scene {i+1}/{len(scene_plan)}: generating image via Imagen")
        result_path = _google_imagen_generate(image_prompt, out, aspect_ratio="16:9")
        if result_path:
            images.append(result_path)
            continue

        _log(f"Scene {i+1}: Imagen failed — no image obtained")

    return images


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
# Step 5: Compose video (FFmpeg — Ken Burns + crossfades + subtitles)
# ---------------------------------------------------------------------------

def step_compose_slideshow(
    images: list[str],
    audio_path: str | None,
    output_path: Path,
    seconds_per_image: float = 6.0,
    sections: list[str] | None = None,
) -> str | None:
    """Compose video: Ken Burns motion, crossfade transitions, subtitles."""
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

        if audio_path and Path(audio_path).exists():
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_track),
                "-i", audio_path,
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


# ---------------------------------------------------------------------------
# Main pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(prompt_file: str) -> None:
    _log(f"Starting pipeline runner (cwd={os.getcwd()})")
    _log(f"PROJ_ROOT={PROJ_ROOT}")
    _log(f"Python={sys.executable} {sys.version_info.major}.{sys.version_info.minor}")
    _log(f"Google API: {'available' if _google_available() else 'NOT available'}")

    _discover_tools()

    available = registry.get_available()
    unavailable = registry.get_unavailable()
    _log(f"Tools discovered: {len(available)} available, {len(unavailable)} unavailable")
    for t in available:
        _log(f"  OK   {t.name} ({t.provider})")

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

    # --- Phase 0: Reference video analysis ---
    ref_summary = None
    ref_path = step_download_reference(ref_url, project_dir) if ref_url else None
    if ref_path:
        _log(f"Reference available at: {ref_path}")
        analysis = step_analyze_reference(ref_path, project_dir)
        if analysis:
            ref_summary = _summarize_reference(analysis)
            _log(f"Reference summary:\n{ref_summary[:300]}...")

    # --- Phase 1: Parse production intent (Gemini) ---
    intent = _parse_production_intent(raw_prompt, ref_summary)
    content_prompt = intent["content_prompt"]
    audio_lang = form_audio_lang or intent["audio_language"]
    subtitle_lang = form_subtitle_lang or intent["subtitle_language"]
    target_dur = intent["target_duration"]
    reference_driven = intent["reference_driven"]

    if audio_lang not in SUPPORTED_LANGUAGES and audio_lang != "english":
        audio_lang = "english"
    if subtitle_lang and subtitle_lang not in SUPPORTED_LANGUAGES and subtitle_lang != "english":
        subtitle_lang = ""

    _log(f"Content prompt: {content_prompt[:200]}...")
    _log(f"Audio language: {audio_lang}")
    _log(f"Subtitle language: {subtitle_lang or '(same as audio)'}")
    _log(f"Target duration: {target_dur}s")
    _log(f"Reference-driven: {reference_driven}")

    # --- Phase 1b: Transcribe uploaded audio (Gemini multimodal) ---
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

    # --- Phase 2: Script generation (Gemini) ---
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

    # --- Phase 3: Scene plan (Gemini) ---
    scene_count = max(2, min(10, int(target_dur / 5)))
    scene_plan = step_generate_scene_plan(
        script, content_prompt, title, ref_summary, scene_count=scene_count,
    )
    _log(f"Scene plan: {len(scene_plan)} scenes")
    for i, s in enumerate(scene_plan):
        _log(f"  Scene {i+1}: query='{s.get('search_query', '?')}' mood={s.get('mood', '?')}")

    narration_sections = [s.get("narration", "") for s in scene_plan]

    # Subtitles: skip if custom audio with no transcript
    if has_custom_audio and not audio_transcript:
        _log("Skipping subtitles — no transcript available for uploaded audio")
        subtitle_sections = [""] * len(narration_sections)
    elif subtitle_lang and subtitle_lang != audio_lang:
        subtitle_sections = step_translate_subtitles(
            narration_sections, audio_lang, subtitle_lang,
        )
    else:
        subtitle_sections = narration_sections

    # --- Phase 4: TTS — Google Cloud TTS (skipped if custom audio) ---
    if uploaded_audio and Path(uploaded_audio).exists():
        _log(f"Using uploaded audio: {uploaded_audio}")
        audio_path = uploaded_audio
    else:
        audio_path = step_tts(script, assets_dir / "narration.mp3", language=audio_lang)

    # --- Phase 5: Scene-matched images (Google Imagen) ---
    images = step_fetch_images(scene_plan, assets_dir / "images")

    if not images:
        _log("No images generated — creating gradient placeholders")
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

    # --- Phase 6: Compose final video (FFmpeg) ---
    final_path = renders_dir / "final.mp4"

    video_path = step_compose_slideshow(
        images, audio_path, final_path,
        sections=subtitle_sections,
    )

    if video_path and Path(video_path).exists():
        _log("Pipeline completed successfully")
        print(f"OUTPUT_VIDEO={video_path}")
        sys.exit(0)
    else:
        _log("Pipeline failed — no output video produced")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Google-only AI video pipeline")
    parser.add_argument("--pipeline", required=True, help="Pipeline name")
    parser.add_argument("--project", required=True, help="Project ID")
    parser.add_argument("--prompt-file", required=True, help="Path to prompt JSON")
    args = parser.parse_args()
    run_pipeline(args.prompt_file)


if __name__ == "__main__":
    main()
