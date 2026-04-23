"""Shared utilities, constants, and Google API wrappers used by all pipeline workers.

Centralises: logging, progress emission, Gemini/Imagen/TTS API calls,
text helpers, FFmpeg probe, and project-wide constants.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Project root bootstrap (same as the old monolith so tool discovery works)
# ---------------------------------------------------------------------------

PROJ_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))
os.chdir(str(PROJ_ROOT))

from tools.tool_registry import registry      # noqa: E402
from tools.base_tool import ToolStatus         # noqa: E402
from webapp.pipeline_stages import PIPELINE_STAGE_ORDER  # noqa: E402

_PIPELINE_STAGE_IDS: list[str] = [s[0] for s in PIPELINE_STAGE_ORDER]
_N_PIPELINE_STAGES: int = len(_PIPELINE_STAGE_IDS)

# ---------------------------------------------------------------------------
# API usage tracking
# ---------------------------------------------------------------------------

_api_usage: dict[str, Any] = {
    "gemini_calls": 0,
    "imagen_calls": 0,
    "tts_calls": 0,
    "tts_characters": 0,
    "estimated_cost_usd": 0.0,
}

_api_usage_lock = threading.Lock()


def _track_api(service: str, cost_estimate: float = 0.0, chars: int = 0) -> None:
    with _api_usage_lock:
        if service == "gemini":
            _api_usage["gemini_calls"] += 1
            _api_usage["estimated_cost_usd"] += cost_estimate
        elif service == "imagen":
            _api_usage["imagen_calls"] += 1
            _api_usage["estimated_cost_usd"] += cost_estimate
        elif service == "gemini_image":
            _api_usage["imagen_calls"] += 1
            _api_usage["estimated_cost_usd"] += cost_estimate
        elif service == "tts":
            _api_usage["tts_calls"] += 1
            _api_usage["tts_characters"] += chars
            _api_usage["estimated_cost_usd"] += cost_estimate


def get_api_usage() -> dict[str, Any]:
    return dict(_api_usage)


def _reset_api_usage() -> None:
    _api_usage["gemini_calls"] = 0
    _api_usage["imagen_calls"] = 0
    _api_usage["tts_calls"] = 0
    _api_usage["tts_characters"] = 0
    _api_usage["estimated_cost_usd"] = 0.0


# ---------------------------------------------------------------------------
# Logging / progress
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[pipeline-runner] {msg}", flush=True)


def _emit_progress_snapshot(completed: list[str], current: str | None, overall_pct: int) -> None:
    payload = {
        "completed": completed,
        "current": current,
        "overallPct": max(0, min(100, overall_pct)),
        "stageLabels": dict(PIPELINE_STAGE_ORDER),
    }
    print(f"STEP_PROGRESS={json.dumps(payload)}", flush=True)


def _progress_pct(completed_len: int, within_stage: float = 0.0) -> int:
    if _N_PIPELINE_STAGES <= 0:
        return 0
    frac = max(0.0, min(0.999, within_stage))
    return max(0, min(99, int(100 * (completed_len + frac) / _N_PIPELINE_STAGES)))


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
    api_key = _google_api_key()
    if not api_key:
        return None

    models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"]

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
                    _log(f"Gemini rate-limited (429) on {model}, waiting {wait}s (attempt {attempt+1}/{retries})")
                    time.sleep(wait)
                    continue
                if resp.status_code == 404:
                    _log(f"Gemini model {model} not found (404), trying next")
                    break
                resp.raise_for_status()
                data = resp.json()
                _track_api("gemini", cost_estimate=0.0001)
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except requests.exceptions.HTTPError as e:
                if "429" in str(e) and attempt < retries - 1:
                    wait = min(5 * (attempt + 1), 60)
                    _log(f"Gemini rate-limited on {model}, waiting {wait}s")
                    time.sleep(wait)
                    continue
                if "404" in str(e):
                    break
                _log(f"Gemini API error ({model}): {e}")
                break
            except Exception as e:
                _log(f"Gemini API error ({model}): {e}")
                break

    return None


# ---------------------------------------------------------------------------
# Google Imagen
# ---------------------------------------------------------------------------

IMAGEN_MODELS = [
    "imagen-4.0-fast-generate-001",
    "imagen-4.0-generate-001",
]
IMAGEN_ENDPOINT_TMPL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:predict"
)


def _google_imagen_generate(
    prompt: str, output_path: Path, aspect_ratio: str = "16:9"
) -> str | None:
    api_key = _google_api_key()
    if not api_key:
        return None

    for model in IMAGEN_MODELS:
        url = IMAGEN_ENDPOINT_TMPL.format(model=model) + f"?key={api_key}"
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
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": api_key,
                    },
                    timeout=90,
                )
                if resp.status_code == 429:
                    wait = 5 * (attempt + 1)
                    _log(f"Imagen rate-limited on {model}, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if resp.status_code == 404:
                    _log(f"Imagen model {model} not found (404), trying next")
                    break
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
                    _track_api("imagen", cost_estimate=0.04)
                    return str(output_path)
                return None

            except requests.exceptions.HTTPError as e:
                if "404" in str(e):
                    _log(f"Imagen model {model} not found, trying next")
                    break
                _log(f"Imagen API error ({model}, attempt {attempt+1}): {e}")
                if attempt < 2:
                    time.sleep(3)
            except Exception as e:
                _log(f"Imagen error: {e}")
                break

    return None


# ---------------------------------------------------------------------------
# Gemini native image generation (Nano Banana family)
# ---------------------------------------------------------------------------

def _gemini_response_first_image_b64(data: dict[str, Any]) -> tuple[str | None, str]:
    for cand in data.get("candidates") or []:
        content = cand.get("content") or {}
        for part in content.get("parts") or []:
            inline = part.get("inlineData") or part.get("inline_data")
            if not inline or not isinstance(inline, dict):
                continue
            b64 = inline.get("data")
            if not b64:
                continue
            mime = (
                inline.get("mimeType")
                or inline.get("mime_type")
                or "image/png"
            )
            return str(b64), str(mime)
    return None, ""


def _google_gemini_native_image_generate(
    prompt: str,
    output_path: Path,
    aspect_ratio: str = "16:9",
) -> str | None:
    api_key = _google_api_key()
    if not api_key:
        return None

    models_raw = os.environ.get(
        "GEMINI_NATIVE_IMAGE_MODELS",
        "gemini-2.0-flash-preview-image-generation,"
        "gemini-2.5-flash-image-preview,"
        "gemini-3.1-flash-image-preview",
    )
    models = [m.strip() for m in models_raw.split(",") if m.strip()]

    modality_variants: list[list[str]] = [["TEXT", "IMAGE"], ["IMAGE"]]

    for model in models:
        url = GEMINI_ENDPOINT.format(model=model) + f"?key={api_key}"
        model_not_found = False

        for modalities in modality_variants:
            if model_not_found:
                break
            gen_cfg: dict[str, Any] = {
                "responseModalities": modalities,
                "temperature": 0.85,
            }
            if aspect_ratio:
                gen_cfg["imageConfig"] = {"aspectRatio": aspect_ratio}

            body: dict[str, Any] = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": gen_cfg,
            }

            for attempt in range(3):
                try:
                    resp = requests.post(
                        url,
                        json=body,
                        headers={
                            "Content-Type": "application/json",
                            "x-goog-api-key": api_key,
                        },
                        timeout=120,
                    )
                    if resp.status_code == 429:
                        wait = min(5 * (attempt + 1), 60)
                        _log(
                            f"Gemini image rate-limited (429) on {model}, waiting {wait}s "
                            f"(attempt {attempt + 1}/3)",
                        )
                        time.sleep(wait)
                        continue
                    if resp.status_code == 404:
                        _log(f"Gemini image model {model} not found (404), trying next")
                        model_not_found = True
                        break
                    if resp.status_code >= 400:
                        err = resp.text[:400] if resp.text else ""
                        _log(
                            f"Gemini image HTTP {resp.status_code} on {model} "
                            f"modalities={modalities}: {err}",
                        )
                        break

                    data = resp.json()
                    img_b64, _mime = _gemini_response_first_image_b64(data)
                    if not img_b64:
                        fb = data.get("promptFeedback") or data.get("prompt_feedback")
                        _log(
                            f"Gemini image {model}: no inline image; feedback={fb}",
                        )
                        break

                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(base64.b64decode(img_b64))
                    if output_path.exists() and output_path.stat().st_size > 500:
                        _track_api("gemini_image", cost_estimate=0.04)
                        return str(output_path)
                    return None

                except requests.exceptions.HTTPError as e:
                    _log(f"Gemini image API error ({model}): {e}")
                    if attempt < 2:
                        time.sleep(3)
                except Exception as e:
                    _log(f"Gemini image error ({model}): {e}")
                    break

        if model_not_found:
            continue

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


def _google_tts(text: str, output_path: Path, language: str = "english") -> str | None:
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
            char_count = len(text)
            _track_api("tts", cost_estimate=char_count * 0.000016, chars=char_count)
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


def _elevenlabs_api_key() -> str:
    return os.environ.get("ELEVENLABS_API_KEY", "").strip()


def _elevenlabs_available() -> bool:
    return bool(_elevenlabs_api_key())


def _elevenlabs_tts(
    text: str,
    output_path: "Path",
    *,
    voice_id: str,
    language: str = "english",
) -> "str | None":
    """Generate TTS audio via ElevenLabs using a given voice_id.

    Uses eleven_multilingual_v2 which supports 29 languages including
    Tamil, Hindi, Telugu, Malayalam, Kannada, etc.
    Falls back to None if the ElevenLabs API key is missing or package absent.
    """
    api_key = _elevenlabs_api_key()
    if not api_key:
        _log("ElevenLabs API key not set — skipping ElevenLabs TTS")
        return None

    try:
        from elevenlabs.client import ElevenLabs  # type: ignore
        from elevenlabs import save, VoiceSettings  # type: ignore
    except ImportError:
        _log("elevenlabs package not installed — run: pip install elevenlabs>=1.9")
        return None

    _log(f"Generating {language} narration via ElevenLabs (voice_id={voice_id[:12]}...)...")
    try:
        client = ElevenLabs(api_key=api_key)
        audio = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.75,
                similarity_boost=0.90,
                style=0.0,
                speed=1.0,
            ),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save(audio, str(output_path))
        if output_path.exists() and output_path.stat().st_size > 500:
            char_count = len(text)
            _track_api("tts", cost_estimate=char_count * 0.000030, chars=char_count)
            _log(f"ElevenLabs narration generated ({language}): {output_path}")
            return str(output_path)
        _log("ElevenLabs TTS output too small — may have failed")
    except Exception as e:
        _log(f"ElevenLabs TTS error: {e}")
    return None


# ---------------------------------------------------------------------------
# Style / prompt constants
# ---------------------------------------------------------------------------

DEFAULT_STYLE: dict[str, str] = {
    "art_style": (
        "2D digital illustration with soft oil-painting textures, "
        "clean character outlines with expressive faces, warm rich color palette, "
        "detailed Indian/South Asian cultural setting, storybook quality"
    ),
    "image_type": "2D illustrated storybook",
    "editing_style": "slow Ken Burns zoom/pan, soft dissolve transitions, emotional pacing",
    "color_palette": "warm earth tones — golden yellows, deep oranges, rich reds, warm browns, soft greens",
    "mood": "warm emotional storytelling",
}

# ---------------------------------------------------------------------------
# Style profile helpers
# ---------------------------------------------------------------------------

def _get_image_style_prefix() -> str:
    """Return the IMAGE_STYLE_PREFIX for the current IMAGE_STYLE_PROFILE.

    Set IMAGE_STYLE_PROFILE environment variable to select a preset:
      illustrated — 2D oil-painting storybook illustration (DEFAULT)
      cinematic   — photorealistic 3D renders, dramatic lighting
      anime       — anime/manga cel-shaded style
      minimal     — clean minimal flat design
    """
    profile = (os.environ.get("IMAGE_STYLE_PROFILE") or "illustrated").strip().lower()
    if profile in ("cinematic", "3d", "realistic", "photorealistic"):
        return (
            "Cinematic photorealistic 3D illustration, ultra-detailed, "
            "dramatic volumetric lighting with warm golden rays and deep atmospheric shadows, "
            "highly detailed realistic characters with expressive faces and natural skin textures, "
            "richly detailed Indian/South Asian cultural setting and period-accurate costumes, "
            "16:9 wide cinematic composition with shallow depth of field, "
            "emotionally resonant scene, Unreal Engine quality render. "
        )
    if profile in ("anime", "manga"):
        return (
            "Anime/manga style illustration, clean cel-shading, vivid saturated colors, "
            "detailed expressive character designs, dynamic composition, "
            "16:9 cinematic widescreen. "
        )
    if profile in ("minimal", "flat"):
        return (
            "Clean minimalist flat design illustration, bold simple shapes, "
            "limited color palette, modern graphic style, 16:9 composition. "
        )
    # Default: 2D illustrated storybook (warm, hand-painted quality)
    return (
        "2D digital illustration with soft oil-painting textures, "
        "clean confident character outlines, expressive detailed faces, "
        "warm rich color palette (golden yellows, deep oranges, rich reds, warm browns, soft greens), "
        "beautifully detailed Indian/South Asian cultural settings and period-accurate costumes, "
        "soft ambient lighting with golden hour warmth, "
        "storybook quality, 16:9 wide composition. "
    )


IMAGE_STYLE_PREFIX: str = _get_image_style_prefix()

TTS_CHAR_LIMIT = 4800


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


def _parse_json_response(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```\w*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```$', '', cleaned)
    return json.loads(cleaned)


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
    return sections[:target_count + 4]


def _env_truthy(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _meta_ai_style_extra() -> str:
    """Return an additional style prefix string for special profiles.

    For the main style profiles (cinematic, illustrated, etc.), this is empty
    because the style is already embedded in IMAGE_STYLE_PREFIX.
    Extra modifiers only apply for social/feed optimized variants.
    """
    prof = (os.environ.get("IMAGE_STYLE_PROFILE") or "illustrated").strip().lower()
    if prof in ("meta", "meta-ai", "meta_ai", "social", "feed"):
        return (
            "Social short-form polish: bold readable composition, vivid balanced colors, "
            "clean negative space, modern AI-video look with soft cinematic depth. "
        )
    return ""


# ---------------------------------------------------------------------------
# FFmpeg probe helpers
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
