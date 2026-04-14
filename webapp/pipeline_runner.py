"""Google-only AI video pipeline runner.

All AI services use a single GOOGLE_API_KEY:
  - Gemini API: script, scene planning, intent parsing, audio transcription,
                character extraction, style analysis, post-verification
  - Google Imagen: AI image generation with character/scene consistency
  - Google Cloud TTS: narration in 30+ languages
  - FFmpeg: video composition (Ken Burns, crossfades, subtitles, audio mixing)
  - yt-dlp: reference video download
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
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import requests

PROJ_ROOT = Path(__file__).resolve().parent.parent

# API usage tracking — accumulated during run, saved at end
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
            # Native Gemini image (incl. "Nano Banana"); count toward image quota in dashboards
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

if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))
os.chdir(str(PROJ_ROOT))

from tools.tool_registry import registry
from tools.base_tool import ToolStatus
from webapp.pipeline_stages import PIPELINE_STAGE_ORDER

_PIPELINE_STAGE_IDS: list[str] = [s[0] for s in PIPELINE_STAGE_ORDER]
_N_PIPELINE_STAGES: int = len(_PIPELINE_STAGE_IDS)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[pipeline-runner] {msg}", flush=True)


def _emit_progress_snapshot(completed: list[str], current: str | None, overall_pct: int) -> None:
    """Emit machine-readable progress for the worker UI (step-by-step stages)."""
    payload = {
        "completed": completed,
        "current": current,
        "overallPct": max(0, min(100, overall_pct)),
        "stageLabels": dict(PIPELINE_STAGE_ORDER),
    }
    print(f"STEP_PROGRESS={json.dumps(payload)}", flush=True)


def _progress_pct(completed_len: int, within_stage: float = 0.0) -> int:
    """Map completed stage count + 0..1 fraction of current stage to 0..99 (100 reserved for final)."""
    if _N_PIPELINE_STAGES <= 0:
        return 0
    frac = max(0.0, min(0.999, within_stage))
    return max(0, min(99, int(100 * (completed_len + frac) / _N_PIPELINE_STAGES)))


def _cpu_parallel_cap() -> int:
    """Target max logical cores for parallel pools: CPU_PARALLEL_FRACTION * cpu_count (OS buffer)."""
    try:
        frac = float(os.environ.get("CPU_PARALLEL_FRACTION", "0.85"))
    except ValueError:
        frac = 0.85
    frac = max(0.35, min(0.95, frac))
    n = os.cpu_count() or 2
    return max(1, int(n * frac))


def _clamp_parallel_workers(requested: int, *, per_job_threads: int = 1) -> int:
    """Clamp pool size so (workers × encoder threads) stays within ~CPU cap."""
    cap = _cpu_parallel_cap()
    thr = max(1, int(per_job_threads))
    max_by_cpu = max(1, cap // thr)
    return max(1, min(int(requested), max_by_cpu))


def _merge_timings_for_budget(
    timings: list[dict[str, Any]],
    max_scenes: int = 48,
) -> list[dict[str, Any]]:
    """Merge consecutive timed segments so we have at most max_scenes (faster, fewer images)."""
    if not timings:
        return []
    norm: list[dict[str, Any]] = []
    for t in timings:
        start = float(t.get("start", 0))
        end = float(t.get("end", start + 0.5))
        dur = float(t.get("duration", max(0.1, end - start)))
        text = (t.get("text") or "").strip()
        text_en = (t.get("text_en") or text).strip()
        norm.append({"start": start, "end": end, "duration": max(0.15, dur), "text": text, "text_en": text_en})
    if len(norm) <= max_scenes:
        return norm
    merged: list[dict[str, Any]] = []
    n = len(norm)
    idx = 0
    remaining_slots = max_scenes
    while idx < n and remaining_slots > 0:
        items_left = n - idx
        take = max(1, (items_left + remaining_slots - 1) // remaining_slots)
        chunk = norm[idx : idx + take]
        idx += take
        remaining_slots -= 1
        merged.append({
            "start": chunk[0]["start"],
            "end": chunk[-1]["end"],
            "duration": sum(c["duration"] for c in chunk),
            "text": " ".join(c["text"] for c in chunk if c["text"]),
            "text_en": " ".join(c["text_en"] for c in chunk if c["text_en"]),
        })
    return merged


def _add_english_to_timings(timings: list[dict[str, Any]], source_lang: str) -> list[dict[str, Any]]:
    """Add text_en per segment for image/scene planning (Imagen prompts in English)."""
    if not timings:
        return timings
    low = (source_lang or "english").lower()
    if low in ("english", "en", ""):
        for t in timings:
            t["text_en"] = (t.get("text") or "").strip()
        return timings
    if not _google_available():
        for t in timings:
            t["text_en"] = (t.get("text") or "").strip()
        return timings

    batch_size = 35
    all_en: list[str] = []
    for batch_start in range(0, len(timings), batch_size):
        batch = timings[batch_start : batch_start + batch_size]
        numbered = "\n".join(f'{i}: {t.get("text", "")}' for i, t in enumerate(batch))
        prompt = f"""Translate each numbered line from {source_lang} to clear English for AI image generation.
Preserve names, story meaning, and emotional tone. One translation per line, same order.
Lines:
{numbered}

Respond ONLY with a JSON array of strings (length {len(batch)})."""
        _log(f"Translating {len(batch)} timed lines to English for visual alignment...")
        result = _gemini_generate(prompt, max_tokens=4096)
        if not result:
            for t in batch:
                all_en.append((t.get("text") or "").strip())
            continue
        try:
            arr = _parse_json_response(result)
            if isinstance(arr, list) and len(arr) >= len(batch):
                for i in range(len(batch)):
                    all_en.append(str(arr[i]).strip() if i < len(arr) else batch[i].get("text", ""))
            elif isinstance(arr, list):
                for i, t in enumerate(batch):
                    all_en.append(str(arr[i]).strip() if i < len(arr) else (t.get("text") or ""))
            else:
                for t in batch:
                    all_en.append((t.get("text") or "").strip())
        except (json.JSONDecodeError, ValueError, TypeError):
            for t in batch:
                all_en.append((t.get("text") or "").strip())

    for i, t in enumerate(timings):
        t["text_en"] = all_en[i] if i < len(all_en) else (t.get("text") or "").strip()
    return timings

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


def _build_character_consistency_block(
    character_data: dict[str, Any] | None,
    *,
    max_chars: int = 3200,
) -> str:
    """Compact bible for prompts — same text prepended / paired with reference for every shot."""
    if not character_data:
        return ""
    chars = character_data.get("characters") or []
    if not chars:
        return ""
    lines: list[str] = [
        "LOCKED SERIES DESIGNS — use identically in every frame (faces, hair, skin tone, "
        "age, body proportions, signature outfit colors and patterns):",
    ]
    for ch in chars[:14]:
        if not isinstance(ch, dict):
            continue
        name = (ch.get("name") or "Character").strip()
        desc = (ch.get("description") or "").strip().replace("\n", " ")
        role = (ch.get("role") or "").strip()
        if len(desc) > 420:
            desc = desc[:417] + "..."
        bit = f"• {name}"
        if role:
            bit += f" ({role})"
        bit += f": {desc}" if desc else ""
        lines.append(bit)
    locs = character_data.get("locations") or []
    if locs:
        lines.append("RECURRING LOCATIONS (keep visual identity when reused):")
        for loc in locs[:6]:
            if not isinstance(loc, dict):
                continue
            ln = (loc.get("name") or "").strip()
            ld = (loc.get("description") or "").strip().replace("\n", " ")
            if len(ld) > 200:
                ld = ld[:197] + "..."
            if ln or ld:
                lines.append(f"• {ln}: {ld}" if ln else f"• {ld}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text


_GEMINI_REF_IMAGE_INSTRUCTION = """The FIRST attached image is the OFFICIAL CHARACTER REFERENCE for this video.
Every illustrated scene MUST keep the same character designs: identical faces, hairstyles and colors,
skin tones, ages, body shapes, and recurring costume details as shown in that reference.
Match the same 2D illustrated art style. Only change pose, expression, camera angle, and background
to fit the scene description below. Do not redesign or swap characters."""


def _gemini_response_first_image_b64(data: dict[str, Any]) -> tuple[str | None, str]:
    """Parse generateContent response for inline image bytes (camelCase or snake_case)."""
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
    *,
    reference_image_b64: str | None = None,
    reference_mime: str = "image/png",
) -> str | None:
    """Image generation via Gemini native image models (aka Nano Banana family) — generateContent + IMAGE."""
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

    user_parts: list[dict[str, Any]] = []
    if reference_image_b64:
        user_parts.append({
            "inline_data": {
                "mime_type": reference_mime,
                "data": reference_image_b64,
            },
        })
    user_parts.append({"text": prompt})

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
                "contents": [{"role": "user", "parts": user_parts}],
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


def _generate_character_reference_sheet(
    character_data: dict[str, Any],
    ref_style: dict[str, str] | None,
    output_path: Path,
) -> str | None:
    """One Nano Banana still: lineup / sheet so later multimodal calls can lock designs."""
    bible = _build_character_consistency_block(character_data, max_chars=3000)
    if not bible:
        return None
    art = (ref_style or {}).get("art_style", "stylized illustration")
    prompt = (
        f"{IMAGE_STYLE_PREFIX}"
        "Create a single wide 16:9 CHARACTER REFERENCE SHEET for an animated video. "
        "Show ALL main characters together in one frame — clear faces, readable outfits, "
        "neutral or slight smile poses, even lighting, no story action, no text labels. "
        "This image will be reused to keep the same designs in every scene.\n\n"
        f"Art direction: {art}.\n\n"
        f"{bible}\n"
        "Output one polished lineup illustration."
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return _google_gemini_native_image_generate(
        prompt, output_path, aspect_ratio="16:9",
    )


def _meta_ai_style_extra() -> str:
    """Optional prompt bias toward short-form / Meta-style AI polish (env IMAGE_STYLE_PROFILE)."""
    prof = (os.environ.get("IMAGE_STYLE_PROFILE") or "").strip().lower()
    if prof in ("meta", "meta-ai", "meta_ai", "social", "feed"):
        return (
            "Social short-form polish: bold readable composition, vivid balanced colors, "
            "clean negative space, modern AI-video look with soft cinematic depth. "
        )
    return ""


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


DEFAULT_STYLE: dict[str, str] = {
    "art_style": (
        "2D digital illustration with soft oil-painting textures, "
        "clean outlines, expressive character faces, warm skin tones, "
        "soft gradient background washes, storybook quality"
    ),
    "image_type": "2D illustrated oil-painting style",
    "editing_style": "slow Ken Burns zoom/pan, soft dissolve transitions, gentle sparkle particle effects",
    "color_palette": "warm earth tones — golden yellows, soft oranges, muted greens, warm browns, golden hour lighting",
    "mood": "cinematic emotional storybook",
}

IMAGE_STYLE_PREFIX = (
    "2D digital illustration with soft oil-painting textures, "
    "clean character outlines, expressive faces, warm earth-tone palette "
    "(golden yellows, soft oranges, muted greens, warm browns), "
    "soft gradient background, golden hour warm lighting, "
    "storybook quality, 16:9 wide cinematic composition. "
)

def _analyze_reference_style(ref_summary: str) -> dict[str, str]:
    """Use Gemini to identify the visual art style, image type, and editing approach."""
    if not _google_available() or not ref_summary:
        return dict(DEFAULT_STYLE)

    llm_prompt = f"""Analyze this reference video and identify its visual style.

Reference video analysis:
{ref_summary[:800]}

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
# Audio transcription (Gemini multimodal)
# ---------------------------------------------------------------------------

def step_transcribe_audio(audio_path: str, language: str = "english") -> str | None:
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

        multimodal_models = ["gemini-2.5-flash", "gemini-2.5-pro"]
        for mm_model in multimodal_models:
            gemini_url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{mm_model}:generateContent"
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
            if resp.status_code == 404:
                _log(f"Transcription model {mm_model} not found, trying next")
                continue
            break
        if resp.status_code == 200:
            data = resp.json()
            _track_api("gemini", cost_estimate=0.001)
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


def _transcribe_with_timestamps(audio_path: str, language: str = "english") -> list[dict[str, Any]]:
    """Transcribe audio and estimate per-sentence timing using Gemini."""
    api_key = _google_api_key()
    if not api_key:
        return []

    _log("Extracting sentence-level timestamps via Gemini...")
    try:
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")

        ext = Path(audio_path).suffix.lower().lstrip(".")
        mime_map = {
            "mp3": "audio/mpeg", "m4a": "audio/mp4", "wav": "audio/wav",
            "ogg": "audio/ogg", "aac": "audio/aac", "flac": "audio/flac",
        }
        mime_type = mime_map.get(ext, "audio/mpeg")

        multimodal_models = ["gemini-2.5-flash", "gemini-2.5-pro"]
        resp = None
        for mm_model in multimodal_models:
            gemini_url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{mm_model}:generateContent"
                f"?key={api_key}"
            )
            body = {
                "contents": [{
                    "parts": [
                        {"inline_data": {"mime_type": mime_type, "data": audio_b64}},
                        {"text": f"""Transcribe this {language} audio sentence by sentence.
For each sentence, estimate its start and end time in seconds.

Return a JSON array where each element has:
- "text": the transcribed sentence
- "start": start time in seconds (float)
- "end": end time in seconds (float)
- "duration": duration in seconds (float)

Be as accurate as possible with timing. Group words into natural sentences.
Respond ONLY with the JSON array."""},
                    ]
                }],
                "generationConfig": {"maxOutputTokens": 4096, "temperature": 0.1},
            }
            resp = requests.post(gemini_url, json=body, timeout=120)
            if resp.status_code == 404:
                _log(f"Timestamp model {mm_model} not found, trying next")
                continue
            break
        if resp and resp.status_code == 200:
            data = resp.json()
            _track_api("gemini", cost_estimate=0.001)
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = _parse_json_response(text)
            if isinstance(parsed, list) and len(parsed) >= 1:
                _log(f"Sentence-level timestamps extracted: {len(parsed)} sentences")
                return parsed
    except Exception as e:
        _log(f"Timestamp extraction error: {e}")

    return []


# ---------------------------------------------------------------------------
# Character & scene extraction (Gemini)
# ---------------------------------------------------------------------------

def _extract_characters_and_scenes(
    transcript: str, title: str, ref_style: dict[str, str]
) -> dict[str, Any]:
    """Identify characters, locations, and scene descriptions from the transcript."""
    if not _google_available():
        return {"characters": [], "locations": [], "scenes": []}

    art_style = ref_style.get("art_style", "realistic")
    image_type = ref_style.get("image_type", "AI-generated")

    llm_prompt = f"""Analyze this story/narration transcript and extract characters and scenes.

Title: {title}
Art style to use: {art_style}
Image type: {image_type}

Transcript:
{transcript[:3000]}

Return a JSON object with:
- "characters": Array of character objects, each with:
  - "name": character name
  - "description": detailed visual description (age, gender, clothing, hair, distinguishing features)
  - "role": their role in the story (protagonist, friend, narrator, etc.)
- "locations": Array of location objects, each with:
  - "name": location name
  - "description": detailed visual description (type, size, setting — village/city, colors, atmosphere)
- "scenes": Array of scene objects (one per story beat), each with:
  - "sentence": the narration text for this scene
  - "characters_present": list of character names in this scene
  - "location": where this scene takes place
  - "action": what is happening (e.g. "playing near the house", "walking through the forest")
  - "mood": emotional tone of the scene
  - "camera_suggestion": suggested framing (e.g. "wide shot", "medium close-up", "aerial view")

Be very specific with descriptions — they will be used to generate consistent AI images.

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
                return parsed
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
) -> list[dict[str, Any]]:
    if _google_available():
        ref_context = ""
        if ref_summary:
            ref_context = f"\nReference video style to match:\n{ref_summary[:400]}\n"

        style_context = ""
        if ref_style:
            style_context = f"""
Visual style requirements:
- Art style: {ref_style.get('art_style', 'realistic')}
- Image type: {ref_style.get('image_type', 'AI-generated')}
- Color palette: {ref_style.get('color_palette', 'natural')}
- Mood: {ref_style.get('mood', 'professional')}
"""

        char_context = ""
        if character_data and character_data.get("characters"):
            char_descs = []
            for ch in character_data["characters"]:
                char_descs.append(f"  - {ch.get('name', 'Unknown')}: {ch.get('description', 'No description')}")
            char_context = f"\nCharacters (use EXACT same descriptions for consistency):\n" + "\n".join(char_descs) + "\n"

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

        # For very large scene counts, batch the request
        if scene_count > 20:
            return _generate_scene_plan_batched(
                script, prompt, title, scene_count,
                ref_context, style_context, char_context, timing_context, ref_style,
            )

        llm_prompt = f"""You are a video scene planner. Break this narration script into {scene_count} visual scenes.

Title: {title}
Topic: {prompt[:200]}
{ref_context}{style_context}{char_context}{timing_context}
Script:
{script[:2000]}

For each scene, provide a JSON array with exactly {scene_count} objects. Each object must have:
- "narration": the exact portion of the script for this scene (1-3 sentences)
- "image_prompt": A DETAILED prompt for AI image generation. CRITICAL RULES:
  * EVERY prompt MUST start with: "2D digital illustration with soft oil-painting textures, clean character outlines, expressive faces, warm earth-tone palette (golden yellows, soft oranges, muted greens, warm browns), soft gradient background, golden hour warm lighting, storybook quality, 16:9 wide cinematic composition."
  * Then describe the SPECIFIC scene content after the style prefix
  * Describe the EXACT same character appearance every time a character appears
  * For characters: include ALL physical features (age, hair color, clothing, build, expression)
  * For animals: describe exact species, colors, markings consistently
  * For locations: include specific environmental details (trees, water, sky, buildings)
  * Style reference: like Indian Tamil YouTube story channels — warm, emotional, expressive 2D illustrated art
- "search_query": 2-4 word search query for the scene content
- "mood": one word describing the scene mood
- "duration": estimated seconds this scene should be shown (match audio timing if available)
- "transition": "dissolve" for emotional scenes, "fade" for scene changes, "zoom" for dramatic moments

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
    """Generate scene plans in batches for long videos (20+ scenes)."""
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

        llm_prompt = f"""You are a video scene planner. Create scenes {start_scene+1} to {end_scene} of a {total_scenes}-scene video.

Title: {title}
Topic: {prompt[:200]}
{ref_context}{style_context}{char_context}
Script portion (scenes {start_scene+1}-{end_scene}):
{script_chunk[:3000]}

Create a JSON array with exactly {count_this_batch} scene objects. Each must have:
- "narration": portion of script for this scene
- "image_prompt": EVERY prompt MUST start with "2D digital illustration with soft oil-painting textures, clean character outlines, expressive faces, warm earth-tone palette, soft gradient background, golden hour warm lighting, storybook quality, 16:9 wide cinematic composition." Then describe the specific scene.
- "search_query": 2-4 word search query
- "mood": one word mood
- "duration": seconds (float)
- "transition": "dissolve" for emotional, "fade" for scene changes, "zoom" for dramatic

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


def step_generate_scene_plan_timeline(
    segments: list[dict[str, Any]],
    character_data: dict[str, Any],
    ref_style: dict[str, str],
    title: str,
) -> list[dict[str, Any]]:
    """Build one scene per timed segment: durations match audio; image prompts from English story text."""
    if not segments:
        return []

    style_line = ref_style.get("art_style", DEFAULT_STYLE["art_style"])
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
        prompt = f"""You create still-image scenes for a narrated video. Return a JSON array of EXACTLY {len(batch)} objects, same order as the lines below.

Title: {title}
Art direction: {IMAGE_STYLE_PREFIX}{style_line}

Characters (keep consistent when the same person or animal reappears):
{char_snip}

Timed scenes (each line has duration_sec — copy it EXACTLY into the "duration" field as a float):
{block}

Rules:
- "narration": use the ORIG text for that scene (subtitle / spoken line).
- "image_prompt": MUST begin with the same 2D oil-painting illustrated style wording, then describe ONLY what the EN line says is happening — literal visuals, not metaphors unrelated to the sentence.
- "duration": must equal duration_sec from that line (float).
- "transition": "dissolve" or "fade".
- "search_query": 2-4 English keywords.

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
                        if "2d" not in ip.lower() and "illustrat" not in ip.lower():
                            ip = f"{IMAGE_STYLE_PREFIX}{batch[j].get('text_en', '')}"
                        s["image_prompt"] = ip
                        s.setdefault("transition", "dissolve")
                        s.setdefault("search_query", "story")
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


# ---------------------------------------------------------------------------
# Subtitle translation (Gemini)
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
            translated = _parse_json_response(result)
            if isinstance(translated, list) and len(translated) == len(narration_sections):
                _log(f"Subtitles translated: {len(translated)} sections -> {target_language}")
                return translated
        except (json.JSONDecodeError, ValueError) as e:
            _log(f"Subtitle translation parse failed: {e}")

    _log("Subtitle translation failed — using original narration")
    return narration_sections


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

TTS_CHAR_LIMIT = 4800


def _split_tts_text(text: str, limit: int = TTS_CHAR_LIMIT) -> list[str]:
    """Split long text into chunks that fit within TTS character limits."""
    if len(text) <= limit:
        return [text]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > limit and current:
            chunks.append(current.strip())
            current = sent
        else:
            current = f"{current} {sent}" if current else sent
    if current.strip():
        chunks.append(current.strip())
    return chunks if chunks else [text]


def step_tts(text: str, output_path: Path, language: str = "english") -> str | None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = _split_tts_text(text)
    if len(chunks) <= 1:
        return _google_tts(text, output_path, language=language)

    _log(f"Long text detected ({len(text)} chars) — splitting into {len(chunks)} TTS chunks")
    chunk_paths: list[str] = []
    for i, chunk in enumerate(chunks):
        chunk_path = output_path.parent / f"tts_chunk_{i:03d}.mp3"
        result = _google_tts(chunk, chunk_path, language=language)
        if result:
            chunk_paths.append(result)
        else:
            _log(f"TTS chunk {i+1} failed — skipping")

    if not chunk_paths:
        return None

    if len(chunk_paths) == 1:
        import shutil
        shutil.copy2(chunk_paths[0], str(output_path))
        return str(output_path)

    concat_file = output_path.parent / "tts_concat.txt"
    with open(concat_file, "w") as f:
        for cp in chunk_paths:
            safe = Path(cp).resolve().as_posix()
            f.write(f"file '{safe}'\n")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(concat_file), "-c", "copy", str(output_path)],
            capture_output=True, text=True, timeout=120, check=True,
        )
        _log(f"TTS chunks concatenated: {output_path}")
        return str(output_path)
    except subprocess.CalledProcessError as e:
        _log(f"TTS concat failed: {e.stderr[:200] if e.stderr else ''}")
        import shutil
        shutil.copy2(chunk_paths[0], str(output_path))
        return str(output_path)


# ---------------------------------------------------------------------------
# Image generation (Gemini native / Nano Banana, or Imagen)
# ---------------------------------------------------------------------------

def step_fetch_images(
    scene_plan: list[dict[str, Any]],
    output_dir: Path,
    *,
    character_data: dict[str, Any] | None = None,
    ref_style: dict[str, str] | None = None,
    on_image_progress: Callable[[int, int], None] | None = None,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images: list[str | None] = [None] * len(scene_plan)

    if not _google_available():
        _log("No GOOGLE_API_KEY — cannot generate images")
        return []

    backend = (os.environ.get("IMAGE_BACKEND") or "gemini").strip().lower()
    if backend not in ("gemini", "imagen", "auto"):
        _log(f"IMAGE_BACKEND={backend!r} invalid — use gemini, imagen, or auto; defaulting to gemini")
        backend = "gemini"

    try:
        req_img = int(os.environ.get("IMAGEN_PARALLEL", "3"))
    except ValueError:
        req_img = 3
    parallel = _clamp_parallel_workers(max(1, min(8, req_img)), per_job_threads=1)
    _log(
        f"Generating images (backend={backend}, parallel={parallel}, "
        f"CPU cap={_cpu_parallel_cap()} logical cores) — "
        f"Gemini native = Nano Banana when backend=gemini",
    )

    bible_full = _build_character_consistency_block(character_data)
    use_ref_image = (
        bool(bible_full)
        and os.environ.get("GEMINI_CHARACTER_REF_IMAGE", "1").strip().lower()
        not in ("0", "false", "no", "off")
    )
    ref_sheet_b64: str | None = None
    ref_sheet_mime = "image/png"
    ref_path = output_dir / "_character_reference.png"
    if use_ref_image and character_data and character_data.get("characters"):
        _log("Generating character reference sheet (Nano Banana) for visual consistency...")
        sheet = _generate_character_reference_sheet(
            character_data, ref_style, ref_path,
        )
        if sheet and Path(sheet).exists():
            ref_sheet_b64 = base64.b64encode(Path(sheet).read_bytes()).decode("ascii")
            _log("Character reference sheet ready — scene stills will use multimodal locking")
        else:
            _log("Character reference sheet failed — falling back to text-only consistency")

    def _wrap_prompt_for_backend(raw: str) -> str:
        if not bible_full:
            return raw
        if backend == "imagen" or (
            backend == "auto" and not ref_sheet_b64
        ):
            return (
                "CHARACTER AND LOCATION CONSISTENCY (mandatory for every shot in this video):\n"
                f"{bible_full}\n\n"
                f"SCENE:\n{raw}"
            )
        if ref_sheet_b64:
            backup = bible_full if len(bible_full) <= 1600 else bible_full[:1597] + "..."
            return (
                f"{_GEMINI_REF_IMAGE_INSTRUCTION}\n\n"
                "DESIGN LOCK (text backup — must match reference image):\n"
                f"{backup}\n\n"
                f"SCENE ILLUSTRATION:\n{raw}"
            )
        return (
            "CHARACTER CONSISTENCY (mandatory across all shots):\n"
            f"{bible_full}\n\n"
            f"SCENE:\n{raw}"
        )

    def _generate_still(image_prompt: str, out: Path) -> str | None:
        if backend == "imagen":
            return _google_imagen_generate(image_prompt, out, aspect_ratio="16:9")
        if backend == "gemini":
            return _google_gemini_native_image_generate(
                image_prompt,
                out,
                aspect_ratio="16:9",
                reference_image_b64=ref_sheet_b64,
                reference_mime=ref_sheet_mime,
            )
        r = _google_gemini_native_image_generate(
            image_prompt,
            out,
            aspect_ratio="16:9",
            reference_image_b64=ref_sheet_b64,
            reference_mime=ref_sheet_mime,
        )
        return r or _google_imagen_generate(image_prompt, out, aspect_ratio="16:9")

    meta_extra = _meta_ai_style_extra()

    def _one_scene(i: int, scene: dict[str, Any]) -> tuple[int, str | None]:
        out = output_dir / f"scene_{i:02d}.png"
        image_prompt = scene.get("image_prompt", "")
        if not image_prompt:
            image_prompt = scene.get("search_query", "abstract background")

        if "2d" not in image_prompt.lower() and "illustrat" not in image_prompt.lower() and "oil" not in image_prompt.lower():
            image_prompt = f"{IMAGE_STYLE_PREFIX}{image_prompt}"
        if meta_extra and meta_extra.lower() not in image_prompt.lower():
            image_prompt = f"{meta_extra}{image_prompt}"

        image_prompt = _wrap_prompt_for_backend(image_prompt)

        _log(f"Scene {i+1}/{len(scene_plan)}: queued image ({backend})")
        result_path = _generate_still(image_prompt, out)
        if not result_path:
            _log(f"Scene {i+1}: image generation failed — no image obtained")
        return i, result_path

    total = len(scene_plan)
    done_lock = threading.Lock()
    done_count = 0

    def _track_done() -> None:
        nonlocal done_count
        with done_lock:
            done_count += 1
            d, t = done_count, total
        if on_image_progress:
            try:
                on_image_progress(d, t)
            except Exception as e:
                _log(f"Image progress callback error: {e}")

    if total == 0:
        return []

    if parallel <= 1:
        for i, scene in enumerate(scene_plan):
            idx, path = _one_scene(i, scene)
            images[idx] = path
            _track_done()
    else:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futures = [ex.submit(_one_scene, i, scene) for i, scene in enumerate(scene_plan)]
            for fut in as_completed(futures):
                idx, path = fut.result()
                images[idx] = path
                _track_done()

    out_list = [p for p in images if p]
    _log(f"Image generation complete: {len(out_list)}/{total} images")
    return out_list


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


_KB_PATTERNS: list[tuple[float, float, str]] = [
    (1.0, 1.15, "center"),
    (1.15, 1.0, "center"),
    (1.0, 1.12, "left"),
    (1.12, 1.0, "right"),
]


def _ffmpeg_one_slideshow_segment(
    payload: dict[str, Any],
) -> tuple[int, str, bool]:
    """Encode one Ken-Burns (or fallback) clip — module-level for ProcessPoolExecutor."""
    i = int(payload["i"])
    img = str(payload["img"])
    seg = Path(payload["seg"])
    seg_dur = float(payload["seg_dur"])
    W = int(payload["W"])
    H = int(payload["H"])
    FPS = int(payload["FPS"])
    x264_preset = str(payload["x264_preset"])
    x264_threads = int(payload.get("x264_threads", 1))
    enable_subtitles = bool(payload.get("enable_subtitles"))
    section = str(payload.get("section") or "")
    pat_idx = int(payload.get("pat_idx", 0))
    z_start, z_end, anchor = _KB_PATTERNS[pat_idx % len(_KB_PATTERNS)]

    dur_frames = max(1, int(seg_dur * FPS))
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
    if enable_subtitles and section.strip():
        sub_text = _escape_drawtext(section.strip()[:120])
        sub_end = max(0.15, seg_dur - min(0.5, seg_dur * 0.2))
        vf_parts.append(
            f"drawtext=text='{sub_text}'"
            f":fontsize=32:fontcolor=white"
            f":borderw=2:bordercolor=black@0.8"
            f":x=(w-tw)/2:y=h-70"
            f":enable='between(t,0.05,{sub_end})'"
        )
    vf = ",".join(vf_parts)
    seg.parent.mkdir(parents=True, exist_ok=True)
    enc_threads: list[str] = []
    if x264_threads > 0:
        enc_threads = ["-threads", str(x264_threads)]
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", img,
        "-t", str(seg_dur),
        "-vf", vf,
        "-c:v", "libx264",
        *enc_threads,
        "-crf", "20", "-preset", x264_preset,
        "-r", str(FPS), "-pix_fmt", "yuv420p",
        str(seg),
    ]
    timeout_per_scene = max(180, int(seg_dur * 10))
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_per_scene, check=True)
        if seg.exists() and seg.stat().st_size > 100:
            return i, str(seg), True
    except subprocess.CalledProcessError as e:
        err = e.stderr[:200] if e.stderr else ""
        try:
            simple_cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", img,
                "-t", str(seg_dur),
                "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
                       f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p",
                "-c:v", "libx264",
                *enc_threads,
                "-crf", "20", "-preset", x264_preset,
                "-r", str(FPS), "-pix_fmt", "yuv420p",
                str(seg),
            ]
            subprocess.run(simple_cmd, capture_output=True, text=True, timeout=120, check=True)
            if seg.exists() and seg.stat().st_size > 100:
                return i, str(seg), True
        except subprocess.CalledProcessError:
            pass
        print(f"[pipeline-runner] segment {i+1} ffmpeg failed: {err}", flush=True)
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"[pipeline-runner] segment {i+1} error: {e}", flush=True)
    return i, str(seg), False


# ---------------------------------------------------------------------------
# Compose video (FFmpeg with audio-synced timing)
# ---------------------------------------------------------------------------

def step_compose_slideshow(
    images: list[str],
    audio_path: str | None,
    output_path: Path,
    scene_plan: list[dict[str, Any]] | None = None,
    sections: list[str] | None = None,
    enable_subtitles: bool = False,
    *,
    min_scene_duration: float = 3.0,
    x264_preset: str = "fast",
) -> str | None:
    if not images:
        _log("No images to compose")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    W, H, FPS = 1920, 1080, 30
    FADE_DUR = 1.0

    floor_dur = max(0.12, float(min_scene_duration))
    durations: list[float] = []
    if scene_plan:
        for s in scene_plan:
            d = s.get("duration", 6.0)
            try:
                durations.append(max(floor_dur, float(d)))
            except (ValueError, TypeError):
                durations.append(max(floor_dur, 6.0))

    if audio_path and Path(audio_path).exists():
        total_dur = _probe_duration(audio_path)
        if total_dur > 0:
            if not durations or abs(sum(durations) - total_dur) > total_dur * 0.3:
                per_image = max(floor_dur, total_dur / len(images))
                durations = [per_image] * len(images)
            else:
                ratio = total_dur / sum(durations) if sum(durations) > 0 else 1.0
                durations = [d * ratio for d in durations]

    while len(durations) < len(images):
        durations.append(6.0)

    temp_dir = output_path.parent / ".runner_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        x264_thr_cfg = int(os.environ.get("FFMPEG_X264_THREADS", "1"))
    except ValueError:
        x264_thr_cfg = 1

    cpu_cap = _cpu_parallel_cap()
    seg_par_raw = os.environ.get("FFMPEG_SEGMENT_PARALLEL", "").strip()
    if not seg_par_raw or seg_par_raw == "0":
        requested = min(len(images), cpu_cap, 8)
    else:
        try:
            requested = int(seg_par_raw)
        except ValueError:
            requested = cpu_cap
        requested = max(1, min(len(images), requested))

    thr_per_job = max(1, x264_thr_cfg) if x264_thr_cfg > 0 else 1
    seg_workers = _clamp_parallel_workers(requested, per_job_threads=thr_per_job)
    if len(images) < 3:
        seg_workers = 1

    if seg_workers <= 1:
        if x264_thr_cfg <= 0:
            x264_threads_eff = 0
        else:
            x264_threads_eff = max(1, min(x264_thr_cfg, cpu_cap))
    else:
        x264_threads_eff = thr_per_job

    try:
        frac = float(os.environ.get("CPU_PARALLEL_FRACTION", "0.85"))
    except ValueError:
        frac = 0.85
    _log(
        f"FFmpeg segments: workers={seg_workers}, libx264 threads/segment={x264_threads_eff or 'auto'}, "
        f"CPU budget {cpu_cap}/{os.cpu_count() or '?'} logical cores ({frac * 100:.0f}% fraction)",
    )

    try:
        payloads: list[dict[str, Any]] = []
        for i, img in enumerate(images):
            section = ""
            if enable_subtitles and sections and i < len(sections):
                section = sections[i] or ""
            payloads.append({
                "i": i,
                "img": img,
                "seg": str(temp_dir / f"seg_{i:04d}.mp4"),
                "seg_dur": durations[i],
                "W": W,
                "H": H,
                "FPS": FPS,
                "x264_preset": x264_preset,
                "x264_threads": x264_threads_eff,
                "enable_subtitles": enable_subtitles,
                "section": section,
                "pat_idx": i,
            })

        segment_slots: list[Path | None] = [None] * len(images)
        if seg_workers <= 1:
            for p in payloads:
                idx, path_str, ok = _ffmpeg_one_slideshow_segment(p)
                _log(f"Rendering scene {idx + 1}/{len(images)} ({p['seg_dur']:.1f}s)")
                if not ok:
                    _log(f"FFmpeg segment {idx + 1} failed")
                    return None
                segment_slots[idx] = Path(path_str)
        else:
            _log(
                f"Rendering {len(images)} scenes with parallel FFmpeg "
                f"({seg_workers} workers)",
            )
            with ProcessPoolExecutor(max_workers=seg_workers) as pool:
                futures = [
                    pool.submit(_ffmpeg_one_slideshow_segment, p) for p in payloads
                ]
                for fut in as_completed(futures):
                    idx, path_str, ok = fut.result()
                    if not ok:
                        _log(f"FFmpeg segment {idx + 1} failed")
                        return None
                    segment_slots[idx] = Path(path_str)

        segments = [segment_slots[i] for i in range(len(images)) if segment_slots[i]]

        if len(segments) != len(images):
            _log("FFmpeg segment count mismatch")
            return None

        if len(segments) > 1:
            _log("Applying crossfade transitions")
            prev = segments[0]
            cumulative_offset = durations[0] - FADE_DUR
            for i in range(1, len(segments)):
                xfade_out = temp_dir / f"xfade_{i:04d}.mp4"

                transition = "fade"
                if scene_plan and i < len(scene_plan):
                    t = scene_plan[i].get("transition", "fade")
                    if t in ("dissolve", "fade", "slideright", "slideleft", "wiperight", "wipeleft"):
                        transition = t

                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(prev), "-i", str(segments[i]),
                    "-filter_complex",
                    f"xfade=transition={transition}:duration={FADE_DUR}:offset={cumulative_offset:.2f},format=yuv420p",
                    "-c:v", "libx264", "-crf", "20", "-preset", x264_preset,
                    str(xfade_out),
                ]
                xfade_timeout = max(300, len(segments) * 30)
                try:
                    subprocess.run(cmd, capture_output=True, text=True, timeout=xfade_timeout, check=True)
                    prev = xfade_out
                    cumulative_offset += durations[i] - FADE_DUR
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
            mux_timeout = max(120, len(images) * 15)
            subprocess.run(cmd, capture_output=True, text=True, timeout=mux_timeout, check=True)
        else:
            import shutil
            shutil.copy2(str(video_track), str(output_path))

        size_mb = output_path.stat().st_size / (1024 * 1024)
        dur = _probe_duration(str(output_path))
        _log(f"Video composed: {output_path} ({size_mb:.1f} MB, {dur:.1f}s)")
        return str(output_path)

    except subprocess.CalledProcessError as e:
        _log(f"FFmpeg composition failed: {e.stderr[:500] if e.stderr else e}")
        return None
    finally:
        import shutil as _shutil
        _shutil.rmtree(str(temp_dir), ignore_errors=True)


# ---------------------------------------------------------------------------
# Post-creation verification (Gemini)
# ---------------------------------------------------------------------------

def _verify_output(
    video_path: str,
    script: str,
    scene_plan: list[dict[str, Any]],
    audio_path: str | None,
) -> dict[str, Any]:
    """Basic verification of the output video."""
    result: dict[str, Any] = {"passed": True, "checks": []}

    video_dur = _probe_duration(video_path)
    result["video_duration"] = video_dur

    if video_dur < 5:
        result["passed"] = False
        result["checks"].append("FAIL: Video too short (<5s)")
    else:
        result["checks"].append(f"OK: Video duration {video_dur:.1f}s")

    if audio_path and Path(audio_path).exists():
        audio_dur = _probe_duration(audio_path)
        result["audio_duration"] = audio_dur
        diff = abs(video_dur - audio_dur)
        if diff > audio_dur * 0.2 and diff > 5:
            result["checks"].append(f"WARN: Video/audio duration mismatch (video={video_dur:.1f}s, audio={audio_dur:.1f}s)")
        else:
            result["checks"].append(f"OK: Audio/video sync within tolerance")

    file_size = Path(video_path).stat().st_size / (1024 * 1024)
    result["file_size_mb"] = round(file_size, 1)
    if file_size < 0.1:
        result["passed"] = False
        result["checks"].append("FAIL: Video file too small")
    else:
        result["checks"].append(f"OK: File size {file_size:.1f} MB")

    result["checks"].append(f"OK: {len(scene_plan)} scenes planned")

    for check in result["checks"]:
        _log(f"  Verify: {check}")

    return result


def _cleanup_intermediate_assets(project_dir: Path) -> None:
    """Remove heavy intermediates after final.mp4 succeeds (keeps prompts, final render)."""
    img_dir = project_dir / "assets" / "images"
    if img_dir.is_dir():
        for p in img_dir.iterdir():
            try:
                if p.is_file():
                    p.unlink()
            except OSError as e:
                _log(f"Cleanup skip {p}: {e}")
    assets = project_dir / "assets"
    if assets.is_dir():
        for p in assets.glob("tts_chunk_*.mp3"):
            try:
                p.unlink()
            except OSError:
                pass
        for name in ("tts_concat.txt",):
            tp = assets / name
            if tp.is_file():
                try:
                    tp.unlink()
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Main pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(prompt_file: str) -> None:
    _reset_api_usage()
    pipeline_start = time.time()

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
    enable_subtitles = payload.get("enableSubtitles", False)
    enable_music = payload.get("enableMusic", False)

    _log(f"Pipeline: {pipeline_name}")
    _log(f"Project:  {project_id}")
    _log(f"Title:    {title}")
    _log(f"Raw prompt: {raw_prompt[:300]}...")
    _log(f"Subtitles: {'enabled' if enable_subtitles else 'disabled'}")
    _log(f"Background music: {'enabled' if enable_music else 'disabled'}")
    if uploaded_audio:
        _log(f"Custom audio: {uploaded_audio}")

    project_dir = Path("projects") / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = project_dir / "assets"
    renders_dir = project_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    done_stages: list[str] = []
    done_stages.append("setup")
    _emit_progress_snapshot(done_stages, "reference", _progress_pct(len(done_stages), 0))

    # --- Phase 0: Reference video analysis ---
    ref_summary = None
    ref_path = step_download_reference(ref_url, project_dir) if ref_url else None
    if ref_path:
        _log(f"Reference available at: {ref_path}")
        analysis = step_analyze_reference(ref_path, project_dir)
        if analysis:
            ref_summary = _summarize_reference(analysis)
            _log(f"Reference summary:\n{ref_summary[:300]}...")

    # --- Phase 0b: Reference style analysis ---
    ref_style = _analyze_reference_style(ref_summary or "")

    done_stages.append("reference")
    _emit_progress_snapshot(done_stages, "transcribe", _progress_pct(len(done_stages), 0))

    # --- Phase 1: Parse production intent ---
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

    # --- Phase 1b: Transcribe uploaded audio with timestamps ---
    has_custom_audio = bool(uploaded_audio and Path(uploaded_audio).exists())
    audio_transcript = None
    sentence_timings: list[dict[str, Any]] = []
    if has_custom_audio:
        audio_dur = _probe_duration(uploaded_audio)
        _log(f"Custom audio duration: {audio_dur:.1f}s")
        target_dur = max(audio_dur, 15.0)

        sentence_timings = _transcribe_with_timestamps(uploaded_audio, language=audio_lang)
        if sentence_timings:
            audio_transcript = " ".join(t.get("text", "") for t in sentence_timings)
            _log(f"Audio transcript with {len(sentence_timings)} timed sentences")
        else:
            audio_transcript = step_transcribe_audio(uploaded_audio, language=audio_lang)
            if audio_transcript:
                _log(f"Audio transcript ({len(audio_transcript)} chars, no timestamps)")
            else:
                _log("WARNING: Could not transcribe audio — images may not match audio content")

    done_stages.append("transcribe")
    _emit_progress_snapshot(done_stages, "english", _progress_pct(len(done_stages), 0))

    # --- Phase 2: Script generation ---
    if has_custom_audio:
        if audio_transcript:
            script = audio_transcript
        else:
            _log("WARNING: No transcript — using title-based script")
            script = title if title and title != project_id else "Visual presentation"
    else:
        script = step_generate_script(
            content_prompt, title, ref_summary,
            target_duration=target_dur,
            language=audio_lang,
            reference_driven=reference_driven,
        )
    _log(f"Script ({len(script)} chars):\n{script[:300]}...")

    max_scenes_budget = max(6, min(80, int(os.environ.get("PIPELINE_MAX_SCENES", "48"))))
    merged_timings: list[dict[str, Any]] = []
    timeline_mode = bool(has_custom_audio and sentence_timings)
    if timeline_mode:
        merged_timings = _merge_timings_for_budget(sentence_timings, max_scenes=max_scenes_budget)
        _log(f"Timeline segments after budget merge: {len(merged_timings)} (cap {max_scenes_budget})")
        merged_timings = _add_english_to_timings(merged_timings, audio_lang)

    done_stages.append("english")
    _emit_progress_snapshot(done_stages, "characters", _progress_pct(len(done_stages), 0))

    script_for_visuals = (
        " ".join((t.get("text_en") or t.get("text", "")) for t in merged_timings).strip()
        if merged_timings
        else script
    )
    if not script_for_visuals:
        script_for_visuals = script

    # --- Phase 2b: Character & scene extraction ---
    character_data = _extract_characters_and_scenes(script_for_visuals, title, ref_style)

    done_stages.append("characters")
    _emit_progress_snapshot(done_stages, "scenes", _progress_pct(len(done_stages), 0))

    # --- Phase 3: Scene plan with character consistency ---
    if merged_timings:
        scene_plan = step_generate_scene_plan_timeline(
            merged_timings, character_data, ref_style, title,
        )
    else:
        # Scale scene count based on duration: ~1 scene per 8s for short, per 25s for long videos
        if target_dur <= 120:
            scene_count = max(2, min(15, int(target_dur / 6)))
        elif target_dur <= 600:
            scene_count = max(10, min(40, int(target_dur / 12)))
        else:
            scene_count = max(30, min(80, int(target_dur / 20)))
        scene_plan = step_generate_scene_plan(
            script, content_prompt, title, ref_summary,
            ref_style=ref_style,
            character_data=character_data,
            scene_count=scene_count,
            sentence_timings=sentence_timings if sentence_timings else None,
        )
    done_stages.append("scenes")
    _emit_progress_snapshot(done_stages, "subtitles", _progress_pct(len(done_stages), 0))

    _log(f"Scene plan: {len(scene_plan)} scenes")
    for i, s in enumerate(scene_plan):
        _log(f"  Scene {i+1}: mood={s.get('mood', '?')} dur={s.get('duration', '?')}s trans={s.get('transition', '?')}")

    narration_sections = [s.get("narration", "") for s in scene_plan]

    # Subtitles
    if not enable_subtitles:
        _log("Subtitles disabled by user")
        subtitle_sections = [""] * len(narration_sections)
    elif has_custom_audio and not audio_transcript:
        _log("Skipping subtitles — no transcript available for uploaded audio")
        subtitle_sections = [""] * len(narration_sections)
    elif subtitle_lang and subtitle_lang != audio_lang:
        subtitle_sections = step_translate_subtitles(
            narration_sections, audio_lang, subtitle_lang,
        )
    else:
        subtitle_sections = narration_sections

    done_stages.append("subtitles")
    _emit_progress_snapshot(done_stages, "tts", _progress_pct(len(done_stages), 0))

    # --- Phase 4: TTS (skipped if custom audio) ---
    if uploaded_audio and Path(uploaded_audio).exists():
        _log(f"Using uploaded audio: {uploaded_audio}")
        audio_path: str | None = uploaded_audio
    else:
        audio_path = step_tts(script, assets_dir / "narration.mp3", language=audio_lang)

    done_stages.append("tts")
    _emit_progress_snapshot(done_stages, "images", _progress_pct(len(done_stages), 0))

    def _on_img_progress(d: int, t: int) -> None:
        frac = (d / t) if t else 1.0
        _emit_progress_snapshot(
            list(done_stages),
            "images",
            _progress_pct(len(done_stages), min(0.95, 0.05 + 0.9 * frac)),
        )

    # --- Phase 5: Character-consistent images ---
    images = step_fetch_images(
        scene_plan,
        assets_dir / "images",
        character_data=character_data,
        ref_style=ref_style,
        on_image_progress=_on_img_progress,
    )

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

    done_stages.append("images")
    _emit_progress_snapshot(done_stages, "compose", _progress_pct(len(done_stages), 0))

    # --- Phase 6: Compose final video ---
    final_path = renders_dir / "final.mp4"

    if os.environ.get("SLIDESHOW_MIN_SCENE_SEC"):
        min_scene_sec = float(os.environ["SLIDESHOW_MIN_SCENE_SEC"])
    else:
        min_scene_sec = 0.35 if merged_timings else 3.0
    ff_preset = os.environ.get("FFMPEG_SEGMENT_PRESET", "fast")

    video_path = step_compose_slideshow(
        images, audio_path, final_path,
        scene_plan=scene_plan,
        sections=subtitle_sections,
        enable_subtitles=enable_subtitles,
        min_scene_duration=min_scene_sec,
        x264_preset=ff_preset,
    )

    done_stages.append("compose")
    _emit_progress_snapshot(done_stages, "verify", _progress_pct(len(done_stages), 0))

    # --- Phase 7: Post-creation verification ---
    if video_path and Path(video_path).exists():
        _log("Running post-creation verification...")
        verification = _verify_output(video_path, script, scene_plan, audio_path)
        if verification["passed"]:
            _log("Verification PASSED")
        else:
            _log("Verification FAILED — but video was still created")

        done_stages.append("verify")
        _emit_progress_snapshot(done_stages, None, 100)
        _cleanup_intermediate_assets(project_dir)

        elapsed = time.time() - pipeline_start
        usage = get_api_usage()
        _log(f"Pipeline completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
        _log(f"API_USAGE={json.dumps(usage)}")
        _log("Pipeline completed successfully")
        print(f"OUTPUT_VIDEO={video_path}")
        sys.exit(0)
    else:
        elapsed = time.time() - pipeline_start
        usage = get_api_usage()
        _log(f"Pipeline failed after {elapsed:.0f}s")
        _log(f"API_USAGE={json.dumps(usage)}")
        _log("Pipeline failed — no output video produced")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Google AI video pipeline")
    parser.add_argument("--pipeline", required=True, help="Pipeline name")
    parser.add_argument("--project", required=True, help="Project ID")
    parser.add_argument("--prompt-file", required=True, help="Path to prompt JSON")
    args = parser.parse_args()
    run_pipeline(args.prompt_file)


if __name__ == "__main__":
    main()
