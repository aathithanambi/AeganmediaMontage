"""STEP 1 — Audio to Transcript with Timestamps, TTS, and Translation.

Responsible for:
  - Transcribing audio (with and without timestamps) via Gemini multimodal
  - Merging timed segments to fit scene budget
  - Adding English translations per segment
  - Subtitle translation
  - Text-to-Speech generation (Google Cloud TTS) with chunk splitting
"""
from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import requests

from webapp.workers.shared import (
    SUPPORTED_LANGUAGES,
    TTS_CHAR_LIMIT,
    _gemini_generate,
    _google_api_key,
    _google_available,
    _google_tts,
    _log,
    _parse_json_response,
    _track_api,
)

# ---------------------------------------------------------------------------
# Transcription (Gemini multimodal)
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
                        {"text": f"Transcribe this audio to text. The audio is in {language}. "
                                 f"Return ONLY the transcribed text, nothing else."},
                    ]
                }],
                "generationConfig": {"maxOutputTokens": 4096, "temperature": 0.1},
            }
            resp = requests.post(gemini_url, json=body, timeout=300)
            if resp.status_code == 404:
                _log(f"Transcription model {mm_model} not found, trying next")
                continue
            break
        if resp and resp.status_code == 200:
            data = resp.json()
            _track_api("gemini", cost_estimate=0.001)
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            if text and len(text.strip()) > 10:
                _log(f"Transcription complete: {len(text)} chars")
                return text.strip()
            _log("Transcription returned insufficient text")
        elif resp:
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
            resp = requests.post(gemini_url, json=body, timeout=300)
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
# Timing merge / English translation
# ---------------------------------------------------------------------------

def _merge_timings_for_budget(
    timings: list[dict[str, Any]],
    max_scenes: int = 48,
) -> list[dict[str, Any]]:
    """Merge consecutive timed segments so we have at most max_scenes."""
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
    """Add text_en per segment for image/scene planning."""
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

def _split_tts_text(text: str, limit: int = TTS_CHAR_LIMIT) -> list[str]:
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
        shutil.copy2(chunk_paths[0], str(output_path))
        return str(output_path)
