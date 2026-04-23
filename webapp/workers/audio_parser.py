"""STEP 1 — Audio to Transcript with Timestamps, TTS, and Translation.

Responsible for:
  - Transcribing audio via Google Speech-to-Text API (word-level timestamps)
    with Gemini multimodal estimation as fallback
  - Merging timed segments to fit scene budget
  - Adding English translations per segment
  - Timestamp re-alignment after language translation (pace ratio correction)
  - Subtitle translation
  - Text-to-Speech generation (Google Cloud TTS) with chunk splitting
  - Voice cloning from reference video (ElevenLabs IVC) + multilingual TTS

Agent checkpoints:
  transcript_v1  — raw sentence timings from Speech API or Gemini
  translation_v1 — merged timings with English translations added
"""
from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import requests

from webapp.workers.shared import (
    SUPPORTED_LANGUAGES,
    TTS_CHAR_LIMIT,
    _elevenlabs_available,
    _elevenlabs_tts,
    _gemini_generate,
    _google_api_key,
    _google_available,
    _google_tts,
    _log,
    _parse_json_response,
    _probe_duration,
    _track_api,
)

# ---------------------------------------------------------------------------
# Language constants
# ---------------------------------------------------------------------------

# BCP-47 language codes for Google Speech-to-Text API
SPEECH_LANGUAGE_CODES: dict[str, str] = {
    "english": "en-US", "tamil": "ta-IN", "hindi": "hi-IN",
    "telugu": "te-IN", "kannada": "kn-IN", "malayalam": "ml-IN",
    "bengali": "bn-IN", "marathi": "mr-IN", "gujarati": "gu-IN",
    "punjabi": "pa-IN", "urdu": "ur-IN", "arabic": "ar-XA",
    "spanish": "es-ES", "french": "fr-FR", "german": "de-DE",
    "italian": "it-IT", "portuguese": "pt-BR", "japanese": "ja-JP",
    "korean": "ko-KR", "chinese": "cmn-CN", "russian": "ru-RU",
    "dutch": "nl-NL", "polish": "pl-PL", "turkish": "tr-TR",
    "thai": "th-TH", "vietnamese": "vi-VN", "indonesian": "id-ID",
    "malay": "ms-MY", "swedish": "sv-SE", "norwegian": "nb-NO",
    "danish": "da-DK", "finnish": "fi-FI",
}

# Relative speech pace vs English (values > 1.0 = spoken slower than English).
# Used to re-scale image display windows after translating narration into another language.
LANGUAGE_PACE_RATIO: dict[str, float] = {
    "english": 1.00, "tamil": 1.15, "hindi": 1.08, "telugu": 1.12,
    "kannada": 1.10, "malayalam": 1.13, "bengali": 1.05, "marathi": 1.07,
    "gujarati": 1.06, "punjabi": 1.04, "urdu": 1.09, "arabic": 1.10,
    "spanish": 0.95, "french": 0.97, "german": 1.02, "italian": 0.96,
    "portuguese": 0.95, "japanese": 0.90, "korean": 0.92, "chinese": 0.88,
    "russian": 1.05, "dutch": 0.98, "turkish": 1.03, "thai": 1.05,
}

# ---------------------------------------------------------------------------
# Google Speech-to-Text API — true word-level timestamps
# ---------------------------------------------------------------------------

def _convert_to_flac_for_speech(audio_path: str) -> str | None:
    """Convert audio to mono 16 kHz FLAC for optimal Speech API accuracy.
    Returns the temp FLAC path on success, or None if ffmpeg is unavailable."""
    try:
        out = Path(audio_path).with_suffix(".speech_tmp.flac")
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path,
             "-ar", "16000", "-ac", "1", "-c:a", "flac", str(out)],
            capture_output=True, text=True, timeout=180,
        )
        if result.returncode == 0 and out.exists() and out.stat().st_size > 200:
            return str(out)
    except Exception:
        pass
    return None


def _get_speech_encoding(audio_path: str) -> str:
    """Map file extension to a Google Speech API AudioEncoding string."""
    return {
        "flac": "FLAC", "wav": "LINEAR16", "mp3": "MP3",
        "ogg": "OGG_OPUS", "m4a": "MP3", "aac": "MP3",
    }.get(Path(audio_path).suffix.lower().lstrip("."), "MP3")


def _parse_speech_time(t: Any) -> float:
    """Convert '1.200s' or 1.2 (seconds) from Speech API word offset to float."""
    if isinstance(t, (int, float)):
        return float(t)
    return float(str(t).rstrip("s")) if t else 0.0


def _speech_results_to_segments(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Google Speech API result blocks into sentence-level timed segments."""
    segments: list[dict[str, Any]] = []
    for result in results:
        alts = result.get("alternatives", [])
        if not alts:
            continue
        best = alts[0]
        transcript = best.get("transcript", "").strip()
        words = best.get("words", [])
        if not transcript:
            continue
        if words:
            start = _parse_speech_time(words[0].get("startTime", "0s"))
            end   = _parse_speech_time(words[-1].get("endTime", "0s"))
            word_list = [
                {
                    "word": w.get("word", ""),
                    "start": _parse_speech_time(w.get("startTime", "0s")),
                    "end":   _parse_speech_time(w.get("endTime", "0s")),
                }
                for w in words
            ]
        else:
            start, end, word_list = 0.0, 0.0, []
        segments.append({
            "text": transcript,
            "start": start,
            "end": end,
            "duration": max(0.1, end - start),
            "words": word_list,
            "source": "google_speech_api",
        })
    return segments


def _speech_longrunning(
    base_url: str,
    api_key: str,
    body: dict[str, Any],
) -> list[dict[str, Any]]:
    """Submit an async LongRunningRecognize job and poll until complete (≤10 min)."""
    url = f"{base_url}/speech:longrunningrecognize?key={api_key}"
    resp = requests.post(url, json=body, timeout=120)
    if resp.status_code != 200:
        _log(f"LongRunningRecognize submit failed: {resp.status_code} {resp.text[:200]}")
        return []
    op_name = resp.json().get("name", "")
    if not op_name:
        _log("No operation name returned from longrunningrecognize")
        return []
    _log(f"Speech operation started: {op_name} — polling...")
    poll_url = f"{base_url}/operations/{op_name}?key={api_key}"
    for attempt in range(60):
        time.sleep(10)
        try:
            pr = requests.get(poll_url, timeout=30)
        except Exception as e:
            _log(f"Poll request error: {e}")
            continue
        if pr.status_code != 200:
            _log(f"Poll error: {pr.status_code}")
            continue
        op = pr.json()
        if op.get("done"):
            if "error" in op:
                _log(f"Speech operation error: {op['error']}")
                return []
            return op.get("response", {}).get("results", [])
        if attempt % 3 == 0:
            _log(f"Speech still processing... ({(attempt + 1) * 10}s elapsed)")
    _log("Speech operation timed out after 10 minutes")
    return []


def step_google_speech_transcribe(
    audio_path: str,
    language: str = "english",
) -> list[dict[str, Any]]:
    """Transcribe audio using Google Speech-to-Text API with measured word-level timestamps.

    Delivers ±50ms per-word accuracy vs Gemini's ±1-2s estimated sentence timing.
    Automatically selects synchronous (<55s) or async long-running (≥55s) recognition.
    Falls back to Gemini timestamp estimation when the Speech API is unavailable.

    Checkpoint key: transcript_v1
    """
    api_key = _google_api_key()
    if not api_key:
        _log("No GOOGLE_API_KEY — falling back to Gemini timestamp estimation")
        return _transcribe_with_timestamps(audio_path, language)

    lang_code = SPEECH_LANGUAGE_CODES.get(language.lower(), "en-US")
    _log(f"[Speech API] Transcribing with word-level timestamps ({lang_code})...")

    flac_path: str | None = None
    try:
        audio_dur = _probe_duration(audio_path)
        flac_path = _convert_to_flac_for_speech(audio_path)
        use_path = flac_path or audio_path

        with open(use_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")

        encoding = "FLAC" if flac_path else _get_speech_encoding(audio_path)
        config: dict[str, Any] = {
            "encoding": encoding,
            "languageCode": lang_code,
            "enableWordTimeOffsets": True,
            "enableAutomaticPunctuation": True,
            "model": "latest_long",
        }
        if encoding == "LINEAR16":
            config["sampleRateHertz"] = 16000

        body = {"config": config, "audio": {"content": audio_b64}}
        base_url = "https://speech.googleapis.com/v1"

        if audio_dur > 55:
            _log(f"[Speech API] Long audio ({audio_dur:.0f}s) — using async recognition")
            results = _speech_longrunning(base_url, api_key, body)
        else:
            resp = requests.post(
                f"{base_url}/speech:recognize?key={api_key}",
                json=body, timeout=120,
            )
            if resp.status_code != 200:
                _log(
                    f"[Speech API] Error {resp.status_code}: {resp.text[:300]}\n"
                    "Falling back to Gemini timestamp estimation"
                )
                return _transcribe_with_timestamps(audio_path, language)
            results = resp.json().get("results", [])

        if not results:
            _log("[Speech API] No results returned — falling back to Gemini")
            return _transcribe_with_timestamps(audio_path, language)

        segments = _speech_results_to_segments(results)
        _track_api("speech", cost_estimate=max(0.004, audio_dur * 0.00001))
        _log(f"[Speech API] {len(segments)} segments with measured timestamps ✓")
        return segments

    except Exception as e:
        _log(f"[Speech API] Unexpected error: {e} — falling back to Gemini")
        return _transcribe_with_timestamps(audio_path, language)
    finally:
        if flac_path:
            try:
                Path(flac_path).unlink(missing_ok=True)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Transcription (Gemini multimodal — fallback / no-audio-upload path)
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
    """Transcribe audio and ESTIMATE per-sentence timing using Gemini multimodal.

    This is the fallback path. Prefer step_google_speech_transcribe() for
    measured word-level timestamps (±50ms vs ±1-2s estimation accuracy).
    """
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
# Long-segment splitter
# ---------------------------------------------------------------------------

def _split_long_segments(
    timings: list[dict[str, Any]],
    max_segment_dur: float = 6.0,
) -> list[dict[str, Any]]:
    """Split any segment longer than max_segment_dur into equal sub-segments.

    Without this, a long opening sentence (e.g. 12s) produces a single image
    that stays on screen while the audio narrates multiple story beats.
    Each sub-segment inherits the parent's text and English translation so
    that the scene planner can still generate a meaningful image prompt.

    Example: one 12s segment → two 6s sub-segments, each with the same text.
    The scene planner will generate two slightly different images for visual variety.
    """
    if not timings:
        return timings
    result: list[dict[str, Any]] = []
    for t in timings:
        dur = float(t.get("duration", max(0.1, t.get("end", 0) - t.get("start", 0))))
        if dur <= max_segment_dur:
            result.append(t)
            continue
        n_splits = max(2, int(dur / max_segment_dur) + (1 if dur % max_segment_dur > 0.5 else 0))
        sub_dur = dur / n_splits
        start = float(t.get("start", 0))
        for k in range(n_splits):
            sub_start = round(start + k * sub_dur, 3)
            sub_end = round(start + (k + 1) * sub_dur, 3)
            result.append({
                **t,
                "start": sub_start,
                "end": sub_end,
                "duration": round(sub_dur, 3),
            })
        _log(f"[Segment split] {dur:.1f}s segment → {n_splits} × {sub_dur:.1f}s sub-scenes")
    return result


# ---------------------------------------------------------------------------
# Timestamp re-alignment after language translation
# ---------------------------------------------------------------------------

def realign_timestamps_for_target_language(
    timings: list[dict[str, Any]],
    source_lang: str,
    target_lang: str,
) -> list[dict[str, Any]]:
    """Re-scale segment time windows to match the target language's speech pace.

    When translating audio from English to Tamil (for TTS replacement), Tamil
    speech runs ~15% slower. Each image display window must expand proportionally
    so images stay in sync with the generated Tamil narration.

    Use case: uploaded English audio → Tamil TTS → Tamil timestamps for scene sync.
    If source and target pace are within 1%, no re-alignment is applied.

    Args:
        timings:     Sentence segments from source-language transcription.
        source_lang: Language of the original transcription (e.g. "english").
        target_lang: Language of the TTS that will replace the audio (e.g. "tamil").

    Returns:
        New list with re-computed start / end / duration. All text fields preserved.
    """
    if not timings:
        return timings

    src_ratio = LANGUAGE_PACE_RATIO.get(source_lang.lower(), 1.0)
    tgt_ratio = LANGUAGE_PACE_RATIO.get(target_lang.lower(), 1.0)

    if abs(src_ratio - tgt_ratio) < 0.01:
        return timings

    scale = tgt_ratio / src_ratio
    direction = f"+{(scale - 1) * 100:.1f}%" if scale > 1 else f"{(scale - 1) * 100:.1f}%"
    _log(
        f"[Realign] {source_lang}→{target_lang}: scale={scale:.3f} ({direction} duration). "
        f"Tamil is ~15% slower than English."
    )

    realigned: list[dict[str, Any]] = []
    cursor = float(timings[0].get("start", 0.0))

    for t in timings:
        orig_dur = float(t.get("duration", max(0.1, t.get("end", 0) - t.get("start", 0))))
        new_dur = max(0.15, orig_dur * scale)
        realigned.append({
            **t,
            "start": round(cursor, 3),
            "end": round(cursor + new_dur, 3),
            "duration": round(new_dur, 3),
        })
        cursor += new_dur

    total_orig = sum(float(t.get("duration", 0)) for t in timings)
    total_new = sum(t["duration"] for t in realigned)
    _log(f"[Realign] Total duration: {total_orig:.1f}s → {total_new:.1f}s")
    return realigned


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


# ---------------------------------------------------------------------------
# Voice cloning from reference video
# ---------------------------------------------------------------------------

def step_clone_voice_from_reference(
    ref_video_path: str,
    assets_dir: Path,
    clone_name: str = "ref_voice",
) -> "str | None":
    """Extract narration audio from the reference video and create an ElevenLabs
    Instant Voice Clone.

    Process:
      1. ffmpeg: strip music/sfx by extracting the full audio track.
         (A clean 30-90s sample is enough for a good IVC.)
      2. ElevenLabs IVC API: create a temporary cloned voice.
      3. Return the ElevenLabs voice_id string for reuse in TTS generation.

    Returns voice_id string on success, None on any failure.
    The caller should handle None gracefully (fall back to Google TTS).
    """
    if not _elevenlabs_available():
        _log("ElevenLabs key not set — cannot clone voice from reference")
        return None

    try:
        from elevenlabs.client import ElevenLabs  # type: ignore
    except ImportError:
        _log("elevenlabs package not installed — run: pip install elevenlabs>=1.9")
        return None

    # Step 1: Extract audio from reference video (first 120s — enough for IVC)
    ref_audio = assets_dir / "ref_voice_sample.mp3"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", ref_video_path,
                "-t", "120",              # take up to 120 seconds
                "-vn",                    # no video
                "-acodec", "libmp3lame",
                "-q:a", "2",
                "-ar", "22050",
                str(ref_audio),
            ],
            capture_output=True, text=True, timeout=120, check=True,
        )
    except subprocess.CalledProcessError as e:
        _log(f"Voice sample extraction failed: {e.stderr[:200] if e.stderr else e}")
        return None

    if not ref_audio.exists() or ref_audio.stat().st_size < 10_000:
        _log("Reference audio sample too small — skipping voice clone")
        return None

    _log(f"Reference voice sample extracted: {ref_audio} ({ref_audio.stat().st_size // 1024} KB)")

    # Step 2: Create ElevenLabs Instant Voice Clone
    try:
        from webapp.workers.shared import _elevenlabs_api_key  # local import avoids circular
        client = ElevenLabs(api_key=_elevenlabs_api_key())
        with open(ref_audio, "rb") as f:
            voice = client.voices.ivc.create(
                name=clone_name,
                files=[f],
                remove_background_noise=True,
            )
        voice_id = voice.voice_id
        _log(f"Voice cloned successfully: voice_id={voice_id}")
        # Persist voice_id so we can skip re-cloning on checkpoint resume
        (assets_dir / "cloned_voice_id.txt").write_text(voice_id, encoding="utf-8")
        return voice_id
    except Exception as e:
        _log(f"ElevenLabs voice clone failed: {e}")
        return None


def step_tts_elevenlabs(
    text: str,
    output_path: Path,
    language: str,
    voice_id: str,
) -> "str | None":
    """Generate TTS narration using ElevenLabs with a cloned voice.

    Handles long texts by splitting into ≤2500-char chunks, generating each,
    and concatenating them with ffmpeg (same pattern as step_tts).
    Falls back to None so orchestrator can fall back to Google TTS.
    """
    MAX_CHUNK = 2500  # ElevenLabs recommended max per request
    if len(text) <= MAX_CHUNK:
        return _elevenlabs_tts(text, output_path, voice_id=voice_id, language=language)

    # Split into chunks on sentence boundaries
    sentences = re.split(r'(?<=[.!?।])\s+', text)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > MAX_CHUNK and current:
            chunks.append(current.strip())
            current = sent
        else:
            current = f"{current} {sent}" if current else sent
    if current.strip():
        chunks.append(current.strip())

    _log(f"ElevenLabs TTS: {len(text)} chars split into {len(chunks)} chunks")
    chunk_paths: list[str] = []
    for i, chunk in enumerate(chunks):
        chunk_path = output_path.parent / f"el_chunk_{i:03d}.mp3"
        result = _elevenlabs_tts(chunk, chunk_path, voice_id=voice_id, language=language)
        if result:
            chunk_paths.append(result)
        else:
            _log(f"ElevenLabs TTS chunk {i + 1} failed")

    if not chunk_paths:
        return None
    if len(chunk_paths) == 1:
        shutil.copy2(chunk_paths[0], str(output_path))
        return str(output_path)

    concat_file = output_path.parent / "el_concat.txt"
    with open(concat_file, "w") as f:
        for cp in chunk_paths:
            safe = Path(cp).resolve().as_posix()
            f.write(f"file '{safe}'\n")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(concat_file), "-c", "copy", str(output_path)],
            capture_output=True, text=True, timeout=180, check=True,
        )
        _log(f"ElevenLabs TTS chunks concatenated: {output_path}")
        return str(output_path)
    except subprocess.CalledProcessError as e:
        _log(f"ElevenLabs TTS concat failed: {e.stderr[:200] if e.stderr else ''}")
        shutil.copy2(chunk_paths[0], str(output_path))
        return str(output_path)
