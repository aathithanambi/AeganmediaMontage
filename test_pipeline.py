"""Pipeline validation test — checks each Google API step independently.

Run:  python test_pipeline.py
Requires: GOOGLE_API_KEY in environment or .env file
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import requests

# Load .env if present
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
BASE = "https://generativelanguage.googleapis.com/v1beta/models"

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

results: list[tuple[str, str, str]] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    status = PASS if ok else FAIL
    results.append((name, "PASS" if ok else "FAIL", detail))
    print(f"  [{status}] {name}{f' — {detail}' if detail else ''}")


def test_api_key() -> bool:
    print("\n1. API Key Check")
    if not API_KEY:
        report("GOOGLE_API_KEY set", False, "not found in env")
        return False
    masked = API_KEY[:8] + "..." + API_KEY[-4:]
    report("GOOGLE_API_KEY set", True, masked)
    return True


def test_gemini_text() -> bool:
    print("\n2. Gemini Text Generation")
    models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"]
    for model in models:
        url = f"{BASE}/{model}:generateContent?key={API_KEY}"
        body = {
            "contents": [{"parts": [{"text": "Say hello in one sentence."}]}],
            "generationConfig": {"maxOutputTokens": 100, "temperature": 0.5},
        }
        try:
            resp = requests.post(url, json=body, timeout=30)
            if resp.status_code == 200:
                text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                report(f"Gemini {model}", True, f'"{text[:60]}..."')
                return True
            elif resp.status_code == 404:
                report(f"Gemini {model}", False, "404 — model not available")
            elif resp.status_code == 429:
                report(f"Gemini {model}", False, "429 — rate limited")
            else:
                report(f"Gemini {model}", False, f"{resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            report(f"Gemini {model}", False, str(e)[:100])
    return False


def test_gemini_json() -> bool:
    print("\n3. Gemini JSON Parsing (scene plan simulation)")
    url = f"{BASE}/gemini-2.5-flash:generateContent?key={API_KEY}"
    body = {
        "contents": [{"parts": [{"text":
            'Return a JSON array with 2 objects. Each has "narration", "mood", "duration". '
            'Topic: a sunset. Respond ONLY with the JSON array.'
        }]}],
        "generationConfig": {"maxOutputTokens": 500, "temperature": 0.3},
    }
    try:
        resp = requests.post(url, json=body, timeout=30)
        if resp.status_code != 200:
            report("Gemini JSON output", False, f"HTTP {resp.status_code}")
            return False
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        cleaned = text.strip()
        if cleaned.startswith("```"):
            import re
            cleaned = re.sub(r'^```\w*\n?', '', cleaned)
            cleaned = re.sub(r'\n?```$', '', cleaned)
        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and len(parsed) >= 2:
            report("Gemini JSON output", True, f"{len(parsed)} scenes parsed")
            return True
        report("Gemini JSON output", False, f"unexpected structure: {type(parsed)}")
    except json.JSONDecodeError as e:
        report("Gemini JSON output", False, f"JSON parse error: {e}")
    except Exception as e:
        report("Gemini JSON output", False, str(e)[:100])
    return False


def test_imagen() -> bool:
    print("\n4. Imagen Image Generation")
    models = ["imagen-4.0-fast-generate-001", "imagen-4.0-generate-001"]
    for model in models:
        url = f"{BASE}/{model}:predict?key={API_KEY}"
        body = {
            "instances": [{"prompt": "A beautiful sunrise over mountains, landscape photography"}],
            "parameters": {"aspectRatio": "16:9", "sampleCount": 1},
        }
        try:
            resp = requests.post(
                url, json=body,
                headers={"Content-Type": "application/json", "x-goog-api-key": API_KEY},
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                predictions = data.get("predictions", [])
                if predictions:
                    img_b64 = predictions[0].get("bytesBase64Encoded", "")
                    if img_b64:
                        img_bytes = base64.b64decode(img_b64)
                        size_kb = len(img_bytes) / 1024
                        report(f"Imagen {model}", True, f"image generated ({size_kb:.0f} KB)")

                        out = Path("test_imagen_output.png")
                        out.write_bytes(img_bytes)
                        report("Image saved", True, str(out))
                        return True
                    report(f"Imagen {model}", False, "empty image data")
                else:
                    report(f"Imagen {model}", False, "no predictions returned")
            elif resp.status_code == 404:
                report(f"Imagen {model}", False, "404 — model not available")
            else:
                report(f"Imagen {model}", False, f"{resp.status_code}: {resp.text[:150]}")
        except Exception as e:
            report(f"Imagen {model}", False, str(e)[:100])
    return False


def test_tts() -> bool:
    print("\n5. Google Cloud Text-to-Speech")
    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={API_KEY}"
    body = {
        "input": {"text": "Hello, this is a test of the text to speech API."},
        "voice": {"languageCode": "en-US", "name": "en-US-Studio-Q"},
        "audioConfig": {"audioEncoding": "MP3"},
    }
    try:
        resp = requests.post(url, json=body, headers={"Content-Type": "application/json"}, timeout=30)
        if resp.status_code == 200:
            audio_b64 = resp.json().get("audioContent", "")
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
                size_kb = len(audio_bytes) / 1024
                report("Google Cloud TTS", True, f"audio generated ({size_kb:.0f} KB)")

                out = Path("test_tts_output.mp3")
                out.write_bytes(audio_bytes)
                report("Audio saved", True, str(out))
                return True
            report("Google Cloud TTS", False, "empty audio")
        elif resp.status_code == 403:
            report("Google Cloud TTS", False,
                   "403 Forbidden — enable 'Cloud Text-to-Speech API' at "
                   "https://console.cloud.google.com/apis/library/texttospeech.googleapis.com")
        else:
            report("Google Cloud TTS", False, f"{resp.status_code}: {resp.text[:150]}")
    except Exception as e:
        report("Google Cloud TTS", False, str(e)[:100])
    return False


def test_ffmpeg() -> bool:
    print("\n6. FFmpeg Availability")
    import subprocess
    try:
        proc = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10)
        if proc.returncode == 0:
            version_line = proc.stdout.splitlines()[0] if proc.stdout else "unknown"
            report("FFmpeg installed", True, version_line[:60])
            return True
        report("FFmpeg installed", False, "non-zero exit code")
    except FileNotFoundError:
        report("FFmpeg installed", False, "ffmpeg not found in PATH")
    except Exception as e:
        report("FFmpeg installed", False, str(e)[:100])
    return False


def test_ytdlp() -> bool:
    print("\n7. yt-dlp Availability")
    import subprocess
    try:
        proc = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True, timeout=10)
        if proc.returncode == 0:
            report("yt-dlp installed", True, f"v{proc.stdout.strip()}")
            return True
        report("yt-dlp installed", False, "non-zero exit code")
    except FileNotFoundError:
        report("yt-dlp installed", False, "yt-dlp not found in PATH")
    except Exception as e:
        report("yt-dlp installed", False, str(e)[:100])
    return False


def main() -> None:
    print("=" * 60)
    print("AeganMedia Montage — Pipeline Validation Test")
    print("=" * 60)

    if not test_api_key():
        print("\nCANNOT CONTINUE: Set GOOGLE_API_KEY in your .env file or environment.")
        sys.exit(1)

    gemini_ok = test_gemini_text()
    json_ok = test_gemini_json()
    imagen_ok = test_imagen()
    tts_ok = test_tts()
    ffmpeg_ok = test_ffmpeg()
    ytdlp_ok = test_ytdlp()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = len(results)
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"  {passed}/{total} checks passed, {failed} failed\n")

    critical_ok = gemini_ok and imagen_ok
    if critical_ok:
        print("  CORE PIPELINE: Ready (Gemini + Imagen working)")
    else:
        print("  CORE PIPELINE: NOT READY")
        if not gemini_ok:
            print("    - Gemini API not working — ALL AI features broken")
            print("      Check your API key at https://aistudio.google.com/apikey")
        if not imagen_ok:
            print("    - Imagen not working — images won't generate (will use placeholders)")
            print("      Ensure Generative Language API is enabled")

    if not tts_ok:
        print("  TTS: Not available — enable Cloud Text-to-Speech API in Google Cloud Console")
    if not ffmpeg_ok:
        print("  FFmpeg: Not installed — video composition will fail")
    if not ytdlp_ok:
        print("  yt-dlp: Not installed — reference video download will fail")

    print()
    # Clean up test files
    for f in ["test_imagen_output.png", "test_tts_output.mp3"]:
        p = Path(f)
        if p.exists():
            print(f"  Test output: {p.resolve()}")

    sys.exit(0 if critical_ok else 1)


if __name__ == "__main__":
    main()
