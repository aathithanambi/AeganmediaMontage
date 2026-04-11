"""AI-powered pipeline runner for web-submitted video production jobs.

Orchestrates OpenMontage tools with LLM intelligence:
  1. (optional) Download + analyze reference video
  2. Generate professional narration script via LLM
  3. Create scene-by-scene plan with specific image prompts
  4. Generate/fetch scene-matched images
  5. Generate narration audio (TTS)
  6. Fetch mood-matched background music
  7. Compose final video with Ken Burns motion, crossfades, subtitles

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


def _gemini_generate(prompt: str, max_tokens: int = 4096) -> str | None:
    """Call Google Gemini and return the text response."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    url = GEMINI_ENDPOINT.format(model="gemini-2.0-flash") + f"?key={api_key}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.7,
        },
    }

    try:
        resp = requests.post(url, json=body, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        _log(f"Gemini API error: {e}")
        return None


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _extract_keywords(text: str, max_words: int = 4) -> list[str]:
    """Pull meaningful keywords from prompt text."""
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
        "minute", "seconds", "minutes",
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
) -> str:
    """Generate a professional narration script using LLM or template fallback."""
    if _gemini_available():
        ref_context = ""
        if ref_summary:
            ref_context = (
                f"\n\nReference video analysis (match this style and content approach):"
                f"\n{ref_summary}\n"
            )

        llm_prompt = f"""You are a professional video scriptwriter. Write a narration script for a {target_duration}-second video.

Topic/Prompt: {prompt}
Title: {title}
{ref_context}
Requirements:
- Write ONLY the narration text that will be spoken aloud
- Target approximately {target_duration} seconds of speech (~{target_duration * 2} words)
- Start with a compelling hook (first sentence should grab attention)
- Use clear, conversational language suitable for text-to-speech
- Include specific facts, examples, or details relevant to the topic
- End with a strong closing statement
- Do NOT include stage directions, timestamps, or [brackets]
- Do NOT include "Welcome to" or generic filler phrases
- Make every sentence informative and relevant to: {prompt}

Write the script now:"""

        _log("Generating script via Gemini...")
        result = _gemini_generate(llm_prompt, max_tokens=2048)
        if result and len(result.strip()) > 50:
            script = result.strip()
            script = re.sub(r'\[.*?\]', '', script)
            script = re.sub(r'\(.*?\)', '', script)
            script = re.sub(r'\n{3,}', '\n\n', script)
            _log(f"LLM script generated: {len(script)} chars, ~{len(script.split())} words")
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
        keywords = ["technology", "innovation", "digital"]

    scenes: list[dict[str, str]] = []
    for i, section in enumerate(sections):
        section_kw = _extract_keywords(section, max_words=3)
        query = " ".join(section_kw[:3]) if section_kw else keywords[i % len(keywords)]
        scenes.append({
            "narration": section,
            "image_prompt": (
                f"Professional cinematic photograph, 16:9, dramatic lighting, "
                f"shallow depth of field. Subject: {query}. "
                f"Context: {section[:100]}. Photo-realistic, high detail."
            ),
            "search_query": query,
            "mood": "professional",
        })
    return scenes


# ---------------------------------------------------------------------------
# Step 3: TTS
# ---------------------------------------------------------------------------

def step_tts(text: str, output_path: Path) -> str | None:
    """Generate narration audio. Prefers high-quality cloud voices."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tts_tools = ["google_tts", "elevenlabs_tts", "openai_tts", "piper_tts"]
    for name in tts_tools:
        tool = _get_tool(name)
        if tool is None:
            continue
        _log(f"Generating narration with {name}")
        params: dict[str, Any] = {"text": text, "output_path": str(output_path)}
        if name == "google_tts":
            params["voice"] = "en-US-Studio-Q"
            params["speaking_rate"] = 0.95
        elif name == "piper_tts":
            params["model"] = "en_US-lessac-medium"
        result = tool.execute(params)
        if result.success and result.artifacts:
            _log(f"Narration generated: {result.artifacts[0]}")
            return result.artifacts[0]
        _log(f"{name} failed: {result.error}")
    _log("No TTS tool available — narration skipped")
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

    music_tools = ["pixabay_music", "freesound_music", "music_gen", "suno_music"]
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
) -> str | None:
    """Compose a high-quality slideshow: Ken Burns motion, crossfades, subtitles."""
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

            vf_parts = [
                f"scale=-1:{H * 2}",
                f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}'"
                f":d={dur_frames}:s={W}x{H}:fps={FPS}",
                "format=yuv420p",
            ]

            if sections and i < len(sections):
                sub_text = _escape_drawtext(sections[i][:120])
                vf_parts.append(
                    f"drawtext=text='{sub_text}'"
                    f":fontsize=36:fontcolor=white"
                    f":borderw=3:bordercolor=black"
                    f":x=(w-tw)/2:y=h-80"
                    f":enable='between(t,0.5,{seconds_per_image - 0.5})'"
                )

            vf = ",".join(vf_parts)

            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", img,
                "-t", str(seconds_per_image),
                "-vf", vf,
                "-c:v", "libx264", "-crf", "18", "-preset", "slow",
                "-r", str(FPS), "-pix_fmt", "yuv420p",
                seg,
            ]
            _log(f"Rendering scene {i+1}/{len(images)} (Ken Burns + subtitle)")
            subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=True)
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
                    "-c:v", "libx264", "-crf", "18", "-preset", "slow",
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

        if has_narration and has_music:
            _log("Mixing narration + background music")
            mixed_audio = temp_dir / "mixed_audio.aac"
            subprocess.run([
                "ffmpeg", "-y",
                "-i", audio_path, "-i", music_path,
                "-filter_complex",
                "[0:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,volume=1.0[voice];"
                "[1:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,volume=0.15[bgm];"
                "[voice][bgm]amix=inputs=2:duration=first:dropout_transition=3[out]",
                "-map", "[out]", "-c:a", "aac", "-b:a", "192k",
                str(mixed_audio),
            ], capture_output=True, text=True, timeout=120, check=True)
            audio_final = str(mixed_audio)
        elif has_narration:
            audio_final = audio_path
        elif has_music:
            audio_final = music_path
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
        if sections and i < len(sections):
            scene["subtitle"] = sections[i]
        composition_data["scenes"].append(scene)

    if audio_path:
        composition_data["audioPath"] = str(Path(audio_path).resolve())
    if music_path and Path(music_path).exists():
        composition_data["musicPath"] = str(Path(music_path).resolve())

    props_file = output_path.parent / "remotion_props.json"
    props_file.write_text(json.dumps(composition_data, indent=2), encoding="utf-8")

    total_frames = sum(s.get("durationInFrames", 180) for s in composition_data["scenes"])

    cmd = [
        "npx", "remotion", "render",
        "src/index.tsx", "Explainer",
        str(output_path),
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
    prompt = payload.get("prompt", "")
    ref_url = payload.get("referenceUrl", "")

    _log(f"Pipeline: {pipeline_name}")
    _log(f"Project:  {project_id}")
    _log(f"Title:    {title}")
    _log(f"Prompt:   {prompt[:200]}...")

    project_dir = Path("projects") / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = project_dir / "assets"
    renders_dir = project_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Reference analysis ---
    ref_summary = None
    ref_path = step_download_reference(ref_url, project_dir) if ref_url else None
    if ref_path:
        _log(f"Reference available at: {ref_path}")
        analysis = step_analyze_reference(ref_path, project_dir)
        if analysis:
            ref_summary = _summarize_reference(analysis)
            _log(f"Reference summary:\n{ref_summary[:300]}...")

    # --- Phase 2: Script generation (LLM-powered) ---
    script = step_generate_script(prompt, title, ref_summary, target_duration=60)
    _log(f"Script ({len(script.split())} words):\n{script[:300]}...")

    # --- Phase 3: Scene plan (LLM-powered) ---
    scene_plan = step_generate_scene_plan(
        script, prompt, title, ref_summary, scene_count=6,
    )
    _log(f"Scene plan: {len(scene_plan)} scenes")
    for i, s in enumerate(scene_plan):
        _log(f"  Scene {i+1}: query='{s.get('search_query', '?')}' "
             f"mood={s.get('mood', '?')}")

    narration_sections = [s.get("narration", "") for s in scene_plan]
    keywords = _extract_keywords(prompt)
    if not keywords:
        keywords = _extract_keywords(script)
    if not keywords:
        keywords = ["technology", "innovation", "digital"]

    # --- Phase 4: TTS ---
    audio_path = step_tts(script, assets_dir / "narration.wav")

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
    target_dur = max(narration_dur, len(images) * 6.0, 30.0)
    music_path = step_fetch_music(
        scene_plan, keywords, assets_dir / "music",
        target_duration=target_dur,
    )

    # --- Phase 7: Compose final video ---
    final_path = renders_dir / "final.mp4"

    video_path = step_compose_remotion(
        images, audio_path, final_path, title, narration_sections, music_path,
    )

    if not video_path:
        video_path = step_compose_slideshow(
            images, audio_path, final_path,
            sections=narration_sections, music_path=music_path,
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
