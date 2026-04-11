"""Simplified pipeline runner for web-submitted jobs.

Orchestrates OpenMontage tools in a linear sequence:
  1. (optional) Download reference video
  2. Generate narration via TTS
  3. Fetch stock images for scene visuals
  4. Compose final video (Remotion or FFmpeg)

This is the "free tier" runner — works with only local tools + free
stock API keys.  For full agent-driven pipelines (research, proposals,
multi-stage review), configure PIPELINE_RUN_COMMAND to point to your
AI agent endpoint instead.
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

PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))
os.chdir(str(PROJ_ROOT))

from tools.tool_registry import registry
from tools.base_tool import ToolStatus


def _log(msg: str) -> None:
    print(f"[pipeline-runner] {msg}", flush=True)


def _discover_tools() -> None:
    registry.discover("tools")


def _get_tool(name: str) -> Any:
    tool = registry.get(name)
    if tool is None:
        return None
    if tool.get_status() != ToolStatus.AVAILABLE:
        return None
    return tool


def _extract_keywords(text: str, max_words: int = 4) -> list[str]:
    """Pull meaningful keywords from prompt text for image search."""
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
        "your", "create", "video", "make", "please", "need", "want",
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


def _split_script_sections(text: str, target_count: int = 4) -> list[str]:
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


def _generate_narration_text(prompt: str, title: str | None) -> str:
    """Create narration from the user prompt."""
    if len(prompt) > 200:
        return prompt
    heading = title or "Video"
    return textwrap.dedent(f"""\
        Welcome to {heading}.
        {prompt}
        Thank you for watching. This video was produced by AeganMedia Montage.
    """).strip()


def step_download_reference(
    ref_url: str, project_dir: Path
) -> str | None:
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


def step_tts(text: str, output_path: Path) -> str | None:
    """Generate narration audio. Returns path or None."""
    tts_tools = ["piper_tts", "google_tts", "elevenlabs_tts", "openai_tts"]
    for name in tts_tools:
        tool = _get_tool(name)
        if tool is None:
            continue
        _log(f"Generating narration with {name}")
        params: dict[str, Any] = {"text": text, "output_path": str(output_path)}
        if name == "piper_tts":
            params["model"] = "en_US-lessac-medium"
        result = tool.execute(params)
        if result.success and result.artifacts:
            _log(f"Narration generated: {result.artifacts[0]}")
            return result.artifacts[0]
        _log(f"{name} failed: {result.error}")
    _log("No TTS tool available — narration skipped")
    return None


def step_fetch_images(
    keywords: list[str],
    count: int,
    output_dir: Path,
) -> list[str]:
    """Fetch stock images for scenes. Returns list of paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_tools = ["pexels_image", "pixabay_image"]
    available_tool = None
    for name in image_tools:
        t = _get_tool(name)
        if t is not None:
            available_tool = t
            break
    if available_tool is None:
        _log("No stock image tool available — images skipped")
        return []

    images: list[str] = []
    kw_groups = keywords[:count] if len(keywords) >= count else keywords + keywords[:count]
    for i in range(count):
        query = kw_groups[i % len(kw_groups)] if kw_groups else "technology"
        out = output_dir / f"scene_{i:02d}.jpg"
        _log(f"Fetching image {i+1}/{count}: '{query}'")
        result = available_tool.execute({
            "query": query,
            "orientation": "landscape",
            "output_path": str(out),
        })
        if result.success and result.artifacts:
            images.append(result.artifacts[0])
        else:
            _log(f"Image fetch failed for '{query}': {result.error}")
    return images


def step_compose_slideshow(
    images: list[str],
    audio_path: str | None,
    output_path: Path,
    seconds_per_image: float = 5.0,
) -> str | None:
    """Compose a slideshow video from images + audio using FFmpeg."""
    if not images:
        _log("No images to compose")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if audio_path and Path(audio_path).exists():
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, timeout=30,
        )
        try:
            total_dur = float(probe.stdout.strip())
            seconds_per_image = max(3.0, total_dur / len(images))
        except (ValueError, ZeroDivisionError):
            pass

    temp_dir = output_path.parent / ".runner_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    segments: list[Path] = []

    try:
        for i, img in enumerate(images):
            seg = temp_dir / f"seg_{i:04d}.mp4"
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", img,
                "-t", str(seconds_per_image),
                "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p",
                "-c:v", "libx264", "-crf", "23", "-preset", "medium",
                "-r", "30",
                seg,
            ]
            subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
            segments.append(seg)

        concat_file = temp_dir / "concat.txt"
        with open(concat_file, "w") as f:
            for seg in segments:
                safe = str(seg.resolve()).replace("\\", "/")
                f.write(f"file '{safe}'\n")

        no_audio_out = temp_dir / "slideshow_silent.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(concat_file), "-c", "copy", str(no_audio_out)],
            capture_output=True, text=True, timeout=120, check=True,
        )

        if audio_path and Path(audio_path).exists():
            cmd = [
                "ffmpeg", "-y",
                "-i", str(no_audio_out),
                "-i", audio_path,
                "-c:v", "copy", "-c:a", "aac",
                "-map", "0:v", "-map", "1:a",
                "-shortest",
                str(output_path),
            ]
            subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
        else:
            import shutil
            shutil.copy2(str(no_audio_out), str(output_path))

        _log(f"Video composed: {output_path}")
        return str(output_path)

    except subprocess.CalledProcessError as e:
        _log(f"FFmpeg composition failed: {e.stderr[:500] if e.stderr else e}")
        return None
    finally:
        for f in segments:
            f.unlink(missing_ok=True)
        for f in temp_dir.iterdir():
            f.unlink(missing_ok=True)
        temp_dir.rmdir()


def step_compose_remotion(
    images: list[str],
    audio_path: str | None,
    output_path: Path,
    title: str | None = None,
    sections: list[str] | None = None,
) -> str | None:
    """Attempt Remotion render for richer animations. Falls back to FFmpeg slideshow."""
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
            "durationInFrames": 150,
        }
        if sections and i < len(sections):
            scene["subtitle"] = sections[i]
        composition_data["scenes"].append(scene)

    if audio_path:
        composition_data["audioPath"] = str(Path(audio_path).resolve())

    props_file = output_path.parent / "remotion_props.json"
    props_file.write_text(json.dumps(composition_data, indent=2), encoding="utf-8")

    total_frames = sum(s.get("durationInFrames", 150) for s in composition_data["scenes"])

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


def run_pipeline(prompt_file: str) -> None:
    """Main pipeline execution."""
    _log(f"Starting pipeline runner (cwd={os.getcwd()})")
    _log(f"PROJ_ROOT={PROJ_ROOT}")
    _log(f"Python={sys.executable} {sys.version_info.major}.{sys.version_info.minor}")

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
    _log(f"Prompt:   {prompt[:120]}...")

    project_dir = Path("projects") / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = project_dir / "assets"
    renders_dir = project_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    ref_path = step_download_reference(ref_url, project_dir) if ref_url else None
    if ref_path:
        _log(f"Reference available at: {ref_path}")

    narration_text = _generate_narration_text(prompt, title)
    sections = _split_script_sections(narration_text)
    keywords = _extract_keywords(prompt)
    if not keywords:
        keywords = _extract_keywords(narration_text)
    if not keywords:
        keywords = ["technology", "innovation", "digital"]

    _log(f"Keywords: {keywords[:6]}")
    _log(f"Sections: {len(sections)}")

    audio_path = step_tts(narration_text, assets_dir / "narration.wav")

    image_count = max(3, min(8, len(sections)))
    images = step_fetch_images(keywords, image_count, assets_dir / "images")

    if not images:
        _log("No images fetched — generating placeholder")
        placeholder = assets_dir / "images" / "placeholder.jpg"
        placeholder.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i",
             f"color=c=0x1a1a2e:s=1920x1080:d=1",
             "-frames:v", "1", str(placeholder)],
            capture_output=True, text=True, timeout=30,
        )
        if placeholder.exists():
            images = [str(placeholder)] * image_count

    final_path = renders_dir / "final.mp4"

    video_path = step_compose_remotion(
        images, audio_path, final_path, title, sections
    )

    if not video_path:
        video_path = step_compose_slideshow(images, audio_path, final_path)

    if video_path and Path(video_path).exists():
        _log("Pipeline completed successfully")
        print(f"OUTPUT_VIDEO={video_path}")
        sys.exit(0)
    else:
        _log("Pipeline failed — no output video produced")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenMontage simplified pipeline runner")
    parser.add_argument("--pipeline", required=True, help="Pipeline name")
    parser.add_argument("--project", required=True, help="Project ID")
    parser.add_argument("--prompt-file", required=True, help="Path to prompt JSON")
    args = parser.parse_args()
    run_pipeline(args.prompt_file)


if __name__ == "__main__":
    main()
