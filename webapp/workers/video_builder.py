"""STEP 4+5 — FFmpeg Video Composition, Verification, Asset Packaging, Cleanup.

Responsible for:
  - Composing video from images + audio (Ken Burns, crossfades, subtitles, audio mux)
  - Post-creation verification
  - Zipping scene images for download
  - Copying images to renders/
  - Cleaning up intermediate files

All FFmpeg timeouts scale dynamically to support videos up to 20+ minutes.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

from webapp.workers.shared import (
    _escape_drawtext,
    _log,
    _probe_duration,
)

# ---------------------------------------------------------------------------
# Font path resolution for subtitle rendering
# ---------------------------------------------------------------------------

# Noto Sans font paths installed by fonts-noto-core + fonts-noto-hinted on Debian/Ubuntu.
# Key: language name (lowercase, as used in audio_lang).
# Value: list of candidate font paths tried in order; first existing file wins.
_NOTO_FONT_BASE = "/usr/share/fonts/truetype/noto"
_LANG_FONT_MAP: dict[str, list[str]] = {
    "tamil":     [f"{_NOTO_FONT_BASE}/NotoSansTamil-Regular.ttf",
                  f"{_NOTO_FONT_BASE}/NotoSansTamil[wdth,wght].ttf"],
    "hindi":     [f"{_NOTO_FONT_BASE}/NotoSansDevanagari-Regular.ttf",
                  f"{_NOTO_FONT_BASE}/NotoSansDevanagari[wdth,wght].ttf"],
    "telugu":    [f"{_NOTO_FONT_BASE}/NotoSansTelugu-Regular.ttf",
                  f"{_NOTO_FONT_BASE}/NotoSansTelugu[wdth,wght].ttf"],
    "kannada":   [f"{_NOTO_FONT_BASE}/NotoSansKannada-Regular.ttf",
                  f"{_NOTO_FONT_BASE}/NotoSansKannada[wdth,wght].ttf"],
    "malayalam": [f"{_NOTO_FONT_BASE}/NotoSansMalayalam-Regular.ttf",
                  f"{_NOTO_FONT_BASE}/NotoSansMalayalam[wdth,wght].ttf"],
    "bengali":   [f"{_NOTO_FONT_BASE}/NotoSansBengali-Regular.ttf",
                  f"{_NOTO_FONT_BASE}/NotoSansBengali[wdth,wght].ttf"],
    "gujarati":  [f"{_NOTO_FONT_BASE}/NotoSansGujarati-Regular.ttf",
                  f"{_NOTO_FONT_BASE}/NotoSansGujarati[wdth,wght].ttf"],
    "punjabi":   [f"{_NOTO_FONT_BASE}/NotoSansGurmukhi-Regular.ttf",
                  f"{_NOTO_FONT_BASE}/NotoSansGurmukhi[wdth,wght].ttf"],
    "marathi":   [f"{_NOTO_FONT_BASE}/NotoSansDevanagari-Regular.ttf"],  # same script as Hindi
    "arabic":    [f"{_NOTO_FONT_BASE}/NotoNaskhArabic-Regular.ttf",
                  f"{_NOTO_FONT_BASE}/NotoSansArabic-Regular.ttf"],
    "urdu":      [f"{_NOTO_FONT_BASE}/NotoNaskhArabic-Regular.ttf"],
    "japanese":  [f"{_NOTO_FONT_BASE}/NotoSansCJK-Regular.ttc",
                  f"{_NOTO_FONT_BASE}/NotoSansJP-Regular.ttf"],
    "korean":    [f"{_NOTO_FONT_BASE}/NotoSansCJK-Regular.ttc",
                  f"{_NOTO_FONT_BASE}/NotoSansKR-Regular.ttf"],
}
# Fallback Latin/common font
_NOTO_SANS_FALLBACK = [
    f"{_NOTO_FONT_BASE}/NotoSans-Regular.ttf",
    f"{_NOTO_FONT_BASE}/NotoSans[Condensed,ExtraCondensed]-Regular.ttf",
]


def _get_subtitle_font(language: str) -> str:
    """Return the best available fontfile path for the given language.

    Checks candidates in order and returns the first file that exists on disk.
    Falls back to NotoSans (Latin) if no specific font is found.
    Returns an empty string when no font file exists (drawtext uses system default).
    """
    lang = (language or "").strip().lower()
    candidates = _LANG_FONT_MAP.get(lang, []) + _NOTO_SANS_FALLBACK
    for path in candidates:
        if Path(path).exists():
            return path
    _log(f"No Noto font found for '{lang}' — subtitle will use system default font")
    return ""

# ---------------------------------------------------------------------------
# Video composition (FFmpeg with audio-synced timing)
# ---------------------------------------------------------------------------

def _build_watermark_filter(
    watermark_text: str,
    watermark_image: str | None = None,
) -> str:
    """Build an FFmpeg drawtext filter string for a channel watermark.

    Adds a semi-transparent text watermark in the top-right corner,
    matching the style of Tamil story YouTube channels (e.g. Thagaval Thalam).
    Set WATERMARK_TEXT env var to enable (e.g. "My Channel").
    Set WATERMARK_IMAGE env var to path of a .png logo to overlay instead.
    """
    if watermark_image and Path(watermark_image).exists():
        # Image watermark — caller must add the image as an extra -i input
        return f"overlay=W-w-20:20:format=auto,format=yuv420p"
    # Text watermark — top-right corner, semi-transparent white with shadow
    text = _escape_drawtext(watermark_text)
    return (
        f"drawtext=text='{text}'"
        f":fontsize=28:fontcolor=white@0.75"
        f":borderw=1:bordercolor=black@0.5"
        f":x=w-tw-16:y=16"
    )


def step_compose_slideshow(
    images: list[str],
    audio_path: str | None,
    output_path: Path,
    scene_plan: list[dict[str, Any]] | None = None,
    sections: list[str] | None = None,
    enable_subtitles: bool = False,
    *,
    subtitle_language: str = "english",
    min_scene_duration: float = 3.0,
    x264_preset: str = "fast",
) -> str | None:
    if not images:
        _log("No images to compose")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    W, H, FPS = 1920, 1080, 30
    # Crossfade duration in seconds. Shorter = crisper cuts, less double-exposure blur.
    # 0.15s is the sweet spot: invisible at normal playback speed but still softer than
    # a hard cut. 0.3s and above creates a visible ghost overlay (double-exposure effect)
    # when two similar-background scenes cross-dissolve.
    # Set XFADE_DURATION=0.3 for a softer dissolve, XFADE_DURATION=0.08 for near-instant cuts.
    FADE_DUR = float(os.environ.get("XFADE_DURATION", "0.15"))

    floor_dur = max(0.12, float(min_scene_duration))
    durations: list[float] = []
    if scene_plan:
        for s in scene_plan:
            d = s.get("duration", 6.0)
            try:
                durations.append(max(floor_dur, float(d)))
            except (ValueError, TypeError):
                durations.append(max(floor_dur, 6.0))

    audio_total: float = 0.0
    if audio_path and Path(audio_path).exists():
        audio_total = _probe_duration(audio_path)
        if audio_total > 0:
            # Each xfade transition removes FADE_DUR seconds from the final output.
            # To compensate, inflate the raw per-segment durations so that:
            #   sum(durations) - (n_images - 1) * FADE_DUR == audio_total
            # This ensures the composed video track matches the uploaded audio length.
            n_fades = max(0, len(images) - 1)
            xfade_overhead = n_fades * FADE_DUR
            target_raw_dur = audio_total + xfade_overhead

            if not durations or abs(sum(durations) - audio_total) > audio_total * 0.3:
                per_image = max(floor_dur, target_raw_dur / len(images))
                durations = [per_image] * len(images)
            else:
                ratio = target_raw_dur / sum(durations) if sum(durations) > 0 else 1.0
                durations = [d * ratio for d in durations]

            # Correct any floating-point rounding drift against the inflated target
            gap = target_raw_dur - sum(durations)
            if abs(gap) > 0.01 and durations:
                durations[-1] = max(floor_dur, durations[-1] + gap)
            _log(
                f"Audio-synced durations: raw_sum={sum(durations):.3f}s "
                f"xfade_overhead={xfade_overhead:.1f}s "
                f"expected_output={sum(durations) - xfade_overhead:.3f}s "
                f"audio={audio_total:.3f}s ({len(durations)} segs)"
            )

    while len(durations) < len(images):
        durations.append(6.0)

    n_images = len(images)
    temp_dir = output_path.parent / ".runner_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    segments: list[Path] = []

    # 8 Ken Burns patterns: (zoom_start, zoom_end, anchor)
    # Cinematic variety: slow dramatic push-ins, pull-outs, and diagonal pans.
    # Longer zoom ranges (1.0→1.25) give the photorealistic 3D images a
    # much more dynamic feel than the old shallow 1.0→1.12 range.
    kb_patterns = [
        (1.0,  1.20, "center"),   # slow zoom in  — revealing scene
        (1.20, 1.0,  "center"),   # pull back      — widening to context
        (1.0,  1.25, "left"),     # push in from left — following action
        (1.25, 1.0,  "right"),    # pull out right  — expanding view
        (1.05, 1.22, "center"),   # gentle push in  — emotional moment
        (1.22, 1.05, "left"),     # gentle pull left — contemplative
        (1.0,  1.18, "right"),    # drift right     — panning reveal
        (1.18, 1.0,  "center"),   # slow zoom out   — closing beat
    ]

    # Resolve subtitle font once for all segments
    sub_font_path = _get_subtitle_font(subtitle_language) if enable_subtitles else ""
    _log(f"Subtitle font: {sub_font_path or '(system default)'} for language '{subtitle_language}'")

    try:
        for i, img in enumerate(images):
            seg = temp_dir / f"seg_{i:04d}.mp4"
            seg_dur = durations[i]
            dur_frames = int(seg_dur * FPS)
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

            if enable_subtitles and sections and i < len(sections) and sections[i].strip():
                raw_sub = sections[i].strip()[:160]
                sub_text = _escape_drawtext(raw_sub)
                sub_end = max(0.15, seg_dur - min(0.4, seg_dur * 0.15))
                # Build drawtext with language-specific Noto font so Indic
                # scripts (Tamil, Hindi, Telugu, etc.) render correctly.
                font_clause = f":fontfile='{sub_font_path}'" if sub_font_path else ""
                vf_parts.append(
                    f"drawtext=text='{sub_text}'"
                    f"{font_clause}"
                    f":fontsize=40:fontcolor=white@0.97"
                    f":borderw=3:bordercolor=black@0.90"
                    f":box=1:boxcolor=black@0.45:boxborderw=10"
                    f":x=(w-tw)/2:y=h*0.87"
                    f":enable='between(t,0.08,{sub_end:.2f})'"
                )

            # Channel watermark overlay (top-right corner)
            wm_text = (os.environ.get("WATERMARK_TEXT") or "").strip()
            wm_img = (os.environ.get("WATERMARK_IMAGE") or "").strip()
            if wm_text and not wm_img:
                vf_parts.append(_build_watermark_filter(wm_text))

            vf = ",".join(vf_parts)

            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", img,
                "-t", str(seg_dur),
                "-vf", vf,
                "-c:v", "libx264", "-crf", "20", "-preset", x264_preset,
                "-r", str(FPS), "-pix_fmt", "yuv420p",
                seg,
            ]
            timeout_per_scene = max(300, int(seg_dur * 15))
            _log(f"Rendering scene {i+1}/{n_images} ({seg_dur:.1f}s)")
            try:
                subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_per_scene, check=True)
                segments.append(seg)
            except subprocess.CalledProcessError as e:
                _log(f"Ken Burns failed for scene {i+1}, trying simple scale: "
                     f"{e.stderr[:150] if e.stderr else ''}")
                simple_cmd = [
                    "ffmpeg", "-y",
                    "-loop", "1", "-i", img,
                    "-t", str(seg_dur),
                    "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
                           f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p",
                    "-c:v", "libx264", "-crf", "20", "-preset", x264_preset,
                    "-r", str(FPS), "-pix_fmt", "yuv420p",
                    seg,
                ]
                subprocess.run(simple_cmd, capture_output=True, text=True,
                               timeout=120, check=True)
                segments.append(seg)

        if len(segments) > 1:
            max_crossfade_scenes = int(os.environ.get("MAX_CROSSFADE_SCENES", "30"))
            use_crossfade = (
                max_crossfade_scenes <= 0 or len(segments) <= max_crossfade_scenes
            )
            if use_crossfade:
                _log("Applying crossfade transitions")
            else:
                _log(
                    f"Skipping crossfade: {len(segments)} scenes exceed "
                    f"MAX_CROSSFADE_SCENES={max_crossfade_scenes}; using concat"
                )
            prev = segments[0]
            cumulative_offset = durations[0] - FADE_DUR
            if use_crossfade:
                for i in range(1, len(segments)):
                    xfade_out = temp_dir / f"xfade_{i:04d}.mp4"

                    transition = "fade"
                    if scene_plan and i < len(scene_plan):
                        t = scene_plan[i].get("transition", "fade")
                        if t in ("dissolve", "fade", "slideright", "slideleft", "wiperight", "wipeleft"):
                            transition = t

                    _log(f"Crossfade transition {i}/{len(segments) - 1}")
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", str(prev), "-i", str(segments[i]),
                        "-filter_complex",
                        f"xfade=transition={transition}:duration={FADE_DUR}:offset={cumulative_offset:.2f},format=yuv420p",
                        "-c:v", "libx264", "-crf", "20", "-preset", x264_preset,
                        str(xfade_out),
                    ]
                    xfade_timeout = max(600, n_images * 45)
                    try:
                        subprocess.run(cmd, capture_output=True, text=True, timeout=xfade_timeout, check=True)
                        prev = xfade_out
                        cumulative_offset += durations[i] - FADE_DUR
                    except subprocess.CalledProcessError as e:
                        _log(f"Crossfade failed at segment {i}, falling back to concat: "
                             f"{e.stderr[:200] if e.stderr else ''}")
                        prev = None
                        break
                    except subprocess.TimeoutExpired:
                        _log(f"Crossfade timed out at segment {i}, falling back to concat")
                        prev = None
                        break
            else:
                prev = None

            if prev and prev != segments[0]:
                video_track = prev
            else:
                concat_file = temp_dir / "concat.txt"
                with open(concat_file, "w") as f:
                    for seg in segments:
                        safe = str(seg.resolve()).replace("\\", "/")
                        f.write(f"file '{safe}'\n")
                concat_out = temp_dir / "concat_out.mp4"
                concat_timeout = max(600, n_images * 5)
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                     "-i", str(concat_file), "-c", "copy", str(concat_out)],
                    capture_output=True, text=True, timeout=concat_timeout, check=True,
                )
                video_track = concat_out
        else:
            video_track = segments[0]

        if audio_path and Path(audio_path).exists():
            # Use explicit -t to pin output to the true audio duration.
            # Avoids -shortest silently truncating the video when the composed
            # video track is shorter than the audio due to xfade overlap loss.
            mux_t_args = ["-t", f"{audio_total:.3f}"] if audio_total > 0 else []
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_track),
                "-i", audio_path,
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v", "-map", "1:a",
                *mux_t_args,
                "-movflags", "+faststart",
                str(output_path),
            ]
            mux_timeout = max(600, n_images * 20)
            subprocess.run(cmd, capture_output=True, text=True, timeout=mux_timeout, check=True)
        else:
            shutil.copy2(str(video_track), str(output_path))

        size_mb = output_path.stat().st_size / (1024 * 1024)
        dur = _probe_duration(str(output_path))
        _log(f"Video composed: {output_path} ({size_mb:.1f} MB, {dur:.1f}s)")
        return str(output_path)

    except subprocess.CalledProcessError as e:
        _log(f"FFmpeg composition failed: {e.stderr[:500] if e.stderr else e}")
        return None
    finally:
        shutil.rmtree(str(temp_dir), ignore_errors=True)


# ---------------------------------------------------------------------------
# Post-creation verification
# ---------------------------------------------------------------------------

def _verify_output(
    video_path: str,
    script: str,
    scene_plan: list[dict[str, Any]],
    audio_path: str | None,
) -> dict[str, Any]:
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
        pct = (diff / audio_dur * 100) if audio_dur > 0 else 0
        if diff > 1.5:
            result["passed"] = False
            result["checks"].append(
                f"FAIL: Video/audio duration mismatch — "
                f"video={video_dur:.1f}s audio={audio_dur:.1f}s diff={diff:.1f}s ({pct:.0f}%). "
                f"Expected video≈audio after xfade compensation."
            )
        else:
            result["checks"].append(
                f"OK: Audio/video sync (video={video_dur:.1f}s, audio={audio_dur:.1f}s, diff={diff:.1f}s)"
            )

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


# ---------------------------------------------------------------------------
# Asset packaging
# ---------------------------------------------------------------------------

def _zip_scene_images(images: list[str], renders_dir: Path) -> str | None:
    if not images:
        return None
    zip_path = renders_dir / "scene_images.zip"
    try:
        with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
            for i, img_path in enumerate(images):
                p = Path(img_path)
                if p.exists():
                    ext = p.suffix or ".png"
                    zf.write(str(p), f"scene_{i + 1:03d}{ext}")
        if zip_path.exists() and zip_path.stat().st_size > 0:
            _log(f"Scene images zipped: {zip_path} ({zip_path.stat().st_size / 1024:.0f} KB, {len(images)} images)")
            return str(zip_path)
    except Exception as e:
        _log(f"Failed to zip images: {e}")
    return None


def _copy_images_to_renders(images: list[str], renders_dir: Path) -> None:
    dest = renders_dir / "images"
    dest.mkdir(parents=True, exist_ok=True)
    for i, img_path in enumerate(images):
        p = Path(img_path)
        if p.exists():
            ext = p.suffix or ".png"
            shutil.copy2(str(p), str(dest / f"scene_{i + 1:03d}{ext}"))


# ---------------------------------------------------------------------------
# Intermediate cleanup
# ---------------------------------------------------------------------------

def _cleanup_intermediate_assets(project_dir: Path) -> None:
    """Remove heavy intermediates (keeps renders/ with video, images, and zip)."""
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
    img_dir = project_dir / "assets" / "images"
    if img_dir.is_dir():
        for p in img_dir.iterdir():
            try:
                if p.is_file():
                    p.unlink()
            except OSError as e:
                _log(f"Cleanup skip {p}: {e}")
