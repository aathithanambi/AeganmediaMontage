"""Pipeline orchestrator — wires all worker modules together.

Contains:
  - run_pipeline(): main orchestration function that calls audio_parser,
    scene_generator, image_renderer, and video_builder in sequence
  - main(): CLI entry point for subprocess invocation
  - OUTPUT_SCENES= JSON output for DB storage
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from webapp.workers.shared import (
    SUPPORTED_LANGUAGES,
    _discover_tools,
    _emit_progress_snapshot,
    _env_truthy,
    _google_available,
    _log,
    _probe_duration,
    _progress_pct,
    _reset_api_usage,
    get_api_usage,
    registry,
)

from webapp.workers.audio_parser import (
    _add_english_to_timings,
    _merge_timings_for_budget,
    _transcribe_with_timestamps,
    step_tts,
    step_transcribe_audio,
    step_translate_subtitles,
)

from webapp.workers.scene_generator import (
    _analyze_reference_style,
    _character_lock_fingerprint,
    _extract_characters_and_scenes,
    _load_character_lock,
    _normalize_character_entries,
    _parse_production_intent,
    _save_character_lock,
    _summarize_reference,
    step_analyze_reference,
    step_download_reference,
    step_generate_scene_plan,
    step_generate_scene_plan_timeline,
    step_generate_script,
)

from webapp.workers.image_renderer import (
    step_fetch_images,
)

from webapp.workers.video_builder import (
    _cleanup_intermediate_assets,
    _copy_images_to_renders,
    _verify_output,
    _zip_scene_images,
    step_compose_slideshow,
)

# ---------------------------------------------------------------------------
# Main pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(prompt_file: str) -> None:
    _reset_api_usage()
    pipeline_start = time.time()

    _log(f"Starting pipeline runner (cwd={os.getcwd()})")
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

    done_stages.append("reference")
    _emit_progress_snapshot(done_stages, "transcribe", _progress_pct(len(done_stages), 0))

    # --- Phase 1: Parse production intent ---
    intent = _parse_production_intent(raw_prompt, ref_summary)
    content_prompt = intent["content_prompt"]
    style_notes = str(intent.get("style_notes") or "").strip()
    audio_lang = form_audio_lang or intent["audio_language"]
    subtitle_lang = form_subtitle_lang or intent["subtitle_language"]
    target_dur = intent["target_duration"]
    reference_driven = intent["reference_driven"]

    ref_style = _analyze_reference_style(ref_summary or "", style_notes)

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

    max_scenes_budget = max(6, min(200, int(os.environ.get("PIPELINE_MAX_SCENES", "120"))))
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

    # --- Phase 2b: Character & scene extraction (cached by transcript fingerprint) ---
    char_fp = _character_lock_fingerprint(
        script_for_visuals,
        title,
        str(ref_style.get("art_style", "")),
        creator_topic=content_prompt,
        style_notes=style_notes,
    )
    if _env_truthy("CHARACTER_CACHE", True):
        cached_chars = _load_character_lock(assets_dir, char_fp)
        if cached_chars is not None:
            character_data = _normalize_character_entries(dict(cached_chars))
        else:
            character_data = _extract_characters_and_scenes(
                script_for_visuals, title, ref_style, creator_topic=content_prompt,
            )
            _save_character_lock(assets_dir, char_fp, character_data)
    else:
        character_data = _extract_characters_and_scenes(
            script_for_visuals, title, ref_style, creator_topic=content_prompt,
        )
        _save_character_lock(assets_dir, char_fp, character_data)

    done_stages.append("characters")
    _emit_progress_snapshot(done_stages, "scenes", _progress_pct(len(done_stages), 0))

    # --- Phase 3: Scene plan with character consistency ---
    if merged_timings:
        scene_plan = step_generate_scene_plan_timeline(
            merged_timings,
            character_data,
            ref_style,
            title,
            creator_topic=content_prompt,
            style_notes=style_notes,
        )
    else:
        if target_dur <= 120:
            scene_count = max(2, min(15, int(target_dur / 6)))
        elif target_dur <= 600:
            scene_count = max(10, min(40, int(target_dur / 12)))
        else:
            scene_count = max(30, min(200, int(target_dur / 20)))
        scene_plan = step_generate_scene_plan(
            script, content_prompt, title, ref_summary,
            ref_style=ref_style,
            character_data=character_data,
            scene_count=scene_count,
            sentence_timings=sentence_timings if sentence_timings else None,
            style_notes=style_notes,
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

    # --- Phase 5: Character-consistent images (lock block + optional prompt cache) ---
    images = step_fetch_images(
        scene_plan, assets_dir / "images",
        character_data=character_data,
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

    _copy_images_to_renders(images, renders_dir)
    images_zip_path = _zip_scene_images(images, renders_dir)
    if images_zip_path:
        print(f"OUTPUT_IMAGES_ZIP={images_zip_path}")

    # Emit OUTPUT_SCENES= for DB storage
    scenes_data: list[dict[str, Any]] = []
    for i, s in enumerate(scene_plan):
        img_file = images[i] if i < len(images) else ""
        scenes_data.append({
            "scene": i + 1,
            "image": Path(img_file).name if img_file else "",
            "start": float(s.get("start", 0)),
            "end": float(s.get("end", 0)),
            "duration": float(s.get("duration", 0)),
            "prompt": (s.get("image_prompt") or s.get("search_query") or "")[:300],
            "narration": (s.get("narration") or "")[:500],
        })
    try:
        print(f"OUTPUT_SCENES={json.dumps({'scenes': scenes_data})}")
    except (TypeError, ValueError):
        pass

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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Google AI video pipeline")
    parser.add_argument("--pipeline", required=True, help="Pipeline name")
    parser.add_argument("--project", required=True, help="Project ID")
    parser.add_argument("--prompt-file", required=True, help="Path to prompt JSON")
    args = parser.parse_args()
    run_pipeline(args.prompt_file)


if __name__ == "__main__":
    main()
