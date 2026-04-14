"""STEP 3 — Parallel Image Generation with Character Consistency.

Responsible for:
  - Building the character lock block prepended to every image prompt
  - Parallel image generation (ThreadPoolExecutor) with cache support
  - Backend selection (Gemini native / Imagen / auto)
"""
from __future__ import annotations

import hashlib
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from webapp.workers.shared import (
    IMAGE_STYLE_PREFIX,
    _env_truthy,
    _google_available,
    _google_gemini_native_image_generate,
    _google_imagen_generate,
    _log,
    _meta_ai_style_extra,
)

# ---------------------------------------------------------------------------
# Character lock block (prepended to every image prompt)
# ---------------------------------------------------------------------------

def _build_character_lock_block(
    character_data: dict[str, Any] | None,
    *,
    max_chars: int = 3600,
) -> str:
    if not character_data:
        return ""
    chars = character_data.get("characters") or []
    if not chars:
        return ""
    lines: list[str] = [
        "SERIES CHARACTER LOCK — obey in EVERY frame:",
        "• Keep the SAME face, hair, skin color, age, and body type for each named character.",
        "• Keep the SAME signature outfit and the SAME main garment colors unless the narration "
        "explicitly says they changed clothes.",
        "• Do NOT recolor skin, outfits, swap palette, or invent new costumes between scenes.",
        "",
        "LOCKED CAST:",
    ]
    for ch in chars[:16]:
        if not isinstance(ch, dict):
            continue
        name = (ch.get("name") or "Character").strip()
        role = (ch.get("role") or "").strip()
        desc = (ch.get("description") or "").strip().replace("\n", " ")
        outfit = (ch.get("signature_outfit") or "").strip()
        skin = (ch.get("skin_color") or "").strip()
        colors = ch.get("main_colors") or []
        color_line = ", ".join(str(c) for c in colors[:10]) if colors else ""
        bit = f"• {name}"
        if role:
            bit += f" ({role})"
        lines.append(bit + ":")
        if skin:
            lines.append(f"  SKIN COLOR (never change): {skin}")
        if color_line:
            lines.append(f"  MAIN COLORS (fixed): {color_line}")
        if outfit:
            lines.append(f"  SIGNATURE OUTFIT (fixed): {outfit}")
        if desc:
            if len(desc) > 520:
                desc = desc[:517] + "..."
            lines.append(f"  LOOK: {desc}")
    lines.append("")
    locs = character_data.get("locations") or []
    if locs:
        lines.append("RECURRING PLACES (keep look consistent when reused):")
        for loc in locs[:6]:
            if not isinstance(loc, dict):
                continue
            ln = (loc.get("name") or "").strip()
            ld = (loc.get("description") or "").strip().replace("\n", " ")
            if len(ld) > 200:
                ld = ld[:197] + "..."
            if ln or ld:
                lines.append(f"• {ln}: {ld}" if ln else f"• {ld}")
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text


# ---------------------------------------------------------------------------
# Image generation orchestrator
# ---------------------------------------------------------------------------

def step_fetch_images(
    scene_plan: list[dict[str, Any]],
    output_dir: Path,
    *,
    character_data: dict[str, Any] | None = None,
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

    parallel = max(1, min(8, int(os.environ.get("IMAGEN_PARALLEL", "3"))))
    _log(
        f"Generating images (backend={backend}, parallel={parallel}) — "
        f"Gemini native = Nano Banana family when backend=gemini",
    )

    lock_block = _build_character_lock_block(character_data)
    cache_enabled = _env_truthy("IMAGE_PROMPT_CACHE", True)
    cache_root = output_dir.parent / "image_gen_cache"
    if cache_enabled:
        cache_root.mkdir(parents=True, exist_ok=True)

    def _generate_still(image_prompt: str, out: Path) -> str | None:
        if backend == "imagen":
            return _google_imagen_generate(image_prompt, out, aspect_ratio="16:9")
        if backend == "gemini":
            return _google_gemini_native_image_generate(
                image_prompt, out, aspect_ratio="16:9",
            )
        r = _google_gemini_native_image_generate(
            image_prompt, out, aspect_ratio="16:9",
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

        if lock_block:
            image_prompt = f"{lock_block}\n\n--- SCENE ---\n{image_prompt}"

        cache_key = hashlib.sha256(
            f"{backend}|{image_prompt}".encode("utf-8"),
        ).hexdigest()
        cache_file = cache_root / f"{cache_key}.png"

        if cache_enabled and cache_file.is_file() and cache_file.stat().st_size > 500:
            try:
                shutil.copy2(cache_file, out)
                _log(f"Scene {i+1}/{len(scene_plan)}: image cache HIT")
                return i, str(out)
            except OSError as e:
                _log(f"Scene {i+1}: cache copy failed ({e}), regenerating")

        _log(f"Scene {i+1}/{len(scene_plan)}: queued image ({backend})")
        result_path = _generate_still(image_prompt, out)
        if result_path and cache_enabled and Path(result_path).is_file():
            try:
                shutil.copy2(result_path, cache_file)
            except OSError:
                pass
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
