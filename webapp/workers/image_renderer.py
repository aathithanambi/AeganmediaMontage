"""STEP 3 — Character-Consistent Image Generation.

Responsible for:
  - Building the character lock block prepended to every image prompt
  - Sequential generation within character groups (eliminates visual drift between scenes)
  - Per-character first-appearance anchoring ("same as scene N")
  - Outfit change detection from scene narration
  - Parallel image generation (ThreadPoolExecutor) with cache support
  - Backend selection (Gemini native / Imagen / auto)

Character consistency strategy
───────────────────────────────
1. Every image prompt receives the full CHARACTER LOCK block (skin, outfit, colors).
2. Scenes that share a named character are generated sequentially, ordered by scene
   index, so the text lock is applied consistently without racing conditions.
3. After the first image for a character is generated, every subsequent scene that
   includes that character appends "PREVIOUS APPEARANCE: [scene file]" — a text
   anchor that reinforces the LLM knows a prior reference exists.
4. Outfit-change signals in the narration ("wore", "put on", "changed into …") are
   detected and the updated outfit description is injected into the prompt, overriding
   the default signature_outfit for that scene only.
"""
from __future__ import annotations

import hashlib
import os
import re
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
# Outfit-change detection
# ---------------------------------------------------------------------------

# Words in a scene narration that signal the character has changed clothes.
_OUTFIT_CHANGE_PATTERNS = re.compile(
    r"\b("
    r"wore|wearing|put on|dressed (in|as|up)|changed (into|to|his|her|their)"
    r"|changed clothes|new (outfit|dress|clothes|costume|robe|armor|uniform)"
    r"|dressed himself|dressed herself|took off his|took off her"
    r"|removed his|removed her|stripped off|donned|adorned"
    r")\b",
    re.IGNORECASE,
)


def _detect_outfit_change(narration: str) -> bool:
    """Return True if the narration implies a costume or appearance change."""
    return bool(_OUTFIT_CHANGE_PATTERNS.search(narration or ""))


# ---------------------------------------------------------------------------
# Character lock block (prepended to every image prompt)
# ---------------------------------------------------------------------------

def _build_character_lock_block(
    character_data: dict[str, Any] | None,
    *,
    outfit_overrides: dict[str, str] | None = None,
    max_chars: int = 4800,
) -> str:
    """Build the CHARACTER LOCK text block prepended to every image prompt.

    Args:
        character_data:   Extracted character/location registry.
        outfit_overrides: Per-character outfit override for this specific scene
                          (populated when outfit-change is detected in narration).
        max_chars:        Maximum block length (extended from 3600 to 4800 so
                          multi-character stories aren't silently truncated).
    """
    if not character_data:
        return ""
    chars = character_data.get("characters") or []
    if not chars:
        return ""
    lines: list[str] = [
        "SERIES CHARACTER LOCK — copy EXACTLY in EVERY frame, NO exceptions:",
        "• The FACE (shape, eyes, nose, jaw) must be IDENTICAL to the first time this character appeared.",
        "• The HAIR COLOR AND STYLE must not change — same color, same cut, same length.",
        "• The SKIN COLOR must not change — not lighter, not darker, not different hue.",
        "• The AGE must not change — same wrinkles, same youthfulness.",
        "• The FULL OUTFIT (top garment + bottom garment + footwear) stays the same unless the story "
        "explicitly says they changed clothes — including pants/trouser colour and shoe/sandal/bare-feet.",
        "• NEVER invent new facial features, recolor skin, change hair color/style, or swap build.",
        "• ONLY ONE instance of each named character per frame — NEVER show the same character twice, "
        "as a reflection, clone, twin, or duplicate. If only one character is named, only one appears.",
        "",
        "LOCKED CAST:",
    ]
    for ch in chars[:16]:
        if not isinstance(ch, dict):
            continue
        name = (ch.get("name") or "Character").strip()
        role = (ch.get("role") or "").strip()
        face_lock = (ch.get("face_lock") or ch.get("description") or "").strip().replace("\n", " ")
        outfit = (ch.get("signature_outfit") or "").strip()
        skin = (ch.get("skin_color") or "").strip()
        hair = (ch.get("hair_color") or "").strip()
        age = (ch.get("age_range") or "").strip()
        colors = ch.get("main_colors") or []
        color_line = ", ".join(str(c) for c in colors[:12]) if colors else ""

        # Apply per-scene outfit override when a change was detected in narration
        if outfit_overrides and name in outfit_overrides:
            outfit = f"[CHANGED THIS SCENE] {outfit_overrides[name]}"

        bit = f"• {name}"
        if role:
            bit += f" ({role})"
        lines.append(bit + ":")
        if age:
            lines.append(f"  AGE (fixed): {age}")
        if skin:
            lines.append(f"  SKIN COLOR (never change): {skin}")
        if hair:
            lines.append(f"  HAIR COLOR & STYLE (never change): {hair}")
        if face_lock:
            lines.append(f"  FACE LOCK (copy exactly): {face_lock[:500]}")
        if color_line:
            lines.append(f"  MAIN COLORS (fixed): {color_line}")
        if outfit:
            lines.append(
                f"  SIGNATURE OUTFIT (top + bottom + footwear — ALL locked, do NOT change): {outfit}"
            )
    lines.append("")
    # Story props — specific objects (food, tools, clothing items) that must look
    # the same in every scene they appear in.  Prevents the AI from substituting
    # a different fruit, colour, or object type across scenes.
    props = character_data.get("story_props") or []
    if props:
        lines.append(
            "STORY PROPS (use EXACTLY this description every time this object appears — "
            "NEVER substitute a different colour, shape, or type):"
        )
        for p in props[:12]:
            if not isinstance(p, dict):
                continue
            pn = (p.get("name") or "").strip()
            pd = (p.get("description") or "").strip().replace("\n", " ")
            if pn or pd:
                lines.append(f"• {pn}: {pd[:200]}" if pn else f"• {pd[:200]}")
        lines.append("")
    locs = character_data.get("locations") or []
    if locs:
        lines.append("RECURRING PLACES (keep look consistent when reused):")
        for loc in locs[:8]:
            if not isinstance(loc, dict):
                continue
            ln = (loc.get("name") or "").strip()
            ld = (loc.get("description") or "").strip().replace("\n", " ")
            if ln or ld:
                lines.append(f"• {ln}: {ld[:250]}" if ln else f"• {ld[:250]}")
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text


# ---------------------------------------------------------------------------
# Character-to-scene mapping helpers
# ---------------------------------------------------------------------------

def _extract_scene_character_names(
    scene: dict[str, Any],
    character_data: dict[str, Any] | None,
) -> list[str]:
    """Return the names of known characters that appear in this scene.

    Looks at 'characters_present' field first (set during extraction),
    then falls back to scanning the image_prompt and narration text.
    """
    known = {
        (ch.get("name") or "").strip().lower()
        for ch in (character_data or {}).get("characters", [])
        if isinstance(ch, dict) and ch.get("name")
    }
    if not known:
        return []

    # Prefer explicit field set during _extract_characters_and_scenes
    explicit = scene.get("characters_present") or []
    if explicit:
        return [n for n in explicit if n.strip().lower() in known]

    # Fallback: scan text
    text = (
        (scene.get("narration") or "") + " " + (scene.get("image_prompt") or "")
    ).lower()
    return [name for name in known if name in text]


def _build_scene_groups(
    scene_plan: list[dict[str, Any]],
    character_data: dict[str, Any] | None,
) -> list[list[int]]:
    """Group scene indices so scenes sharing a character are in the same group.

    Scenes within a group are generated sequentially (preserving visual
    continuity). Groups with no shared characters can run in parallel.

    Returns a list of groups, each group being an ordered list of scene indices.
    """
    n = len(scene_plan)
    # Each scene starts in its own component
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Map each character name → list of scene indices where they appear
    char_scenes: dict[str, list[int]] = {}
    for i, scene in enumerate(scene_plan):
        for name in _extract_scene_character_names(scene, character_data):
            char_scenes.setdefault(name, []).append(i)

    # Union-find: connect all scenes that share a character
    for indices in char_scenes.values():
        for j in range(1, len(indices)):
            union(indices[0], indices[j])

    # Build groups (preserve original scene order within each group)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    return [sorted(g) for g in groups.values()]


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
    """Generate one image per scene with cross-scene character consistency.

    Consistency strategy (applied in order):
    1. CHARACTER LOCK block (skin / outfit / colors) prepended to every prompt.
    2. Scenes sharing a named character are generated SEQUENTIALLY within a
       thread — no racing between "same character, different scene" renders.
    3. After a character's first image is produced, every later scene that
       includes that character appends a "PREVIOUS APPEARANCE" anchor note.
    4. Outfit-change signals in narration are detected; overriding outfit
       text is injected into the lock block for that scene only.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images: list[str | None] = [None] * len(scene_plan)

    if not _google_available():
        _log("No GOOGLE_API_KEY — cannot generate images")
        return []

    backend = (os.environ.get("IMAGE_BACKEND") or "gemini").strip().lower()
    if backend not in ("gemini", "imagen", "auto"):
        _log(f"IMAGE_BACKEND={backend!r} invalid — defaulting to gemini")
        backend = "gemini"

    # Groups of scenes that share characters → generate each group sequentially.
    # Independent groups run in parallel across threads.
    scene_groups = _build_scene_groups(scene_plan, character_data)
    n_groups = len(scene_groups)
    parallel = max(1, min(8, int(os.environ.get("IMAGEN_PARALLEL", "3")), n_groups))
    _log(
        f"Generating {len(scene_plan)} images: "
        f"{n_groups} character-group(s), parallel={parallel} groups at a time "
        f"(sequential within each group)"
    )

    cache_enabled = _env_truthy("IMAGE_PROMPT_CACHE", True)
    cache_root = output_dir.parent / "image_gen_cache"
    if cache_enabled:
        cache_root.mkdir(parents=True, exist_ok=True)

    meta_extra = _meta_ai_style_extra()

    # Per-character first-appearance image path (written by first scene, read by later ones)
    char_first_image: dict[str, str] = {}
    char_first_lock = threading.Lock()

    def _generate_still(prompt: str, out: Path) -> str | None:
        if backend == "imagen":
            return _google_imagen_generate(prompt, out, aspect_ratio="16:9")
        if backend == "gemini":
            return _google_gemini_native_image_generate(prompt, out, aspect_ratio="16:9")
        r = _google_gemini_native_image_generate(prompt, out, aspect_ratio="16:9")
        return r or _google_imagen_generate(prompt, out, aspect_ratio="16:9")

    def _one_scene(i: int, scene: dict[str, Any]) -> tuple[int, str | None]:
        out = output_dir / f"scene_{i:02d}.png"

        # ── Step A: base image prompt ────────────────────────────────────────
        image_prompt = scene.get("image_prompt") or scene.get("search_query") or "abstract background"
        # Prepend style prefix only if the prompt doesn't already contain it.
        # Check the first 80 chars against the first 20 chars of the prefix to be style-agnostic.
        style_anchor = IMAGE_STYLE_PREFIX[:20].lower()
        if style_anchor not in image_prompt.lower()[:80]:
            image_prompt = f"{IMAGE_STYLE_PREFIX}{image_prompt}"
        if meta_extra and meta_extra.lower() not in image_prompt.lower():
            image_prompt = f"{meta_extra}{image_prompt}"

        # ── Step B: detect outfit change for this scene ──────────────────────
        narration = scene.get("narration", "")
        outfit_overrides: dict[str, str] | None = None
        if _detect_outfit_change(narration) and character_data:
            # Extract new outfit description from narration for each character
            outfit_overrides = {}
            for ch in character_data.get("characters", []):
                if not isinstance(ch, dict):
                    continue
                name = (ch.get("name") or "").strip()
                if name and name.lower() in narration.lower():
                    # Extract a concise new outfit from the sentence
                    # (keep existing if we can't parse a better one)
                    outfit_overrides[name] = _extract_outfit_from_narration(narration, name)
            if not any(outfit_overrides.values()):
                outfit_overrides = None

        # ── Step C: prepend character lock block ─────────────────────────────
        lock_block = _build_character_lock_block(character_data, outfit_overrides=outfit_overrides)
        camera_angle = scene.get("camera_angle", "").strip()
        scene_header = f"--- SCENE {i + 1}"
        if camera_angle:
            scene_header += f" [{camera_angle.upper()} SHOT]"
        scene_header += " ---"
        if lock_block:
            image_prompt = f"{lock_block}\n\n{scene_header}\n{image_prompt}"
        elif camera_angle:
            image_prompt = f"{scene_header}\n{image_prompt}"

        # ── Step D: append "previous appearance" anchor for returning characters ──
        scene_chars = _extract_scene_character_names(scene, character_data)
        with char_first_lock:
            prev_refs = [
                f"  • {name}: previously shown in {Path(char_first_image[name]).name} — "
                f"render IDENTICAL face, skin, and outfit."
                for name in scene_chars
                if name in char_first_image
            ]
        if prev_refs:
            image_prompt += (
                "\n\nCHARACTER CONTINUITY (copy exactly from previous scene):\n"
                + "\n".join(prev_refs)
            )

        # ── Step E: cache lookup ─────────────────────────────────────────────
        cache_key = hashlib.sha256(f"{backend}|{image_prompt}".encode()).hexdigest()
        cache_file = cache_root / f"{cache_key}.png"
        if cache_enabled and cache_file.is_file() and cache_file.stat().st_size > 500:
            try:
                shutil.copy2(cache_file, out)
                _log(f"Scene {i + 1}/{len(scene_plan)}: cache HIT")
                _record_first_appearances(i, scene_chars, str(out), char_first_image, char_first_lock)
                return i, str(out)
            except OSError as e:
                _log(f"Scene {i + 1}: cache copy failed ({e}), regenerating")

        # ── Step F: generate ─────────────────────────────────────────────────
        _log(f"Scene {i + 1}/{len(scene_plan)}: generating ({backend})")
        result_path = _generate_still(image_prompt, out)
        if result_path:
            if cache_enabled:
                try:
                    shutil.copy2(result_path, cache_file)
                except OSError:
                    pass
            _record_first_appearances(i, scene_chars, result_path, char_first_image, char_first_lock)
        else:
            _log(f"Scene {i + 1}: generation failed")
        return i, result_path

    def _run_group(group: list[int]) -> None:
        """Generate all scenes in a group sequentially (shared character chain)."""
        for i in group:
            idx, path = _one_scene(i, scene_plan[i])
            images[idx] = path
            _track_done()

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
                _log(f"Progress callback error: {e}")

    if total == 0:
        return []

    if parallel <= 1 or n_groups == 1:
        for group in scene_groups:
            _run_group(group)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            for _ in as_completed([ex.submit(_run_group, g) for g in scene_groups]):
                pass  # progress tracked inside _run_group

    out_list = [p for p in images if p]
    _log(f"Image generation complete: {len(out_list)}/{total} images")
    return out_list


# ---------------------------------------------------------------------------
# First-appearance tracker
# ---------------------------------------------------------------------------

def _record_first_appearances(
    scene_idx: int,
    char_names: list[str],
    image_path: str,
    registry: dict[str, str],
    lock: threading.Lock,
) -> None:
    """Register the first generated image for each character in this scene."""
    with lock:
        for name in char_names:
            if name not in registry:
                registry[name] = image_path
                _log(f"  Character anchor set: '{name}' → scene_{scene_idx:02d}.png")


# ---------------------------------------------------------------------------
# Outfit extraction from narration text
# ---------------------------------------------------------------------------

def _extract_outfit_from_narration(narration: str, character_name: str) -> str:
    """Attempt to extract a new outfit description from a narration sentence.

    Returns a short description if found, or an empty string so the caller
    can fall back to the default signature_outfit.
    """
    # Match patterns like "wore a red robe", "put on golden armor", "dressed in silk"
    patterns = [
        re.compile(
            r"(?:wore?|wearing|put on|dressed in|changed into|donned|adorned with)\s+(.{10,80}?)(?:[,.]|$)",
            re.IGNORECASE,
        ),
    ]
    for pat in patterns:
        m = pat.search(narration)
        if m:
            return m.group(1).strip().rstrip(",.")
    return ""
