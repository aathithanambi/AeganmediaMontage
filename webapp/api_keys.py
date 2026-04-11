"""API key status checker for admin dashboard.

Checks which API keys are configured, their availability,
and fetches credit/usage info where provider APIs support it.
"""

from __future__ import annotations

import os
from typing import Any

import requests


API_KEY_REGISTRY: list[dict[str, Any]] = [
    {
        "env": "GOOGLE_API_KEY",
        "name": "Google AI",
        "category": "LLM / TTS / Images",
        "tier": "free",
        "tools": ["Gemini LLM (script writing)", "Google TTS", "Google Imagen"],
        "url": "https://aistudio.google.com/apikey",
        "balance_check": None,
    },
    {
        "env": "ELEVENLABS_API_KEY",
        "name": "ElevenLabs",
        "category": "TTS / Music / SFX",
        "tier": "free_tier",
        "tools": ["ElevenLabs TTS (best voice)", "Music Gen", "Sound Effects"],
        "url": "https://elevenlabs.io/",
        "balance_check": "elevenlabs",
    },
    {
        "env": "FAL_KEY",
        "name": "fal.ai",
        "category": "Images / Video",
        "tier": "free_tier",
        "tools": ["FLUX Images (best quality)", "Kling Video", "Veo Video",
                  "MiniMax Video", "Recraft Images"],
        "url": "https://fal.ai/dashboard/keys",
        "balance_check": None,
    },
    {
        "env": "PEXELS_API_KEY",
        "name": "Pexels",
        "category": "Stock Media",
        "tier": "free",
        "tools": ["Stock Photos", "Stock Video"],
        "url": "https://www.pexels.com/api/",
        "balance_check": None,
    },
    {
        "env": "PIXABAY_API_KEY",
        "name": "Pixabay",
        "category": "Stock Media",
        "tier": "free",
        "tools": ["Stock Photos", "Stock Video", "Stock Music"],
        "url": "https://pixabay.com/api/docs/",
        "balance_check": None,
    },
    {
        "env": "FREESOUND_API_KEY",
        "name": "Freesound",
        "category": "Music / SFX",
        "tier": "free",
        "tools": ["Background Music Search", "Sound Effects Search"],
        "url": "https://freesound.org/apiv2/apply",
        "balance_check": None,
    },
    {
        "env": "OPENAI_API_KEY",
        "name": "OpenAI",
        "category": "TTS / Images",
        "tier": "paid",
        "tools": ["OpenAI TTS", "DALL-E Image Gen"],
        "url": "https://platform.openai.com/api-keys",
        "balance_check": None,
    },
    {
        "env": "XAI_API_KEY",
        "name": "xAI / Grok",
        "category": "Images / Video",
        "tier": "free_tier",
        "tools": ["Grok Image Gen/Edit", "Grok Video Gen"],
        "url": "https://console.x.ai/",
        "balance_check": None,
    },
    {
        "env": "HEYGEN_API_KEY",
        "name": "HeyGen",
        "category": "Avatar Video",
        "tier": "paid",
        "tools": ["Avatar Videos", "Video Translation", "Face Swap"],
        "url": "https://app.heygen.com/settings/api",
        "balance_check": "heygen",
    },
    {
        "env": "RUNWAY_API_KEY",
        "name": "Runway",
        "category": "Video Generation",
        "tier": "paid",
        "tools": ["Runway Gen-4 Video"],
        "url": "https://app.runwayml.com/settings/api-keys",
        "balance_check": None,
    },
    {
        "env": "SUNO_API_KEY",
        "name": "Suno",
        "category": "Music Generation",
        "tier": "paid",
        "tools": ["Suno AI Music (full songs)"],
        "url": "https://sunoapi.org/api-key",
        "balance_check": None,
    },
    {
        "env": "HF_TOKEN",
        "name": "HuggingFace",
        "category": "Analysis",
        "tier": "free",
        "tools": ["Speaker Diarization (WhisperX)"],
        "url": "https://huggingface.co/settings/tokens",
        "balance_check": None,
    },
]

TIER_LABELS = {
    "free": "Free",
    "free_tier": "Free Tier",
    "paid": "Paid",
}


def _mask_key(key: str) -> str:
    """Show first 4 and last 4 chars, mask the rest."""
    if len(key) <= 10:
        return key[:2] + "***" + key[-2:]
    return key[:4] + "***" + key[-4:]


def _check_elevenlabs(api_key: str) -> dict[str, Any]:
    """Fetch ElevenLabs subscription info."""
    try:
        resp = requests.get(
            "https://api.elevenlabs.io/v1/user/subscription",
            headers={"xi-api-key": api_key},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {
                "plan": data.get("tier", "unknown"),
                "characters_used": data.get("character_count", 0),
                "characters_limit": data.get("character_limit", 0),
                "characters_remaining": max(
                    0, data.get("character_limit", 0) - data.get("character_count", 0)
                ),
                "next_reset": data.get("next_character_count_reset_unix"),
            }
        return {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def _check_heygen(api_key: str) -> dict[str, Any]:
    """Fetch HeyGen remaining quota."""
    try:
        resp = requests.get(
            "https://api.heygen.com/v1/user/remaining_quota",
            headers={"X-Api-Key": api_key},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            return {
                "remaining_quota": data.get("remaining_quota", 0),
                "plan": "active",
            }
        return {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


BALANCE_CHECKERS = {
    "elevenlabs": _check_elevenlabs,
    "heygen": _check_heygen,
}


def get_api_key_status() -> list[dict[str, Any]]:
    """Return status of all API keys for dashboard display."""
    results: list[dict[str, Any]] = []

    for entry in API_KEY_REGISTRY:
        env_name = entry["env"]
        raw_value = os.environ.get(env_name, "")
        is_set = bool(raw_value and raw_value.strip())

        item: dict[str, Any] = {
            "env": env_name,
            "name": entry["name"],
            "category": entry["category"],
            "tier": TIER_LABELS.get(entry["tier"], entry["tier"]),
            "tools": entry["tools"],
            "url": entry["url"],
            "configured": is_set,
            "masked_value": _mask_key(raw_value) if is_set else "",
            "balance": None,
        }

        if is_set and entry.get("balance_check"):
            checker = BALANCE_CHECKERS.get(entry["balance_check"])
            if checker:
                item["balance"] = checker(raw_value)

        results.append(item)

    return results


def get_api_summary() -> dict[str, Any]:
    """Return summary counts for the dashboard."""
    keys = get_api_key_status()
    configured = sum(1 for k in keys if k["configured"])
    total = len(keys)
    tools_available = sum(len(k["tools"]) for k in keys if k["configured"])
    tools_total = sum(len(k["tools"]) for k in keys)
    return {
        "configured": configured,
        "total": total,
        "tools_available": tools_available,
        "tools_total": tools_total,
    }
