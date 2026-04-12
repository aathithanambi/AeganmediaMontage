"""Google AI API key status checker for admin dashboard.

Only GOOGLE_API_KEY is needed — it powers Gemini, Imagen, and Cloud TTS.
"""

from __future__ import annotations

import os
from typing import Any

import requests


API_KEY_REGISTRY: list[dict[str, Any]] = [
    {
        "env": "GOOGLE_API_KEY",
        "name": "Google AI",
        "category": "LLM / TTS / Images / Transcription",
        "tier": "free",
        "tools": [
            "Gemini LLM (script writing, scene planning, intent parsing)",
            "Gemini Audio Transcription (multimodal)",
            "Google Imagen (AI image generation)",
            "Google Cloud TTS (50+ languages, 700+ voices)",
            "Subtitle Translation (Gemini)",
        ],
        "url": "https://aistudio.google.com/apikey",
        "balance_check": "google",
        "apis_required": [
            "Generative Language API (Gemini)",
            "Cloud Text-to-Speech API",
        ],
    },
]

TIER_LABELS = {
    "free": "Free",
    "free_tier": "Free Tier",
    "paid": "Paid",
}


def _mask_key(key: str) -> str:
    if len(key) <= 10:
        return key[:2] + "***" + key[-2:]
    return key[:4] + "***" + key[-4:]


def _check_google(api_key: str) -> dict[str, Any]:
    """Verify Google API key by calling Gemini with a tiny test prompt."""
    try:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.0-flash-lite:generateContent?key={api_key}"
        )
        body = {
            "contents": [{"parts": [{"text": "Say OK"}]}],
            "generationConfig": {"maxOutputTokens": 10},
        }
        resp = requests.post(url, json=body, timeout=15)
        if resp.status_code == 200:
            return {"status": "active", "gemini": "OK"}
        return {"status": "error", "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


BALANCE_CHECKERS = {
    "google": _check_google,
}


def get_api_key_status() -> list[dict[str, Any]]:
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
            "apis_required": entry.get("apis_required", []),
        }

        if is_set and entry.get("balance_check"):
            checker = BALANCE_CHECKERS.get(entry["balance_check"])
            if checker:
                item["balance"] = checker(raw_value)

        results.append(item)

    return results


def get_api_summary() -> dict[str, Any]:
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
