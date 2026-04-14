"""Ordered pipeline stages for progress UI (run detail page)."""

# id -> human label (order matters for display)
PIPELINE_STAGE_ORDER: list[tuple[str, str]] = [
    ("setup", "Setup & tools"),
    ("reference", "Reference video analysis"),
    ("transcribe", "Transcribe audio & timeline"),
    ("english", "English text for visuals"),
    ("characters", "Characters & locations"),
    ("scenes", "Scene plan (timeline)"),
    ("subtitles", "Subtitles"),
    ("tts", "Text-to-speech narration"),
    ("images", "AI images (Imagen)"),
    ("compose", "Compose video"),
    ("verify", "Quality check"),
]
