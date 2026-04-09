from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from webapp.database import get_db


def cleanup_expired_videos() -> int:
    db = get_db()
    now = datetime.now(UTC)
    expired_jobs = db.video_jobs.find(
        {
            "expiresAt": {"$lte": now},
            "videoExists": True,
        }
    )
    cleaned_count = 0
    for job in expired_jobs:
        video_path = job.get("videoPath")
        if video_path:
            path = Path(video_path)
            if path.exists() and path.is_file():
                path.unlink(missing_ok=True)
        db.video_jobs.update_one(
            {"_id": job["_id"]},
            {
                "$set": {
                    "videoExists": False,
                    "videoPath": None,
                    "status": "deleted",
                    "deletedAt": now,
                    "updatedAt": now,
                }
            },
        )
        cleaned_count += 1
    return cleaned_count

