from __future__ import annotations

import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

from webapp.database import get_db

log = logging.getLogger(__name__)


def _delete_project_folder(video_path: str) -> None:
    """Delete the entire project folder that contains the video.

    Project layout: projects/<id>/renders/final.mp4
    We walk up from the video to find the project root and remove it entirely.
    """
    vp = Path(video_path).resolve()
    # renders/final.mp4 -> renders -> project_dir
    project_dir = vp.parent.parent if vp.parent.name == "renders" else vp.parent
    # Safety: only delete if it looks like a project folder inside "projects/"
    try:
        rel = project_dir.relative_to(Path("projects").resolve())
        if rel.parts:
            shutil.rmtree(str(project_dir), ignore_errors=True)
            log.info("Deleted project folder: %s", project_dir)
    except (ValueError, OSError):
        # Not inside projects/ — just delete the single file
        if vp.exists():
            vp.unlink(missing_ok=True)


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
            _delete_project_folder(video_path)

        # Also clean imagesZipPath if stored separately
        zip_path = job.get("imagesZipPath")
        if zip_path:
            zp = Path(zip_path)
            if zp.exists():
                zp.unlink(missing_ok=True)

        db.video_jobs.update_one(
            {"_id": job["_id"]},
            {
                "$set": {
                    "videoExists": False,
                    "videoPath": None,
                    "imagesZipPath": None,
                    "status": "deleted",
                    "deletedAt": now,
                    "updatedAt": now,
                }
            },
        )

        # Also clear zip path on the pipeline_runs record
        if job.get("projectId"):
            db.pipeline_runs.update_many(
                {"projectId": job["projectId"]},
                {"$set": {"imagesZipPath": None, "updatedAt": now}},
            )

        cleaned_count += 1
    return cleaned_count

