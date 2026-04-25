from __future__ import annotations

import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

from webapp.config import settings
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


def _remove_tree_contents(dir_path: Path) -> int:
    """Delete all children of a directory, preserving the directory itself."""
    if not dir_path.exists() or not dir_path.is_dir():
        return 0
    removed = 0
    for child in dir_path.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(str(child), ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
            removed += 1
        except OSError:
            continue
    return removed


def cleanup_all_project_files() -> dict[str, int]:
    """Purge generated project files to reclaim disk.

    Safety guard: skips currently queued/running project IDs so active jobs are
    not destroyed mid-run.
    """
    db = get_db()
    now = datetime.now(UTC)

    active_project_ids = set(
        pid
        for pid in db.pipeline_runs.distinct(
            "projectId", {"status": {"$in": ["queued", "running"]}}
        )
        if isinstance(pid, str) and pid
    )

    projects_root = Path("projects").resolve()
    videos_root = Path(settings.videos_root).resolve()
    projects_root.mkdir(parents=True, exist_ok=True)
    videos_root.mkdir(parents=True, exist_ok=True)

    removed_project_dirs = 0
    skipped_project_dirs = 0
    removed_video_files = 0

    for project_dir in projects_root.iterdir():
        if not project_dir.is_dir():
            continue
        if project_dir.name in active_project_ids:
            skipped_project_dirs += 1
            continue
        try:
            shutil.rmtree(str(project_dir), ignore_errors=True)
            removed_project_dirs += 1
        except OSError:
            continue

    for media_file in videos_root.iterdir():
        if media_file.is_dir():
            removed_video_files += _remove_tree_contents(media_file)
            continue
        file_name = media_file.name
        if any(file_name.startswith(f"{pid}_") for pid in active_project_ids):
            continue
        try:
            media_file.unlink(missing_ok=True)
            removed_video_files += 1
        except OSError:
            continue

    run_filter: dict[str, object] = {}
    if active_project_ids:
        run_filter = {"projectId": {"$nin": list(active_project_ids)}}

    job_update = db.video_jobs.update_many(
        {**run_filter, "videoExists": True},
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
    run_update = db.pipeline_runs.update_many(
        run_filter,
        {
            "$set": {
                "outputVideoPath": None,
                "imagesZipPath": None,
                "subtitlesEnPath": None,
                "subtitlesLangPath": None,
                "updatedAt": now,
            }
        },
    )

    return {
        "active_projects_skipped": len(active_project_ids),
        "project_dirs_removed": removed_project_dirs,
        "project_dirs_skipped": skipped_project_dirs,
        "videos_removed": removed_video_files,
        "video_jobs_marked_deleted": job_update.modified_count,
        "pipeline_runs_updated": run_update.modified_count,
    }

