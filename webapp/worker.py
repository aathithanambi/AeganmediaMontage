from __future__ import annotations

import json
import logging
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path

from bson import ObjectId
from pymongo import ReturnDocument
from pymongo.errors import PyMongoError

from webapp.config import settings
from webapp.database import get_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [worker] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("pipeline-worker")


def _claim_next_run() -> dict | None:
    db = get_db()
    run = db.pipeline_runs.find_one_and_update(
        {"status": "queued"},
        {
            "$set": {
                "status": "running",
                "startedAt": datetime.now(UTC),
                "updatedAt": datetime.now(UTC),
            }
        },
        sort=[("createdAt", 1)],
        return_document=ReturnDocument.AFTER,
    )
    if run:
        log.info(
            "Claimed run %s | pipeline=%s project=%s",
            run["_id"], run.get("pipelineName"), run.get("projectId"),
        )
    return run


def _mark_failed(run_id, message: str) -> None:
    log.error("Run %s FAILED: %s", run_id, message[:300])
    db = get_db()
    run = db.pipeline_runs.find_one_and_update(
        {"_id": run_id},
        {"$set": {"status": "failed", "error": message, "updatedAt": datetime.now(UTC)}},
    )
    if run and run.get("creditsCharged", 0) > 0 and run.get("requestedBy"):
        db.users.update_one(
            {"_id": ObjectId(run["requestedBy"])},
            {"$inc": {"credits": run["creditsCharged"]}, "$set": {"updatedAt": datetime.now(UTC)}},
        )
        log.info("Refunded %d credit(s) to user %s", run["creditsCharged"], run["requestedBy"])


def _copy_to_videos_root(source_path: str, project_id: str) -> str | None:
    src = Path(source_path)
    if not src.exists():
        log.warning("Source video not found for copy: %s", source_path)
        return None
    dest_dir = Path(settings.videos_root)
    dest_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    dest = dest_dir / f"{project_id}_{ts}{src.suffix}"
    try:
        shutil.copy2(str(src), str(dest))
        log.info("Copied video to %s", dest)
        return str(dest)
    except OSError as exc:
        log.error("Failed to copy video to videos_root: %s", exc)
        return source_path


def _mark_completed(run: dict, stdout: str, stderr: str) -> None:
    db = get_db()
    now = datetime.now(UTC)
    raw_video_path = _resolve_output_path(run, stdout)
    video_path = raw_video_path
    if raw_video_path:
        copied = _copy_to_videos_root(raw_video_path, run["projectId"])
        if copied:
            video_path = copied
    log.info(
        "Run %s COMPLETED | video=%s",
        run["_id"], video_path or "(none)",
    )
    db.pipeline_runs.update_one(
        {"_id": run["_id"]},
        {
            "$set": {
                "status": "completed",
                "stdout": stdout[-12000:],
                "stderr": stderr[-12000:],
                "completedAt": now,
                "updatedAt": now,
                "outputVideoPath": video_path,
            }
        },
    )
    if video_path:
        db.video_jobs.insert_one(
            {
                "projectId": run["projectId"],
                "title": run.get("title", run["projectId"]),
                "status": "processed",
                "videoPath": video_path,
                "videoExists": True,
                "requestedBy": run.get("requestedBy"),
                "createdAt": now,
                "updatedAt": now,
                "expiresAt": now + timedelta(hours=settings.cleanup_after_hours),
                "deletedAt": None,
            }
        )


def _resolve_output_path(run: dict, stdout: str) -> str | None:
    for line in stdout.splitlines():
        if line.startswith("OUTPUT_VIDEO="):
            raw = line.split("=", 1)[1].strip()
            if raw:
                return raw
    fallback = Path("projects") / run["projectId"] / "renders" / "final.mp4"
    return str(fallback) if fallback.exists() else None


def _execute_run(run: dict) -> None:
    cmd_template = settings.pipeline_run_command.strip()
    if not cmd_template:
        _mark_failed(
            run["_id"],
            "PIPELINE_RUN_COMMAND is empty. Configure command to execute pipeline runner/agent on this server.",
        )
        return

    prompt_dir = Path("projects") / run["projectId"] / "artifacts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = prompt_dir / "web_prompt.json"
    prompt_file.write_text(
        json.dumps(
            {
                "pipeline": run["pipelineName"],
                "projectId": run["projectId"],
                "title": run.get("title"),
                "prompt": run.get("prompt"),
                "referenceUrl": run.get("referenceUrl"),
                "requestedBy": run.get("requestedBy"),
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    command = cmd_template.format(
        pipeline=run["pipelineName"],
        project_id=run["projectId"],
        prompt_file=str(prompt_file),
    )
    log.info("Executing: %s", command)

    args = shlex.split(command)
    start_ts = time.monotonic()
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=settings.worker_timeout_seconds,
        check=False,
    )
    elapsed = round(time.monotonic() - start_ts, 1)

    log.info(
        "Command finished | exit=%d elapsed=%.1fs stdout=%d bytes stderr=%d bytes",
        proc.returncode, elapsed, len(proc.stdout), len(proc.stderr),
    )

    if proc.stdout:
        for line in proc.stdout.splitlines()[-30:]:
            log.info("  stdout> %s", line)
    if proc.stderr:
        for line in proc.stderr.splitlines()[-20:]:
            log.warning("  stderr> %s", line)

    if proc.returncode != 0:
        _mark_failed(
            run["_id"],
            f"Pipeline command failed (exit={proc.returncode}). stderr={proc.stderr[-2000:]}",
        )
        db = get_db()
        db.pipeline_runs.update_one(
            {"_id": run["_id"]},
            {
                "$set": {
                    "stdout": proc.stdout[-12000:],
                    "stderr": proc.stderr[-12000:],
                    "updatedAt": datetime.now(UTC),
                }
            },
        )
        return
    _mark_completed(run, proc.stdout, proc.stderr)


def run_forever() -> None:
    log.info("pipeline-worker started")
    log.info("poll interval: %ds", settings.worker_poll_seconds)
    log.info("PIPELINE_RUN_COMMAND: %s", settings.pipeline_run_command or "(empty)")
    log.info("VIDEOS_ROOT: %s", settings.videos_root)
    log.info("WORKER_TIMEOUT: %ds", settings.worker_timeout_seconds)

    while True:
        try:
            run = _claim_next_run()
        except PyMongoError as exc:
            log.error("Mongo connection issue while polling: %s", exc)
            time.sleep(settings.worker_poll_seconds)
            continue
        if run is None:
            time.sleep(settings.worker_poll_seconds)
            continue
        log.info("=" * 60)
        log.info("Processing run %s", run["_id"])
        try:
            _execute_run(run)
        except subprocess.TimeoutExpired:
            _mark_failed(run["_id"], "Pipeline command timed out.")
        except Exception as exc:
            log.error("Worker exception: %s\n%s", exc, traceback.format_exc())
            _mark_failed(run["_id"], f"Worker exception: {exc}")
        log.info("=" * 60)


if __name__ == "__main__":
    run_forever()
