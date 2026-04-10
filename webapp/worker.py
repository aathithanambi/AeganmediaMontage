from __future__ import annotations

import json
import shlex
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pymongo import ReturnDocument
from pymongo.errors import PyMongoError

from webapp.config import settings
from webapp.database import get_db


def _claim_next_run() -> dict | None:
    db = get_db()
    return db.pipeline_runs.find_one_and_update(
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


def _mark_failed(run_id, message: str) -> None:
    db = get_db()
    db.pipeline_runs.update_one(
        {"_id": run_id},
        {"$set": {"status": "failed", "error": message, "updatedAt": datetime.now(UTC)}},
    )


def _mark_completed(run: dict, stdout: str, stderr: str) -> None:
    db = get_db()
    now = datetime.now(UTC)
    video_path = _resolve_output_path(run, stdout)
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
    if not settings.pipeline_run_command.strip():
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
                "requestedBy": run.get("requestedBy"),
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    command = settings.pipeline_run_command.format(
        pipeline=run["pipelineName"],
        project_id=run["projectId"],
        prompt_file=str(prompt_file),
    )
    args = shlex.split(command)
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=settings.worker_timeout_seconds,
        check=False,
    )
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
    print("pipeline-worker started")
    print(f"poll interval: {settings.worker_poll_seconds}s")
    while True:
        try:
            run = _claim_next_run()
        except PyMongoError as exc:
            print(f"Mongo connection issue while polling queue: {exc}")
            time.sleep(settings.worker_poll_seconds)
            continue
        if run is None:
            time.sleep(settings.worker_poll_seconds)
            continue
        try:
            _execute_run(run)
        except subprocess.TimeoutExpired:
            _mark_failed(run["_id"], "Pipeline command timed out.")
        except Exception as exc:  # noqa: BLE001
            _mark_failed(run["_id"], f"Worker exception: {exc}")


if __name__ == "__main__":
    run_forever()

