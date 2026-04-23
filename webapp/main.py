from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from bson import ObjectId
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jwt import InvalidTokenError
from pymongo.errors import PyMongoError

from webapp.bootstrap import run_bootstrap
from webapp.cleanup import cleanup_expired_videos
from webapp.config import settings
from webapp.database import get_db
from webapp.api_keys import get_api_key_status, get_api_summary
from webapp.security import create_access_token, decode_access_token, hash_password, verify_password
from webapp.pipeline_stages import PIPELINE_STAGE_ORDER

app = FastAPI(title=settings.app_name)

base_dir = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(base_dir / "templates"))
app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")
_cleanup_task: asyncio.Task | None = None

_PIPELINE_CATALOG: list[dict[str, str]] = []


def _load_pipeline_catalog() -> list[dict[str, str]]:
    """Scan pipeline_defs/*.yaml once and cache name + description + stability."""
    catalog: list[dict[str, str]] = []
    defs_dir = Path("pipeline_defs")
    if not defs_dir.is_dir():
        return catalog
    for yf in sorted(defs_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(yf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        name = data.get("name", yf.stem)
        desc = data.get("description", "")
        if isinstance(desc, str):
            desc = re.sub(r"\s+", " ", desc).strip()
        catalog.append({
            "name": name,
            "description": desc,
            "category": data.get("category", ""),
            "stability": data.get("stability", ""),
        })
    return catalog


def _current_user(request: Request) -> dict[str, Any] | None:
    token = request.cookies.get("access_token")
    if not token:
        return None
    try:
        payload = decode_access_token(token)
    except InvalidTokenError:
        return None
    try:
        user_id = ObjectId(payload["sub"])
    except Exception:
        return None
    db = get_db()
    user = db.users.find_one({"_id": user_id})
    if not user or not user.get("isActive", True):
        return None
    return user


def _require_user(request: Request) -> dict[str, Any]:
    user = _current_user(request)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return user


def _require_role(user: dict[str, Any], allowed: set[str]) -> None:
    if user.get("role") not in allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")


def _template_context(request: Request, **kwargs: Any) -> dict[str, Any]:
    user = _current_user(request)
    base = {
        "request": request,
        "current_user": user,
        "app_name": settings.app_name,
        "credit_cost": settings.credit_cost_per_run,
    }
    base.update(kwargs)
    return base


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(job["_id"]),
        "projectId": job.get("projectId", ""),
        "title": job.get("title", ""),
        "status": job.get("status", "unknown"),
        "videoExists": job.get("videoExists", False),
        "videoPath": job.get("videoPath"),
        "createdAt": job.get("createdAt"),
        "expiresAt": job.get("expiresAt"),
        "deletedAt": job.get("deletedAt"),
    }


def _public_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(run["_id"]),
        "pipelineName": run.get("pipelineName"),
        "projectId": run.get("projectId"),
        "title": run.get("title"),
        "status": run.get("status"),
        "error": run.get("error"),
        "outputVideoPath": run.get("outputVideoPath"),
        "createdAt": run.get("createdAt"),
        "startedAt": run.get("startedAt"),
        "completedAt": run.get("completedAt"),
        "elapsedSeconds": run.get("elapsedSeconds"),
        "apiUsage": run.get("apiUsage"),
        "requestedBy": run.get("requestedBy"),
    }


def _is_privileged(user: dict[str, Any]) -> bool:
    return user.get("role") in {"admin", "manager"}


@app.on_event("startup")
async def _on_startup() -> None:
    global _cleanup_task, _PIPELINE_CATALOG
    Path(settings.videos_root).mkdir(parents=True, exist_ok=True)
    if settings.init_db_on_boot:
        run_bootstrap()
    cleanup_expired_videos()
    _cleanup_task = asyncio.create_task(_cleanup_loop())
    _PIPELINE_CATALOG = _load_pipeline_catalog()


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    global _cleanup_task
    if _cleanup_task is not None:
        _cleanup_task.cancel()
        _cleanup_task = None


async def _cleanup_loop() -> None:
    while True:
        await asyncio.sleep(3600)
        cleanup_expired_videos()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    db = get_db()
    try:
        db.command("ping")
    except PyMongoError as exc:
        return JSONResponse({"status": "degraded", "db": "down", "error": str(exc)}, status_code=503)
    return JSONResponse({"status": "ok", "db": "up"})


# ---------------------------------------------------------------------------
# Public pages
# ---------------------------------------------------------------------------

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request, "home.html",
        _template_context(request, pipelines=_PIPELINE_CATALOG),
    )


@app.get("/signup")
def signup_page(request: Request):
    return templates.TemplateResponse(request, "signup.html", _template_context(request, error=None))


@app.post("/signup")
def signup(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
):
    db = get_db()
    existing = db.users.find_one({"email": email.strip().lower()})
    if existing:
        return templates.TemplateResponse(
            request, "signup.html", _template_context(request, error="Email already exists"), status_code=400
        )
    user_doc = {
        "name": email.split("@")[0].capitalize(),
        "email": email.strip().lower(),
        "passwordHash": hash_password(password),
        "role": "user",
        "credits": 0,
        "isActive": True,
        "isApproved": False,
        "createdAt": datetime.now(UTC),
        "updatedAt": datetime.now(UTC),
    }
    result = db.users.insert_one(user_doc)
    token = create_access_token(str(result.inserted_id), "user")
    response = RedirectResponse("/dashboard", status_code=303)
    response.set_cookie("access_token", token, httponly=True, samesite="lax")
    return response


@app.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", _template_context(request, error=None))


@app.post("/login")
def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
):
    db = get_db()
    user = db.users.find_one({"email": email.strip().lower()})
    if not user or not verify_password(password, user["passwordHash"]):
        return templates.TemplateResponse(
            request, "login.html", _template_context(request, error="Invalid credentials"), status_code=400
        )
    token = create_access_token(str(user["_id"]), user["role"])
    response = RedirectResponse("/dashboard", status_code=303)
    response.set_cookie("access_token", token, httponly=True, samesite="lax")
    return response


@app.post("/logout")
def logout():
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie("access_token")
    return response


@app.get("/forgot-password")
def forgot_password_page(request: Request):
    return templates.TemplateResponse(request, "forgot_password.html", _template_context(request, message=None))


@app.post("/forgot-password")
def forgot_password(request: Request, email: str = Form(...)):
    db = get_db()
    user = db.users.find_one({"email": email.strip().lower()})
    if user:
        db.password_reset_requests.insert_one(
            {
                "userId": user["_id"],
                "email": user["email"],
                "status": "pending",
                "requestedAt": datetime.now(UTC),
            }
        )
    return templates.TemplateResponse(
        request,
        "forgot_password.html",
        _template_context(
            request,
            message="Request submitted to admin. They can reset your password from admin dashboard.",
        ),
    )


# ---------------------------------------------------------------------------
# Dashboard (all authenticated users)
# ---------------------------------------------------------------------------

@app.get("/dashboard")
def dashboard(request: Request):
    user = _require_user(request)
    db = get_db()
    user_id_str = str(user["_id"])
    jobs = [
        _public_job(job)
        for job in db.video_jobs.find({"requestedBy": user_id_str}).sort("createdAt", -1).limit(50)
    ]
    runs = [
        _public_run(r)
        for r in db.pipeline_runs.find({"requestedBy": user_id_str}).sort("createdAt", -1).limit(50)
    ]
    return templates.TemplateResponse(
        request, "dashboard.html",
        _template_context(request, jobs=jobs, runs=runs, pipelines=_PIPELINE_CATALOG, settings=settings),
    )


MAX_AUDIO_SIZE = 200 * 1024 * 1024  # 200 MB (supports 20-min WAV / FLAC)
ALLOWED_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".aac", ".flac"}


@app.post("/dashboard/create-project")
async def create_project_from_form(
    request: Request,
    pipeline_name: str = Form(...),
    title: str = Form(...),
    prompt: str = Form(...),
    reference_url: str = Form(""),
    audio_language: str = Form(""),
    subtitle_language: str = Form(""),
    enable_subtitles: str = Form("no"),
    enable_watermark: str = Form("no"),
    clone_voice: str = Form("no"),
    enable_music: str = Form("no"),
    audio_file: UploadFile | None = File(None),
):
    """Form-based project creation with optional audio upload."""
    user = _require_user(request)

    pipeline_file = Path("pipeline_defs") / f"{pipeline_name}.yaml"
    if not pipeline_file.exists():
        raise HTTPException(status_code=400, detail=f"Unknown pipeline: {pipeline_name}")

    db = get_db()
    if not _is_privileged(user):
        if not user.get("isApproved", False):
            raise HTTPException(
                status_code=403,
                detail="Your account is pending approval. An admin must approve your account before you can create videos.",
            )

        credits = user.get("credits", 0)
        if credits <= 0:
            existing_runs = db.pipeline_runs.count_documents({"requestedBy": str(user["_id"])})
            if existing_runs >= settings.free_tier_max_runs:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"Free tier limit reached ({settings.free_tier_max_runs} video). "
                        f"Contact admin/manager to get credits for more videos."
                    ),
                )
        else:
            cost = settings.credit_cost_per_run
            if credits < cost:
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient credits. You need {cost} credit(s) but have {credits}.",
                )
            db.users.update_one(
                {"_id": user["_id"], "credits": {"$gte": cost}},
                {"$inc": {"credits": -cost}, "$set": {"updatedAt": datetime.now(UTC)}},
            )

    full_prompt = prompt
    if reference_url.strip():
        full_prompt = f"Reference: {reference_url.strip()}\n\n{prompt}"

    project_id = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "untitled"

    uploaded_audio_path: str | None = None
    if audio_file and audio_file.filename:
        ext = Path(audio_file.filename).suffix.lower()
        if ext not in ALLOWED_AUDIO_EXTS:
            raise HTTPException(
                status_code=400,
                detail=f"Audio format {ext} not supported. Use: {', '.join(ALLOWED_AUDIO_EXTS)}",
            )
        audio_data = await audio_file.read()
        if len(audio_data) > MAX_AUDIO_SIZE:
            raise HTTPException(status_code=400, detail="Audio file exceeds 50 MB limit")

        audio_dir = Path("projects") / project_id / "assets"
        audio_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", audio_file.filename)
        dest = audio_dir / f"uploaded_audio{ext}"
        dest.write_bytes(audio_data)
        uploaded_audio_path = str(dest)

    now = datetime.now(UTC)
    run_doc = {
        "pipelineName": pipeline_name,
        "projectId": project_id,
        "title": title,
        "prompt": full_prompt,
        "referenceUrl": reference_url.strip() or None,
        "audioLanguage": audio_language.strip() or None,
        "subtitleLanguage": subtitle_language.strip() or None,
        "enableSubtitles": enable_subtitles.strip().lower() == "yes",
        "enableWatermark": enable_watermark.strip().lower() == "yes",
        "cloneVoice": clone_voice.strip().lower() == "yes",
        "enableMusic": enable_music.strip().lower() == "yes",
        "uploadedAudioPath": uploaded_audio_path,
        "status": "queued",
        "requestedBy": str(user["_id"]),
        "creditsCharged": 0 if (_is_privileged(user) or user.get("credits", 0) <= 0) else settings.credit_cost_per_run,
        "createdAt": now,
        "updatedAt": now,
        "startedAt": None,
        "completedAt": None,
        "outputVideoPath": None,
        "error": None,
    }
    db.pipeline_runs.insert_one(run_doc)
    return RedirectResponse("/dashboard", status_code=303)


@app.post("/profile")
def update_profile(request: Request, email: str = Form(...), password: str = Form("")):
    user = _require_user(request)
    db = get_db()
    update_doc: dict[str, Any] = {"email": email.strip().lower(), "updatedAt": datetime.now(UTC)}
    if password.strip():
        update_doc["passwordHash"] = hash_password(password.strip())
    db.users.update_one({"_id": user["_id"]}, {"$set": update_doc})
    return RedirectResponse("/dashboard", status_code=303)


# ---------------------------------------------------------------------------
# Video download
# ---------------------------------------------------------------------------

def _safe_video_path(raw_path: str) -> Path | None:
    """Resolve a video path and ensure it's within an allowed directory."""
    resolved = Path(raw_path).resolve()
    allowed_roots = [
        Path(settings.videos_root).resolve(),
        Path("/app/projects").resolve(),
        Path("projects").resolve(),
    ]
    for root in allowed_roots:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    return None


@app.get("/download/{job_id}")
def download_video(request: Request, job_id: str):
    user = _require_user(request)
    db = get_db()
    job = db.video_jobs.find_one({"_id": ObjectId(job_id)})
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")

    if not _is_privileged(user) and job.get("requestedBy") != str(user["_id"]):
        raise HTTPException(status_code=403, detail="Forbidden")

    if not job.get("videoExists") or not job.get("videoPath"):
        raise HTTPException(status_code=410, detail="Video has been deleted. Metadata is still available.")

    safe_path = _safe_video_path(job["videoPath"])
    if not safe_path or not safe_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    filename = f"{job.get('projectId', 'video')}_{job.get('title', 'output')}.mp4"
    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
    return FileResponse(
        path=str(safe_path),
        media_type="video/mp4",
        filename=filename,
    )


@app.get("/download/run/{run_id}")
def download_run_video(request: Request, run_id: str):
    user = _require_user(request)
    db = get_db()
    run = db.pipeline_runs.find_one({"_id": ObjectId(run_id)})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    if not _is_privileged(user) and run.get("requestedBy") != str(user["_id"]):
        raise HTTPException(status_code=403, detail="Forbidden")

    if not run.get("outputVideoPath"):
        raise HTTPException(status_code=404, detail="No output video for this run")

    safe_path = _safe_video_path(run["outputVideoPath"])
    if not safe_path or not safe_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    filename = f"{run.get('projectId', 'video')}_{run.get('title', 'output')}.mp4"
    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
    return FileResponse(
        path=str(safe_path),
        media_type="video/mp4",
        filename=filename,
    )


@app.get("/download/run/{run_id}/images")
def download_run_images(request: Request, run_id: str):
    """Download the scene images zip for a pipeline run."""
    user = _require_user(request)
    db = get_db()
    run = db.pipeline_runs.find_one({"_id": ObjectId(run_id)})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    if not _is_privileged(user) and run.get("requestedBy") != str(user["_id"]):
        raise HTTPException(status_code=403, detail="Forbidden")

    zip_path_str = run.get("imagesZipPath")
    if not zip_path_str:
        raise HTTPException(status_code=404, detail="No images zip available for this run")

    safe_path = _safe_video_path(zip_path_str)
    if not safe_path or not safe_path.exists():
        raise HTTPException(status_code=404, detail="Images zip not found on disk")

    filename = f"{run.get('projectId', 'video')}_{run.get('title', 'output')}_images.zip"
    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
    return FileResponse(
        path=str(safe_path),
        media_type="application/zip",
        filename=filename,
    )


@app.get("/download/run/{run_id}/subtitles/en")
def download_run_subtitles_en(request: Request, run_id: str):
    """Download the English subtitle SRT file for a pipeline run."""
    user = _require_user(request)
    db = get_db()
    run = db.pipeline_runs.find_one({"_id": ObjectId(run_id)})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if not _is_privileged(user) and run.get("requestedBy") != str(user["_id"]):
        raise HTTPException(status_code=403, detail="Forbidden")

    srt_path_str = run.get("subtitlesEnPath")
    if not srt_path_str:
        raise HTTPException(status_code=404, detail="English subtitles not available for this run")

    safe_path = _safe_video_path(srt_path_str)
    if not safe_path or not safe_path.exists():
        raise HTTPException(status_code=404, detail="Subtitle file not found on disk")

    filename = f"{run.get('projectId', 'video')}_{run.get('title', 'output')}_en.srt"
    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
    return FileResponse(path=str(safe_path), media_type="text/plain", filename=filename)


@app.get("/download/run/{run_id}/subtitles/lang")
def download_run_subtitles_lang(request: Request, run_id: str):
    """Download the narration-language subtitle SRT file for a pipeline run."""
    user = _require_user(request)
    db = get_db()
    run = db.pipeline_runs.find_one({"_id": ObjectId(run_id)})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if not _is_privileged(user) and run.get("requestedBy") != str(user["_id"]):
        raise HTTPException(status_code=403, detail="Forbidden")

    srt_path_str = run.get("subtitlesLangPath")
    if not srt_path_str:
        raise HTTPException(status_code=404, detail="Narration language subtitles not available for this run")

    safe_path = _safe_video_path(srt_path_str)
    if not safe_path or not safe_path.exists():
        raise HTTPException(status_code=404, detail="Subtitle file not found on disk")

    lang = run.get("subtitlesLang") or run.get("audioLanguage") or "lang"
    filename = f"{run.get('projectId', 'video')}_{run.get('title', 'output')}_{lang}.srt"
    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
    return FileResponse(path=str(safe_path), media_type="text/plain", filename=filename)


# ---------------------------------------------------------------------------
# Run detail page + progress API + video preview
# ---------------------------------------------------------------------------

@app.get("/api/run/{run_id}/progress")
def run_progress_api(request: Request, run_id: str):
    """JSON endpoint for polling progress from the frontend."""
    user = _require_user(request)
    db = get_db()
    run = db.pipeline_runs.find_one(
        {"_id": ObjectId(run_id)},
        {"status": 1, "progress": 1, "error": 1, "outputVideoPath": 1,
         "requestedBy": 1, "elapsedSeconds": 1, "etaString": 1,
         "progressSnapshot": 1},
    )
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if not _is_privileged(user) and run.get("requestedBy") != str(user["_id"]):
        raise HTTPException(status_code=403, detail="Forbidden")

    video_ready = False
    if run.get("outputVideoPath"):
        safe = _safe_video_path(run["outputVideoPath"])
        video_ready = safe is not None and safe.exists()

    payload: dict[str, Any] = {
        "status": run.get("status", "unknown"),
        "progress": run.get("progress", 0),
        "error": run.get("error"),
        "videoReady": video_ready,
        "elapsedSeconds": run.get("elapsedSeconds", 0),
        "eta": run.get("etaString", ""),
    }
    if run.get("progressSnapshot"):
        payload["progressSnapshot"] = run["progressSnapshot"]
    return JSONResponse(payload)


@app.post("/api/run/{run_id}/cancel")
def cancel_run(request: Request, run_id: str):
    """Cancel a queued or running pipeline run."""
    user = _require_user(request)
    db = get_db()
    run = db.pipeline_runs.find_one({"_id": ObjectId(run_id)})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if not _is_privileged(user) and run.get("requestedBy") != str(user["_id"]):
        raise HTTPException(status_code=403, detail="Forbidden")
    if run["status"] not in ("queued", "running"):
        raise HTTPException(status_code=400, detail=f"Cannot cancel run in '{run['status']}' state")

    db.pipeline_runs.update_one(
        {"_id": ObjectId(run_id)},
        {"$set": {
            "status": "cancelled",
            "error": f"Cancelled by {user.get('email', 'user')}",
            "updatedAt": datetime.now(UTC),
        }},
    )

    if run.get("creditsCharged", 0) > 0 and run.get("requestedBy"):
        db.users.update_one(
            {"_id": ObjectId(run["requestedBy"])},
            {"$inc": {"credits": run["creditsCharged"]}, "$set": {"updatedAt": datetime.now(UTC)}},
        )

    return JSONResponse({"status": "cancelled", "refunded": run.get("creditsCharged", 0)})


@app.get("/preview/run/{run_id}")
def preview_run_video(request: Request, run_id: str):
    """Serve video for in-browser preview (same auth as download)."""
    user = _require_user(request)
    db = get_db()
    run = db.pipeline_runs.find_one({"_id": ObjectId(run_id)})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if not _is_privileged(user) and run.get("requestedBy") != str(user["_id"]):
        raise HTTPException(status_code=403, detail="Forbidden")
    if not run.get("outputVideoPath"):
        raise HTTPException(status_code=404, detail="No output video")

    safe_path = _safe_video_path(run["outputVideoPath"])
    if not safe_path or not safe_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    return FileResponse(path=str(safe_path), media_type="video/mp4")


@app.get("/dashboard/run/{run_id}")
def run_detail(request: Request, run_id: str):
    user = _require_user(request)
    db = get_db()
    run = db.pipeline_runs.find_one({"_id": ObjectId(run_id)})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    if not _is_privileged(user) and run.get("requestedBy") != str(user["_id"]):
        raise HTTPException(status_code=403, detail="Forbidden")

    video_available = False
    if run.get("outputVideoPath"):
        safe = _safe_video_path(run["outputVideoPath"])
        video_available = safe is not None and safe.exists()

    images_zip_available = False
    if run.get("imagesZipPath"):
        safe_zip = _safe_video_path(run["imagesZipPath"])
        images_zip_available = safe_zip is not None and safe_zip.exists()

    # Subtitle download availability
    subtitles_en_available = False
    if run.get("subtitlesEnPath"):
        sp = _safe_video_path(run["subtitlesEnPath"])
        subtitles_en_available = sp is not None and sp.exists()

    subtitles_lang_available = False
    audio_lang = (run.get("audioLanguage") or "english").strip().lower()
    # Resolve the display name for the narration language button.
    # Always derive it from audioLanguage so the grayed placeholder shows even
    # before the SRT file has been generated.
    if audio_lang and audio_lang not in ("english", "en", ""):
        subtitles_lang_name = (run.get("subtitlesLang") or audio_lang).capitalize()
        if run.get("subtitlesLangPath"):
            sp2 = _safe_video_path(run["subtitlesLangPath"])
            subtitles_lang_available = sp2 is not None and sp2.exists()
    else:
        subtitles_lang_name = ""   # English projects: no separate narration-language button

    return templates.TemplateResponse(
        request, "run_detail.html",
        _template_context(
            request,
            run=run,
            video_available=video_available,
            images_zip_available=images_zip_available,
            subtitles_en_available=subtitles_en_available,
            subtitles_lang_available=subtitles_lang_available,
            subtitles_lang_name=subtitles_lang_name,
            pipeline_stages=PIPELINE_STAGE_ORDER,
        ),
    )


# ---------------------------------------------------------------------------
# Admin dashboard
# ---------------------------------------------------------------------------

def _get_usage_stats(db) -> dict[str, Any]:
    """Aggregate API usage stats for admin dashboard."""
    now = datetime.now(UTC)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    def _agg(match_filter: dict) -> dict:
        pipeline = [
            {"$match": match_filter},
            {"$group": {
                "_id": None,
                "total_gemini": {"$sum": "$gemini_calls"},
                "total_imagen": {"$sum": "$imagen_calls"},
                "total_tts": {"$sum": "$tts_calls"},
                "total_tts_chars": {"$sum": "$tts_characters"},
                "total_cost": {"$sum": "$estimated_cost_usd"},
                "run_count": {"$sum": 1},
            }},
        ]
        results = list(db.api_usage.aggregate(pipeline))
        if results:
            r = results[0]
            return {
                "gemini_calls": r.get("total_gemini", 0),
                "imagen_calls": r.get("total_imagen", 0),
                "tts_calls": r.get("total_tts", 0),
                "tts_characters": r.get("total_tts_chars", 0),
                "estimated_cost": round(r.get("total_cost", 0), 4),
                "run_count": r.get("run_count", 0),
            }
        return {"gemini_calls": 0, "imagen_calls": 0, "tts_calls": 0,
                "tts_characters": 0, "estimated_cost": 0, "run_count": 0}

    return {
        "today": _agg({"createdAt": {"$gte": today_start}}),
        "month": _agg({"createdAt": {"$gte": month_start}}),
        "all_time": _agg({}),
    }


def _get_per_user_usage(db) -> list[dict[str, Any]]:
    """Get API usage breakdown per user."""
    pipeline = [
        {"$group": {
            "_id": "$requestedBy",
            "total_gemini": {"$sum": "$gemini_calls"},
            "total_imagen": {"$sum": "$imagen_calls"},
            "total_tts": {"$sum": "$tts_calls"},
            "total_tts_chars": {"$sum": "$tts_characters"},
            "total_cost": {"$sum": "$estimated_cost_usd"},
            "run_count": {"$sum": 1},
            "last_used": {"$max": "$createdAt"},
        }},
        {"$sort": {"total_cost": -1}},
    ]
    results = list(db.api_usage.aggregate(pipeline))

    user_ids = [ObjectId(r["_id"]) for r in results if r["_id"]]
    users_map = {}
    if user_ids:
        for u in db.users.find({"_id": {"$in": user_ids}}, {"email": 1, "name": 1, "role": 1, "credits": 1}):
            users_map[str(u["_id"])] = u

    enriched = []
    for r in results:
        uid = r["_id"]
        user_info = users_map.get(uid, {})
        enriched.append({
            "user_id": uid,
            "email": user_info.get("email", "unknown"),
            "name": user_info.get("name", ""),
            "role": user_info.get("role", "user"),
            "credits": user_info.get("credits", 0),
            "gemini_calls": r.get("total_gemini", 0),
            "imagen_calls": r.get("total_imagen", 0),
            "tts_calls": r.get("total_tts", 0),
            "tts_characters": r.get("total_tts_chars", 0),
            "estimated_cost": round(r.get("total_cost", 0), 4),
            "run_count": r.get("run_count", 0),
            "last_used": r.get("last_used"),
        })
    return enriched


@app.get("/admin-dashboard")
def admin_dashboard(request: Request):
    user = _require_user(request)
    _require_role(user, {"admin", "manager"})
    db = get_db()
    users = list(db.users.find().sort("createdAt", -1))
    reset_requests = list(db.password_reset_requests.find({"status": "pending"}).sort("requestedAt", -1))
    all_runs = [_public_run(r) for r in db.pipeline_runs.find().sort("createdAt", -1).limit(200)]
    api_keys = get_api_key_status()
    api_summary = get_api_summary()
    usage_stats = _get_usage_stats(db)
    per_user_usage = _get_per_user_usage(db)
    return templates.TemplateResponse(
        request,
        "admin_dashboard.html",
        _template_context(
            request, users=users, reset_requests=reset_requests,
            all_runs=all_runs, api_keys=api_keys, api_summary=api_summary,
            usage_stats=usage_stats, per_user_usage=per_user_usage,
            settings=settings,
        ),
    )


@app.post("/admin/users/{user_id}/role")
def update_user_role(request: Request, user_id: str, role: str = Form(...)):
    actor = _require_user(request)
    _require_role(actor, {"admin"})
    if role not in {"admin", "manager", "user"}:
        raise HTTPException(status_code=400, detail="Invalid role")
    db = get_db()
    db.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"role": role, "updatedAt": datetime.now(UTC)}})
    return RedirectResponse("/admin-dashboard", status_code=303)


@app.post("/admin/users/{user_id}/approve")
def approve_user(request: Request, user_id: str):
    actor = _require_user(request)
    _require_role(actor, {"admin"})
    db = get_db()
    db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"isApproved": True, "updatedAt": datetime.now(UTC)}},
    )
    return RedirectResponse("/admin-dashboard", status_code=303)


@app.post("/admin/users/{user_id}/reject")
def reject_user(request: Request, user_id: str):
    actor = _require_user(request)
    _require_role(actor, {"admin"})
    db = get_db()
    db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"isApproved": False, "updatedAt": datetime.now(UTC)}},
    )
    return RedirectResponse("/admin-dashboard", status_code=303)


@app.post("/admin/users/{user_id}/credits")
def update_user_credits(request: Request, user_id: str, credits: int = Form(...)):
    actor = _require_user(request)
    _require_role(actor, {"admin", "manager"})
    db = get_db()
    db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"credits": max(0, credits), "updatedAt": datetime.now(UTC)}},
    )
    return RedirectResponse("/admin-dashboard", status_code=303)


@app.post("/admin/users/{user_id}/delete")
def delete_user(request: Request, user_id: str):
    actor = _require_user(request)
    _require_role(actor, {"admin"})
    if str(actor["_id"]) == user_id:
        raise HTTPException(status_code=400, detail="Admin cannot delete self")
    db = get_db()
    db.users.delete_one({"_id": ObjectId(user_id)})
    return RedirectResponse("/admin-dashboard", status_code=303)


@app.post("/admin/password-requests/{request_id}/resolve")
def resolve_password_request(request: Request, request_id: str, status_value: str = Form(...)):
    actor = _require_user(request)
    _require_role(actor, {"admin"})
    if status_value not in {"approved", "rejected"}:
        raise HTTPException(status_code=400, detail="Invalid status")
    db = get_db()
    db.password_reset_requests.update_one(
        {"_id": ObjectId(request_id)},
        {"$set": {"status": status_value, "handledBy": str(actor["_id"]), "handledAt": datetime.now(UTC)}},
    )
    return RedirectResponse("/admin-dashboard", status_code=303)


@app.post("/admin/run-cleanup")
def run_cleanup(request: Request):
    actor = _require_user(request)
    _require_role(actor, {"admin", "manager"})
    cleaned = cleanup_expired_videos()
    return JSONResponse({"cleaned": cleaned})


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/pipelines")
def list_pipelines():
    return JSONResponse({"items": _PIPELINE_CATALOG})


@app.post("/api/jobs")
def create_job(
    request: Request,
    project_id: str = Form(...),
    title: str = Form(...),
    video_path: str = Form(...),
    status_value: str = Form("processed"),
):
    actor = _require_user(request)
    _require_role(actor, {"admin", "manager"})
    now = datetime.now(UTC)
    db = get_db()
    job_doc = {
        "projectId": project_id,
        "title": title,
        "status": status_value,
        "videoPath": video_path,
        "videoExists": True,
        "requestedBy": str(actor["_id"]),
        "createdAt": now,
        "updatedAt": now,
        "expiresAt": now + timedelta(hours=settings.cleanup_after_hours),
        "deletedAt": None,
    }
    result = db.video_jobs.insert_one(job_doc)
    return JSONResponse({"id": str(result.inserted_id)})


@app.post("/api/pipeline-runs")
def enqueue_pipeline_run(
    request: Request,
    pipeline_name: str = Form(...),
    project_id: str = Form(...),
    title: str = Form(...),
    prompt: str = Form(...),
    reference_url: str = Form(""),
):
    actor = _require_user(request)

    pipeline_file = Path("pipeline_defs") / f"{pipeline_name}.yaml"
    if not pipeline_file.exists():
        raise HTTPException(status_code=400, detail=f"Unknown pipeline: {pipeline_name}")

    db = get_db()
    if not _is_privileged(actor):
        if not actor.get("isApproved", False):
            raise HTTPException(
                status_code=403,
                detail="Account pending admin approval.",
            )
        credits = actor.get("credits", 0)
        if credits <= 0:
            existing_runs = db.pipeline_runs.count_documents({"requestedBy": str(actor["_id"])})
            if existing_runs >= settings.free_tier_max_runs:
                raise HTTPException(
                    status_code=403,
                    detail=f"Free tier limit reached ({settings.free_tier_max_runs} video).",
                )
        else:
            cost = settings.credit_cost_per_run
            if credits < cost:
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient credits ({credits}/{cost}).",
                )
            db.users.update_one(
                {"_id": actor["_id"], "credits": {"$gte": cost}},
                {"$inc": {"credits": -cost}, "$set": {"updatedAt": datetime.now(UTC)}},
            )

    full_prompt = prompt
    if reference_url.strip():
        full_prompt = f"Reference: {reference_url.strip()}\n\n{prompt}"

    now = datetime.now(UTC)
    run_doc = {
        "pipelineName": pipeline_name,
        "projectId": project_id,
        "title": title,
        "prompt": full_prompt,
        "referenceUrl": reference_url.strip() or None,
        "status": "queued",
        "requestedBy": str(actor["_id"]),
        "creditsCharged": 0 if _is_privileged(actor) else settings.credit_cost_per_run,
        "createdAt": now,
        "updatedAt": now,
        "startedAt": None,
        "completedAt": None,
        "outputVideoPath": None,
        "error": None,
    }
    result = db.pipeline_runs.insert_one(run_doc)
    return JSONResponse({"id": str(result.inserted_id), "status": "queued"})


@app.get("/api/pipeline-runs")
def list_pipeline_runs(request: Request):
    actor = _require_user(request)
    db = get_db()
    if _is_privileged(actor):
        query: dict[str, Any] = {}
    else:
        query = {"requestedBy": str(actor["_id"])}
    runs = [_public_run(r) for r in db.pipeline_runs.find(query).sort("createdAt", -1).limit(100)]
    return JSONResponse({"items": runs})
