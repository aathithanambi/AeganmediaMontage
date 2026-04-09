from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from bson import ObjectId
from fastapi import FastAPI, Form, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jwt import InvalidTokenError

from webapp.bootstrap import run_bootstrap
from webapp.cleanup import cleanup_expired_videos
from webapp.config import settings
from webapp.database import get_db
from webapp.security import create_access_token, decode_access_token, hash_password, verify_password

app = FastAPI(title=settings.app_name)

base_dir = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(base_dir / "templates"))
app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")
_cleanup_task: asyncio.Task | None = None


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
    base = {"request": request, "current_user": user, "app_name": settings.app_name}
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
    }


@app.on_event("startup")
async def _on_startup() -> None:
    global _cleanup_task
    Path(settings.videos_root).mkdir(parents=True, exist_ok=True)
    if settings.init_db_on_boot:
        run_bootstrap()
    cleanup_expired_videos()
    _cleanup_task = asyncio.create_task(_cleanup_loop())


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


@app.get("/")
def home(request: Request):
    db = get_db()
    jobs = [_public_job(job) for job in db.video_jobs.find().sort("createdAt", -1).limit(100)]
    return templates.TemplateResponse("home.html", _template_context(request, jobs=jobs))


@app.get("/signup")
def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", _template_context(request, error=None))


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
            "signup.html", _template_context(request, error="Email already exists"), status_code=400
        )
    user_doc = {
        "email": email.strip().lower(),
        "passwordHash": hash_password(password),
        "role": "user",
        "credits": 0,
        "isActive": True,
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
    return templates.TemplateResponse("login.html", _template_context(request, error=None))


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
            "login.html", _template_context(request, error="Invalid credentials"), status_code=400
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
    return templates.TemplateResponse("forgot_password.html", _template_context(request, message=None))


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
        "forgot_password.html",
        _template_context(
            request,
            message="Request submitted to admin. They can reset your password from admin dashboard.",
        ),
    )


@app.get("/dashboard")
def dashboard(request: Request):
    user = _require_user(request)
    db = get_db()
    jobs = [
        _public_job(job)
        for job in db.video_jobs.find({"requestedBy": str(user["_id"])}).sort("createdAt", -1).limit(50)
    ]
    return templates.TemplateResponse("dashboard.html", _template_context(request, jobs=jobs))


@app.post("/profile")
def update_profile(request: Request, email: str = Form(...), password: str = Form("")):
    user = _require_user(request)
    db = get_db()
    update_doc: dict[str, Any] = {"email": email.strip().lower(), "updatedAt": datetime.now(UTC)}
    if password.strip():
        update_doc["passwordHash"] = hash_password(password.strip())
    db.users.update_one({"_id": user["_id"]}, {"$set": update_doc})
    return RedirectResponse("/dashboard", status_code=303)


@app.get("/admin-dashboard")
def admin_dashboard(request: Request):
    user = _require_user(request)
    _require_role(user, {"admin", "manager"})
    db = get_db()
    users = list(db.users.find().sort("createdAt", -1))
    reset_requests = list(db.password_reset_requests.find({"status": "pending"}).sort("requestedAt", -1))
    return templates.TemplateResponse(
        "admin_dashboard.html",
        _template_context(request, users=users, reset_requests=reset_requests),
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
):
    actor = _require_user(request)
    _require_role(actor, {"admin", "manager"})

    pipeline_file = Path("pipeline_defs") / f"{pipeline_name}.yaml"
    if not pipeline_file.exists():
        raise HTTPException(status_code=400, detail=f"Unknown pipeline: {pipeline_name}")

    db = get_db()
    now = datetime.now(UTC)
    run_doc = {
        "pipelineName": pipeline_name,
        "projectId": project_id,
        "title": title,
        "prompt": prompt,
        "status": "queued",
        "requestedBy": str(actor["_id"]),
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
    _require_role(actor, {"admin", "manager"})
    db = get_db()
    runs = [_public_run(r) for r in db.pipeline_runs.find().sort("createdAt", -1).limit(100)]
    return JSONResponse({"items": runs})

