from __future__ import annotations

from datetime import UTC, datetime

from pymongo import ASCENDING, DESCENDING

from webapp.config import settings
from webapp.database import get_db
from webapp.security import hash_password


def _ensure_indexes() -> None:
    db = get_db()
    db.users.create_index([("email", ASCENDING)], unique=True)
    db.video_jobs.create_index([("createdAt", DESCENDING)])
    db.video_jobs.create_index([("expiresAt", ASCENDING)])
    db.password_reset_requests.create_index([("requestedAt", DESCENDING)])
    db.pipeline_runs.create_index([("createdAt", DESCENDING)])
    db.pipeline_runs.create_index([("status", ASCENDING), ("createdAt", ASCENDING)])
    db.api_usage.create_index([("createdAt", DESCENDING)])
    db.api_usage.create_index([("requestedBy", ASCENDING)])


def _seed_user(email: str, password: str, role: str, name: str = "") -> None:
    db = get_db()
    if db.users.find_one({"email": email}):
        return
    is_privileged = role in ("admin", "manager")
    db.users.insert_one(
        {
            "name": name or role.capitalize(),
            "email": email,
            "passwordHash": hash_password(password),
            "role": role,
            "credits": 0,
            "isActive": True,
            "isApproved": is_privileged,
            "createdAt": datetime.now(UTC),
            "updatedAt": datetime.now(UTC),
        }
    )


def run_bootstrap() -> None:
    _ensure_indexes()
    if settings.init_seed_users:
        _seed_user(settings.seed_admin_email, settings.seed_admin_password, "admin", "Admin")
        _seed_user(settings.seed_manager_email, settings.seed_manager_password, "manager", "Manager")
        _seed_user(settings.seed_user_email, settings.seed_user_password, "user", "User")


if __name__ == "__main__":
    run_bootstrap()
    print("Bootstrap completed.")

