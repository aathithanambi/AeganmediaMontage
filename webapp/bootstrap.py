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


def _seed_user(email: str, password: str, role: str) -> None:
    db = get_db()
    if db.users.find_one({"email": email}):
        return
    db.users.insert_one(
        {
            "email": email,
            "passwordHash": hash_password(password),
            "role": role,
            "credits": 0,
            "isActive": True,
            "createdAt": datetime.now(UTC),
            "updatedAt": datetime.now(UTC),
        }
    )


def run_bootstrap() -> None:
    _ensure_indexes()
    if settings.init_seed_users:
        _seed_user(settings.seed_admin_email, settings.seed_admin_password, "admin")
        _seed_user(settings.seed_manager_email, settings.seed_manager_password, "manager")


if __name__ == "__main__":
    run_bootstrap()
    print("Bootstrap completed.")

