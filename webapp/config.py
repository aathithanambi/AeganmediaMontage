from __future__ import annotations

import os
from dataclasses import dataclass


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "AeganMediaMontage")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "41006"))
    jwt_secret: str = os.getenv("JWT_SECRET", "change-this-in-production")
    jwt_exp_minutes: int = int(os.getenv("JWT_EXP_MINUTES", "1440"))

    mongodb_uri: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    mongodb_db: str = os.getenv("MONGODB_DB", "aeganmediamontage")

    videos_root: str = os.getenv("VIDEOS_ROOT", "/var/aeganmediamontage/videos")
    cleanup_after_hours: int = int(os.getenv("VIDEO_RETENTION_HOURS", "48"))
    init_db_on_boot: bool = _to_bool(os.getenv("INIT_DB_ON_BOOT", "false"))
    init_seed_users: bool = _to_bool(os.getenv("INIT_SEED_USERS", "true"))

    seed_admin_email: str = os.getenv("SEED_ADMIN_EMAIL", "admin@aegan.local")
    seed_admin_password: str = os.getenv("SEED_ADMIN_PASSWORD", "Admin@123")
    seed_manager_email: str = os.getenv("SEED_MANAGER_EMAIL", "manager@aegan.local")
    seed_manager_password: str = os.getenv("SEED_MANAGER_PASSWORD", "Manager@123")

    worker_poll_seconds: int = int(os.getenv("WORKER_POLL_SECONDS", "5"))
    worker_timeout_seconds: int = int(os.getenv("WORKER_TIMEOUT_SECONDS", "10800"))
    # command mode executes subprocess without shell; placeholders are resolved by worker.
    # supported placeholders: {pipeline}, {project_id}, {prompt_file}
    pipeline_run_command: str = os.getenv("PIPELINE_RUN_COMMAND", "")


settings = Settings()

