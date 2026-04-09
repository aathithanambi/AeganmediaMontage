# AeganMediaMontage Web App Deploy Guide

## What was added

- FastAPI web app with pages:
  - `/`
  - `/login`
  - `/signup`
  - `/forgot-password`
  - `/dashboard`
  - `/admin-dashboard`
- Role model:
  - `admin` (full rights)
  - `manager` (no delete user, can update credits)
  - `user` (self profile + dashboard)
- MongoDB collections:
  - `users`
  - `video_jobs`
  - `password_reset_requests`
- Video metadata stays in DB even after file is deleted.
- Latest processed jobs are listed first.
- Auto cleanup deletes expired video files from disk after 48 hours.

## App + Admin in one container?

Yes. Admin dashboard + backend API are in one container (`aeganmediamontage-web`).

MongoDB should remain a separate service (recommended) or Atlas cluster.

## First-time DB initialization

Two controls are available:

1. `docker/mongo/init.db.js`
   - Runs once when Mongo data volume is empty (standard Mongo behavior).
2. `INIT_DB_ON_BOOT=true|false`
   - Controls app-side bootstrap/index creation and user seeding.
   - Set `true` only for first deployment, then set `false`.

## Local Docker Compose

```bash
cp .env.example .env
# Set MONGODB_URI for local mongo:
# mongodb://aegan_app_user:change-me@mongo:27017/aeganmediamontage?authSource=aeganmediamontage

docker compose up -d --build
```

App is available at: `http://127.0.0.1:51003`

## Register a processed job from pipeline

Use authenticated admin/manager session and POST:

- endpoint: `/api/jobs`
- form fields:
  - `project_id`
  - `title`
  - `video_path`
  - `status_value` (optional, default `processed`)

This stores DB details, sets TTL timestamp, and makes it visible on `/`.

