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
  - `pipeline_runs`
- Video metadata stays in DB even after file is deleted.
- Latest processed jobs are listed first.
- Auto cleanup deletes expired video files from disk after 48 hours.

## App + Admin in one container?

Yes. Admin dashboard + backend API are in one container (`aeganmediamontage-web`).

MongoDB should remain a separate service (recommended) or Atlas cluster.

## Pipeline worker

`pipeline-worker` is added as a dedicated process/container (`aeganmediamontage-worker`).

- It polls `pipeline_runs` queue in MongoDB Atlas.
- It executes `PIPELINE_RUN_COMMAND`.
- On success, it writes output video metadata into `video_jobs`.
- It runs on your server, not on GitHub-hosted runners.

### Queue endpoints

- `POST /api/pipeline-runs` (admin/manager) enqueue a run
  - fields: `pipeline_name`, `project_id`, `title`, `prompt`
- `GET /api/pipeline-runs` list recent runs (latest first)


### Important

You must configure `PIPELINE_RUN_COMMAND` in `.env`/GitHub secrets.
This command should call your pipeline execution entrypoint on the server.
Supported placeholders in command:

- `{pipeline}`
- `{project_id}`
- `{prompt_file}` (JSON payload saved by worker)

## First-time DB initialization

Use `INIT_DB_ON_BOOT=true|false`:

- `true` on first deployment to create indexes and seed admin/manager users.
- `false` after first successful startup.

## Local Docker Compose

```bash
cp .env.example .env
# Set MONGODB_URI to your Atlas connection string.

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

