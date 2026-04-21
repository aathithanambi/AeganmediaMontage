# AeganMedia Montage

**AI-powered video production platform using Google AI services.**

Create professional videos from audio files, reference videos, or text prompts. The platform uses Google Gemini, Google Imagen, and Google Cloud TTS to handle everything from script writing to final video composition.

---

## What It Does

1. **Audio-to-Video** — Upload your narration audio and the AI transcribes it sentence-by-sentence, extracts characters and scenes, generates matching visuals, and composes a video with audio-synced timing.

2. **Reference Video Matching** — Paste a YouTube URL and the AI analyzes the art style, editing approach, color palette, and mood to create similar content with your own topic.

3. **Text-to-Video** — Describe what you want and the AI writes the script, generates narration, creates images, and composes the final video.

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **AI Engine** | Google Gemini 2.0 Flash | Script writing, scene planning, audio transcription, character extraction, intent parsing, subtitle translation, post-creation verification |
| **Image Generation** | Google Imagen 3.0 | Character-consistent AI images with art style matching |
| **Voice Generation** | Google Cloud TTS | Professional narration in 30+ languages (700+ voices) |
| **Video Composition** | FFmpeg | Ken Burns motion, crossfade transitions, subtitle burn-in, audio mixing |
| **Reference Analysis** | yt-dlp + Gemini | Download and analyze reference videos for style matching |
| **Web Framework** | FastAPI + Jinja2 | Dashboard, project management, user auth |
| **Database** | MongoDB Atlas | User accounts, pipeline runs, video metadata |
| **Deployment** | Docker + GitHub Actions | Automated deployment to Ubuntu server |

---

## Pipeline Workflow

```
Audio/Prompt Input
    |
    v
Phase 0: Reference Video Analysis
    - Download reference video (yt-dlp)
    - Analyze scenes, transcript, keyframes
    - Identify art style, image type, color palette, mood
    |
    v
Phase 1: Production Intent Parsing (Gemini)
    - Separate instructions from creative content
    - Detect audio language, subtitle language, target duration
    - Identify if reference-driven
    |
    v
Phase 1b: Audio Transcription with Timestamps (Gemini)
    - Sentence-by-sentence transcription
    - Start/end time per sentence
    - Duration per sentence for image timing
    |
    v
Phase 2: Script Generation (Gemini)
    - Use transcript as script (if custom audio)
    - Or generate script from prompt + reference
    - Support for 30+ languages
    |
    v
Phase 2b: Character & Scene Extraction (Gemini)
    - Identify characters with detailed visual descriptions
    - Identify locations with environmental details
    - Map each story beat to characters, locations, actions
    |
    v
Phase 3: Scene Plan with Character Consistency (Gemini)
    - Detailed image prompts with same character descriptions
    - Art style from reference applied to every prompt
    - Per-scene duration matched to audio sentence timing
    - Transition suggestions per scene
    |
    v
Phase 4: TTS Narration (Google Cloud TTS)
    - Skip if custom audio uploaded
    - Generate in detected/selected language
    |
    v
Phase 5: Image Generation (Google Imagen)
    - Character-consistent prompts
    - Art style from reference
    - 16:9 aspect ratio for video
    |
    v
Phase 6: Video Composition (FFmpeg)
    - Ken Burns motion per scene
    - Audio-synced image durations
    - Crossfade transitions between scenes
    - Optional subtitle burn-in
    |
    v
Phase 7: Post-Creation Verification
    - Video duration check
    - Audio/video sync validation
    - File size validation
```

---

## Features

### Audio-Synced Image Timing
Each image duration matches its corresponding sentence in the audio. No more generic 6-second-per-image slideshows — the visuals change exactly when the narration moves to the next topic.

### Character Consistency
Gemini extracts character names, physical descriptions, and roles from the transcript. Every image prompt includes the exact same character description, so characters look similar across all scenes.

### Reference Style Matching
When you provide a reference video URL, Gemini analyzes:
- **Art style** — oil painting, realistic photography, anime, watercolor, etc.
- **Image type** — original photos, AI-generated, hand-drawn, mixed media
- **Editing style** — slow dissolves, quick cuts, ken burns, zoom transitions
- **Color palette** — warm earthy, cool cinematic, vibrant saturated, dark moody
- **Mood** — dramatic, cheerful, nostalgic, educational, epic

This style information is embedded into every image generation prompt.

### Multi-Language Support
- **Narration**: 30+ languages via Google Cloud TTS
- **Indian languages**: Tamil, Hindi, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi, Urdu
- **International**: Spanish, French, German, Japanese, Korean, Arabic, Portuguese, and more
- **Subtitles**: Optional, translated to any supported language via Gemini

### Optional Controls
- **Subtitles**: Default off. Enable and choose a translation language.
- **Background Music**: Default off. (Coming soon)
- **SFX**: Default off. (Coming soon)

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **FFmpeg** — `sudo apt install ffmpeg` or [ffmpeg.org](https://ffmpeg.org/download.html)
- **Google API Key** — Get free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### Required Google Cloud APIs

Enable these in your [Google Cloud Console](https://console.cloud.google.com/apis/library):

1. **Generative Language API** — Gemini for script, scenes, transcription
2. **Cloud Text-to-Speech API** — Voice narration generation
3. **Imagen API** — AI image generation (via Gemini endpoint)

### Setup

```bash
git clone https://github.com/yourusername/AeganmediaMontage.git
cd AeganmediaMontage
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Run Locally

```bash
# Start the web app
python -m uvicorn webapp.main:app --host 0.0.0.0 --port 41006

# In another terminal, start the worker
python -m webapp.worker
```

Visit [http://localhost:41006](http://localhost:41006)

### Docker

```bash
docker build -t aeganmediamontage .

# Web app
docker run -d --name web \
  --env-file .env \
  -p 41006:41006 \
  -v /var/aeganmediamontage/videos:/var/aeganmediamontage/videos \
  aeganmediamontage

# Worker
docker run -d --name worker \
  --env-file .env \
  -v /var/aeganmediamontage/videos:/var/aeganmediamontage/videos \
  aeganmediamontage python -m webapp.worker
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Google AI Studio API key — powers Gemini, Imagen, Cloud TTS |
| `MONGODB_URI` | Yes | MongoDB Atlas connection string |
| `JWT_SECRET` | Yes | Secret for JWT token signing |
| `APP_NAME` | No | Application name (default: AeganMediaMontage) |
| `PORT` | No | Server port (default: 41006) |
| `VIDEOS_ROOT` | No | Video storage path (default: /var/aeganmediamontage/videos) |
| `VIDEO_RETENTION_HOURS` | No | Auto-cleanup interval (default: 48) |
| `INIT_DB_ON_BOOT` | No | Initialize DB on startup (default: false) |
| `CREDIT_COST_PER_RUN` | No | Credits per project (default: 1) |

---

## User Roles

| Role | Permissions |
|------|------------|
| **Admin** | Full access: manage users, roles, credits, API keys, all runs |
| **Manager** | Manage credits, view all runs, unlimited project creation |
| **User** | Create projects (requires credits), view own runs |

### Default Seed Users

| User | Email | Password |
|------|-------|----------|
| Admin | admin@aegan.local | Admin@123 |
| Manager | manager@aegan.local | Manager@123 |
| User | user@aegan.local | User@2026 |

---

## Project Structure

```
AeganmediaMontage/
├── webapp/
│   ├── main.py              # FastAPI routes (dashboard, auth, API)
│   ├── pipeline_runner.py    # Google AI video pipeline engine
│   ├── worker.py             # Background job processor
│   ├── config.py             # Settings from environment
│   ├── database.py           # MongoDB connection
│   ├── security.py           # JWT + password hashing
│   ├── bootstrap.py          # DB seed users
│   ├── cleanup.py            # Expired video cleanup
│   ├── api_keys.py           # Google API key status checker
│   ├── templates/            # Jinja2 HTML templates
│   │   ├── base.html         # Layout with nav + footer
│   │   ├── home.html         # Landing page
│   │   ├── dashboard.html    # User dashboard + project form
│   │   ├── run_detail.html   # Pipeline run progress + video preview
│   │   ├── admin_dashboard.html  # Admin panel
│   │   ├── login.html        # Sign in
│   │   ├── signup.html       # Create account
│   │   └── forgot_password.html  # Password reset request
│   └── static/
│       └── css/app.css       # Styles
├── tools/                    # OpenMontage tool registry
├── pipeline_defs/            # Pipeline YAML manifests
├── docker/
│   ├── app-entrypoint.sh     # Docker entrypoint
│   └── pre-deploy.sh         # Pre-deployment cleanup
├── .github/workflows/
│   └── main.yml              # CI/CD deployment workflow
├── Dockerfile
├── requirements.txt
├── .env.example
└── .dockerignore
```

---

## Deployment

### GitHub Actions CI/CD

The project deploys automatically via GitHub Actions on push to `main` or `feature/development`.

**Required GitHub Secrets:**
- `HOST` — Server IP/hostname
- `USERNAME` — SSH username
- `SSH_KEY` — SSH private key
- `MONGODB_URI` — MongoDB Atlas URI
- `JWT_SECRET` — JWT signing secret
- `GOOGLE_API_KEY` — Google AI Studio API key
- `SEED_ADMIN_EMAIL`, `SEED_ADMIN_PASSWORD`
- `SEED_MANAGER_EMAIL`, `SEED_MANAGER_PASSWORD`
- `SEED_USER_EMAIL`, `SEED_USER_PASSWORD`

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name montage.aeganmedia.com;

    location / {
        proxy_pass http://127.0.0.1:41006;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        client_max_body_size 50M;
    }
}
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Landing page |
| `/login` | GET/POST | User login |
| `/signup` | GET/POST | User registration |
| `/dashboard` | GET | User dashboard |
| `/dashboard/create-project` | POST | Create new video project |
| `/dashboard/run/{id}` | GET | Pipeline run detail |
| `/api/run/{id}/progress` | GET | Progress polling (JSON) |
| `/preview/run/{id}` | GET | Video preview stream |
| `/download/run/{id}` | GET | Video download |
| `/admin-dashboard` | GET | Admin panel |
| `/health` | GET | Health check |

---

## License

[GNU AGPLv3](LICENSE)

---

**AeganMedia Montage** — Professional AI video production powered by Google AI.

Built by [Aegan Media](https://www.aeganmedia.com).
