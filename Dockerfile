FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/appuser

WORKDIR /app

# System deps: ffmpeg, CA certs, Noto fonts for Indic subtitle rendering
# fonts-noto-core  — NotoSans base (Latin, Greek, Cyrillic, etc.)
# fonts-noto-hinted — Tamil, Devanagari (Hindi), Telugu, Kannada, Malayalam,
#                    Bengali, Gujarati, Punjabi, Arabic and more
# Without these fonts ffmpeg drawtext renders Indic characters as □□□□ boxes.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl ffmpeg \
    fonts-noto-core \
    fonts-noto-hinted \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd -g 10001 appgroup && useradd -m -u 10001 -g appgroup -s /usr/sbin/nologin appuser

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy full source
COPY . /app
RUN chmod +x /app/docker/app-entrypoint.sh && \
    mkdir -p /app/projects /data/videos && \
    chown -R appuser:appgroup /app /data

USER appuser:appgroup

EXPOSE 51004

ENTRYPOINT ["/app/docker/app-entrypoint.sh"]
