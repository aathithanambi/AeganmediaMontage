FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/appuser \
    NODE_OPTIONS="--max-old-space-size=2048"

WORKDIR /app

# System deps: ffmpeg, Node.js 22 LTS, CA certs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg ffmpeg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
       | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" \
       > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update && apt-get install -y --no-install-recommends nodejs \
    && apt-get purge -y gnupg && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd -g 10001 appgroup && useradd -m -u 10001 -g appgroup -s /usr/sbin/nologin appuser

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir piper-tts 2>/dev/null || echo "piper-tts optional, skipping"

# Pre-download Piper voice model
RUN mkdir -p /home/appuser/.local/share/piper_models && \
    python -c "from piper.download import ensure_voice_exists; ensure_voice_exists('en_US-lessac-medium', ['/home/appuser/.local/share/piper_models'], '/home/appuser/.local/share/piper_models')" 2>/dev/null \
    || echo "Piper voice download skipped (will use cloud TTS)"

# Remotion composer (Node.js deps)
COPY remotion-composer/package.json remotion-composer/package-lock.json* /app/remotion-composer/
RUN cd /app/remotion-composer && npm install --omit=dev 2>/dev/null || npm install 2>/dev/null || echo "remotion install skipped"

# Copy full source
COPY . /app
RUN chmod +x /app/docker/app-entrypoint.sh && \
    mkdir -p /app/projects /var/aeganmediamontage/videos && \
    chown -R appuser:appgroup /app /var/aeganmediamontage

USER appuser:appgroup

EXPOSE 41006

ENTRYPOINT ["/app/docker/app-entrypoint.sh"]
