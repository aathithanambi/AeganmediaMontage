FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/appuser

WORKDIR /app

# System deps: ffmpeg, CA certs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd -g 10001 appgroup && useradd -m -u 10001 -g appgroup -s /usr/sbin/nologin appuser

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy full source
COPY . /app
RUN chmod +x /app/docker/app-entrypoint.sh && \
    mkdir -p /app/projects /var/aeganmediamontage/videos && \
    chown -R appuser:appgroup /app /var/aeganmediamontage

USER appuser:appgroup

EXPOSE 41006

ENTRYPOINT ["/app/docker/app-entrypoint.sh"]
