# UploadBot Dockerfile

FROM python:3.10-slim

LABEL maintainer=" UploadBot"
LABEL description="Telegram bot for creating Shorts/TikTok from YouTube videos"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r botuser && useradd -r -g botuser botuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R botuser:botuser /app

USER botuser

EXPOSE 8080

CMD ["python", "bot.py"]
