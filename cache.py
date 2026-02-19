# cache.py — кэширование транскрипций и результатов

import hashlib
import json
import os
import logging
import time
import shutil

logger = logging.getLogger(__name__)

CACHE_DIR = "cache"
TRANSCRIPTIONS_DIR = os.path.join(CACHE_DIR, "transcriptions")
METADATA_DIR = os.path.join(CACHE_DIR, "metadata")
PROCESSED_DIR = os.path.join(CACHE_DIR, "processed")
PROCESSED_VIDEOS_DIR = os.path.join(PROCESSED_DIR, "videos")
PROCESSED_INDEX_PATH = os.path.join(PROCESSED_DIR, "index.json")


def ensure_cache_dirs():
    """Создает директории для кэша."""
    os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_VIDEOS_DIR, exist_ok=True)


def get_file_hash(file_path, chunk_size=1024 * 1024):
    """
    Вычисляет хэш файла для идентификации.
    Читает файл чанками, чтобы не загружать всё видео в RAM.
    """
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Ошибка вычисления хэша: {e}")
        return None


def get_url_hash(url):
    """
    Вычисляет хэш URL для кэширования.
    """
    return hashlib.md5(url.encode()).hexdigest()


def get_transcript_cache_path(url_hash):
    """Возвращает путь к кэшированной транскрипции."""
    return os.path.join(TRANSCRIPTIONS_DIR, f"{url_hash}.txt")


def get_metadata_cache_path(url_hash):
    """Возвращает путь к метаданным."""
    return os.path.join(METADATA_DIR, f"{url_hash}.json")


def cache_transcription(url, transcript):
    """
    Сохраняет транскрипцию в кэш.
    """
    ensure_cache_dirs()
    url_hash = get_url_hash(url)
    cache_path = get_transcript_cache_path(url_hash)

    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        logger.info(f"Транскрипция закэширована: {url_hash[:8]}...")
        return True
    except Exception as e:
        logger.error(f"Ошибка кэширования транскрипции: {e}")
        return False


def get_cached_transcription(url):
    """
    Получает транскрипцию из кэша.
    """
    ensure_cache_dirs()
    url_hash = get_url_hash(url)
    cache_path = get_transcript_cache_path(url_hash)

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                transcript = f.read()
            logger.info(f"Транскрипция найдена в кэше: {url_hash[:8]}...")
            return transcript
        except Exception as e:
            logger.error(f"Ошибка чтения кэша: {e}")
    return None


def cache_metadata(url, metadata):
    """
    Сохраняет метаданные в кэш.
    """
    ensure_cache_dirs()
    url_hash = get_url_hash(url)
    metadata_path = get_metadata_cache_path(url_hash)

    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Ошибка кэширования метаданных: {e}")
        return False


def get_cached_metadata(url):
    """
    Получает метаданные из кэша.
    """
    ensure_cache_dirs()
    url_hash = get_url_hash(url)
    metadata_path = get_metadata_cache_path(url_hash)

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка чтения метаданных: {e}")
    return None


def is_cached(url):
    """
    Проверяет, есть ли URL в кэше.
    """
    ensure_cache_dirs()
    url_hash = get_url_hash(url)
    transcript_path = get_transcript_cache_path(url_hash)
    return os.path.exists(transcript_path)


def get_cache_size():
    """
    Возвращает размер кэша в байтах.
    """
    ensure_cache_dirs()
    total_size = 0
    for dir_path in [TRANSCRIPTIONS_DIR, METADATA_DIR, PROCESSED_DIR]:
        if os.path.exists(dir_path):
            for root, dirs, files in os.walk(dir_path):
                for f in files:
                    fp = os.path.join(root, f)
                    total_size += os.path.getsize(fp)
    return total_size


def format_cache_size(size_bytes):
    """Форматирует размер в человекочитаемый вид."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def clear_cache(max_age_days=30):
    """
    Очищает старые файлы кэша.
    """
    ensure_cache_dirs()
    current_time = time.time()
    deleted_count = 0
    deleted_size = 0

    for dir_path in [TRANSCRIPTIONS_DIR, METADATA_DIR, PROCESSED_VIDEOS_DIR]:
        if os.path.exists(dir_path):
            for root, dirs, files in os.walk(dir_path):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        file_mtime = os.path.getmtime(fp)
                        if (current_time - file_mtime) > (max_age_days * 86400):
                            deleted_size += os.path.getsize(fp)
                            os.remove(fp)
                            deleted_count += 1
                    except Exception:
                        pass

    logger.info(f"Удалено {deleted_count} файлов кэша ({format_cache_size(deleted_size)})")
    return deleted_count, format_cache_size(deleted_size)


def _load_processed_index():
    """Читает индекс кэша обработанных видео."""
    ensure_cache_dirs()
    if not os.path.exists(PROCESSED_INDEX_PATH):
        return {}

    try:
        with open(PROCESSED_INDEX_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        logger.warning(f"Не удалось прочитать индекс кэша обработанных видео: {e}")
    return {}


def _save_processed_index(index):
    """Сохраняет индекс кэша обработанных видео."""
    ensure_cache_dirs()
    try:
        with open(PROCESSED_INDEX_PATH, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Не удалось сохранить индекс кэша обработанных видео: {e}")
        return False


def build_processing_cache_key(source_hash, settings_signature, random_cut=False):
    """
    Строит ключ кэша готовых результатов.
    random_cut включен в ключ, чтобы не конфликтовать с full-режимом.
    """
    payload = {
        "source_hash": source_hash,
        "settings_signature": settings_signature,
        "random_cut": bool(random_cut),
        "pipeline_version": 1,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.md5(encoded).hexdigest()


def get_cached_processed_result(cache_key):
    """
    Возвращает кэшированный результат обработки, если файлы существуют.
    """
    index = _load_processed_index()
    entry = index.get(cache_key)
    if not entry:
        return None

    videos = entry.get("videos", [])
    if not videos:
        return None

    existing = [v for v in videos if os.path.exists(v)]
    if not existing:
        # Авто-очистка битой записи.
        del index[cache_key]
        _save_processed_index(index)
        return None

    result = entry.copy()
    result["videos"] = existing
    return result


def cache_processed_result(cache_key, videos, tags, source_hash=None, settings_signature=None, random_cut=False):
    """
    Сохраняет готовые видео и метаданные в кэш.
    """
    ensure_cache_dirs()
    if not videos:
        return None

    cached_videos = []
    for idx, video_path in enumerate(videos, 1):
        if not os.path.exists(video_path):
            continue
        ext = os.path.splitext(video_path)[1] or ".mp4"
        cache_video_path = os.path.join(PROCESSED_VIDEOS_DIR, f"{cache_key}_{idx}{ext}")
        try:
            shutil.copy2(video_path, cache_video_path)
        except Exception as e:
            logger.warning(f"Не удалось скопировать видео в кэш: {video_path} -> {cache_video_path}: {e}")
            continue
        cached_videos.append(cache_video_path)

    if not cached_videos:
        return None

    index = _load_processed_index()
    index[cache_key] = {
        "videos": cached_videos,
        "tags": tags,
        "source_hash": source_hash,
        "settings_signature": settings_signature,
        "random_cut": bool(random_cut),
        "updated_at": time.time(),
    }
    _save_processed_index(index)
    logger.info("Готовые видео сохранены в кэш: key=%s videos=%s", cache_key[:8], len(cached_videos))
    return index[cache_key]
