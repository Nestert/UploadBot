# download.py — модуль загрузки YouTube-видео с помощью yt-dlp

import yt_dlp
import os
import uuid
import logging
from utils import ensure_videos_dir

def download_video(youtube_url, output_dir=None):
    """
    Скачивает видео с YouTube по ссылке.
    Возвращает имя скачанного файла (mp4).
    """
    # Генерируем уникальное имя файла и сохраняем в папку videos/ или указанный output_dir
    if output_dir:
        videos_dir = output_dir
    else:
        videos_dir = ensure_videos_dir()
    video_id = str(uuid.uuid4())[:8]
    output_filename = os.path.join(videos_dir, f'yt_{video_id}.mp4')
    
    # Увеличенные таймауты для нестабильного соединения
    timeout_seconds = 300

    ydl_opts = {
        'outtmpl': output_filename,
        'format': 'best[ext=mp4]/best',
        'merge_output_format': 'mp4',
        'quiet': False,
        'verbose': True,
        'noprogress': True,
        'retries': 15,
        'fragment_retries': 10,
        'file_access_retries': 10,
        'ignoreerrors': False,
        'socket_timeout': timeout_seconds,
        'http_chunk_size': 10485760,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        },
        'extractor_args': {
            'youtube': {
                'player_client': ['web'],
            }
        },
    }

    try:
        logging.info(f"Начинаю загрузку видео с YouTube (таймаут: {timeout_seconds}с)...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        if os.path.exists(output_filename):
            file_size = os.path.getsize(output_filename)
            logging.info(f"Видео успешно загружено: {output_filename} ({file_size} байт)")
            return output_filename
        raise Exception("Не удалось скачать видео.")
    except yt_dlp.utils.DownloadError as e:
        logging.error(f"yt-dlp DownloadError: {e}")
        raise Exception(
            f"Ошибка загрузки видео. Возможные причины: медленное соединение, ограничения доступа к YouTube, "
            f"или проблемы с видео. Попробуйте другое видео или повторите попытку позже. Детали: {e}"
        )
    except Exception as e:
        logging.error(f"Ошибка загрузки видео: {e}")
        raise Exception(f"Ошибка загрузки видео: {e}")
