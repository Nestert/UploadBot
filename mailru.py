# mailru.py — работа с облаком Mail.ru (Облако Mail.ru)

import os
import re
import json
import logging
import subprocess
import uuid
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

MAILRU_SHARED_PATTERN = r'(?:cloud\.mail\.ru|my\.mail\.ru)/.*[?&]share='


def is_mailru_url(url):
    """Проверяет, является ли URL ссылкой на Mail.ru Cloud."""
    patterns = [
        r'cloud\.mail\.ru',
        r'my\.mail\.ru',
        r'files\.mail\.ru'
    ]
    return any(re.search(p, url) for p in patterns)


def extract_public_link_id(url):
    """
    Извлекает ID публичной ссылки Mail.ru Cloud.
    """
    try:
        if 'share=' in url:
            match = re.search(r'[?&]share=([^&]+)', url)
            if match:
                return match.group(1)
        if '/public/' in url:
            match = re.search(r'/public/([a-zA-Z0-9]+)', url)
            if match:
                return match.group(1)
        return None
    except Exception as e:
        logger.error(f"Ошибка извлечения ID ссылки: {e}")
        return None


def download_from_mailru_public(url, output_dir, progress_callback=None):
    """
    Скачивает файл из публичной ссылки Mail.ru Cloud.
    Использует yt-dlp для загрузки.
    
    Args:
        url: Публичная ссылка на файл
        output_dir: Директория для сохранения
        progress_callback: Функция обратного вызова для прогресса
    
    Returns:
        Путь к скачанному файлу или None при ошибке
    """
    try:
        output_template = os.path.join(output_dir, f"%(title)s_%(id)s.%(ext)s")

        import yt_dlp

        ydl_opts = {
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True,
            'format': 'best',
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)

            if os.path.exists(filename):
                logger.info(f"Файл скачан: {filename}")
                return filename
            else:
                logger.error(f"Файл не найден после загрузки: {filename}")
                return None

    except Exception as e:
        logger.error(f"Ошибка загрузки с Mail.ru: {e}")
        return None


def get_mailru_file_info(url):
    """
    Получает информацию о файле из ссылки Mail.ru Cloud.
    """
    try:
        import yt_dlp

        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title', 'unknown'),
                'ext': info.get('ext', 'mp4'),
                'size': info.get('filesize', 0),
                'duration': info.get('duration', 0),
                'url': url
            }
    except Exception as e:
        logger.error(f"Ошибка получения информации о файле: {e}")
        return None


def get_auth_token(client_id, client_secret, redirect_uri, code):
    """
    Получает OAuth-токен для доступа к API Mail.ru Cloud.
    """
    import requests

    url = "https://oauth.mail.ru/token"
    data = {
        'grant_type': 'authorization_code',
        'client_id': client_id,
        'client_secret': client_secret,
        'redirect_uri': redirect_uri,
        'code': code
    }

    try:
        response = requests.post(url, data=data)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Ошибка получения токена: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Ошибка запроса токена: {e}")
        return None


def list_files(token, path='/'):
    """
    Получает список файлов в директории.
    Требует OAuth-токен.
    """
    import requests

    url = "https://cloud-api.mail.ru/v2/file/list"
    headers = {'Authorization': f'Bearer {token}'}
    params = {'path': path}

    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Ошибка получения списка файлов: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Ошибка запроса списка файлов: {e}")
        return None


def download_file(token, file_path, output_dir):
    """
    Скачивает файл из облака с использованием OAuth-токена.
    """
    import requests

    download_url = f"https://cloud-api.mail.ru/v2/file/link"
    headers = {'Authorization': f'Bearer {token}'}
    params = {'path': file_path}

    try:
        response = requests.get(download_url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'body' in data:
                file_url = data['body']['url']

                filename = os.path.basename(file_path)
                output_path = os.path.join(output_dir, f"mailru_{uuid.uuid4().hex[:8]}_{filename}")

                resp = requests.get(file_url, stream=True)
                if resp.status_code == 200:
                    with open(output_path, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    return output_path

        logger.error(f"Ошибка получения ссылки для скачивания: {response.text}")
        return None
    except Exception as e:
        logger.error(f"Ошибка скачивания файла: {e}")
        return None
