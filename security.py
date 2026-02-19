# security.py — функции безопасности

"""
Модуль безопасности: шифрование токенов и валидация входных данных.
"""

import os
import re
import base64
import hashlib
import logging
from typing import Optional

# Попытка импорта cryptography для шифрования
try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logging.warning("cryptography не установлен. Шифрование токенов недоступно.")


# === Шифрование токенов ===

_KEY_FILE = ".encryption.key"
_fernet: Optional['Fernet'] = None


def _get_or_create_key() -> bytes:
    """Получает или создаёт ключ шифрования."""
    if os.path.exists(_KEY_FILE):
        with open(_KEY_FILE, 'rb') as f:
            return f.read()
    else:
        key = Fernet.generate_key()
        with open(_KEY_FILE, 'wb') as f:
            f.write(key)
        os.chmod(_KEY_FILE, 0o600)  # Только владелец может читать
        return key


def _get_fernet() -> Optional['Fernet']:
    """Возвращает Fernet-объект для шифрования/дешифрования."""
    global _fernet
    if not CRYPTO_AVAILABLE:
        return None
    if _fernet is None:
        key = _get_or_create_key()
        _fernet = Fernet(key)
    return _fernet


def encrypt_token(token: str) -> str:
    """
    Шифрует токен.
    
    Args:
        token: Токен в открытом виде
        
    Returns:
        Зашифрованный токен в base64 или исходный токен если шифрование недоступно
    """
    if not token:
        return token
        
    fernet = _get_fernet()
    if fernet is None:
        logging.warning("Шифрование недоступно, токен сохранён без шифрования")
        return token
    
    try:
        encrypted = fernet.encrypt(token.encode('utf-8'))
        return f"ENC:{base64.urlsafe_b64encode(encrypted).decode('ascii')}"
    except Exception as e:
        logging.error(f"Ошибка шифрования токена: {e}")
        return token


def decrypt_token(encrypted_token: str) -> str:
    """
    Расшифровывает токен.
    
    Args:
        encrypted_token: Зашифрованный токен
        
    Returns:
        Расшифрованный токен или исходная строка если дешифрование не удалось
    """
    if not encrypted_token:
        return encrypted_token
    
    # Проверяем, зашифрован ли токен
    if not encrypted_token.startswith("ENC:"):
        return encrypted_token
    
    fernet = _get_fernet()
    if fernet is None:
        logging.warning("Шифрование недоступно, возвращаем токен как есть")
        return encrypted_token
    
    try:
        encoded = encrypted_token[4:]  # Убираем "ENC:"
        encrypted = base64.urlsafe_b64decode(encoded.encode('ascii'))
        return fernet.decrypt(encrypted).decode('utf-8')
    except Exception as e:
        logging.error(f"Ошибка расшифровки токена: {e}")
        return encrypted_token


def is_encrypted(value: str) -> bool:
    """Проверяет, зашифровано ли значение."""
    return value.startswith("ENC:") if value else False


# === Валидация URL ===

# Паттерны для различных поддерживаемых источников
YOUTUBE_PATTERN = re.compile(
    r'^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+$',
    re.IGNORECASE
)

MAILRU_PATTERN = re.compile(
    r'^(https?://)?(cloud\.mail\.ru|files\.mail\.ru)/.+$',
    re.IGNORECASE
)


def is_valid_youtube_url(url: str) -> bool:
    """
    Проверяет валидность YouTube URL.
    
    Args:
        url: URL для проверки
        
    Returns:
        True если URL валиден
    """
    if not url or not isinstance(url, str):
        return False
    return bool(YOUTUBE_PATTERN.match(url.strip()))


def is_valid_mailru_url(url: str) -> bool:
    """
    Проверяет валидность Mail.ru Cloud URL.
    
    Args:
        url: URL для проверки
        
    Returns:
        True если URL валиден
    """
    if not url or not isinstance(url, str):
        return False
    return bool(MAILRU_PATTERN.match(url.strip()))


def is_valid_video_url(url: str) -> bool:
    """
    Проверяет, является ли URL поддерживаемым видео-источником.
    
    Args:
        url: URL для проверки
        
    Returns:
        True если URL — поддерживаемый видео-источник
    """
    return is_valid_youtube_url(url) or is_valid_mailru_url(url)


def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """
    Очищает имя файла от потенциально опасных символов.
    
    Args:
        filename: Исходное имя файла
        max_length: Максимальная длина
        
    Returns:
        Безопасное имя файла
    """
    if not filename:
        return "unnamed"
    
    # Удаляем опасные символы
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    # Убираем точки в начале (скрытые файлы)
    safe = safe.lstrip('.')
    # Ограничиваем длину
    if len(safe) > max_length:
        name, ext = os.path.splitext(safe)
        safe = name[:max_length - len(ext)] + ext
    
    return safe or "unnamed"


def validate_user_id(user_id) -> bool:
    """
    Проверяет валидность Telegram user_id.
    
    Args:
        user_id: ID пользователя
        
    Returns:
        True если ID валиден
    """
    try:
        uid = int(user_id)
        return uid > 0
    except (ValueError, TypeError):
        return False
