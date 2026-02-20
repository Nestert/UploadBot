# settings.py — управление настройками пользователей

import json
import os
import logging

SETTINGS_FILE = "user_settings.json"

DEFAULT_SETTINGS = {
    "min_clip_duration": 20,
    "max_clip_duration": 90,
    "remove_silence": True,
    "add_subtitles": True,
    "subtitle_style": "subtitle",
    "subtitle_position": "bottom",
    "subtitle_font_size": 32,
    "hashtag_count": 7,
    # Deprecated: оставлено только для обратной совместимости сохраненных настроек.
    "video_quality": "medium",
    "whisper_model": "base",
    "use_gpu": False,
    "vertical_layout_mode": "standard",
    "facecam_subject_side": "left",
    "facecam_detector_backend": "yolo_window_v1",
    "facecam_fallback_mode": "hard_side",
    "facecam_anchor": "edge_middle",
    "use_llm": False,
    "llm_provider": "openai",
    # Deprecated: определение моментов управляется сценариями, а не постоянной настройкой.
    "moment_detection": False,
    "mailru_token": None,
    # Deprecated: refresh token пока не используется в UI-потоке.
    "mailru_refresh_token": None
}

# Ключи настроек, которые сохранены для совместимости, но не участвуют в UI/UX потоке.
DEPRECATED_SETTINGS_KEYS = {"video_quality", "moment_detection", "mailru_refresh_token"}


def load_settings(user_id):
    """
    Загружает настройки пользователя из файла.
    """
    if not os.path.exists(SETTINGS_FILE):
        return DEFAULT_SETTINGS.copy()

    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            all_settings = json.load(f)
        
        stored = all_settings.get(str(user_id), {})
        if not isinstance(stored, dict):
            stored = {}
        user_settings = DEFAULT_SETTINGS.copy()
        user_settings.update(stored)
        
        # Расшифровываем чувствительные данные
        from security import decrypt_token
        if 'mailru_token' in user_settings and user_settings['mailru_token']:
            user_settings['mailru_token'] = decrypt_token(user_settings['mailru_token'])
            
        if 'mailru_refresh_token' in user_settings and user_settings['mailru_refresh_token']:
            user_settings['mailru_refresh_token'] = decrypt_token(user_settings['mailru_refresh_token'])
            
        return user_settings
    except Exception as e:
        logging.warning(f"Ошибка загрузки настроек: {e}")
        return DEFAULT_SETTINGS.copy()


def save_settings(user_id, settings):
    """
    Сохраняет настройки пользователя в файл.
    """
    all_settings = {}

    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                all_settings = json.load(f)
        except Exception:
            pass

    # Создаем копию настроек для сохранения, чтобы не менять объект в памяти
    settings_to_save = settings.copy()
    
    # Шифруем чувствительные данные
    from security import encrypt_token
    if 'mailru_token' in settings_to_save and settings_to_save['mailru_token']:
        settings_to_save['mailru_token'] = encrypt_token(settings_to_save['mailru_token'])
        
    if 'mailru_refresh_token' in settings_to_save and settings_to_save['mailru_refresh_token']:
        settings_to_save['mailru_refresh_token'] = encrypt_token(settings_to_save['mailru_refresh_token'])

    all_settings[str(user_id)] = settings_to_save

    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_settings, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Ошибка сохранения настроек: {e}")


def reset_settings(user_id):
    """
    Сбрасывает настройки пользователя к значениям по умолчанию.
    """
    save_settings(user_id, DEFAULT_SETTINGS.copy())


def get_min_clip_duration(user_id):
    """Получить минимальную длительность клипа."""
    settings = load_settings(user_id)
    return settings.get("min_clip_duration", DEFAULT_SETTINGS["min_clip_duration"])


def get_max_clip_duration(user_id):
    """Получить максимальную длительность клипа."""
    settings = load_settings(user_id)
    return settings.get("max_clip_duration", DEFAULT_SETTINGS["max_clip_duration"])


def get_remove_silence(user_id):
    """Получить настройку удаления тишины."""
    settings = load_settings(user_id)
    return settings.get("remove_silence", DEFAULT_SETTINGS["remove_silence"])


def get_add_subtitles(user_id):
    """Получить настройку добавления субтитров."""
    settings = load_settings(user_id)
    return settings.get("add_subtitles", DEFAULT_SETTINGS["add_subtitles"])


def get_subtitle_style(user_id):
    """Получить стиль субтитров."""
    settings = load_settings(user_id)
    return settings.get("subtitle_style", DEFAULT_SETTINGS["subtitle_style"])


def get_hashtag_count(user_id):
    """Получить количество хештегов."""
    settings = load_settings(user_id)
    return settings.get("hashtag_count", DEFAULT_SETTINGS["hashtag_count"])


def update_setting(user_id, key, value):
    """
    Обновляет отдельную настройку.
    """
    settings = load_settings(user_id)
    settings[key] = value
    save_settings(user_id, settings)
