# errors.py — единая иерархия исключений UploadBot

"""
Кастомные исключения для UploadBot.

Иерархия:
    UploadBotError
    ├── DownloadError         — ошибки загрузки видео (yt-dlp, Telegram)
    ├── TranscriptionError    — ошибки Whisper / аудио
    ├── SceneDetectionError   — ошибки нарезки сцен
    ├── VideoProcessingError  — ошибки обработки видео (ffmpeg, vertical, autoedit)
    ├── SubtitleError         — ошибки субтитров
    ├── CancellationError     — отмена обработки пользователем
    └── SendError             — ошибки отправки в Telegram
"""


class UploadBotError(Exception):
    """Базовый класс для всех ошибок UploadBot."""
    pass


class DownloadError(UploadBotError):
    """Ошибка при скачивании видео."""
    pass


class TranscriptionError(UploadBotError):
    """Ошибка при транскрибации аудио (Whisper)."""
    pass


class SceneDetectionError(UploadBotError):
    """Ошибка при детекции сцен."""
    pass


class VideoProcessingError(UploadBotError):
    """Ошибка при обработке видео (ffmpeg, вертикальный ресайз, autoedit)."""
    pass


class SubtitleError(UploadBotError):
    """Ошибка при работе с субтитрами."""
    pass


class CancellationError(UploadBotError):
    """Пользователь отменил обработку."""
    pass


class SendError(UploadBotError):
    """Ошибка при отправке файла в Telegram."""
    pass
