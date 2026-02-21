# transcribe.py — транскрибация аудио через OpenAI Whisper

import subprocess
import os
import uuid
import logging
from typing import Dict, Any, List

# Workaround для конфликта OpenMP (PyTorch/Numba на macOS).
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Lazy imports
# import whisper
# import torch

logger = logging.getLogger(__name__)

WHISPER_MODELS = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large": "large"
}


def check_gpu_available():
    """Проверяет доступность GPU (CUDA)."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device():
    """Возвращает устройство для inference: 'cuda' или 'cpu'."""
    return "cuda" if check_gpu_available() else "cpu"


def extract_wav(input_video, output_dir=None):
    """
    Извлекает аудиодорожку в формате WAV с требуемыми параметрами для Whisper.
    Возвращает путь к wav-файлу.
    """
    audio_id = str(uuid.uuid4())
    if output_dir:
        wav_file = os.path.join(output_dir, f"{audio_id}.wav")
    else:
        wav_file = f"{audio_id}.wav"
    cmd = [
        "ffmpeg",
        "-i", input_video,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        wav_file,
        "-y"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        if os.path.exists(wav_file):
            return wav_file
        else:
            raise Exception("Не удалось извлечь аудиодорожку.")
    except Exception as e:
        raise Exception(f"Ошибка при извлечении WAV: {e}")


_CACHED_MODEL = None
_CACHED_MODEL_NAME = None


def get_whisper_model(model_name, device):
    """
    Возвращает загруженную модель Whisper, используя кэширование.
    """
    global _CACHED_MODEL, _CACHED_MODEL_NAME
    
    import whisper
    
    if _CACHED_MODEL is not None and _CACHED_MODEL_NAME == model_name:
        return _CACHED_MODEL
    
    logger.info(f"Загрузка модели Whisper '{model_name}' на {device}...")
    _CACHED_MODEL = whisper.load_model(model_name, device=device)
    _CACHED_MODEL_NAME = model_name
    return _CACHED_MODEL


def unload_whisper_model():
    """
    Выгружает модель Whisper из памяти и очищает CUDA кэш.
    """
    global _CACHED_MODEL, _CACHED_MODEL_NAME
    
    if _CACHED_MODEL is not None:
        logger.info("Выгрузка модели Whisper из памяти...")
        del _CACHED_MODEL
        _CACHED_MODEL = None
        _CACHED_MODEL_NAME = None
        
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
    except ImportError:
        pass


def _transcribe_with_model(wav_file, model_name="base", use_gpu=None, fp16=None, word_timestamps=False):
    """
    Запускает Whisper и возвращает сырой result-словарь.
    """
    model_name = WHISPER_MODELS.get(model_name, "base")

    if use_gpu is None:
        use_gpu = check_gpu_available()
    device = "cuda" if use_gpu else "cpu"

    if fp16 is None:
        fp16 = (device == "cuda")

    model = get_whisper_model(model_name, device)
    logger.info(
        "Транскрибация %s (model=%s, device=%s, word_timestamps=%s)...",
        wav_file,
        model_name,
        device,
        word_timestamps
    )
    return model.transcribe(wav_file, fp16=fp16, word_timestamps=word_timestamps)


def run_whisper(wav_file, output_dir=None, model_name="base", use_gpu=None, fp16=None):
    """
    Запускает OpenAI Whisper для транскрибации аудиофайла.

    Args:
        wav_file: Путь к аудиофайлу
        output_dir: Директория для вывода
        model_name: Размер модели (tiny, base, small, medium, large)
        use_gpu: Использовать GPU (None = автоопределение)
        fp16: Использовать FP16 (None = автоопределение)
    """
    txt_id = str(uuid.uuid4())
    if output_dir:
        txt_file = os.path.join(output_dir, f"{txt_id}.txt")
    else:
        txt_file = f"{txt_id}.txt"

    try:
        result = _transcribe_with_model(
            wav_file,
            model_name=model_name,
            use_gpu=use_gpu,
            fp16=fp16,
            word_timestamps=False
        )

        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(result["text"])

        if os.path.exists(txt_file):
            logger.info(f"Транскрибация завершена: {txt_file}")
            return txt_file
        else:
            raise Exception("Whisper не создал итоговый файл.")
    except Exception as e:
        logger.error(f"Ошибка транскрибации Whisper: {e}")
        raise Exception(f"Ошибка транскрибации Whisper: {e}")


def _extract_words(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Извлекает слова с таймкодами из result Whisper."""
    words = []
    for segment in result.get("segments", []) or []:
        for word in segment.get("words", []) or []:
            word_text = (word.get("word") or "").strip()
            start = word.get("start")
            end = word.get("end")
            if not word_text:
                continue
            if start is None or end is None:
                continue
            try:
                start = float(start)
                end = float(end)
            except (TypeError, ValueError):
                continue
            if end <= start:
                end = start + 0.05
            words.append({
                "word": word_text,
                "start": start,
                "end": end
            })
    return words


def _extract_segments(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Нормализует сегменты Whisper до простого формата."""
    segments = []
    for segment in result.get("segments", []) or []:
        start = segment.get("start")
        end = segment.get("end")
        text = (segment.get("text") or "").strip()
        if start is None or end is None:
            continue
        try:
            start = float(start)
            end = float(end)
        except (TypeError, ValueError):
            continue
        if end <= start:
            end = start + 0.05
        segments.append({
            "start": start,
            "end": end,
            "text": text
        })
    return segments


def transcribe_audio_with_timestamps(input_video, output_dir=None, model_name="base", use_gpu=None):
    """
    Транскрибирует видео и возвращает текст + таймкоды сегментов/слов.

    Returns:
        dict: {
            "text": str,
            "segments": [{"start": float, "end": float, "text": str}, ...],
            "words": [{"word": str, "start": float, "end": float}, ...]
        }
    """
    if use_gpu is None:
        use_gpu = check_gpu_available()

    wav_file = extract_wav(input_video, output_dir)
    try:
        try:
            result = _transcribe_with_model(
                wav_file,
                model_name=model_name,
                use_gpu=use_gpu,
                word_timestamps=True
            )
        except Exception as e:
            logger.error("Ошибка транскрибации с таймкодами: %s", e)
            raise Exception(f"Ошибка транскрибации с таймкодами: {e}")
    finally:
        # Убираем временный WAV-файл после транскрибации
        if os.path.exists(wav_file):
            try:
                os.remove(wav_file)
            except OSError as e:
                logger.warning(f"Не удалось удалить временный wav-файл {wav_file}: {e}")

    return {
        "text": (result.get("text") or "").strip(),
        "segments": _extract_segments(result),
        "words": _extract_words(result)
    }


def transcribe_audio(input_video, output_dir=None, model_name="base", use_gpu=None, url=None):
    """
    Основная функция для транскрибации.

    Args:
        input_video: Путь к видеофайлу
        output_dir: Директория для временных файлов
        model_name: Размер модели (tiny, base, small, medium, large)
        use_gpu: Использовать GPU (None = автоопределение)
        url: URL видео для кэширования (опционально)
    """
    import cache

    if use_gpu is None:
        use_gpu = check_gpu_available()
        if use_gpu:
            logger.info("GPU обнаружен! Использую CUDA для транскрибации.")
        else:
            logger.info("GPU не обнаружен. Использую CPU.")

    if url and cache.is_cached(url):
        cached_transcript = cache.get_cached_transcription(url)
        if cached_transcript:
            logger.info(f"Использую кэшированную транскрипцию для {url[:30]}...")
            return cached_transcript

    transcript_data = transcribe_audio_with_timestamps(
        input_video,
        output_dir=output_dir,
        model_name=model_name,
        use_gpu=use_gpu
    )
    transcript = transcript_data["text"]

    if url:
        cache.cache_transcription(url, transcript)

    return transcript


def get_model_info():
    """Возвращает информацию о доступных моделях и GPU."""
    info = {
        "gpu_available": check_gpu_available(),
        "device": get_device(),
        "models": list(WHISPER_MODELS.keys()),
        "recommended_model": "medium" if check_gpu_available() else "base"
    }
    if check_gpu_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return info
