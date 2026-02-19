# llm.py — LLM-суммаризация и генерация описаний

import os
import json
import logging
import requests

logger = logging.getLogger(__name__)


def get_llm_settings():
    """Получает настройки LLM из переменных окружения."""
    return {
        "provider": os.getenv("LLM_PROVIDER", "openai"),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
        "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "llama3"),
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "500"))
    }


def summarize_with_openai(text, api_key=None, model="gpt-3.5-turbo"):
    """
    Создает суммаризацию с помощью OpenAI API.
    """
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Ты — помощник для создания описаний к коротким видео. "
                               "Создай краткое,吸引人的 описание на основе транскрипта. "
                               "Описание должно быть на том же языке, что и транскрипт."
                },
                {
                    "role": "user",
                    "content": f"Создай короткое описание (2-3 предложения) для видео на основе этого текста:\n\n{text[:4000]}"
                }
            ],
            temperature=0.7,
            max_tokens=200
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Ошибка OpenAI API: {e}")
        return None


def summarize_with_ollama(text, model="llama3", ollama_url="http://localhost:11434"):
    """
    Создает суммаризацию с помощью локального Ollama.
    """
    try:
        prompt = f"""Ты — помощник для создания описаний к коротким видео.
Создай краткое,吸引人的 описание (2-3 предложения) для видео на основе этого текста:

{text[:3000]}

Ответь только описанием, без дополнительного текста."""

        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 200
                }
            },
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            logger.error(f"Ollama error: {response.text}")
            return None

    except Exception as e:
        logger.error(f"Ошибка Ollama: {e}")
        return None


def generate_hashtags_with_llm(text, provider="openai", **kwargs):
    """
    Генерирует хештеги с помощью LLM.
    """
    try:
        prompt = """Ты — эксперт по SMM. На основе текста видео:
{text}

Сгенерируй 10 релевантных хештегов на русском и английском.
Формат: только хештеги через пробел, без пояснений.
Пример: #видео #shorts #интересно"""

        if provider == "openai":
            return generate_with_openai(prompt.format(text=text[:2000]), **kwargs)
        elif provider == "ollama":
            return generate_with_ollama(prompt.format(text=text[:2000]), **kwargs)
        else:
            return None

    except Exception as e:
        logger.error(f"Ошибка генерации хештегов: {e}")
        return None


def generate_clickbait_title(text, provider="openai", **kwargs):
    """
    Генерирует кликбейтный заголовок для видео.
    """
    try:
        prompt = """Ты — SMM-специалист. На основе этого текста:
{text}

Создай 3 варианта кликбейтных заголовков для Shorts/TikTok.
Требования:
- На русском языке
- Максимум 80 символов
- Без эмодзи иCaps Lock
- Захватывающий, но не кликбейтный

Формат: только заголовки, каждый на новой строке, без нумерации."""

        if provider == "openai":
            return generate_with_openai(prompt.format(text=text[:2000]), **kwargs)
        elif provider == "ollama":
            return generate_with_ollama(prompt.format(text=text[:2000]), **kwargs)
        else:
            return None

    except Exception as e:
        logger.error(f"Ошибка генерации заголовка: {e}")
        return None


def generate_with_openai(prompt, api_key=None, model="gpt-3.5-turbo"):
    """Универсальная функция генерации с OpenAI."""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Ошибка OpenAI: {e}")
        return None


def generate_with_ollama(prompt, model="llama3", ollama_url="http://localhost:11434"):
    """Универсальная функция генерации с Ollama."""
    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 300
                }
            },
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        return None

    except Exception as e:
        logger.error(f"Ошибка Ollama: {e}")
        return None


def summarize_transcript(transcript, use_llm=True):
    """
    Создает суммаризацию транскрипта.

    Args:
        transcript: Текст транскрипции
        use_llm: Использовать LLM или простой метод

    Returns:
        Словарь с description, hashtags, title
    """
    result = {
        "description": "",
        "hashtags": "",
        "titles": []
    }

    if not transcript or len(transcript) < 50:
        logger.warning("Транскрипт слишком короткий для LLM")
        return result

    if use_llm:
        settings = get_llm_settings()
        provider = settings.get("provider", "openai")

        try:
            if provider == "openai" and settings.get("api_key"):
                result["description"] = summarize_with_openai(
                    transcript,
                    api_key=settings["api_key"],
                    model=settings.get("model", "gpt-3.5-turbo")
                ) or ""

                hashtags = generate_hashtags_with_llm(transcript, provider="openai", api_key=settings["api_key"])
                result["hashtags"] = hashtags or ""

                titles = generate_clickbait_title(transcript, provider="openai", api_key=settings["api_key"])
                if titles:
                    result["titles"] = [t.strip() for t in titles.split('\n') if t.strip()]

            elif provider == "ollama":
                result["description"] = summarize_with_ollama(
                    transcript,
                    model=settings.get("ollama_model", "llama3"),
                    ollama_url=settings.get("ollama_url", "http://localhost:11434")
                ) or ""

                hashtags = generate_hashtags_with_llm(transcript, provider="ollama")
                result["hashtags"] = hashtags or ""

        except Exception as e:
            logger.error(f"Ошибка LLM: {e}")

    return result


def check_llm_available():
    """Проверяет доступность LLM."""
    settings = get_llm_settings()
    provider = settings.get("provider", "openai")

    if provider == "openai" and settings.get("api_key"):
        try:
            import openai
            client = openai.OpenAI(api_key=settings["api_key"])
            client.models.list()
            return {"available": True, "provider": "openai"}
        except Exception:
            pass

    elif provider == "ollama":
        try:
            response = requests.get(
                f"{settings.get('ollama_url', 'http://localhost:11434')}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                return {"available": True, "provider": "ollama"}
        except Exception:
            pass

    return {"available": False, "provider": provider}
