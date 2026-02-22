# Развертывание и настройка

В данном разделе описаны способы запуска проекта **UploadBot**.

## Предварительная подготовка (Общее для всех методов)

1. **Клонирование репозитория**
   ```bash
   git clone https://github.com/Nestert/UploadBot.git
   cd UploadBot
   ```

2. **Настройка переменных окружения**
   Скопируйте пример файла конфигурации:
   ```bash
   cp env_example.txt .env
   ```
   Откройте `.env` и настройте параметры:
   - `TELEGRAM_BOT_TOKEN` — токен вашего бота (получите у @BotFather)
   - `OPENAI_API_KEY` — ключ API OpenAI (опционально для Whisper API/LLM)
   - `LLM_PROVIDER` — выберите провайдера: `openai` или `ollama`

---

## Вариант 1: Запуск через Docker (рекомендуется)

Использование Docker — самый быстрый и надежный способ изоляции окружения.

1. Убедитесь, что у вас установлен **Docker** и **Docker Compose**.
2. В корневой папке проекта выполните сборку и запуск в фоновом режиме:
   ```bash
   docker-compose up -d
   ```
3. Для просмотра логов:
   ```bash
   docker-compose logs -f bot
   ```
4. Для остановки сервиса:
   ```bash
   docker-compose down
   ```

---

## Вариант 2: Локальный запуск на Windows 11

### Требования
- Python 3.10 или выше
- Установленный FFmpeg, прописанный в системный `PATH`
- Git

### Инструкция

1. Откройте **PowerShell** в папке проекта.
2. Создайте и активируйте виртуальное окружение:
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```
3. Установите все зависимости из `requirements.txt`:
   ```powershell
   pip install -r requirements.txt
   ```
4. Запустите бота:
   ```powershell
   python bot.py
   ```
   *Убедитесь, что при первом запуске скачивание моделей Whisper может занять некоторое время.*

---

## Вариант 3: Запуск на Linux / macOS

В целом процесс аналогичен Windows:
1. Установите FFmpeg (через `apt` или `brew`).
2. Создайте окружение:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Установите зависимости и запустите:
   ```bash
   pip install -r requirements.txt
   python bot.py
   ```
