# UploadBot - План развития

## Текущее состояние
Все фазы завершены! UploadBot готов к использованию.

---

## Фаза 1: UX и стабильность (✅ Завершено)

- Прогресс-бар обработки
- Кнопка отмены
- Retry-логика

---

## Фаза 2: Расширение функциональности (✅ Завершено)

- Настройки обработки (/settings)
- Поддержка Mail.ru Cloud (/mailru)
- Превью видео (/preview)

---

## Фаза 3: Оптимизация (✅ Завершено)

- GPU-ускорение Whisper
- Параллельная обработка
- Кэширование транскрипций

---

## Фаза 4: Качество контента (✅ Завершено)

- LLM-суммаризация (OpenAI/Ollama)
- ASS/SSA субтитры (4 стиля)
- AI-выделение лучших моментов

---

## Фаза 5: Инфраструктура (✅ Завершено)

### Dockerfile
```dockerfile
FROM python:3.10-slim
# FFmpeg уже включен
# Изолированная среда
```

### docker-compose.yml
```yaml
services:
  bot:
    build: .
    volumes:
      - ./videos:/app/videos
      - ./cache:/app/cache
    GPU support
```

### Команды Docker
```bash
# Сборка и запуск
docker-compose up -d

# С GPU
docker-compose up -d

# С Ollama
docker-compose --profile llm up -d
```

---

## Команды

| Команда | Описание |
|---------|----------|
| `/start` | Запуск бота |
| `/list_videos` | Список видео |
| `/settings` | Настройки обработки |
| `/preview` | Превью видео |
| `/moments` | Лучшие моменты |
| `/llm` | LLM статус |
| `/gpu` | Проверка GPU |
| `/cache` | Статус кэша |
| `/mailru` | Подключить Mail.ru |

## Источники

- YouTube
- Mail.ru Cloud
- Прямая загрузка файлов

## Переменные окружения

```env
TELEGRAM_BOT_TOKEN=your_token
OPENAI_API_KEY=sk-your-key
LLM_PROVIDER=openai  # или ollama
OLLAMA_URL=http://localhost:11434
```

## Быстрый старт

```bash
git clone <repo>
cd UploadBot
cp env_example.txt .env
# редактируем .env
docker-compose up -d
```
