# bot.py — главный входной файл Telegram-бота

import os
import logging
from dotenv import load_dotenv

from telegram.ext import Application
from telegram.request import HTTPXRequest

# Загружаем переменные окружения из .env файла
load_dotenv()

# Workaround для конфликта OpenMP (PyTorch/Numba на macOS).
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Логгирование для отладки
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Импортируем функцию регистрации обработчиков
from handlers import register_handlers


def main():
    """Запуск бота."""
    # Получаем токен из переменной окружения
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        print("Ошибка: Не установлен TELEGRAM_BOT_TOKEN в переменных окружения")
        return
    
    # Настройка HTTPX с увеличенными таймаутами для стабильности сети
    http_request = HTTPXRequest(
        connect_timeout=30.0,
        read_timeout=300.0,
        write_timeout=300.0,
        pool_timeout=30.0,
    )

    # Создаём приложение
    application = Application.builder().token(token).request(http_request).build()
    
    # Регистрируем все обработчики
    register_handlers(application)
    
    # Запускаем polling
    logging.info("Бот запущен!")
    application.run_polling()


if __name__ == '__main__':
    main()
