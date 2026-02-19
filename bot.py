# bot.py — главный входной файл Telegram-бота

import os
import logging
from dotenv import load_dotenv

from telegram import BotCommand, BotCommandScopeDefault, BotCommandScopeAllPrivateChats
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


async def post_init(application):
    """Регистрирует команды бота в меню Telegram для всех скоупов."""
    commands = [
        BotCommand("start", "Главное меню"),
        BotCommand("list_videos", "Мои видео"),
        BotCommand("ready_videos", "Готовые видео"),
        BotCommand("settings", "Настройки обработки"),
        BotCommand("preview", "Превью видео"),
        BotCommand("moments", "Лучшие моменты"),
        BotCommand("gpu", "Проверка GPU"),
        BotCommand("cache", "Статус кэша"),
        BotCommand("cache_clear", "Очистить кэш"),
        BotCommand("llm", "Статус LLM"),
        BotCommand("mailru", "Подключить Mail.ru"),
        BotCommand("mailru_token", "Токен Mail.ru"),
        BotCommand("mailru_disconnect", "Отключить Mail.ru"),
    ]
    bot = application.bot
    # Сбрасываем старые команды во всех скоупах
    for scope in (BotCommandScopeDefault(), BotCommandScopeAllPrivateChats()):
        await bot.delete_my_commands(scope=scope)
    # Устанавливаем новые команды явно для обоих скоупов
    for scope in (BotCommandScopeDefault(), BotCommandScopeAllPrivateChats()):
        await bot.set_my_commands(commands, scope=scope)
    logging.info("Команды бота зарегистрированы: %s команд", len(commands))


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
    application = Application.builder().token(token).request(http_request).post_init(post_init).build()
    
    # Регистрируем все обработчики
    register_handlers(application)
    
    # Запускаем polling
    logging.info("Бот запущен!")
    application.run_polling()


if __name__ == '__main__':
    main()
