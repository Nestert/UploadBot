# handlers/mailru_handlers.py — обработчики Mail.ru Cloud

import logging
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

import settings
import mailru
from handlers.helpers import set_last_source
from handlers.processing import process_source


async def mailru_link(update, context):
    """Показывает инструкцию по подключению Mail.ru Cloud."""
    user_id = update.message.chat_id
    user_settings = settings.load_settings(user_id)

    mailru_connected = user_settings.get('mailru_token') is not None

    text = "☁️ Mail.ru Cloud\n\n"

    if mailru_connected:
        text += "✅ Облако подключено\n\n"
        text += "Команды:\n"
        text += "/mailru_token <код> - Обновить токен\n"
        text += "/mailru_disconnect - Отключить облако\n\n"
        text += "Публичные ссылки Mail.ru можно отправлять прямо в чат."
    else:
        text += "Для доступа к вашим файлам в облаке:\n\n"
        text += "1. Перейдите по ссылке для авторизации:\n"
        text += "https://oauth.mail.ru/login\n\n"
        text += "2. Введите полученный код командой:\n"
        text += "/mailru_token <ваш_код>\n\n"
        text += "Или просто отправьте публичную ссылку на файл\n"
        text += "для скачивания (не требует авторизации)."

    keyboard = []
    if mailru_connected:
        keyboard.append([InlineKeyboardButton("❌ Отключить облако", callback_data="mailru_disconnect")])

    keyboard.append([InlineKeyboardButton("🔙 В главное меню", callback_data="cmd_start")])
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(text, reply_markup=reply_markup)


async def mailru_token(update, context):
    """Сохраняет OAuth-токен Mail.ru."""
    if not context.args:
        await update.message.reply_text(
            "❌ Использование: /mailru_token <код>\n\n"
            "Получите код на https://oauth.mail.ru/login"
        )
        return

    token_code = context.args[0]
    user_id = update.message.chat_id

    user_settings = settings.load_settings(user_id)
    user_settings['mailru_token'] = token_code
    settings.save_settings(user_id, user_settings)

    await update.message.reply_text("✅ Токен Mail.ru Cloud сохранён!")


async def mailru_disconnect(update, context):
    """Удаляет токен Mail.ru (callback и command alias)."""
    query = getattr(update, 'callback_query', None)

    if query:
        await query.answer()
        user_id = query.message.chat_id
    else:
        user_id = update.message.chat_id

    user_settings = settings.load_settings(user_id)

    if 'mailru_token' in user_settings:
        del user_settings['mailru_token']
        settings.save_settings(user_id, user_settings)

    if query:
        try:
            await query.edit_message_text("✅ Облако Mail.ru отключено")
        except Exception:
            await context.bot.send_message(user_id, "✅ Облако Mail.ru отключено")
    else:
        await update.message.reply_text("✅ Облако Mail.ru отключено")


async def handle_mailru_link(update, context):
    """Обрабатывает ссылки на Mail.ru Cloud через общий пайплайн обработки."""
    url = update.message.text.strip()

    from security import is_valid_mailru_url

    if not is_valid_mailru_url(url):
        return

    logging.info("event=source_received chat_id=%s source_type=mailru_url", update.message.chat.id)
    set_last_source(context, url, 'mailru_url')

    await update.message.reply_text("☁️ Получаю файл из Mail.ru Cloud...")

    try:
        file_info = mailru.get_mailru_file_info(url)
        if file_info:
            await update.message.reply_text(
                f"📄 Найден файл: {file_info['title']}\n"
                f"Размер: {file_info['size'] / 1024 / 1024:.1f} МБ"
            )
    except Exception:
        # Нефатально: даже без метаданных продолжаем обработку.
        pass

    await process_source(update, context, url, source_type='mailru_url', random_cut=False)
