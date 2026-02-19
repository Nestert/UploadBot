# handlers/message_handlers.py — обработчики сообщений

import logging
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from security import is_valid_youtube_url, is_valid_mailru_url
from handlers.commands import (
    ACTION_PREVIEW,
    ACTION_MOMENTS,
    preview_from_source,
    moments_from_source,
)
from handlers.helpers import set_last_source
from handlers.mailru_handlers import handle_mailru_link


async def process_link(update, context):
    """Обрабатывает YouTube-ссылки и предлагает режим обработки."""
    youtube_url = update.message.text.strip()

    if not is_valid_youtube_url(youtube_url):
        return False

    context.user_data['pending_source'] = youtube_url
    context.user_data['pending_type'] = 'url'
    set_last_source(context, youtube_url, 'url')

    logging.info("event=source_received chat_id=%s source_type=url", update.message.chat.id)

    keyboard = [
        [InlineKeyboardButton("▶️ Обработать полностью", callback_data="mode_full")],
        [InlineKeyboardButton("🎲 Случайный фрагмент (60s)", callback_data="mode_random")],
        [InlineKeyboardButton("❌ Отмена", callback_data=f"cancel_{update.message.chat.id}")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        f"📹 Ссылка принята: {youtube_url}\n\nВыберите режим обработки:",
        reply_markup=reply_markup,
        disable_web_page_preview=True,
    )
    return True


async def process_video(update, context):
    """Обрабатывает загруженное пользователем видео-файл."""
    video = update.message.video

    context.user_data['pending_source'] = video.file_id
    context.user_data['pending_type'] = 'file'
    context.user_data['pending_filename'] = video.file_name or "telegram_video.mp4"
    set_last_source(context, video.file_id, 'file')

    logging.info("event=source_received chat_id=%s source_type=file", update.message.chat.id)

    keyboard = [
        [InlineKeyboardButton("▶️ Обработать полностью", callback_data="mode_full")],
        [InlineKeyboardButton("🎲 Случайный фрагмент (60s)", callback_data="mode_random")],
        [InlineKeyboardButton("❌ Отмена", callback_data=f"cancel_{update.message.chat.id}")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        f"📹 Видео получено: {video.file_name or 'video.mp4'}\n\nВыберите режим обработки:",
        reply_markup=reply_markup,
    )


async def route_text_message(update, context):
    """Единый роутер текстовых сообщений: awaiting_source -> youtube -> mailru -> fallback."""
    text = (update.message.text or "").strip()
    if not text:
        return

    chat_id = update.message.chat.id

    if context.user_data.get('awaiting_source'):
        action = context.user_data.get('awaiting_action')
        logging.info("event=source_received chat_id=%s source_type=awaiting action=%s", chat_id, action)

        if action == ACTION_PREVIEW:
            await preview_from_source(update, context, text)
            return
        if action == ACTION_MOMENTS:
            await moments_from_source(update, context, text)
            return

        logging.info("event=dead_end reason=unknown_awaiting_action chat_id=%s action=%s", chat_id, action)
        await update.message.reply_text(
            "❌ Не удалось определить ожидаемое действие. Открой /start и выбери пункт заново."
        )
        return

    if await process_link(update, context):
        return

    if is_valid_mailru_url(text):
        logging.info("event=source_received chat_id=%s source_type=mailru_url", chat_id)
        await handle_mailru_link(update, context)
        return

    logging.info("event=dead_end reason=unsupported_text chat_id=%s", chat_id)
    await update.message.reply_text(
        "Не понял источник. Отправьте ссылку YouTube/Mail.ru, видео-файл или используйте /start."
    )


async def error_handler(update, context):
    """Глобальный обработчик исключений для Telegram-бота."""
    logging.exception("Исключение в обработчике обновления", exc_info=context.error)
    try:
        if update and hasattr(update, 'message') and update.message:
            await update.message.reply_text("Произошла внутренняя ошибка. Повторите попытку позже.")
    except Exception:
        pass
