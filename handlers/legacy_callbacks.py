# handlers/legacy_callbacks.py — legacy callback-обработчики

import os
import logging
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from utils import get_video_path
from handlers.callbacks import handle_processing_mode
from handlers.helpers import set_last_source


async def process_existing_video(update, context):
    """Legacy: обработка видео по старому callback формата process_<video_name>."""
    query = update.callback_query
    await query.answer()

    video_name = query.data.replace("process_", "", 1)
    video_path = get_video_path(video_name)

    if not os.path.exists(video_path):
        try:
            await query.edit_message_text(
                f"❌ Ошибка: видео {video_name} не найдено.\n"
                "Используй /list_videos чтобы увидеть актуальный список."
            )
        except Exception:
            try:
                await query.message.reply_text(
                    f"❌ Ошибка: видео {video_name} не найдено.\n"
                    "Используй /list_videos чтобы увидеть актуальный список."
                )
            except Exception:
                pass
        return

    context.user_data['pending_source'] = video_path
    context.user_data['pending_type'] = 'local_path'
    set_last_source(context, video_path, 'local_path')

    keyboard = [
        [InlineKeyboardButton("▶️ Обработать полностью", callback_data="mode_full")],
        [InlineKeyboardButton("🎲 Случайный фрагмент (60s)", callback_data="mode_random")],
        [InlineKeyboardButton("❌ Отмена", callback_data=f"cancel_{query.message.chat.id}")],
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    try:
        await query.edit_message_text(
            f"🎬 Выбрано видео: {video_name}\n\nВыберите режим обработки:",
            reply_markup=reply_markup,
        )
    except Exception as e:
        logging.warning(f"Не удалось обновить legacy меню режима: {e}")
        try:
            await query.message.reply_text(
                f"🎬 Выбрано видео: {video_name}\n\nВыберите режим обработки:",
                reply_markup=reply_markup,
            )
        except Exception:
            pass


async def handle_legacy_processing_mode(update, context):
    """Legacy: process_full/process_random."""
    await handle_processing_mode(update, context)
