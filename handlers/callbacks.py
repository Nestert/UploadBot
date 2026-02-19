# handlers/callbacks.py — обработчики callback-кнопок

import os
import asyncio
import logging
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import TimedOut

from download import download_video
from utils import get_video_path, TempFileManager, list_available_videos, cut_video_chunk
import settings
import mailru
from settings import save_settings, DEFAULT_SETTINGS

from handlers.processing import (
    processing_jobs,
    cancel_events,
    process_source,
)
from handlers.helpers import (
    create_callback_update,
    set_last_source,
    get_last_source,
    clear_awaiting_source,
)
from handlers.commands import (
    ACTION_PREVIEW,
    ACTION_MOMENTS,
    prompt_source_for_action,
    preview_from_source,
    moments_from_source,
)


def _duration_keyboard(current_duration):
    options = [15, 20, 30, 45, 60, 90]
    rows = []
    for duration in options:
        marker = "✅ " if duration == current_duration else ""
        rows.append([InlineKeyboardButton(f"{marker}{duration} сек", callback_data=f"duration_{duration}")])
    rows.append([
        InlineKeyboardButton("🔙 В настройки", callback_data="setting_back"),
        InlineKeyboardButton("🏠 В меню", callback_data="cmd_start"),
    ])
    return InlineKeyboardMarkup(rows)


def _hashtags_keyboard(current_count):
    options = [3, 5, 7, 10]
    rows = []
    for count in options:
        marker = "✅ " if count == current_count else ""
        rows.append([InlineKeyboardButton(f"{marker}{count} хештегов", callback_data=f"hashtags_{count}")])
    rows.append([
        InlineKeyboardButton("🔙 В настройки", callback_data="setting_back"),
        InlineKeyboardButton("🏠 В меню", callback_data="cmd_start"),
    ])
    return InlineKeyboardMarkup(rows)


def _whisper_keyboard(current_model):
    models = ["tiny", "base", "small", "medium", "large"]
    rows = []
    for model in models:
        marker = "✅ " if model == current_model else ""
        rows.append([InlineKeyboardButton(f"{marker}{model}", callback_data=f"whisper_{model}")])
    rows.append([
        InlineKeyboardButton("🔙 В настройки", callback_data="setting_back"),
        InlineKeyboardButton("🏠 В меню", callback_data="cmd_start"),
    ])
    return InlineKeyboardMarkup(rows)


async def _run_awaiting_action(chat_id, context, source):
    action = context.user_data.get('awaiting_action')
    callback_update = create_callback_update(chat_id, context.bot)

    if action == ACTION_PREVIEW:
        await preview_from_source(callback_update, context, source)
        return
    if action == ACTION_MOMENTS:
        await moments_from_source(callback_update, context, source)
        return

    logging.info("event=dead_end reason=missing_awaiting_action chat_id=%s", chat_id)
    clear_awaiting_source(context)
    await context.bot.send_message(
        chat_id,
        "❌ Действие устарело. Нажмите Превью или Моменты заново.",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 В главное меню", callback_data="cmd_start")]]),
    )


async def handle_cancel(update, context):
    """Обработчик отмены обработки."""
    query = update.callback_query

    try:
        await query.answer()
    except Exception as e:
        logging.warning(f"Не удалось ответить на callback: {e}")

    chat_id = int(query.data.replace("cancel_", ""))

    if chat_id in cancel_events:
        try:
            cancel_events[chat_id].set()
        except Exception as e:
            logging.warning(f"Не удалось установить cancel event: {e}")

    if chat_id in processing_jobs:
        tracker = processing_jobs[chat_id]
        tracker.cancel()
        del processing_jobs[chat_id]

    try:
        await query.edit_message_text("❌ Обработка отменена")
    except Exception as e:
        logging.warning(f"Не удалось обновить сообщение отмены: {e}")


async def handle_retry(update, context):
    """Обработчик повторной обработки."""
    query = update.callback_query

    try:
        await query.answer()
    except Exception as e:
        logging.warning(f"Не удалось ответить на callback retry: {e}")

    try:
        await query.edit_message_text("🔄 Повторите отправку ссылки или видео для обработки")
    except Exception as e:
        logging.warning(f"Не удалось обновить сообщение retry: {e}")


async def handle_settings_callback(update, context):
    """Обработчик callback-запросов для настроек."""
    from handlers.commands import show_settings

    query = update.callback_query
    await query.answer()

    user_id = query.message.chat_id
    user_settings = settings.load_settings(user_id)

    if query.data == "setting_silence":
        user_settings['remove_silence'] = not user_settings['remove_silence']
        save_settings(user_id, user_settings)

    elif query.data == "setting_subtitles":
        user_settings['add_subtitles'] = not user_settings['add_subtitles']
        save_settings(user_id, user_settings)

    elif query.data == "setting_subtitle_position":
        positions = ["bottom", "center", "top"]
        current = user_settings.get("subtitle_position", "bottom")
        try:
            idx = positions.index(current)
        except ValueError:
            idx = 0
        user_settings["subtitle_position"] = positions[(idx + 1) % len(positions)]
        save_settings(user_id, user_settings)

    elif query.data == "setting_vertical_layout":
        layouts = ["standard", "facecam_top_split"]
        current = user_settings.get("vertical_layout_mode", "standard")
        try:
            idx = layouts.index(current)
        except ValueError:
            idx = 0
        user_settings["vertical_layout_mode"] = layouts[(idx + 1) % len(layouts)]
        save_settings(user_id, user_settings)

    elif query.data == "setting_facecam_side":
        sides = ["left", "right", "auto"]
        current = user_settings.get("facecam_subject_side", "left")
        try:
            idx = sides.index(current)
        except ValueError:
            idx = 0
        user_settings["facecam_subject_side"] = sides[(idx + 1) % len(sides)]
        save_settings(user_id, user_settings)

    elif query.data == "setting_hashtags":
        text = "🏷️ Выберите количество хештегов:\n\n"
        try:
            await query.message.edit_text(
                text,
                reply_markup=_hashtags_keyboard(user_settings.get('hashtag_count', 7)),
            )
        except Exception as e:
            logging.warning(f"Не удалось открыть меню хештегов: {e}")
        return

    elif query.data.startswith("hashtags_"):
        try:
            count = int(query.data.replace("hashtags_", ""))
        except ValueError:
            count = user_settings.get('hashtag_count', 7)
        user_settings['hashtag_count'] = count
        save_settings(user_id, user_settings)

    elif query.data == "setting_gpu":
        user_settings['use_gpu'] = not user_settings.get('use_gpu', False)
        save_settings(user_id, user_settings)

    elif query.data == "setting_whisper":
        text = "🎤 Выберите модель Whisper:\n\n"
        try:
            await query.message.edit_text(
                text,
                reply_markup=_whisper_keyboard(user_settings.get('whisper_model', 'base')),
            )
        except Exception as e:
            logging.warning(f"Не удалось открыть меню Whisper: {e}")
        return

    elif query.data.startswith("whisper_"):
        model = query.data.replace("whisper_", "")
        if model in {'tiny', 'base', 'small', 'medium', 'large'}:
            user_settings['whisper_model'] = model
            save_settings(user_id, user_settings)

    elif query.data == "setting_llm":
        user_settings['use_llm'] = not user_settings.get('use_llm', False)
        save_settings(user_id, user_settings)

    elif query.data == "setting_duration":
        text = "📏 Выберите максимальную длительность клипа:\n\n"
        try:
            await query.message.edit_text(
                text,
                reply_markup=_duration_keyboard(user_settings.get('max_clip_duration', 90)),
            )
        except Exception as e:
            logging.warning(f"Не удалось обновить меню длительности: {e}")
        return

    elif query.data.startswith("duration_"):
        duration = int(query.data.replace("duration_", ""))
        user_settings['max_clip_duration'] = duration
        if user_settings['min_clip_duration'] >= duration:
            user_settings['min_clip_duration'] = max(5, duration - 10)
        save_settings(user_id, user_settings)

    elif query.data == "setting_reset":
        save_settings(user_id, DEFAULT_SETTINGS.copy())

    elif query.data == "setting_back":
        callback_update = create_callback_update(user_id, context.bot)
        await show_settings(callback_update, context)
        try:
            await query.delete_message()
        except Exception:
            pass
        return

    if not query.data.startswith(("duration_", "hashtags_", "whisper_")):
        try:
            await query.message.delete()
        except Exception as e:
            logging.warning(f"Не удалось удалить сообщение: {e}")

    callback_update = create_callback_update(user_id, context.bot)
    await show_settings(callback_update, context)


async def handle_interface_callback(update, context):
    """Обработник кнопок быстрого доступа."""
    from handlers.commands import (
        start,
        list_videos,
        ready_videos,
        show_settings,
        preview_video,
        moments_detect,
        llm_status,
        check_gpu,
        cache_status,
    )
    from handlers.mailru_handlers import mailru_link

    query = update.callback_query
    await query.answer()

    logging.info("event=menu_click chat_id=%s action=%s", query.message.chat_id, query.data)

    command_map = {
        "cmd_start": (start, "Главное меню"),
        "cmd_list_videos": (list_videos, "Мои видео"),
        "cmd_ready_videos": (ready_videos, "Готовые видео"),
        "cmd_settings": (show_settings, "Настройки"),
        "cmd_preview": (preview_video, "Превью"),
        "cmd_moments": (moments_detect, "Моменты"),
        "cmd_llm": (llm_status, "LLM"),
        "cmd_gpu": (check_gpu, "GPU"),
        "cmd_cache": (cache_status, "Кэш"),
        "cmd_mailru": (mailru_link, "Mail.ru"),
    }

    cmd_info = command_map.get(query.data)
    if not cmd_info:
        try:
            await query.edit_message_text("❌ Неизвестная команда")
        except Exception:
            pass
        return

    func, name = cmd_info
    chat_id = query.message.chat_id

    try:
        callback_update = create_callback_update(chat_id, context.bot)
        await func(callback_update, context)

        try:
            await query.delete_message()
        except Exception:
            pass

    except Exception as e:
        logging.exception(f"Ошибка в {name}: {e}")
        try:
            await context.bot.send_message(
                chat_id,
                f"❌ Ошибка: {str(e)[:100]}",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 В главное меню", callback_data="cmd_start")]]),
            )
        except Exception:
            pass


async def handle_action_entry(update, context):
    """Точки входа action_preview/action_moments."""
    query = update.callback_query
    await query.answer()

    chat_id = query.message.chat_id
    logging.info("event=menu_click chat_id=%s action=%s", chat_id, query.data)

    if query.data == "action_preview":
        callback_update = create_callback_update(chat_id, context.bot)
        await prompt_source_for_action(callback_update, context, ACTION_PREVIEW)
    elif query.data == "action_moments":
        callback_update = create_callback_update(chat_id, context.bot)
        await prompt_source_for_action(callback_update, context, ACTION_MOMENTS)
    else:
        await query.edit_message_text("❌ Неизвестное действие")
        return

    try:
        await query.delete_message()
    except Exception:
        pass


async def handle_source_picker(update, context):
    """Обработчик src_pick_*: выбор источника из списка или последнего."""
    query = update.callback_query
    await query.answer()

    chat_id = query.message.chat_id
    data = query.data

    if data == "src_pick_open":
        videos = list_available_videos()
        if not videos:
            await query.edit_message_text(
                "📂 В списке пока нет видео. Отправьте ссылку или файл.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 В меню", callback_data="cmd_start")]]),
            )
            return

        video_tokens = {}
        keyboard = []
        action = context.user_data.get('awaiting_action', ACTION_PREVIEW)
        action_label = "превью" if action == ACTION_PREVIEW else "моменты"

        for idx, video in enumerate(videos[:10], 1):
            token = f"v{idx}"
            video_tokens[token] = video['path']
            name = video['name']
            if len(name) > 40:
                name = name[:37] + "..."
            keyboard.append([InlineKeyboardButton(f"{idx}. {name}", callback_data=f"src_pick_{token}")])

        context.user_data['video_tokens'] = video_tokens
        keyboard.append([InlineKeyboardButton("🔙 В меню", callback_data="cmd_start")])

        await query.edit_message_text(
            f"📂 Выберите видео для действия: {action_label}",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
        return

    if data == "src_pick_last":
        last_source = get_last_source(context)
        if not last_source:
            await query.edit_message_text(
                "❌ Нет сохраненного источника. Сначала отправьте ссылку/видео.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 В меню", callback_data="cmd_start")]]),
            )
            return

        await _run_awaiting_action(chat_id, context, last_source['source'])
        try:
            await query.delete_message()
        except Exception:
            pass
        return

    token = data.replace("src_pick_", "", 1)
    video_tokens = context.user_data.get('video_tokens', {})
    video_path = video_tokens.get(token)

    if not video_path or not os.path.exists(video_path):
        await query.edit_message_text(
            "❌ Список видео устарел. Открой /list_videos ещё раз.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 В меню", callback_data="cmd_start")]]),
        )
        return

    await _run_awaiting_action(chat_id, context, video_path)

    try:
        await query.delete_message()
    except Exception:
        pass


async def handle_action_process_last(update, context):
    """Быстрый запуск обработки по последнему источнику."""
    query = update.callback_query
    await query.answer()

    chat_id = query.message.chat_id
    logging.info("event=menu_click chat_id=%s action=action_process_last", chat_id)

    last_source = get_last_source(context)
    if not last_source:
        await query.edit_message_text(
            "❌ Нет последнего источника. Сначала отправьте ссылку или выберите видео из списка.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📂 Мои видео", callback_data="cmd_list_videos")]]),
        )
        return

    source = last_source['source']
    source_type = last_source['source_type']

    if source_type not in {'url', 'file', 'local_path', 'mailru_url'}:
        logging.info("event=dead_end reason=unsupported_last_source chat_id=%s source_type=%s", chat_id, source_type)
        await query.edit_message_text(
            "❌ Последний источник нельзя запустить быстро. Выберите видео вручную.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📂 Мои видео", callback_data="cmd_list_videos")]]),
        )
        return

    context.user_data['pending_source'] = source
    context.user_data['pending_type'] = source_type

    await process_source(update, context, source, source_type=source_type, random_cut=False)


async def _resolve_token_video_path(context, token):
    video_tokens = context.user_data.get('video_tokens', {})
    video_path = video_tokens.get(token)
    if video_path and os.path.exists(video_path):
        return video_path
    return None


async def _resolve_token_ready_video_path(context, token):
    ready_tokens = context.user_data.get('ready_video_tokens', {})
    ready_path = ready_tokens.get(token)
    if ready_path and os.path.exists(ready_path):
        return ready_path
    return None


async def handle_start_existing_video(update, context):
    """Запускает обработку локального видео без промежуточного шага."""
    query = update.callback_query
    await query.answer()

    data = query.data
    mode = None
    token = None

    if data.startswith("action_process_"):
        parts = data.split("_", 3)
        if len(parts) == 4:
            mode, token = parts[2], parts[3]
    elif data.startswith("start_"):
        parts = data.split("_", 2)
        if len(parts) == 3:
            mode, token = parts[1], parts[2]

    if mode not in {"full", "random"} or not token:
        try:
            await query.edit_message_text("❌ Некорректные данные кнопки. Открой /list_videos ещё раз.")
        except Exception:
            try:
                await query.message.reply_text("❌ Некорректные данные кнопки. Открой /list_videos ещё раз.")
            except Exception:
                pass
        return

    random_cut = (mode == "random")
    video_path = await _resolve_token_video_path(context, token)

    logging.info(
        "Start existing video: mode=%s random_cut=%s token=%s token_found=%s video=%s",
        mode,
        random_cut,
        token,
        bool(video_path),
        os.path.basename(video_path) if video_path else "n/a",
    )

    if not video_path:
        stale_text = "❌ Список видео устарел. Открой /list_videos ещё раз."
        try:
            await query.edit_message_text(stale_text)
        except Exception:
            try:
                await query.message.reply_text(stale_text)
            except Exception:
                pass
        return

    context.user_data['pending_source'] = video_path
    context.user_data['pending_type'] = 'local_path'
    set_last_source(context, video_path, 'local_path')

    await process_source(update, context, video_path, source_type='local_path', random_cut=random_cut)


async def handle_preview_existing_video(update, context):
    """Создает превью из существующего видео (новый и legacy callbacks)."""
    query = update.callback_query
    await query.answer()

    data = query.data
    video_path = None

    if data.startswith("action_preview_"):
        token = data.replace("action_preview_", "", 1)
        video_path = await _resolve_token_video_path(context, token)
    elif data.startswith("preview_"):
        video_name = data.replace("preview_", "", 1)
        candidate = get_video_path(video_name)
        if os.path.exists(candidate):
            video_path = candidate

    if not video_path:
        await query.edit_message_text(
            "❌ Ошибка: видео не найдено. Используй /list_videos чтобы увидеть актуальный список."
        )
        return

    chat_id = query.message.chat_id
    callback_update = create_callback_update(chat_id, context.bot)
    await preview_from_source(callback_update, context, video_path)

    try:
        await query.delete_message()
    except Exception:
        pass


async def handle_moments_existing_video(update, context):
    """Запускает поиск моментов из списка видео по токену."""
    query = update.callback_query
    await query.answer()

    token = query.data.replace("action_moments_", "", 1)
    video_path = await _resolve_token_video_path(context, token)

    if not video_path:
        await query.edit_message_text(
            "❌ Ошибка: видео не найдено. Используй /list_videos чтобы увидеть актуальный список."
        )
        return

    chat_id = query.message.chat_id
    callback_update = create_callback_update(chat_id, context.bot)
    await moments_from_source(callback_update, context, video_path)

    try:
        await query.delete_message()
    except Exception:
        pass


async def handle_send_ready_video(update, context):
    """Отправляет выбранное готовое видео в чат повторно."""
    query = update.callback_query
    await query.answer()

    token = query.data.replace("action_ready_send_", "", 1)
    ready_path = await _resolve_token_ready_video_path(context, token)
    if not ready_path:
        await query.edit_message_text("❌ Список готовых видео устарел. Открой /ready_videos ещё раз.")
        return

    chat_id = query.message.chat_id
    try:
        sent = False
        upload_caption = "✅ Готовое видео из архива"

        # Первая попытка: send_video с увеличенными таймаутами.
        try:
            with open(ready_path, 'rb') as f:
                await context.bot.send_video(
                    chat_id=chat_id,
                    video=f,
                    caption=upload_caption,
                    read_timeout=300,
                    write_timeout=300,
                    connect_timeout=30,
                    pool_timeout=30,
                )
            sent = True
        except TimedOut:
            logging.warning("send_video timed out for ready video: %s", ready_path)

        # Вторая попытка: send_document как fallback.
        if not sent:
            with open(ready_path, 'rb') as f:
                await context.bot.send_document(
                    chat_id=chat_id,
                    document=f,
                    caption=upload_caption,
                    read_timeout=300,
                    write_timeout=300,
                    connect_timeout=30,
                    pool_timeout=30,
                )
            sent = True

        if not sent:
            raise Exception("Не удалось отправить видео")

        set_last_source(context, ready_path, 'local_path')
        await query.edit_message_text("✅ Видео отправлено в чат")
    except Exception as e:
        logging.exception("Не удалось отправить готовое видео")
        await query.edit_message_text(
            "❌ Ошибка отправки. Файл слишком большой или сеть медленная.\n"
            "Попробуйте снова через 1-2 минуты."
        )


async def handle_preview_ready_video(update, context):
    """Показывает превью выбранного готового видео."""
    query = update.callback_query
    await query.answer()

    token = query.data.replace("action_ready_preview_", "", 1)
    ready_path = await _resolve_token_ready_video_path(context, token)
    if not ready_path:
        await query.edit_message_text("❌ Список готовых видео устарел. Открой /ready_videos ещё раз.")
        return

    chat_id = query.message.chat_id
    callback_update = create_callback_update(chat_id, context.bot)
    await preview_from_source(callback_update, context, ready_path)

    try:
        await query.delete_message()
    except Exception:
        pass


async def handle_extract_moment(update, context):
    """Вырезает выбранный момент из ранее проанализированного видео."""
    query = update.callback_query
    await query.answer()

    token = query.data.replace("extract_moment_", "", 1)
    moment_tokens = context.user_data.get('moment_tokens', {})
    moment_data = moment_tokens.get(token)

    if not moment_data:
        await query.edit_message_text(
            "❌ Данные моментов устарели. Нажмите /moments и запустите анализ заново."
        )
        return

    source = moment_data.get('source')
    source_kind = moment_data.get('source_kind')
    start_time = float(moment_data.get('start', 0))
    duration = max(1.0, float(moment_data.get('duration', 1.0)))

    chat_id = query.message.chat_id
    status_message = None

    try:
        status_message = await query.message.reply_text("✂️ Вырезаю выбранный момент...")
        with TempFileManager() as temp_mgr:
            if source_kind == "local_path":
                video_path = source
            elif source_kind == "youtube_url":
                video_path = await asyncio.to_thread(download_video, source, temp_mgr.temp_dir)
            elif source_kind == "mailru_url":
                video_path = await asyncio.to_thread(mailru.download_from_mailru_public, source, temp_mgr.temp_dir)
            else:
                raise Exception("Неподдерживаемый тип источника")

            if not video_path or not os.path.exists(video_path):
                raise Exception("Не удалось получить видео для нарезки")

            chunk_path = temp_mgr.get_path('video', 'moment_clip.mp4')
            success = await asyncio.to_thread(cut_video_chunk, video_path, chunk_path, start_time, duration)
            if not success or not os.path.exists(chunk_path):
                raise Exception("Не удалось вырезать фрагмент")

            caption = f"🎬 Момент {token}: {start_time:.0f}s - {start_time + duration:.0f}s"
            with open(chunk_path, 'rb') as f:
                await context.bot.send_video(chat_id=chat_id, video=f, caption=caption)

        if status_message:
            await status_message.edit_text("✅ Момент отправлен")

    except Exception as e:
        logging.exception("Ошибка вырезки момента")
        if status_message:
            await status_message.edit_text(f"❌ Ошибка: {e}")
        else:
            await query.message.reply_text(f"❌ Ошибка: {e}")


async def handle_processing_mode(update, context):
    """Обработчик выбора режима (full/random)."""
    query = update.callback_query

    random_cut = (query.data in {"process_random", "mode_random"})

    source = context.user_data.get('pending_source')
    source_type = context.user_data.get('pending_type')

    source_display = "n/a"
    if isinstance(source, str):
        source_display = os.path.basename(source) if source_type == 'local_path' else source[:80]
    elif source is not None:
        source_display = str(source)

    logging.info(
        "Processing mode selected: callback=%s source_type=%s random_cut=%s source=%s",
        query.data,
        source_type,
        random_cut,
        source_display,
    )

    if not source:
        logging.info("event=dead_end reason=missing_pending_source chat_id=%s", query.message.chat_id)
        try:
            await query.message.edit_text("❌ Данные устарели. Отправьте ссылку или видео заново.")
        except Exception:
            await query.message.reply_text("❌ Данные устарели. Отправьте ссылку или видео заново.")
        return

    await process_source(update, context, source, source_type=source_type, random_cut=random_cut)
