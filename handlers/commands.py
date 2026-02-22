# handlers/commands.py — обработчики команд бота

import os
import asyncio
import logging
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from transcribe import get_model_info
from moments import get_best_moments
from download import download_video
from utils import list_available_videos, list_ready_videos, TempFileManager, get_video_thumbnail
from security import is_valid_youtube_url, is_valid_mailru_url
import settings
import cache
import mailru
from handlers.helpers import (
    set_last_source,
    get_last_source,
    set_awaiting_source,
    clear_awaiting_source,
)


ACTION_PREVIEW = "preview"
ACTION_MOMENTS = "moments"


def _get_context_args(context):
    args = getattr(context, "args", None)
    if not args:
        return []
    return args


def _resolve_source_kind(source):
    if not source or not isinstance(source, str):
        return None

    stripped = source.strip()
    if os.path.exists(stripped):
        return "local_path"
    if is_valid_youtube_url(stripped):
        return "youtube_url"
    if is_valid_mailru_url(stripped):
        return "mailru_url"
    return None


def _build_back_to_menu_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔙 В главное меню", callback_data="cmd_start")]
    ])


async def prompt_source_for_action(update, context, action):
    """Переводит пользователя в режим ввода источника для preview/moments."""
    set_awaiting_source(context, action)

    action_label = "превью" if action == ACTION_PREVIEW else "лучших моментов"
    text = (
        f"🧭 Режим {action_label}\n\n"
        "Отправьте один из источников:\n"
        "• ссылку YouTube\n"
        "• публичную ссылку Mail.ru\n"
        "• локальный путь к файлу\n\n"
        "Или выберите источник кнопкой ниже."
    )

    keyboard = [[InlineKeyboardButton("📂 Выбрать из моих видео", callback_data="src_pick_open")]]

    if get_last_source(context):
        keyboard.append([InlineKeyboardButton("⚡ Последний источник", callback_data="src_pick_last")])

    keyboard.append([InlineKeyboardButton("🔙 В главное меню", callback_data="cmd_start")])

    await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))


async def start(update, context):
    """Обработчик команды /start."""
    text = (
        "<b>UploadBot</b> — монтаж Shorts/TikTok в 1-2 шага.\n\n"
        "<b>Основное:</b> Мои видео, Готовые видео, Быстрый запуск, Настройки\n"
        "<b>Креатив:</b> Превью, Лучшие моменты\n"
        "<b>Диагностика:</b> LLM, GPU, Кэш\n\n"
        "Отправьте ссылку YouTube, Mail.ru или загрузите видео в чат."
    )

    keyboard = [
        [InlineKeyboardButton("📂 Мои видео", callback_data="cmd_list_videos")],
        [InlineKeyboardButton("✅ Готовые видео", callback_data="cmd_ready_videos")],
        [InlineKeyboardButton("⚡ Быстрый запуск", callback_data="action_process_last")],
        [InlineKeyboardButton("⚙️ Настройки", callback_data="cmd_settings")],
        [InlineKeyboardButton("🖼️ Превью", callback_data="action_preview")],
        [InlineKeyboardButton("🎯 Моменты", callback_data="action_moments")],
        [InlineKeyboardButton("☁️ Mail.ru", callback_data="cmd_mailru")],
        [InlineKeyboardButton("🤖 LLM", callback_data="cmd_llm")],
        [InlineKeyboardButton("🖥️ GPU", callback_data="cmd_gpu")],
        [InlineKeyboardButton("💾 Кэш", callback_data="cmd_cache")],
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode="HTML")


async def list_videos(update, context):
    """Показывает список исходных (необработанных) видео файлов."""
    videos = list_available_videos()

    if not videos:
        await update.message.reply_text(
            "📂 В папке исходников пока нет видео.\n"
            "Отправь ссылку на YouTube или загрузи видео-файл.",
            reply_markup=_build_back_to_menu_keyboard(),
        )
        return

    keyboard = []
    message_text = "📂 Исходные видео (videos/raw):\n\n"
    video_tokens = {}

    for idx, video in enumerate(videos[:10], 1):
        token = f"v{idx}"
        video_tokens[token] = video['path']
        message_text += f"{idx}. {video['name']}\n   Размер: {video['size']}\n\n"

        keyboard.append([
            InlineKeyboardButton(f"▶️ Full {idx}", callback_data=f"action_process_full_{token}"),
            InlineKeyboardButton(f"🎲 Random 60s {idx}", callback_data=f"action_process_random_{token}"),
        ])
        keyboard.append([
            InlineKeyboardButton(f"🖼️ Превью {idx}", callback_data=f"action_preview_{token}"),
            InlineKeyboardButton(f"🎯 Моменты {idx}", callback_data=f"action_moments_{token}"),
        ])

    # Короткие токены предотвращают переполнение callback_data при длинных именах.
    context.user_data['video_tokens'] = video_tokens

    keyboard.append([InlineKeyboardButton("✅ Показать готовые видео", callback_data="cmd_ready_videos")])
    keyboard.append([InlineKeyboardButton("🔙 В главное меню", callback_data="cmd_start")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(message_text, reply_markup=reply_markup)


async def ready_videos(update, context):
    """Показывает список готовых видео (videos/ready) и позволяет отправить их в чат."""
    videos = list_ready_videos()

    if not videos:
        await update.message.reply_text(
            "✅ В папке готовых видео пока пусто.\n"
            "Сначала обработайте исходник, и результат появится здесь.",
            reply_markup=_build_back_to_menu_keyboard(),
        )
        return

    keyboard = []
    message_text = "✅ Готовые видео (videos/ready):\n\n"
    ready_tokens = {}

    for idx, video in enumerate(videos[:10], 1):
        token = f"r{idx}"
        ready_tokens[token] = video['path']
        
        message_text += f"{idx}. {video['name']}\n"
        message_text += f"   📅 Дата: {video.get('date', 'Неизвестно')}\n"
        message_text += f"   ⏱️ Длительность: {video.get('duration', '? с')}\n"
        message_text += f"   Размер: {video['size']}\n"
        message_text += f"   📝 Описание: {video.get('description', 'Без описания')[:100]}"
        if len(video.get('description', '')) > 100:
            message_text += "...\n\n"
        else:
            message_text += "\n\n"

        keyboard.append([
            InlineKeyboardButton(f"📤 Отправить {idx}", callback_data=f"action_ready_send_{token}"),
            InlineKeyboardButton(f"🖼️ Превью {idx}", callback_data=f"action_ready_preview_{token}"),
            InlineKeyboardButton(f"❌ Удалить {idx}", callback_data=f"action_ready_delete_{token}"),
        ])

    context.user_data['ready_video_tokens'] = ready_tokens

    keyboard.append([InlineKeyboardButton("📂 К исходникам", callback_data="cmd_list_videos")])
    keyboard.append([InlineKeyboardButton("🔙 В главное меню", callback_data="cmd_start")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(message_text, reply_markup=reply_markup)


async def show_settings(update, context):
    """Показывает меню настроек обработки видео."""
    user_id = update.message.chat_id
    user_settings = settings.load_settings(user_id)
    position = user_settings.get('subtitle_position', 'bottom')
    layout_mode = user_settings.get('vertical_layout_mode', 'standard')
    facecam_subject_side = user_settings.get('facecam_subject_side', 'left')
    position_labels = {
        "bottom": "Низ",
        "center": "Центр",
        "top": "Верх",
    }
    layout_labels = {
        "standard": "Стандартный 9:16",
        "facecam_top_split": "Камера сверху 1/3",
    }
    side_labels = {
        "left": "Слева",
        "right": "Справа",
        "auto": "Авто",
    }

    text = "⚙️ Настройки обработки видео:\n\n"
    text += f"📏 Макс. длительность: {user_settings['max_clip_duration']} сек\n"
    text += f"🔇 Удаление тишины: {'✅' if user_settings['remove_silence'] else '❌'}\n"
    text += f"📝 Субтитры: {'✅' if user_settings['add_subtitles'] else '❌'}\n"
    text += f"✂️ Клипов на выходе: {user_settings.get('max_clips_to_process', 3)}\n"
    text += f"📍 Позиция субтитров: {position_labels.get(position, 'Низ')}\n"
    text += f"📱 Режим кадрирования: {layout_labels.get(layout_mode, 'Стандартный 9:16')}\n"
    text += f"🎯 Лицо в кадре: {side_labels.get(facecam_subject_side, 'Слева')}\n"
    text += f"🏷️ Хештегов: {user_settings['hashtag_count']}\n"
    text += f"🖥️ GPU: {'✅' if user_settings.get('use_gpu', False) else '❌'}\n"
    text += f"🎤 Whisper: {user_settings.get('whisper_model', 'base')}\n"
    text += f"🤖 LLM-подписи: {'✅' if user_settings.get('use_llm', False) else '❌'} ({user_settings.get('llm_provider', 'openai')})\n"

    keyboard = [
        [InlineKeyboardButton(f"📏 Макс. длительность: {user_settings['max_clip_duration']}", callback_data="setting_duration")],
        [InlineKeyboardButton(f"🔇 Тишина: {'✅' if user_settings['remove_silence'] else '❌'}", callback_data="setting_silence")],
        [InlineKeyboardButton(f"✂️ Клипов (Top-K): {user_settings.get('max_clips_to_process', 3)}", callback_data="setting_max_clips")],
        [InlineKeyboardButton(f"📝 Субтитры: {'✅' if user_settings['add_subtitles'] else '❌'}", callback_data="setting_subtitles")],
        [InlineKeyboardButton(f"📍 Позиция: {position_labels.get(position, 'Низ')}", callback_data="setting_subtitle_position")],
        [InlineKeyboardButton(f"📱 Кадрирование: {layout_labels.get(layout_mode, 'Стандартный 9:16')}", callback_data="setting_vertical_layout")],
        [InlineKeyboardButton(f"🎯 Лицо: {side_labels.get(facecam_subject_side, 'Слева')}", callback_data="setting_facecam_side")],
        [InlineKeyboardButton(f"🏷️ Хештеги: {user_settings['hashtag_count']}", callback_data="setting_hashtags")],
        [InlineKeyboardButton(f"🖥️ GPU: {'✅' if user_settings.get('use_gpu', False) else '❌'}", callback_data="setting_gpu")],
        [InlineKeyboardButton(f"🎤 Whisper: {user_settings.get('whisper_model', 'base')}", callback_data="setting_whisper")],
        [InlineKeyboardButton(f"🤖 LLM-подписи: {'✅' if user_settings.get('use_llm', False) else '❌'}", callback_data="setting_llm")],
        [InlineKeyboardButton("🔄 Сбросить настройки", callback_data="setting_reset")],
        [InlineKeyboardButton("🔙 В главное меню", callback_data="cmd_start")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(text, reply_markup=reply_markup)


async def check_gpu(update, context):
    """Проверяет доступность GPU и информацию о системе."""
    info = get_model_info()

    text = "🖥️ Системная информация\n\n"
    text += f"GPU доступен: {'✅ Да' if info['gpu_available'] else '❌ Нет'}\n"

    if info['gpu_available']:
        text += f"GPU: {info.get('gpu_name', 'Unknown')}\n"
        text += f"Память: {info.get('gpu_memory', 0):.1f} ГБ\n"

    text += f"Устройство: {info['device']}\n"
    text += f"Рекомендуемая модель Whisper: {info['recommended_model']}\n"
    text += f"Доступные модели: {', '.join(info['models'])}\n\n"
    text += "Изменить модель можно в настройках (/settings)"

    await update.message.reply_text(text, reply_markup=_build_back_to_menu_keyboard())


async def cache_status(update, context):
    """Показывает статус кэша."""
    cache_size = cache.get_cache_size()
    formatted_size = cache.format_cache_size(cache_size)

    text = "💾 Кэш транскрипций\n\n"
    text += f"Размер кэша: {formatted_size}\n\n"
    text += "Команды:\n"
    text += "/cache_clear - Очистить кэш старше 30 дней\n"
    text += f"Директория: {cache.CACHE_DIR}/"

    await update.message.reply_text(text, reply_markup=_build_back_to_menu_keyboard())


async def cache_clear(update, context):
    """Очищает старые файлы кэша."""
    count, size = cache.clear_cache()

    text = "🧹 Кэш очищен\n\n"
    text += f"Удалено файлов: {count}\n"
    text += f"Освобождено: {size}"

    await update.message.reply_text(text)


async def llm_status(update, context):
    """Проверяет доступность LLM и настройки."""
    from llm import check_llm_available
    from subtitles import get_available_styles, get_style_preview

    llm_check = check_llm_available()

    text = "🤖 LLM и субтитры\n\n"
    text += f"LLM доступен: {'✅ Да' if llm_check['available'] else '❌ Нет'}\n"
    text += f"Провайдер: {llm_check.get('provider', 'unknown')}\n\n"

    text += "Доступные стили субтитров:\n"
    styles = get_available_styles()
    for style in styles:
        preview = get_style_preview(style)
        text += f"• {preview['name']}: {preview['preview'][:40]}...\n"

    text += "\nНастройте в /settings\n"
    await update.message.reply_text(text, reply_markup=_build_back_to_menu_keyboard())


async def moments_from_source(update, context, source):
    """Находит лучшие моменты для указанного источника."""
    source = (source or "").strip()
    source_kind = _resolve_source_kind(source)

    if not source_kind:
        logging.info("event=dead_end reason=invalid_source_for_moments source=%s", source[:120])
        await update.message.reply_text(
            "❌ Неподдерживаемый источник для поиска моментов.",
            reply_markup=_build_back_to_menu_keyboard(),
        )
        return

    if source_kind == "local_path":
        set_last_source(context, source, "local_path")
    elif source_kind == "youtube_url":
        set_last_source(context, source, "url")
    elif source_kind == "mailru_url":
        set_last_source(context, source, "mailru_url")

    await update.message.reply_text(f"🎯 Анализирую видео: {source[:60]}...")

    with TempFileManager() as temp_mgr:
        try:
            if source_kind == "youtube_url":
                video_path = await asyncio.to_thread(download_video, source, temp_mgr.temp_dir)
            elif source_kind == "mailru_url":
                video_path = await asyncio.to_thread(mailru.download_from_mailru_public, source, temp_mgr.temp_dir)
            else:
                video_path = source

            if not video_path or not os.path.exists(video_path):
                await update.message.reply_text(
                    "❌ Не удалось загрузить видео.",
                    reply_markup=_build_back_to_menu_keyboard(),
                )
                return

            await update.message.reply_text("🎬 Ищу лучшие моменты...")
            moments = get_best_moments(video_path, num_moments=3)

            if not moments:
                await update.message.reply_text(
                    "❌ Не удалось найти моменты.",
                    reply_markup=_build_back_to_menu_keyboard(),
                )
                return

            text = "🎯 Лучшие моменты:\n\n"
            moment_tokens = {}
            keyboard = []

            for idx, moment in enumerate(moments, 1):
                token = f"m{idx}"
                start = float(moment.get("start", 0))
                end = float(moment.get("end", start))
                duration = max(1.0, end - start)
                moment_type = moment.get("type", "unknown")
                score = float(moment.get("score", 0))

                text += f"{idx}. {moment_type} ({duration:.1f}с) — {start:.0f}s-{end:.0f}s\n"
                text += f"   Оценка: {score:.2f}\n\n"

                moment_tokens[token] = {
                    "source": source,
                    "source_kind": source_kind,
                    "start": start,
                    "end": end,
                    "duration": duration,
                }
                keyboard.append([
                    InlineKeyboardButton(
                        f"✂️ Вырезать момент {idx}",
                        callback_data=f"extract_moment_{token}",
                    )
                ])

            context.user_data['moment_tokens'] = moment_tokens
            keyboard.append([InlineKeyboardButton("🔙 В главное меню", callback_data="cmd_start")])

            clear_awaiting_source(context)
            await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

        except Exception as e:
            logging.exception("Ошибка поиска моментов")
            await update.message.reply_text(f"❌ Ошибка: {e}", reply_markup=_build_back_to_menu_keyboard())


async def moments_detect(update, context):
    """Точка входа команды /moments: интерактивный режим или прямой запуск с аргументом."""
    args = _get_context_args(context)
    if not args:
        await prompt_source_for_action(update, context, ACTION_MOMENTS)
        return

    source = " ".join(args)
    await moments_from_source(update, context, source)


async def preview_from_source(update, context, source):
    """Создает превью для указанного источника."""
    source = (source or "").strip()
    source_kind = _resolve_source_kind(source)

    if not source_kind:
        logging.info("event=dead_end reason=invalid_source_for_preview source=%s", source[:120])
        await update.message.reply_text(
            "❌ Неподдерживаемый источник. Отправьте ссылку YouTube/Mail.ru или путь к файлу.",
            reply_markup=_build_back_to_menu_keyboard(),
        )
        return

    if source_kind == "local_path":
        set_last_source(context, source, "local_path")
        context.user_data['pending_source'] = source
        context.user_data['pending_type'] = 'local_path'
    elif source_kind == "youtube_url":
        set_last_source(context, source, "url")
        context.user_data['pending_source'] = source
        context.user_data['pending_type'] = 'url'
    elif source_kind == "mailru_url":
        set_last_source(context, source, "mailru_url")
        context.user_data['pending_source'] = source
        context.user_data['pending_type'] = 'mailru_url'

    await update.message.reply_text(f"🎬 Создаю превью: {source[:60]}...")

    with TempFileManager() as temp_mgr:
        try:
            if source_kind == "mailru_url":
                file_path = await asyncio.to_thread(mailru.download_from_mailru_public, source, temp_mgr.temp_dir)
            elif source_kind == "youtube_url":
                file_path = await asyncio.to_thread(download_video, source, temp_mgr.temp_dir)
            else:
                file_path = source

            if not file_path or not os.path.exists(file_path):
                await update.message.reply_text("❌ Не удалось скачать/прочитать видео", reply_markup=_build_back_to_menu_keyboard())
                return

            thumbnail = get_video_thumbnail(file_path, temp_mgr.temp_dir)

            keyboard = []
            can_process_now = source_kind in {"local_path", "youtube_url", "mailru_url"}
            if can_process_now:
                keyboard.append([
                    InlineKeyboardButton("▶️ Обработать полностью", callback_data="mode_full"),
                    InlineKeyboardButton("🎲 Random 60s", callback_data="mode_random"),
                ])
            keyboard.append([InlineKeyboardButton("🔙 В главное меню", callback_data="cmd_start")])
            reply_markup = InlineKeyboardMarkup(keyboard)

            if thumbnail:
                with open(thumbnail, 'rb') as f:
                    await update.message.reply_photo(
                        f,
                        caption=(
                            "🖼️ Превью готово.\n\n"
                            "Если всё ок, запустите обработку кнопкой ниже."
                            if can_process_now
                            else "🖼️ Превью готово."
                        ),
                        reply_markup=reply_markup,
                    )
            else:
                await update.message.reply_text("🖼️ Превью создано.", reply_markup=reply_markup)

            clear_awaiting_source(context)

        except Exception as e:
            logging.exception("Ошибка создания превью")
            await update.message.reply_text(f"❌ Ошибка: {e}", reply_markup=_build_back_to_menu_keyboard())


async def preview_video(update, context):
    """Точка входа команды /preview: интерактивный режим или прямой запуск с аргументом."""
    args = _get_context_args(context)
    if not args:
        await prompt_source_for_action(update, context, ACTION_PREVIEW)
        return

    source = " ".join(args)
    await preview_from_source(update, context, source)
