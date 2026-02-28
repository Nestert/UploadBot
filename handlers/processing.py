# handlers/processing.py — логика обработки видео

import asyncio
import logging
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from errors import CancellationError, DownloadError, VideoProcessingError

from download import download_video
from transcribe import transcribe_audio_with_timestamps, quick_word_density
from scenes import detect_scenes
from autoedit import cut_silence
from moments import rank_clips
from vertical import convert_to_vertical
from subtitles import create_ass_subtitles_from_words
from tagging import generate_tags
import mailru
import cache
import settings
import os
import random
from utils import (
    TempFileManager,
    get_video_duration,
    cut_video_chunk,
    ensure_videos_dir,
    persist_unprocessed_video,
)
from handlers.helpers import set_last_source

# Глобальные состояния для обработки
executor = ThreadPoolExecutor(max_workers=4)
processing_jobs = {}
processing_queue = asyncio.Queue(maxsize=10)
cancel_events = {}

STAGES = [
    ("🔍", "Анализ видео"),
    ("🔇", "Удаление тишины"),
    ("📊", "Ранжирование моментов"),
    ("🎤", "Транскрибация клипов и субтитры"),
    ("📱", "Конвертация"),
    ("🏷️", "Генерация хештегов"),
    ("📤", "Отправка результата")
]


def _settings_cache_signature(user_settings):
    """Возвращает стабильный слепок настроек, влияющих на результат рендера."""
    return {
        "min_clip_duration": user_settings.get('min_clip_duration', 20),
        "max_clip_duration": user_settings.get('max_clip_duration', 90),
        "remove_silence": user_settings.get('remove_silence', True),
        "add_subtitles": user_settings.get('add_subtitles', True),
        "subtitle_style": user_settings.get('subtitle_style', 'subtitle'),
        "subtitle_position": user_settings.get('subtitle_position', 'bottom'),
        "hashtag_count": user_settings.get('hashtag_count', 7),
        "whisper_model": user_settings.get('whisper_model', 'base'),
        "use_gpu": user_settings.get('use_gpu', False),
        "vertical_layout_mode": user_settings.get('vertical_layout_mode', 'standard'),
        "facecam_subject_side": user_settings.get('facecam_subject_side', 'left'),
        "facecam_detector_backend": user_settings.get('facecam_detector_backend', 'yolo_window_v1'),
        "facecam_fallback_mode": user_settings.get('facecam_fallback_mode', 'hard_side'),
        "facecam_anchor": user_settings.get('facecam_anchor', 'edge_middle'),
        "use_llm": user_settings.get('use_llm', False),
        "llm_provider": user_settings.get('llm_provider', 'openai'),
    }


class ProgressTracker:
    def __init__(self, chat_id, message):
        self.chat_id = chat_id
        self.message = message
        self.current_stage = 0
        self.cancel_requested = False
        self.cancel_event = asyncio.Event()
        cancel_events[chat_id] = self.cancel_event

    async def update_stage(self, stage_name, stage_emoji):
        self.current_stage += 1
        progress = min(self.current_stage / len(STAGES) * 100, 100)

        keyboard = [[InlineKeyboardButton("❌ Отмена", callback_data=f"cancel_{self.chat_id}")]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        text = f"{stage_emoji} {stage_name}\n\n"
        text += self._create_progress_bar(progress)
        text += f"\n\nЭтап {self.current_stage} из {len(STAGES)}"

        try:
            self.message = await self.message.edit_text(text, reply_markup=reply_markup)
        except Exception as e:
            logging.warning(f"Не удалось обновить прогресс: {e}")

        return not self.cancel_requested

    def _create_progress_bar(self, percentage, length=15):
        filled = int(length * percentage // 100)
        bar = "▓" * filled + "░" * (length - filled)
        return f"[{bar}] {percentage:.0f}%"

    def is_cancelled(self):
        if self.cancel_requested:
            return True
        try:
            if self.cancel_event.is_set():
                self.cancel_requested = True
                return True
        except Exception as e:
            logging.warning(f"Ошибка проверки cancel event: {e}")
        return False

    def cancel(self):
        self.cancel_requested = True
        try:
            self.cancel_event.set()
        except Exception as e:
            logging.warning(f"Не удалось установить cancel event: {e}")

    async def complete(self, final_text="Видео успешно обработано!"):
        if self.chat_id in cancel_events:
            del cancel_events[self.chat_id]

        keyboard = [[InlineKeyboardButton("🔄 Обработать заново", callback_data=f"retry_{self.chat_id}")]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        try:
            await self.message.edit_text(f"✅ {final_text}", reply_markup=reply_markup)
        except Exception as e:
            logging.warning(f"Не удалось обновить сообщение completion: {e}")


async def process_source(update, context, source, source_type='url', random_cut=False):
    """
    Универсальная точка входа для обработки видео.
    :param source: URL, file_id или локальный путь.
    :param source_type: 'url', 'file', 'local_path'.
    :param random_cut: Если True, обрабатывается случайный фрагмент 60с.
    """
    chat_id = update.effective_chat.id
    message = update.effective_message
    source_display = source if isinstance(source, str) else str(source)
    if source_type == 'local_path' and isinstance(source, str):
        source_display = os.path.basename(source)
    elif isinstance(source_display, str):
        source_display = source_display[:80]

    logging.info(
        "process_source start: chat_id=%s source_type=%s random_cut=%s source=%s",
        chat_id,
        source_type,
        random_cut,
        source_display
    )
    if context is not None:
        set_last_source(context, source, source_type)
    logging.info(
        "event=pipeline_started chat_id=%s source_type=%s random_cut=%s",
        chat_id,
        source_type,
        random_cut,
    )
    
    # Хелпер для отправки/редактирования сообщений
    async def send_or_edit(text, reply_markup=None):
        try:
            if update.callback_query:
                try:
                    return await message.edit_text(text, reply_markup=reply_markup)
                except Exception:
                    # Если текст не изменился или сообщение удалено
                    pass
            else:
                return await message.reply_text(text, reply_markup=reply_markup)
        except Exception as e:
            logging.warning(f"Ошибка отправки сообщения: {e}")
            return None

    status_msg = await send_or_edit("⏳ Подготовка к обработке...")
    if status_msg:
        message = status_msg

    try:
        # Используем TempFileManager для скачивания/нарезки
        with TempFileManager() as temp_mgr:
            video_path = None
            source_name_hint = None
            
            if source_type == 'local_path':
                if not os.path.exists(source):
                    await send_or_edit("❌ Файл не найден.")
                    return
                video_path = source
                source_name_hint = os.path.basename(source)
                
            elif source_type == 'url':
                await send_or_edit("📥 Скачиваю видео с YouTube...")
                # Запускаем синхронную загрузку в отдельном потоке
                video_path = await asyncio.to_thread(download_video, source, temp_mgr.temp_dir)
                source_name_hint = os.path.basename(video_path) if video_path else "youtube_video.mp4"

            elif source_type == 'mailru_url':
                await send_or_edit("☁️ Скачиваю видео из Mail.ru Cloud...")
                video_path = await asyncio.to_thread(mailru.download_from_mailru_public, source, temp_mgr.temp_dir)
                source_name_hint = os.path.basename(video_path) if video_path else "mailru_video.mp4"
                
            elif source_type == 'file':
                await send_or_edit("📥 Скачиваю видео из Telegram...")
                try:
                    file_obj = await context.bot.get_file(source)
                    video_path = temp_mgr.get_path('video')
                    await file_obj.download_to_drive(video_path)
                    source_name_hint = context.user_data.get('pending_filename') or "telegram_video.mp4"
                except Exception as e:
                    raise DownloadError(f"Ошибка загрузки из Telegram: {e}")
            
            if not video_path or not os.path.exists(video_path):
                 raise DownloadError("Не удалось получить видео файл.")

            # Сохраняем исходник в videos/raw, чтобы разделить сырье и результат.
            raw_video_path = await asyncio.to_thread(
                persist_unprocessed_video,
                video_path,
                source_name_hint or os.path.basename(video_path)
            )
            if raw_video_path and os.path.exists(raw_video_path):
                video_path = raw_video_path
                if context is not None:
                    # Для быстрых повторов используем локальный исходник из raw.
                    set_last_source(context, raw_video_path, 'local_path')

            user_settings = settings.load_settings(chat_id)
            source_hash = await asyncio.to_thread(cache.get_file_hash, video_path)
            settings_signature = _settings_cache_signature(user_settings)
            cache_key = cache.build_processing_cache_key(
                source_hash or "nohash",
                settings_signature,
                random_cut=random_cut,
            )

            # Для full-режима пробуем вернуть результат из кэша без повторной обработки.
            if not random_cut:
                cached_result = cache.get_cached_processed_result(cache_key)
                if cached_result:
                    await send_or_edit("⚡ Нашел готовые видео в кэше. Отправляю без повторной обработки...")
                    caption = f"🎉 Готово из кэша!\n\n🏷️ {cached_result.get('tags', '')}"
                    if len(caption) > 900:
                        caption = caption[:900] + "..."

                    sent_count = 0
                    for cached_video in cached_result.get("videos", []):
                        try:
                            with open(cached_video, 'rb') as f:
                                await message.reply_video(
                                    f,
                                    caption=caption,
                                    read_timeout=300,
                                    write_timeout=300,
                                    connect_timeout=30,
                                    pool_timeout=30,
                                )
                            sent_count += 1
                        except Exception as video_err:
                            logging.warning("Не удалось отправить кэш как video (%s): %s", cached_video, video_err)
                            try:
                                with open(cached_video, 'rb') as f:
                                    await message.reply_document(
                                        f,
                                        caption=caption,
                                        read_timeout=300,
                                        write_timeout=300,
                                        connect_timeout=30,
                                        pool_timeout=30,
                                    )
                                sent_count += 1
                            except Exception as doc_err:
                                logging.exception(
                                    "Не удалось отправить кэшированное видео (%s): %s",
                                    cached_video,
                                    doc_err,
                                )

                    if sent_count > 0:
                        logging.info(
                            "event=pipeline_finished chat_id=%s success=true cache_hit=true sent_count=%s",
                            chat_id,
                            sent_count,
                        )
                        return

            # Логика Random Chunk
            if random_cut:
                await send_or_edit("✂️ Вырезаю случайный фрагмент (60с)...")
                try:
                    duration = await asyncio.to_thread(get_video_duration, video_path)
                    if duration > 60:
                        start_time = random.uniform(0, max(0, duration - 60))
                        logging.info(
                            "Random chunk selected: video=%s start_time=%.2f duration=60",
                            os.path.basename(video_path),
                            start_time
                        )
                        chunk_path = temp_mgr.get_path('video', 'chunk.mp4')
                        success = await asyncio.to_thread(cut_video_chunk, video_path, chunk_path, start_time, 60)
                        if success:
                            video_path = chunk_path
                        else:
                            await send_or_edit("⚠️ Не удалось вырезать фрагмент, обрабатываю полностью...")
                    else:
                         await send_or_edit("⚠️ Видео короче 60с, обрабатываю полностью...")
                except Exception as e:
                    logging.error(f"Ошибка Random Chunk: {e}")
                    await send_or_edit("⚠️ Ошибка нарезки, обрабатываю полностью...")

            # Запуск основной обработки
            cache_meta = {
                "cache_key": cache_key,
                "source_hash": source_hash,
                "settings_signature": settings_signature,
                "random_cut": random_cut,
            }
            await process_video_task(
                chat_id,
                video_path,
                message,
                context,
                user_settings=user_settings,
                cache_meta=cache_meta if not random_cut else None,
            )

    except Exception as e:
        logging.error(f"Ошибка в process_source: {e}")
        await send_or_edit(f"❌ Ошибка: {e}")


async def queue_worker():
    """
    Фоновый воркер для обработки очереди видео.
    """
    while True:
        try:
            task = await processing_queue.get()
            if task is None:
                break

            chat_id, video_path, message = task
            await process_video_task(chat_id, video_path, message)
            processing_queue.task_done()
        except Exception as e:
            logging.error(f"Ошибка в queue_worker: {e}")
            await asyncio.sleep(1)


async def process_video_task(chat_id, video_path, message, context=None, user_settings=None, cache_meta=None):
    """
    Обрабатывает одно видео из очереди.
    """
    user_settings = user_settings or settings.load_settings(chat_id)

    keyboard = [[InlineKeyboardButton("❌ Отмена", callback_data=f"cancel_{chat_id}")]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    try:
        status_msg = await message.edit_text(
            f"🎬 Обработка...\n\n⏳ Инициализация...",
            reply_markup=reply_markup
        )

        tracker = ProgressTracker(chat_id, status_msg)
        processing_jobs[chat_id] = tracker

        with TempFileManager() as temp_mgr:
            input_file = video_path
            final_videos, tags = await run_processing_pipeline(input_file, tracker, user_settings, temp_mgr)

            if chat_id in processing_jobs:
                del processing_jobs[chat_id]

            await tracker.update_stage("Отправка результата...", "📤")

            output_dir = ensure_videos_dir()
            caption = f"🎉 Готово!\n\n🏷️ {tags}"
            if len(caption) > 900:
                caption = caption[:900] + "..."

            persisted_videos = []
            for fv in final_videos:
                output_name = f"result_{uuid.uuid4().hex[:8]}.mp4"
                output_path = os.path.join(output_dir, output_name)
                shutil.copy2(fv, output_path)
                persisted_videos.append(output_path)
                logging.info("Итоговое видео сохранено: %s", output_path)
                
                # Save tags as description
                try:
                    desc_path = f"{output_path}.txt"
                    with open(desc_path, 'w', encoding='utf-8') as df:
                        df.write(tags)
                    logging.info("Описание видео сохранено: %s", desc_path)
                except Exception as ex:
                    logging.warning("Не удалось сохранить описание для видео %s: %s", output_path, ex)

            sent_count = 0
            for pv in persisted_videos:
                try:
                    with open(pv, 'rb') as f:
                        await message.reply_video(
                            f,
                            caption=caption,
                            read_timeout=300,
                            write_timeout=300,
                            connect_timeout=30,
                            pool_timeout=30,
                        )
                    sent_count += 1
                except Exception as video_err:
                    logging.warning("Не удалось отправить как video (%s): %s", pv, video_err)
                    try:
                        with open(pv, 'rb') as f:
                            await message.reply_document(
                                f,
                                caption=caption,
                                read_timeout=300,
                                write_timeout=300,
                                connect_timeout=30,
                                pool_timeout=30,
                            )
                        sent_count += 1
                    except Exception as doc_err:
                        logging.exception("Не удалось отправить видео в чат (%s): %s", pv, doc_err)

            if sent_count == 0:
                await message.reply_text(
                    "⚠️ Видео обработано и сохранено локально, но отправка в чат не удалась.\n"
                    "Открой /ready_videos — результат уже там."
                )
                logging.info(
                    "event=pipeline_finished chat_id=%s success=false sent_count=%s",
                    chat_id,
                    sent_count,
                )
                await tracker.complete("Обработка завершена, но отправка в чат не удалась.")
            else:
                if cache_meta:
                    cache.cache_processed_result(
                        cache_meta["cache_key"],
                        persisted_videos,
                        tags,
                        source_hash=cache_meta.get("source_hash"),
                        settings_signature=cache_meta.get("settings_signature"),
                        random_cut=cache_meta.get("random_cut", False),
                    )
                logging.info(
                    "event=pipeline_finished chat_id=%s success=true sent_count=%s",
                    chat_id,
                    sent_count,
                )
                await tracker.complete("Видео успешно обработано!")
        
        # Выгружаем модель из памяти после завершения обработки
        from transcribe import unload_whisper_model
        await asyncio.to_thread(unload_whisper_model)

    except Exception as e:
        if chat_id in processing_jobs:
            del processing_jobs[chat_id]
        if isinstance(e, CancellationError) or "отменена" in str(e).lower():
            try:
                await message.edit_text("❌ Обработка отменена")
            except Exception:
                pass
        else:
            logging.exception("Ошибка при обработке видео")
            logging.info("event=pipeline_finished chat_id=%s success=false error=true", chat_id)
            try:
                await message.edit_text(f"❌ Ошибка: {e}")
            except Exception:
                pass


async def run_processing_pipeline(input_file, tracker, user_settings, temp_mgr):
    """
    Единый пайплайн обработки видео.
    Возвращает (final_videos, tags) или выбрасывает исключение.
    """
    min_duration = user_settings.get('min_clip_duration', 20)
    max_duration = user_settings.get('max_clip_duration', 90)
    remove_silence = user_settings.get('remove_silence', True)
    add_subtitles = user_settings.get('add_subtitles', True)
    hashtag_count = user_settings.get('hashtag_count', 7)
    max_clips_to_process = user_settings.get('max_clips_to_process', 3)
    subtitle_style = user_settings.get('subtitle_style', 'subtitle')
    subtitle_position = user_settings.get('subtitle_position', 'bottom')
    whisper_model = user_settings.get('whisper_model', 'base')
    use_gpu = user_settings.get('use_gpu', False)
    vertical_layout_mode = user_settings.get('vertical_layout_mode', 'standard')
    facecam_subject_side = user_settings.get('facecam_subject_side', 'left')
    facecam_detector_backend = user_settings.get('facecam_detector_backend', 'yolo_window_v1')
    facecam_fallback_mode = user_settings.get('facecam_fallback_mode', 'hard_side')
    facecam_anchor = user_settings.get('facecam_anchor', 'edge_middle')

    async def run_with_cancellation_check(func, *args, **kwargs):
        """Запускает функцию с периодической проверкой отмены."""
        loop = asyncio.get_event_loop()
        
        def run():
            return func(*args, **kwargs)
        
        future = loop.run_in_executor(None, run)
        
        while not future.done():
            if tracker.is_cancelled():
                future.cancel()
                raise CancellationError("Обработка отменена")
            await asyncio.sleep(0.5)
        
        return future.result()

    try:
        if not await tracker.update_stage("🔍 Анализ видео (поиск сцен)...", "🔍"):
            raise Exception("Обработка отменена")

        scene_files = await run_with_cancellation_check(
            detect_scenes, input_file, temp_mgr.temp_dir, min_duration=min_duration, max_duration=max_duration
        )

        if tracker.is_cancelled():
            raise Exception("Обработка отменена")
        if not await tracker.update_stage("Удаление тишины...", "🔇"):
            raise Exception("Обработка отменена")
        if remove_silence:
            edited_files = []
            for f in scene_files:
                if tracker.is_cancelled():
                    raise CancellationError("Обработка отменена")
                try:
                    edited_file = await run_with_cancellation_check(cut_silence, f, temp_mgr.temp_dir)
                    edited_files.append(edited_file)
                except Exception as e:
                    # Не роняем весь пайплайн, если auto-editor не смог обработать конкретный клип.
                    logging.warning("Auto-Editor failed for %s, fallback to original clip: %s", f, e)
                    edited_files.append(f)
        else:
            edited_files = scene_files

        if tracker.is_cancelled():
            raise Exception("Обработка отменена")
        if not await tracker.update_stage("Быстрая оценка речи в клипах...", "🔎"):
            raise Exception("Обработка отменена")

        # Быстрая оценка плотности речи через Whisper tiny — только для ранжирования.
        # Это дешевле, чем полная транскрипция, и позволяет отобрать самые «живые» клипы.
        clip_word_density: dict = {}
        if edited_files and len(edited_files) > max_clips_to_process:
            for f in edited_files:
                if tracker.is_cancelled():
                    raise CancellationError("Обработка отменена")
                try:
                    wps = await run_with_cancellation_check(
                        quick_word_density, f, temp_mgr.temp_dir, use_gpu
                    )
                    clip_word_density[f] = wps
                except Exception as _e:
                    logging.warning("quick_word_density failed for %s: %s", f, _e)
                    clip_word_density[f] = 0.0

        if tracker.is_cancelled():
            raise Exception("Обработка отменена")
        if not await tracker.update_stage("Ранжирование моментов...", "📊"):
            raise Exception("Обработка отменена")

        if edited_files:
            edited_files = await run_with_cancellation_check(
                rank_clips,
                edited_files,
                max_clips_to_process,
                25,  # optimal_duration
                clip_word_density or None,
            )

        if tracker.is_cancelled():
            raise Exception("Обработка отменена")
        if not await tracker.update_stage("Транскрибация клипов и подготовка субтитров...", "🎤"):
            raise Exception("Обработка отменена")

        # Полная транскрипция только для отобранных (top_k) клипов.
        clip_transcripts = []
        clip_subtitles = {}
        for f in edited_files:
            if tracker.is_cancelled():
                raise CancellationError("Обработка отменена")

            transcription = await run_with_cancellation_check(
                transcribe_audio_with_timestamps,
                f,
                temp_mgr.temp_dir,
                whisper_model,
                use_gpu
            )

            clip_text = (transcription.get("text") or "").strip()
            if clip_text:
                clip_transcripts.append(clip_text)

            if add_subtitles:
                words = transcription.get("words", []) or []
                if words:
                    subtitles_file = await run_with_cancellation_check(
                        create_ass_subtitles_from_words,
                        words,
                        temp_mgr.temp_dir,
                        subtitle_style,
                        subtitle_position
                    )
                    if subtitles_file:
                        clip_subtitles[f] = subtitles_file
                    else:
                        logging.warning("Не удалось создать субтитры для клипа %s", f)
                else:
                    logging.info("В клипе %s не найдено слов, отправляю без субтитров.", f)

        if tracker.is_cancelled():
            raise Exception("Обработка отменена")
        if not await tracker.update_stage("Конвертация...", "📱"):
            raise Exception("Обработка отменена")

        final_videos = []
        for f in edited_files:
            if tracker.is_cancelled():
                raise CancellationError("Обработка отменена")
            clip_subs = clip_subtitles.get(f) if add_subtitles else None
            convert_kwargs = {
                "layout_mode": vertical_layout_mode,
                "facecam_subject_side": facecam_subject_side,
                "facecam_detector_backend": facecam_detector_backend,
                "facecam_fallback_mode": facecam_fallback_mode,
                "facecam_anchor": facecam_anchor,
            }
            if clip_subs:
                convert_kwargs["subs_file"] = clip_subs

            vert = await run_with_cancellation_check(
                convert_to_vertical,
                f,
                temp_mgr.temp_dir,
                **convert_kwargs
            )
            final_videos.append(vert)

        if tracker.is_cancelled():
            raise Exception("Обработка отменена")
        if not await tracker.update_stage("Генерация хештегов...", "🏷️"):
            raise Exception("Обработка отменена")
        transcript_for_tags = " ".join(clip_transcripts).strip()
        if not transcript_for_tags:
            transcript_for_tags = "video shorts"
        tags = generate_tags(transcript_for_tags, hashtag_count=hashtag_count)

        return final_videos, tags

    except asyncio.CancelledError:
        raise Exception("Обработка отменена")


def run_with_progress(func, tracker, *args, **kwargs):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(func(tracker, *args, **kwargs))
    finally:
        loop.close()


async def process_with_progress(tracker, func, *args, **kwargs):
    result = await asyncio.to_thread(run_with_progress, func, tracker, *args, **kwargs)
    return result


def retry_operation(func, max_retries=3, initial_delay=1.0):
    """
    Декоратор/функция для повтора операции при ошибке.
    """
    def wrapper(*args, **kwargs):
        delay = initial_delay
        last_exception = None

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logging.warning(f"Попытка {attempt + 1} не удалась: {e}. Повтор через {delay:.1f}с...")
                    import time
                    time.sleep(delay)
                    delay *= 2
                else:
                    logging.error(f"Все {max_retries} попыток исчерпаны: {e}")
                    raise VideoProcessingError(f"Операция не удалась после {max_retries} попыток: {e}")

    return wrapper
