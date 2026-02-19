# handlers/__init__.py — регистрация обработчиков


def register_handlers(application):
    """Регистрирует все обработчики в приложении."""
    from telegram.ext import CommandHandler, MessageHandler, CallbackQueryHandler, filters

    from handlers.commands import (
        start,
        list_videos,
        ready_videos,
        show_settings,
        check_gpu,
        cache_status,
        cache_clear,
        llm_status,
        moments_detect,
        preview_video,
    )
    from handlers.callbacks import (
        handle_cancel,
        handle_retry,
        handle_settings_callback,
        handle_interface_callback,
        handle_preview_existing_video,
        handle_start_existing_video,
        handle_processing_mode,
        handle_action_entry,
        handle_source_picker,
        handle_action_process_last,
        handle_moments_existing_video,
        handle_extract_moment,
        handle_send_ready_video,
        handle_preview_ready_video,
    )
    from handlers.legacy_callbacks import (
        process_existing_video,
        handle_legacy_processing_mode,
    )
    from handlers.mailru_handlers import (
        mailru_link,
        mailru_token,
        mailru_disconnect,
    )
    from handlers.message_handlers import (
        route_text_message,
        process_video,
        error_handler,
    )

    # Команды
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('list_videos', list_videos))
    application.add_handler(CommandHandler('ready_videos', ready_videos))
    application.add_handler(CommandHandler('settings', show_settings))
    application.add_handler(CommandHandler('preview', preview_video))
    application.add_handler(CommandHandler('gpu', check_gpu))
    application.add_handler(CommandHandler('cache', cache_status))
    application.add_handler(CommandHandler('cache_clear', cache_clear))
    application.add_handler(CommandHandler('llm', llm_status))
    application.add_handler(CommandHandler('moments', moments_detect))
    application.add_handler(CommandHandler('mailru', mailru_link))
    application.add_handler(CommandHandler('mailru_token', mailru_token))
    application.add_handler(CommandHandler('mailru_disconnect', mailru_disconnect))

    # Callback-кнопки (новый UX)
    application.add_handler(CallbackQueryHandler(handle_action_entry, pattern=r"^action_preview$"))
    application.add_handler(CallbackQueryHandler(handle_action_entry, pattern=r"^action_moments$"))
    application.add_handler(CallbackQueryHandler(handle_action_process_last, pattern=r"^action_process_last$"))
    application.add_handler(CallbackQueryHandler(handle_source_picker, pattern=r"^src_pick_"))

    application.add_handler(CallbackQueryHandler(handle_processing_mode, pattern=r"^mode_(full|random)$"))
    application.add_handler(CallbackQueryHandler(handle_start_existing_video, pattern=r"^action_process_(full|random)_"))
    application.add_handler(CallbackQueryHandler(handle_start_existing_video, pattern=r"^start_(full|random)_"))

    application.add_handler(CallbackQueryHandler(handle_preview_existing_video, pattern=r"^action_preview_"))
    application.add_handler(CallbackQueryHandler(handle_moments_existing_video, pattern=r"^action_moments_"))
    application.add_handler(CallbackQueryHandler(handle_send_ready_video, pattern=r"^action_ready_send_"))
    application.add_handler(CallbackQueryHandler(handle_preview_ready_video, pattern=r"^action_ready_preview_"))
    application.add_handler(CallbackQueryHandler(handle_extract_moment, pattern=r"^extract_moment_"))

    # Legacy callback-кнопки (совместимость со старыми сообщениями)
    application.add_handler(CallbackQueryHandler(handle_legacy_processing_mode, pattern=r"^process_(full|random)$"))
    application.add_handler(CallbackQueryHandler(process_existing_video, pattern=r"^process_"))
    application.add_handler(CallbackQueryHandler(handle_preview_existing_video, pattern=r"^preview_"))

    application.add_handler(CallbackQueryHandler(handle_cancel, pattern=r"^cancel_"))
    application.add_handler(CallbackQueryHandler(handle_retry, pattern=r"^retry_"))
    application.add_handler(CallbackQueryHandler(handle_settings_callback, pattern=r"^setting_"))
    application.add_handler(CallbackQueryHandler(handle_settings_callback, pattern=r"^duration_"))
    application.add_handler(CallbackQueryHandler(handle_settings_callback, pattern=r"^hashtags_"))
    application.add_handler(CallbackQueryHandler(handle_settings_callback, pattern=r"^whisper_"))
    application.add_handler(CallbackQueryHandler(handle_interface_callback, pattern=r"^cmd_"))
    application.add_handler(CallbackQueryHandler(mailru_disconnect, pattern=r"^mailru_disconnect$"))

    # Обработчики сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, route_text_message))
    application.add_handler(MessageHandler(filters.VIDEO, process_video))

    # Глобальный обработчик ошибок
    application.add_error_handler(error_handler)
