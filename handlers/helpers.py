# handlers/helpers.py — вспомогательные классы и функции

"""
Вспомогательные классы для имитации Telegram-объектов.
Используется для вызова обработчиков команд из callback-кнопок.
"""


class CallbackMessage:
    """Эмулирует telegram.Message для использования в callback-обработчиках."""
    
    def __init__(self, chat_id, bot):
        self.chat_id = chat_id
        self._bot = bot

    async def reply_text(self, *args, **kwargs):
        """Отправляет текстовое сообщение."""
        return await self._bot.send_message(self.chat_id, *args, **kwargs)

    async def reply_photo(self, *args, **kwargs):
        """Отправляет фото."""
        return await self._bot.send_photo(self.chat_id, *args, **kwargs)

    async def reply_video(self, *args, **kwargs):
        """Отправляет видео."""
        return await self._bot.send_video(self.chat_id, *args, **kwargs)

    async def reply_document(self, *args, **kwargs):
        """Отправляет документ."""
        return await self._bot.send_document(self.chat_id, *args, **kwargs)


class CallbackUpdate:
    """Эмулирует telegram.Update для использования в callback-обработчиках."""
    
    def __init__(self, chat_id, bot):
        self.message = CallbackMessage(chat_id, bot)


def create_callback_update(chat_id, bot):
    """
    Создает объект CallbackUpdate для вызова command-обработчиков из callbacks.
    
    Args:
        chat_id: ID чата пользователя
        bot: Объект Telegram-бота
        
    Returns:
        CallbackUpdate: Объект, эмулирующий telegram.Update
    """
    return CallbackUpdate(chat_id, bot)


def set_last_source(context, source, source_type):
    """Сохраняет последний источник для быстрого повторного запуска."""
    context.user_data['last_source'] = {
        "source": source,
        "source_type": source_type
    }


def get_last_source(context):
    """Возвращает последний источник или None."""
    last = context.user_data.get('last_source')
    if not isinstance(last, dict):
        return None
    source = last.get("source")
    source_type = last.get("source_type")
    if not source or not source_type:
        return None
    return {
        "source": source,
        "source_type": source_type
    }


def set_awaiting_source(context, action):
    """Переводит пользователя в режим ожидания источника для указанного действия."""
    context.user_data['awaiting_source'] = True
    context.user_data['awaiting_action'] = action


def clear_awaiting_source(context):
    """Сбрасывает режим ожидания источника."""
    context.user_data.pop('awaiting_source', None)
    context.user_data.pop('awaiting_action', None)
