import asyncio
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from handlers.commands import preview_video, moments_detect, list_videos, ready_videos
from handlers.callbacks import handle_extract_moment, handle_start_existing_video, handle_send_ready_video
from handlers.legacy_callbacks import process_existing_video
from handlers.mailru_handlers import mailru_disconnect


class _FakeMessage:
    def __init__(self, chat_id=1):
        self.chat_id = chat_id
        self.replies = []

    async def reply_text(self, text, **kwargs):
        self.replies.append((text, kwargs))
        status = SimpleNamespace(edit_text=AsyncMock())
        return status


class _FakeQueryMessage:
    def __init__(self, chat_id=1):
        self.chat_id = chat_id
        self.chat = SimpleNamespace(id=chat_id)
        self.reply_text = AsyncMock(return_value=SimpleNamespace(edit_text=AsyncMock()))


class _FakeQuery:
    def __init__(self, data, message):
        self.data = data
        self.message = message
        self.edit_message_text = AsyncMock()

    async def answer(self):
        return None


class _FakeUpdate:
    def __init__(self, message=None, callback_query=None):
        self.message = message
        self.callback_query = callback_query


class TestUiUxRedesign(unittest.TestCase):
    def test_preview_without_args_starts_interactive_flow(self):
        message = _FakeMessage(chat_id=101)
        update = _FakeUpdate(message=message)
        context = SimpleNamespace(args=[], user_data={}, bot=SimpleNamespace())

        asyncio.run(preview_video(update, context))

        self.assertTrue(context.user_data.get("awaiting_source"))
        self.assertEqual(context.user_data.get("awaiting_action"), "preview")
        self.assertTrue(message.replies)
        self.assertIn("Режим превью", message.replies[0][0])

    def test_moments_without_args_starts_interactive_flow(self):
        message = _FakeMessage(chat_id=102)
        update = _FakeUpdate(message=message)
        context = SimpleNamespace(args=[], user_data={}, bot=SimpleNamespace())

        asyncio.run(moments_detect(update, context))

        self.assertTrue(context.user_data.get("awaiting_source"))
        self.assertEqual(context.user_data.get("awaiting_action"), "moments")
        self.assertTrue(message.replies)
        self.assertIn("Режим лучших моментов", message.replies[0][0])

    def test_list_videos_uses_short_callback_tokens(self):
        long_name = "x" * 200 + ".mp4"
        message = _FakeMessage(chat_id=103)
        update = _FakeUpdate(message=message)
        context = SimpleNamespace(user_data={})

        with patch("handlers.commands.list_available_videos", return_value=[
            {"path": f"videos/{long_name}", "name": long_name, "size": "1 MB", "mtime": 1.0}
        ]):
            asyncio.run(list_videos(update, context))

        self.assertTrue(message.replies)
        reply_markup = message.replies[0][1].get("reply_markup")
        self.assertIsNotNone(reply_markup)

        callback_data_values = []
        for row in reply_markup.inline_keyboard:
            for button in row:
                if button.callback_data:
                    callback_data_values.append(button.callback_data)

        self.assertTrue(callback_data_values)
        self.assertTrue(all(len(item) <= 64 for item in callback_data_values))

    def test_ready_videos_uses_short_callback_tokens(self):
        long_name = "y" * 200 + ".mp4"
        message = _FakeMessage(chat_id=108)
        update = _FakeUpdate(message=message)
        context = SimpleNamespace(user_data={})

        with patch("handlers.commands.list_ready_videos", return_value=[
            {"path": f"videos/ready/{long_name}", "name": long_name, "size": "2 MB", "mtime": 1.0}
        ]):
            asyncio.run(ready_videos(update, context))

        self.assertTrue(message.replies)
        reply_markup = message.replies[0][1].get("reply_markup")
        self.assertIsNotNone(reply_markup)

        callback_data_values = []
        for row in reply_markup.inline_keyboard:
            for button in row:
                if button.callback_data:
                    callback_data_values.append(button.callback_data)

        self.assertIn("action_ready_send_r1", callback_data_values)
        self.assertTrue(all(len(item) <= 64 for item in callback_data_values))

    def test_extract_moment_callback_sends_video(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as src:
            src.write(b"src")
            src_path = src.name

        query_message = _FakeQueryMessage(chat_id=104)
        query = _FakeQuery("extract_moment_m1", query_message)
        update = _FakeUpdate(callback_query=query)
        bot = SimpleNamespace(send_video=AsyncMock())
        context = SimpleNamespace(
            bot=bot,
            user_data={
                "moment_tokens": {
                    "m1": {
                        "source": src_path,
                        "source_kind": "local_path",
                        "start": 3.0,
                        "duration": 2.0,
                    }
                }
            },
        )

        def _fake_cut_video_chunk(_input, output, _start, _duration):
            with open(output, "wb") as f:
                f.write(b"chunk")
            return True

        try:
            with patch("handlers.callbacks.cut_video_chunk", side_effect=_fake_cut_video_chunk):
                asyncio.run(handle_extract_moment(update, context))

            bot.send_video.assert_awaited_once()
        finally:
            if os.path.exists(src_path):
                os.remove(src_path)

    def test_mailru_disconnect_works_as_command_alias(self):
        message = _FakeMessage(chat_id=105)
        update = _FakeUpdate(message=message)
        context = SimpleNamespace(bot=SimpleNamespace())

        with patch("handlers.mailru_handlers.settings.load_settings", return_value={"mailru_token": "token"}), \
             patch("handlers.mailru_handlers.settings.save_settings") as save_mock:
            asyncio.run(mailru_disconnect(update, context))

        save_mock.assert_called_once()
        save_args = save_mock.call_args.args
        self.assertEqual(save_args[0], 105)
        self.assertNotIn("mailru_token", save_args[1])

    def test_action_process_random_still_runs_pipeline(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as src:
            src.write(b"source")
            src_path = src.name

        query_message = _FakeQueryMessage(chat_id=106)
        query = _FakeQuery("action_process_random_v1", query_message)
        update = _FakeUpdate(callback_query=query)
        context = SimpleNamespace(user_data={"video_tokens": {"v1": src_path}})

        try:
            with patch("handlers.callbacks.process_source", new=AsyncMock()) as process_mock:
                asyncio.run(handle_start_existing_video(update, context))

            process_mock.assert_awaited_once()
            kwargs = process_mock.call_args.kwargs
            self.assertEqual(kwargs["source_type"], "local_path")
            self.assertTrue(kwargs["random_cut"])
        finally:
            if os.path.exists(src_path):
                os.remove(src_path)

    def test_send_ready_video_callback_sends_file(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as ready:
            ready.write(b"ready")
            ready_path = ready.name

        query_message = _FakeQueryMessage(chat_id=109)
        query = _FakeQuery("action_ready_send_r1", query_message)
        update = _FakeUpdate(callback_query=query)
        bot = SimpleNamespace(send_video=AsyncMock())
        context = SimpleNamespace(bot=bot, user_data={"ready_video_tokens": {"r1": ready_path}})

        try:
            asyncio.run(handle_send_ready_video(update, context))
            bot.send_video.assert_awaited_once()
        finally:
            if os.path.exists(ready_path):
                os.remove(ready_path)

    def test_legacy_process_callback_kept_for_backward_compatibility(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as src:
            src.write(b"source")
            src_path = src.name
            src_name = os.path.basename(src_path)

        query_message = _FakeQueryMessage(chat_id=107)
        query = _FakeQuery(f"process_{src_name}", query_message)
        update = _FakeUpdate(callback_query=query)
        context = SimpleNamespace(user_data={})

        try:
            with patch("handlers.legacy_callbacks.get_video_path", return_value=src_path):
                asyncio.run(process_existing_video(update, context))

            self.assertEqual(context.user_data.get("pending_source"), src_path)
            self.assertEqual(context.user_data.get("pending_type"), "local_path")
            query.edit_message_text.assert_awaited_once()
        finally:
            if os.path.exists(src_path):
                os.remove(src_path)


if __name__ == "__main__":
    unittest.main()
