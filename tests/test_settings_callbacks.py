import asyncio
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import settings
from handlers.callbacks import handle_settings_callback


class _FakeMessage:
    def __init__(self, chat_id):
        self.chat_id = chat_id
        self.deleted = False

    async def delete(self):
        self.deleted = True


class _FakeQuery:
    def __init__(self, data, message):
        self.data = data
        self.message = message

    async def answer(self):
        return None


class _FakeUpdate:
    def __init__(self, callback_query):
        self.callback_query = callback_query


class TestSettingsAndCallbacks(unittest.TestCase):
    def test_default_settings_has_subtitle_position(self):
        self.assertIn("subtitle_position", settings.DEFAULT_SETTINGS)
        self.assertEqual(settings.DEFAULT_SETTINGS["subtitle_position"], "bottom")

    def test_default_settings_has_vertical_layout_mode(self):
        self.assertIn("vertical_layout_mode", settings.DEFAULT_SETTINGS)
        self.assertEqual(settings.DEFAULT_SETTINGS["vertical_layout_mode"], "standard")

    def test_default_settings_has_facecam_subject_side(self):
        self.assertIn("facecam_subject_side", settings.DEFAULT_SETTINGS)
        self.assertEqual(settings.DEFAULT_SETTINGS["facecam_subject_side"], "left")

    def test_subtitle_position_cycles_bottom_center_top(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_file = f"{tmp_dir}/user_settings.json"
            user_id = 42

            with patch.object(settings, "SETTINGS_FILE", settings_file):
                message = _FakeMessage(chat_id=user_id)
                update = _FakeUpdate(_FakeQuery("setting_subtitle_position", message))
                context = SimpleNamespace(bot=SimpleNamespace())

                with patch("handlers.commands.show_settings", new=AsyncMock()):
                    # bottom -> center
                    asyncio.run(handle_settings_callback(update, context))
                    self.assertEqual(settings.load_settings(user_id)["subtitle_position"], "center")

                    # center -> top
                    asyncio.run(handle_settings_callback(update, context))
                    self.assertEqual(settings.load_settings(user_id)["subtitle_position"], "top")

                    # top -> bottom
                    asyncio.run(handle_settings_callback(update, context))
                    self.assertEqual(settings.load_settings(user_id)["subtitle_position"], "bottom")

    def test_vertical_layout_cycles_standard_facecam(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_file = f"{tmp_dir}/user_settings.json"
            user_id = 43

            with patch.object(settings, "SETTINGS_FILE", settings_file):
                message = _FakeMessage(chat_id=user_id)
                update = _FakeUpdate(_FakeQuery("setting_vertical_layout", message))
                context = SimpleNamespace(bot=SimpleNamespace())

                with patch("handlers.commands.show_settings", new=AsyncMock()):
                    # standard -> facecam_top_split
                    asyncio.run(handle_settings_callback(update, context))
                    self.assertEqual(
                        settings.load_settings(user_id)["vertical_layout_mode"],
                        "facecam_top_split"
                    )

                    # facecam_top_split -> standard
                    asyncio.run(handle_settings_callback(update, context))
                    self.assertEqual(
                        settings.load_settings(user_id)["vertical_layout_mode"],
                        "standard"
                    )

    def test_facecam_side_cycles_left_right_auto(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_file = f"{tmp_dir}/user_settings.json"
            user_id = 44

            with patch.object(settings, "SETTINGS_FILE", settings_file):
                message = _FakeMessage(chat_id=user_id)
                update = _FakeUpdate(_FakeQuery("setting_facecam_side", message))
                context = SimpleNamespace(bot=SimpleNamespace())

                with patch("handlers.commands.show_settings", new=AsyncMock()):
                    # left -> right
                    asyncio.run(handle_settings_callback(update, context))
                    self.assertEqual(settings.load_settings(user_id)["facecam_subject_side"], "right")

                    # right -> auto
                    asyncio.run(handle_settings_callback(update, context))
                    self.assertEqual(settings.load_settings(user_id)["facecam_subject_side"], "auto")

                    # auto -> left
                    asyncio.run(handle_settings_callback(update, context))
                    self.assertEqual(settings.load_settings(user_id)["facecam_subject_side"], "left")

    def test_load_settings_backfills_facecam_subject_side(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_file = f"{tmp_dir}/user_settings.json"
            user_id = 45
            with open(settings_file, "w", encoding="utf-8") as f:
                f.write('{"45": {"vertical_layout_mode": "facecam_top_split"}}')

            with patch.object(settings, "SETTINGS_FILE", settings_file):
                loaded = settings.load_settings(user_id)
                self.assertEqual(loaded["facecam_subject_side"], "left")


if __name__ == "__main__":
    unittest.main()
