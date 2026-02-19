import asyncio
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import cache
import utils
from handlers.processing import process_source


class _FakeChat:
    def __init__(self, chat_id):
        self.id = chat_id


class _FakeMessage:
    def __init__(self):
        self.sent_videos = 0

    async def reply_text(self, *_args, **_kwargs):
        return self

    async def reply_video(self, *_args, **_kwargs):
        self.sent_videos += 1
        return None

    async def reply_document(self, *_args, **_kwargs):
        self.sent_videos += 1
        return None

    async def edit_text(self, *_args, **_kwargs):
        return self


class _FakeUpdate:
    def __init__(self, chat_id=1):
        self.effective_chat = _FakeChat(chat_id)
        self.effective_message = _FakeMessage()
        self.callback_query = None


class TestVideoStorageAndCache(unittest.TestCase):
    def test_unprocessed_video_is_saved_to_raw_folder(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            videos_dir = os.path.join(tmp_dir, "videos")
            raw_dir = os.path.join(videos_dir, "raw")
            ready_dir = os.path.join(videos_dir, "ready")

            src = os.path.join(tmp_dir, "input.mp4")
            with open(src, "wb") as f:
                f.write(b"video-content")

            with patch.object(utils, "VIDEOS_DIR", videos_dir), \
                 patch.object(utils, "RAW_VIDEOS_DIR", raw_dir), \
                 patch.object(utils, "READY_VIDEOS_DIR", ready_dir):
                saved = utils.persist_unprocessed_video(src, "input.mp4")

                self.assertTrue(saved.startswith(raw_dir))
                self.assertTrue(os.path.exists(saved))
                self.assertTrue(os.path.isdir(raw_dir))
                self.assertTrue(os.path.isdir(ready_dir))

    def test_processed_video_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = os.path.join(tmp_dir, "cache")
            transcriptions_dir = os.path.join(cache_dir, "transcriptions")
            metadata_dir = os.path.join(cache_dir, "metadata")
            processed_dir = os.path.join(cache_dir, "processed")
            processed_videos_dir = os.path.join(processed_dir, "videos")
            processed_index = os.path.join(processed_dir, "index.json")

            src_video = os.path.join(tmp_dir, "result.mp4")
            with open(src_video, "wb") as f:
                f.write(b"final-video")

            with patch.object(cache, "CACHE_DIR", cache_dir), \
                 patch.object(cache, "TRANSCRIPTIONS_DIR", transcriptions_dir), \
                 patch.object(cache, "METADATA_DIR", metadata_dir), \
                 patch.object(cache, "PROCESSED_DIR", processed_dir), \
                 patch.object(cache, "PROCESSED_VIDEOS_DIR", processed_videos_dir), \
                 patch.object(cache, "PROCESSED_INDEX_PATH", processed_index):
                entry = cache.cache_processed_result(
                    "key123",
                    [src_video],
                    "#tags",
                    source_hash="hash",
                    settings_signature={"a": 1},
                    random_cut=False,
                )

                self.assertIsNotNone(entry)
                self.assertEqual(len(entry["videos"]), 1)
                self.assertTrue(os.path.exists(entry["videos"][0]))

                cached = cache.get_cached_processed_result("key123")
                self.assertIsNotNone(cached)
                self.assertEqual(cached.get("tags"), "#tags")

    def test_process_source_uses_cached_ready_video_and_skips_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_path = os.path.join(tmp_dir, "raw.mp4")
            with open(src_path, "wb") as f:
                f.write(b"raw")

            cached_video = os.path.join(tmp_dir, "cached.mp4")
            with open(cached_video, "wb") as f:
                f.write(b"cached")

            update = _FakeUpdate(chat_id=10)
            context = SimpleNamespace(user_data={}, bot=SimpleNamespace())

            with patch("handlers.processing.persist_unprocessed_video", return_value=src_path), \
                 patch("handlers.processing.settings.load_settings", return_value={}), \
                 patch("handlers.processing.cache.get_file_hash", return_value="hash1"), \
                 patch("handlers.processing.cache.build_processing_cache_key", return_value="cachekey"), \
                 patch(
                     "handlers.processing.cache.get_cached_processed_result",
                     return_value={"videos": [cached_video], "tags": "#cached"},
                 ), \
                 patch("handlers.processing.process_video_task", new=AsyncMock()) as process_task_mock:
                asyncio.run(process_source(update, context, src_path, source_type="local_path", random_cut=False))

            process_task_mock.assert_not_awaited()
            self.assertEqual(update.effective_message.sent_videos, 1)


if __name__ == "__main__":
    unittest.main()
