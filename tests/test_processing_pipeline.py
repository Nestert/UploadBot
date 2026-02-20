import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from handlers.processing import run_processing_pipeline


class _DummyTracker:
    def __init__(self):
        self.stages = []

    async def update_stage(self, stage_name, stage_emoji):
        self.stages.append((stage_name, stage_emoji))
        return True

    def is_cancelled(self):
        return False


class TestProcessingPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_transcribe_runs_after_silence_cut_and_per_clip_subtitles(self):
        order = []
        tracker = _DummyTracker()
        temp_mgr = SimpleNamespace(temp_dir="/tmp")
        user_settings = {
            "min_clip_duration": 20,
            "max_clip_duration": 90,
            "remove_silence": True,
            "add_subtitles": True,
            "subtitle_style": "subtitle",
            "subtitle_position": "bottom",
            "vertical_layout_mode": "facecam_top_split",
            "facecam_subject_side": "right",
            "facecam_detector_backend": "yolo_window_v1",
            "facecam_fallback_mode": "hard_side",
            "facecam_anchor": "edge_middle",
            "hashtag_count": 7,
            "whisper_model": "base",
            "use_gpu": False
        }

        def fake_detect(*args, **kwargs):
            order.append("detect")
            return ["scene1.mp4", "scene2.mp4"]

        def fake_cut(path, *_args, **_kwargs):
            order.append(f"cut:{path}")
            return f"edited_{path}"

        def fake_transcribe(path, *_args, **_kwargs):
            order.append(f"transcribe:{path}")
            return {
                "text": f"text_{path}",
                "words": [{"word": "hello", "start": 0.0, "end": 0.3}]
            }

        def fake_subs(*_args, **_kwargs):
            order.append("subs")
            return "clip.ass"

        def fake_convert(path, *_args, **_kwargs):
            order.append(f"convert:{path}")
            return f"vert_{path}"

        tags_mock = Mock(return_value="#tags")

        with patch("handlers.processing.detect_scenes", side_effect=fake_detect), \
             patch("handlers.processing.cut_silence", side_effect=fake_cut), \
             patch("handlers.processing.transcribe_audio_with_timestamps", side_effect=fake_transcribe), \
             patch("handlers.processing.create_ass_subtitles_from_words", side_effect=fake_subs), \
             patch("handlers.processing.convert_to_vertical", side_effect=fake_convert) as convert_mock, \
             patch("handlers.processing.generate_tags", tags_mock):
            final_videos, tags = await run_processing_pipeline(
                "input.mp4",
                tracker,
                user_settings,
                temp_mgr
            )

        self.assertEqual(tags, "#tags")
        self.assertEqual(len(final_videos), 2)
        self.assertTrue(final_videos[0].startswith("vert_"))

        self.assertLess(order.index("cut:scene1.mp4"), order.index("transcribe:edited_scene1.mp4"))
        self.assertLess(order.index("cut:scene2.mp4"), order.index("transcribe:edited_scene2.mp4"))
        self.assertEqual(order.count("subs"), 2)

        tags_mock.assert_called_once_with(
            "text_edited_scene1.mp4 text_edited_scene2.mp4",
            hashtag_count=7
        )
        self.assertEqual(convert_mock.call_count, 2)
        for call in convert_mock.call_args_list:
            self.assertEqual(call.kwargs.get("layout_mode"), "facecam_top_split")
            self.assertEqual(call.kwargs.get("facecam_subject_side"), "right")
            self.assertEqual(call.kwargs.get("facecam_detector_backend"), "yolo_window_v1")
            self.assertEqual(call.kwargs.get("facecam_fallback_mode"), "hard_side")
            self.assertEqual(call.kwargs.get("facecam_anchor"), "edge_middle")
            self.assertEqual(call.kwargs.get("subs_file"), "clip.ass")

    async def test_clip_without_words_is_kept_without_subtitles(self):
        tracker = _DummyTracker()
        temp_mgr = SimpleNamespace(temp_dir="/tmp")
        user_settings = {
            "min_clip_duration": 20,
            "max_clip_duration": 90,
            "remove_silence": True,
            "add_subtitles": True,
            "subtitle_style": "subtitle",
            "subtitle_position": "bottom",
            "vertical_layout_mode": "standard",
            "facecam_subject_side": "left",
            "facecam_detector_backend": "yolo_window_v1",
            "facecam_fallback_mode": "hard_side",
            "facecam_anchor": "edge_middle",
            "hashtag_count": 7,
            "whisper_model": "base",
            "use_gpu": False
        }

        tags_mock = Mock(return_value="#fallback")

        with patch("handlers.processing.detect_scenes", return_value=["scene1.mp4"]), \
             patch("handlers.processing.cut_silence", return_value="edited_scene1.mp4"), \
             patch(
                 "handlers.processing.transcribe_audio_with_timestamps",
                 return_value={"text": "", "words": []}
             ), \
             patch("handlers.processing.create_ass_subtitles_from_words") as subs_mock, \
             patch("handlers.processing.convert_to_vertical", return_value="vert_edited_scene1.mp4"), \
             patch("handlers.processing.generate_tags", tags_mock):
            final_videos, tags = await run_processing_pipeline(
                "input.mp4",
                tracker,
                user_settings,
                temp_mgr
            )

        self.assertEqual(tags, "#fallback")
        self.assertEqual(final_videos, ["vert_edited_scene1.mp4"])
        subs_mock.assert_not_called()
        tags_mock.assert_called_once_with("video shorts", hashtag_count=7)

    async def test_layout_mode_defaults_to_standard_when_missing(self):
        tracker = _DummyTracker()
        temp_mgr = SimpleNamespace(temp_dir="/tmp")
        user_settings = {
            "min_clip_duration": 20,
            "max_clip_duration": 90,
            "remove_silence": False,
            "add_subtitles": False,
            "hashtag_count": 7,
            "whisper_model": "base",
            "use_gpu": False
        }

        with patch("handlers.processing.detect_scenes", return_value=["scene1.mp4"]), \
             patch("handlers.processing.transcribe_audio_with_timestamps", return_value={"text": "", "words": []}), \
             patch("handlers.processing.convert_to_vertical", return_value="vert_scene1.mp4") as convert_mock, \
             patch("handlers.processing.generate_tags", return_value="#fallback"):
            final_videos, tags = await run_processing_pipeline(
                "input.mp4",
                tracker,
                user_settings,
                temp_mgr
            )

        self.assertEqual(final_videos, ["vert_scene1.mp4"])
        self.assertEqual(tags, "#fallback")
        convert_mock.assert_called_once_with(
            "scene1.mp4",
            "/tmp",
            layout_mode="standard",
            facecam_subject_side="left",
            facecam_detector_backend="yolo_window_v1",
            facecam_fallback_mode="hard_side",
            facecam_anchor="edge_middle",
        )


if __name__ == "__main__":
    unittest.main()
