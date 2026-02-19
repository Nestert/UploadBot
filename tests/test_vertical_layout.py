import unittest
from types import SimpleNamespace
from unittest.mock import patch

import vertical


class TestVerticalLayouts(unittest.TestCase):
    def test_standard_layout_filter_keeps_scale_crop_pipeline(self):
        captured = {}

        def fake_run_ffmpeg(cmd, *_args, **_kwargs):
            captured["cmd"] = cmd

        with patch("vertical._run_ffmpeg", side_effect=fake_run_ffmpeg):
            vertical._run_standard_layout("input.mp4", "output.mp4", 1080, 1920)

        self.assertIn("-vf", captured["cmd"])
        vf_value = captured["cmd"][captured["cmd"].index("-vf") + 1]
        self.assertIn("scale=1080:1920", vf_value)
        self.assertIn("crop=1080:1920", vf_value)
        self.assertIn("setsar=1", vf_value)

    def test_convert_facecam_layout_falls_back_to_standard(self):
        with patch("vertical.uuid.uuid4", return_value="test-id"), \
             patch("vertical._run_facecam_top_split_layout", side_effect=Exception("face not found")), \
             patch("vertical._run_standard_layout") as standard_mock, \
             patch("vertical.os.path.exists", return_value=True):
            output = vertical.convert_to_vertical(
                "input.mp4",
                output_dir="/tmp",
                layout_mode="facecam_top_split",
            )

        self.assertEqual(output, "/tmp/vertical_test-id.mp4")
        standard_mock.assert_called_once()

    def test_select_best_face_prefers_stable_larger_candidate(self):
        candidates = [
            {"probe_idx": 0, "frame_idx": 0, "box": (1220, 260, 180, 180), "source": "haarcascade_frontalface_default.xml"},
            {"probe_idx": 1, "frame_idx": 6, "box": (1210, 255, 185, 185), "source": "haarcascade_frontalface_default.xml"},
            {"probe_idx": 5, "frame_idx": 30, "box": (560, 330, 500, 500), "source": "haarcascade_frontalface_alt2.xml"},
            {"probe_idx": 6, "frame_idx": 36, "box": (570, 335, 510, 510), "source": "haarcascade_frontalface_alt2.xml"},
            {"probe_idx": 7, "frame_idx": 42, "box": (580, 332, 505, 505), "source": "haarcascade_frontalface_alt2.xml"},
            {"probe_idx": 8, "frame_idx": 48, "box": (565, 328, 515, 515), "source": "haarcascade_frontalface_alt2.xml"},
            {"probe_idx": 9, "frame_idx": 54, "box": (575, 334, 500, 500), "source": "haarcascade_frontalface_alt2.xml"},
        ]

        selected = vertical._select_best_face_candidate(
            candidates,
            total_probe_frames=12,
            frame_w=1920,
            frame_h=1080,
            subject_side="left",
        )
        best = selected["best"]
        self.assertIsNotNone(best)
        bx, _by, bw, _bh = best["face_box"]
        self.assertLess(bx, 900)
        self.assertGreater(bw, 420)

    def test_side_priority_prefers_left_or_right(self):
        candidates = []
        for probe_idx in range(6):
            candidates.append(
                {
                    "probe_idx": probe_idx,
                    "frame_idx": probe_idx * 6,
                    "box": (250, 270, 320, 320),
                    "source": "haarcascade_frontalface_alt.xml",
                }
            )
            candidates.append(
                {
                    "probe_idx": probe_idx,
                    "frame_idx": probe_idx * 6,
                    "box": (1290, 270, 320, 320),
                    "source": "haarcascade_frontalface_alt.xml",
                }
            )

        best_left = vertical._select_best_face_candidate(
            candidates,
            total_probe_frames=6,
            frame_w=1920,
            frame_h=1080,
            subject_side="left",
        )["best"]["face_box"]
        best_right = vertical._select_best_face_candidate(
            candidates,
            total_probe_frames=6,
            frame_w=1920,
            frame_h=1080,
            subject_side="right",
        )["best"]["face_box"]

        self.assertLess(best_left[0], 800)
        self.assertGreater(best_right[0], 1000)

    def test_profileface_mirror_coordinates_are_reprojected(self):
        class _FakeDetector:
            def __init__(self):
                self.calls = 0

            def detectMultiScale(self, _img, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    return [(10, 20, 30, 40)]
                return [(100, 20, 30, 40)]

        fake_gray = SimpleNamespace(shape=(200, 300))
        fake_cv2 = SimpleNamespace(
            COLOR_BGR2GRAY=1,
            cvtColor=lambda _f, _code: fake_gray,
            equalizeHist=lambda img: img,
            flip=lambda img, _axis: img,
        )
        detector = _FakeDetector()
        detectors = [{"name": "haarcascade_profileface.xml", "detector": detector}]

        with patch.object(vertical, "cv2", fake_cv2):
            detections = vertical._detect_faces_in_resized_frame(object(), detectors)

        self.assertEqual(len(detections), 2)
        direct = detections[0]
        mirrored = detections[1]
        self.assertEqual(direct["box"], (10, 20, 30, 40))
        self.assertEqual(mirrored["box"], (170, 20, 30, 40))
        self.assertTrue(mirrored["mirrored"])

    def test_detect_face_once_low_score_returns_none_and_debug(self):
        frames = [(idx, SimpleNamespace(shape=(720, 1280, 3))) for idx in range(6)]
        detections = [
            [{"box": (100, 100, 130, 130), "source": "haarcascade_frontalface_default.xml"}],
            [],
            [],
            [],
            [],
            [],
        ]

        with patch("vertical._probe_frames_with_indices_from_start", return_value=frames), \
             patch("vertical._get_face_detectors", return_value=[{"name": "haarcascade_frontalface_default.xml", "detector": object()}]), \
             patch("vertical._resize_for_detection", side_effect=lambda frame, max_side=640: (frame, 1.0)), \
             patch("vertical._detect_faces_in_resized_frame", side_effect=detections):
            face_box, probe_frames, detect_debug = vertical._detect_face_once(
                "input.mp4",
                return_probe_frames=True,
                return_debug=True,
            )

        self.assertIsNone(face_box)
        self.assertEqual(len(probe_frames), 6)
        self.assertEqual(detect_debug["fallback_reason"], "low_score")
        self.assertEqual(detect_debug["best_source"], "haarcascade_frontalface_default.xml")

    def test_facecam_layout_builds_640_1280_split_filter(self):
        captured = {}

        def fake_run_ffmpeg(cmd, *_args, **_kwargs):
            captured["cmd"] = cmd

        detect_debug = {
            "subject_side": "left",
            "probe_frames": 2,
            "detector_counts": {"haarcascade_frontalface_alt2.xml": 2},
            "best_score": 0.88,
            "best_source": "haarcascade_frontalface_alt2.xml",
            "best_frame_idx": 12,
            "ranked_candidates": [{"face_box": (50, 50, 100, 100), "score": 0.88}],
            "fallback_reason": None,
        }

        with patch("vertical._probe_video_metadata", return_value=(1920, 1080, 60.0)), \
             patch("vertical._detect_face_once", return_value=((50, 50, 100, 100), [object()], detect_debug)), \
             patch("vertical._heuristic_face_box_from_corners") as heuristic_mock, \
             patch("vertical._build_camera_crop", return_value=(0, 0, 640, 380)) as cam_crop_mock, \
             patch("vertical._camera_crop_sanity_ok", return_value=True), \
             patch("vertical._build_content_crop", return_value=(100, 0, 800, 950)) as content_crop_mock, \
             patch("vertical._run_ffmpeg", side_effect=fake_run_ffmpeg):
            vertical._run_facecam_top_split_layout(
                "input.mp4",
                "output.mp4",
                1080,
                1920,
            )

        heuristic_mock.assert_not_called()
        cam_crop_mock.assert_called_once()
        content_crop_mock.assert_called_once()
        self.assertIn("-filter_complex", captured["cmd"])
        filter_graph = captured["cmd"][captured["cmd"].index("-filter_complex") + 1]
        self.assertIn("scale=1080:640", filter_graph)
        self.assertIn("scale=1080:1280", filter_graph)
        self.assertIn("vstack=inputs=2", filter_graph)

    def test_facecam_layout_tries_second_candidate_if_first_fails_sanity(self):
        detect_debug = {
            "subject_side": "left",
            "probe_frames": 2,
            "detector_counts": {"haarcascade_frontalface_alt2.xml": 2},
            "best_score": 0.9,
            "best_source": "haarcascade_frontalface_alt2.xml",
            "best_frame_idx": 8,
            "ranked_candidates": [
                {"face_box": (1000, 80, 120, 120), "score": 0.9},
                {"face_box": (520, 260, 420, 420), "score": 0.82},
            ],
            "fallback_reason": None,
        }
        cam_rects = [(900, 20, 760, 500), (380, 140, 900, 620)]

        with patch("vertical._probe_video_metadata", return_value=(1920, 1080, 50.0)), \
             patch("vertical._detect_face_once", return_value=((1000, 80, 120, 120), [object()], detect_debug)), \
             patch("vertical._build_camera_crop", side_effect=cam_rects) as cam_crop_mock, \
             patch("vertical._camera_crop_sanity_ok", side_effect=[False, True]), \
             patch("vertical._build_content_crop", return_value=(100, 0, 800, 950)) as content_crop_mock, \
             patch("vertical._run_ffmpeg"):
            vertical._run_facecam_top_split_layout(
                "input.mp4",
                "output.mp4",
                1080,
                1920,
            )

        self.assertEqual(cam_crop_mock.call_count, 2)
        content_crop_mock.assert_called_once_with((520, 260, 420, 420), 1920, 1080, 1080 / 1280)


if __name__ == "__main__":
    unittest.main()
