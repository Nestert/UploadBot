import unittest
from types import SimpleNamespace
from unittest.mock import patch

import vertical
import vertical.detection


class TestVerticalLayouts(unittest.TestCase):
    def test_standard_layout_filter_keeps_scale_crop_pipeline(self):
        captured = {}

        def fake_run_ffmpeg(cmd, *_args, **_kwargs):
            captured["cmd"] = cmd

        with patch("vertical.rendering._run_ffmpeg", side_effect=fake_run_ffmpeg):
            vertical._run_standard_layout("input.mp4", "output.mp4", 1080, 1920)

        self.assertIn("-vf", captured["cmd"])
        vf_value = captured["cmd"][captured["cmd"].index("-vf") + 1]
        self.assertIn("scale=1080:1920", vf_value)
        self.assertIn("crop=1080:1920", vf_value)
        self.assertIn("setsar=1", vf_value)

    def test_convert_facecam_layout_falls_back_to_standard(self):
        with patch("vertical.rendering.uuid.uuid4", return_value="test-id"), \
             patch("vertical.rendering._run_facecam_top_split_layout", side_effect=Exception("face not found")), \
             patch("vertical.rendering._run_standard_layout") as standard_mock, \
             patch("vertical.rendering.os.path.exists", return_value=True):
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

        with patch.object(vertical.detection, "cv2", fake_cv2):
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

        with patch("vertical.detection._probe_frames_with_indices_from_start", return_value=frames), \
             patch("vertical.detection._get_face_detectors", return_value=[{"name": "haarcascade_frontalface_default.xml", "detector": object()}]), \
             patch("vertical.detection._resize_for_detection", side_effect=lambda frame, max_side=640: (frame, 1.0)), \
             patch("vertical.detection._detect_faces_in_resized_frame", side_effect=detections):
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

        webcam_debug = {
            "candidate_count": 3,
            "track_count": 1,
            "best_score": 0.82,
            "preferred_side": "left",
            "fallback_reason": None,
        }

        with patch("vertical.rendering._probe_video_metadata", return_value=(1920, 1080, 60.0)), \
             patch("vertical.rendering._probe_frames_with_indices_from_start", return_value=[(0, object()), (3, object())]), \
             patch("vertical.rendering._detect_webcam_region", return_value=((0, 10, 620, 360), webcam_debug)), \
             patch("vertical.rendering._save_facecam_debug_frames"), \
             patch("vertical.rendering._build_content_crop", return_value=(100, 0, 800, 950)) as content_crop_mock, \
             patch("vertical.rendering._run_ffmpeg", side_effect=fake_run_ffmpeg):
            vertical._run_facecam_top_split_layout(
                "input.mp4",
                "output.mp4",
                1080,
                1920,
            )

        content_crop_mock.assert_called_once()
        self.assertIn("-filter_complex", captured["cmd"])
        filter_graph = captured["cmd"][captured["cmd"].index("-filter_complex") + 1]
        self.assertIn("scale=1080:640", filter_graph)
        self.assertIn("scale=1080:1280", filter_graph)
        self.assertIn("vstack=inputs=2", filter_graph)

    def test_facecam_layout_hard_side_fallback_is_used_when_detection_missing(self):
        captured = {}

        def fake_run_ffmpeg(cmd, *_args, **_kwargs):
            captured["cmd"] = cmd

        with patch("vertical.rendering._probe_video_metadata", return_value=(1920, 1080, 50.0)), \
             patch("vertical.rendering._probe_frames_with_indices_from_start", return_value=[(0, object()), (3, object())]), \
             patch(
                 "vertical.rendering._detect_webcam_region",
                 return_value=(None, {"fallback_reason": "low_score", "preferred_side": "right", "candidate_count": 0, "track_count": 0, "best_score": None}),
             ), \
             patch("vertical.rendering._save_facecam_debug_frames"), \
             patch("vertical.rendering._build_content_crop", return_value=(100, 0, 800, 950)), \
             patch("vertical.rendering._run_ffmpeg", side_effect=fake_run_ffmpeg):
            vertical._run_facecam_top_split_layout(
                "input.mp4",
                "output.mp4",
                1080,
                1920,
                facecam_subject_side="auto",
                facecam_fallback_mode="hard_side",
            )

        self.assertIn("-filter_complex", captured["cmd"])
        filter_graph = captured["cmd"][captured["cmd"].index("-filter_complex") + 1]
        self.assertIn("crop=653", filter_graph)

    def test_detect_webcam_region_finds_corner_overlay(self):
        """Webcam с чёткими границами и другим цветом должен обнаруживаться."""
        import numpy as np
        import cv2
        # Синтетический кадр 1920x1080 с webcam оверлеем в левом верхнем углу
        frame1 = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame1[:, :, :] = 30  # тёмный фон (игра)

        # Webcam overlay 422x237 в top-left с яркой границей
        cam_w, cam_h = 422, 237
        frame1[0:cam_h, 0:cam_w, :] = 200  # яркий webcam
        # Рисуем чёткую границу (как в реальных оверлеях)
        cv2.rectangle(frame1, (0, 0), (cam_w, cam_h), (255, 255, 255), 2)

        # Второй кадр с немного другим контентом внутри webcam
        frame2 = frame1.copy()
        frame2[50:150, 50:200, :] = 180
        frame3 = frame1.copy()
        frame3[60:160, 60:210, :] = 170

        frames = [frame1, frame2, frame3]
        result = vertical._detect_webcam_region(1920, 1080, frames, subject_side="left")
        self.assertIsNotNone(result, "Webcam region должен быть обнаружен")
        rx, ry, rw, rh = result
        # Проверяем что найденный регион в левом верхнем углу
        self.assertLessEqual(rx, 100, "x должен быть близко к левому краю")
        self.assertLessEqual(ry, 100, "y должен быть близко к верхнему краю")

    def test_detect_webcam_region_finds_left_edge_overlay(self):
        """Webcam посередине левого края должен обнаруживаться."""
        import numpy as np
        import cv2
        frame1 = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame1[:, :, :] = 30

        # Webcam 400x250 слева по центру
        cam_w, cam_h = 400, 250
        cam_y = (1080 - cam_h) // 2
        
        # Рисуем webcam
        frame1[cam_y:cam_y+cam_h, 0:cam_w, :] = 200
        cv2.rectangle(frame1, (0, cam_y), (cam_w, cam_y+cam_h), (255, 255, 255), 2)

        # Движение
        frame2 = frame1.copy()
        frame2[cam_y+50:cam_y+150, 50:150, :] = 180
        frames = [frame1, frame2, frame1]

        result = vertical._detect_webcam_region(1920, 1080, frames, subject_side="left")
        self.assertIsNotNone(result, "Webcam на левом краю должен быть найден")
        rx, ry, rw, rh = result
        self.assertLessEqual(rx, 50, "x должен быть у края")
        self.assertTrue(abs(ry - cam_y) < 50, "y должен быть примерно по центру")

    def test_detect_webcam_region_returns_none_on_uniform_frames(self):
        """Однородные кадры без webcam-оверлея — результат None."""
        import numpy as np
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
        frames = [frame.copy(), frame.copy(), frame.copy()]
        result = vertical._detect_webcam_region(1920, 1080, frames, subject_side="auto")
        self.assertIsNone(result, "На однородных кадрах webcam не должен обнаруживаться")


    def test_detect_webcam_region_boosts_score_with_face(self):
        """Webcam с лицом внутри должен получать высокий скор."""
        import numpy as np
        import cv2
        from unittest.mock import MagicMock

        # Создаём кадр с "webcam"
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:, :, :] = 30
        
        cam_w, cam_h = 400, 300
        cam_x, cam_y = 0, 0
        frame[cam_y:cam_y+cam_h, cam_x:cam_x+cam_w, :] = 200
        cv2.rectangle(frame, (cam_x, cam_y), (cam_x+cam_w, cam_y+cam_h), (255, 255, 255), 2)
        
        frames = [frame.copy(), frame.copy()] # Мало движения

        # Mock detector that returns a face inside the webcam rect
        mock_detector = MagicMock()
        # detectMultiScale returns list of (x,y,w,h). 
        # Face inside webcam: 100,100, 100,100 (relative to crop!)
        mock_detector.detectMultiScale.return_value = [(100, 100, 100, 100)]
        
        detectors = [{"name": "mock", "detector": mock_detector}]

        result = vertical._detect_webcam_region(1920, 1080, frames, detectors=detectors, subject_side="left")
        
        self.assertIsNotNone(result)
        # Check that detectMultiScale was called
        self.assertTrue(mock_detector.detectMultiScale.called)

    def test_heuristic_face_box_calls_detect_with_named_subject_side(self):
        with patch("vertical.webcam._detect_webcam_region", return_value=(10, 20, 300, 200)) as detect_mock:
            face_box = vertical._heuristic_face_box_from_corners(1920, 1080, [object()], subject_side="right")

        self.assertIsNotNone(face_box)
        self.assertEqual(face_box[0], 100)
        self.assertEqual(face_box[1], 58)
        self.assertEqual(face_box[2], 120)
        self.assertEqual(face_box[3], 90)
        self.assertEqual(detect_mock.call_count, 1)
        self.assertEqual(detect_mock.call_args.kwargs["subject_side"], "right")
        self.assertIsNone(detect_mock.call_args.kwargs["detectors"])

    def test_detect_webcam_region_finds_overlay_with_horizontal_offset(self):
        import numpy as np
        import cv2

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:, :, :] = 35
        cam_x, cam_y, cam_w, cam_h = 120, 380, 380, 240  # ~6% отступ от левого края
        frame[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w, :] = 205
        cv2.rectangle(frame, (cam_x, cam_y), (cam_x + cam_w, cam_y + cam_h), (255, 255, 255), 2)
        frame2 = frame.copy()
        frame2[cam_y + 40:cam_y + 130, cam_x + 30:cam_x + 140, :] = 175
        frames = [frame, frame2, frame]

        result = vertical._detect_webcam_region(1920, 1080, frames, subject_side="left")
        self.assertIsNotNone(result)
        rx, _ry, _rw, _rh = result
        self.assertGreaterEqual(rx, 60)
        self.assertLessEqual(rx, 180)

    def test_detect_webcam_region_prefers_middle_y_when_side_same(self):
        import numpy as np
        import cv2

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:, :, :] = 30
        # Кандидат A: верхний левый
        frame[40:240, 40:360, :] = 200
        cv2.rectangle(frame, (40, 40), (360, 240), (255, 255, 255), 2)
        # Кандидат B: левый по центру (должен победить)
        cam_y = 430
        frame[cam_y:cam_y + 230, 60:390, :] = 205
        cv2.rectangle(frame, (60, cam_y), (390, cam_y + 230), (255, 255, 255), 2)
        frame2 = frame.copy()
        frame2[cam_y + 40:cam_y + 130, 100:220, :] = 175
        frames = [frame, frame2, frame]

        result = vertical._detect_webcam_region(1920, 1080, frames, subject_side="left")
        self.assertIsNotNone(result)
        _rx, ry, _rw, _rh = result
        self.assertGreater(ry, 320)
        self.assertLess(ry, 620)

    def test_detect_webcam_region_rejects_transient_false_rect(self):
        import numpy as np
        import cv2

        base = np.zeros((1080, 1920, 3), dtype=np.uint8)
        base[:, :, :] = 28
        # Настоящая webcam-подобная область слева по центру (устойчивая)
        stable_x, stable_y, stable_w, stable_h = 80, 420, 360, 230

        f1 = base.copy()
        f2 = base.copy()
        f3 = base.copy()
        for f in (f1, f2, f3):
            f[stable_y:stable_y + stable_h, stable_x:stable_x + stable_w, :] = 200
            cv2.rectangle(f, (stable_x, stable_y), (stable_x + stable_w, stable_y + stable_h), (255, 255, 255), 2)
        # Ложный крупный прямоугольник только на первом кадре справа
        f1[140:600, 1300:1840, :] = 210
        cv2.rectangle(f1, (1300, 140), (1840, 600), (255, 255, 255), 2)
        # Движение внутри устойчивой области
        f2[stable_y + 60:stable_y + 150, stable_x + 60:stable_x + 180, :] = 170
        f3[stable_y + 70:stable_y + 170, stable_x + 70:stable_x + 190, :] = 160

        result = vertical._detect_webcam_region(1920, 1080, [f1, f2, f3], subject_side="left")
        self.assertIsNotNone(result)
        rx, _ry, _rw, _rh = result
        self.assertLess(rx, 400, "должен победить устойчивый регион слева, а не transient справа")


if __name__ == "__main__":
    unittest.main()
