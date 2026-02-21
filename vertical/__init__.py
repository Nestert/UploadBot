"""vertical — пакет для вертикального ресайза видео и субтитров.

Публичный API:
    convert_to_vertical(input_clip, output_dir, ...)
    burn_subtitles(input_clip, subs_file, ...)

Реализация разнесена по подмодулям:
    vertical.geometry   — геометрия crop, IoU, median box
    vertical.detection  — Haar/YOLO детекция лиц, probing
    vertical.webcam     — обнаружение webcam-оверлея
    vertical.rendering  — FFmpeg рендеринг, convert_to_vertical, burn_subtitles

vertical._core реэкспортирует все имена для совместимости тестов.
"""

from vertical.rendering import convert_to_vertical, burn_subtitles  # noqa: F401

# Реэкспорт для vertical._core (тесты используют vertical._core.XXX)
from vertical import _core  # noqa: F401

# Реэкспорт приватных функций для прямого доступа из тестов
from vertical.rendering import (  # noqa: F401
    _run_standard_layout,
    _run_facecam_top_split_layout,
    _probe_video_metadata,
    _run_ffmpeg,
    _encoding_params,
)
from vertical.detection import (  # noqa: F401
    _detect_face_once,
    _select_best_face_candidate,
    _detect_largest_face,
    _get_face_detectors,
    _normalize_subject_side,
    _normalize_facecam_backend,
    _normalize_facecam_fallback_mode,
    _normalize_facecam_anchor,
    _normalize_detector_list,
    _detect_person_boxes_with_yolo,
    _get_facecam_yolo_model,
    _candidate_cascade_paths,
    _detect_faces_in_resized_frame,
    _resize_for_detection,
    _probe_frames_from_start,
    _probe_frames_with_indices_from_start,
)
from vertical.webcam import (  # noqa: F401
    _detect_webcam_region,
    _heuristic_face_box_from_corners,
    _save_facecam_debug_frames,
)
from vertical.geometry import (  # noqa: F401
    _side_score,
    _build_split_filter,
    _build_camera_crop,
    _build_content_crop,
    _camera_crop_sanity_ok,
    _camera_rect_sanity_ok,
    _build_hard_side_camera_rect,
    _pseudo_face_box_from_camera_rect,
)
