"""vertical._core — тонкий реэкспорт из подмодулей для совместимости с тестами.

Вся логика перенесена в vertical.geometry, vertical.detection,
vertical.webcam и vertical.rendering. Этот модуль сохраняет все имена
в пространстве vertical._core, чтобы mock-patch в тестах продолжал работать.
"""

# ── geometry ────────────────────────────────────────────────────────────
from vertical.geometry import (  # noqa: F401
    _clamp,
    _intersection_area,
    _box_iou,
    _median_box,
    _normalize_face_box,
    _is_candidate_face_box_valid,
    _side_score,
    _select_evenly_spaced,
    _person_overlap_ratio,
    _pseudo_face_box_from_camera_rect,
    _camera_rect_sanity_ok,
    _build_hard_side_camera_rect,
    _fit_aspect_crop,
    _build_camera_crop,
    _camera_crop_sanity_ok,
    _build_content_crop,
    _build_split_filter,
)

# ── detection ───────────────────────────────────────────────────────────
from vertical.detection import (  # noqa: F401
    _normalize_subject_side,
    _detector_bonus,
    _normalize_facecam_backend,
    _normalize_facecam_fallback_mode,
    _normalize_facecam_anchor,
    _normalize_detector_list,
    _candidate_cascade_paths,
    _get_face_detectors,
    _detect_largest_face,
    _resize_for_detection,
    _detect_faces_in_resized_frame,
    _get_facecam_yolo_model,
    _detect_person_boxes_with_yolo,
    _select_best_face_candidate,
    _probe_frames_with_indices_from_start,
    _probe_frames_from_start,
    _detect_face_once,
)

# ── webcam ──────────────────────────────────────────────────────────────
from vertical.webcam import (  # noqa: F401
    _skin_ratio_bgr,
    _edge_strength_at_boundary,
    _histogram_divergence,
    _temporal_activity_in_region,
    _boundary_consistency,
    _extract_rect_candidates_from_frame,
    _deduplicate_rect_candidates,
    _build_webcam_tracks,
    _score_webcam_track,
    _detect_webcam_region,
    _heuristic_face_box_from_corners,
    _save_facecam_debug_frames,
)

# ── rendering ───────────────────────────────────────────────────────────
from vertical.rendering import (  # noqa: F401
    _run_ffmpeg,
    _encoding_params,
    _probe_video_metadata,
    _run_standard_layout,
    _run_facecam_top_split_layout,
    convert_to_vertical,
    burn_subtitles,
)

# cv2 и uuid импортируются чтобы тесты могли использовать
# patch("vertical._core.cv2", ...) и patch("vertical._core.uuid.uuid4", ...)
import uuid  # noqa: F401

try:
    import cv2  # noqa: F401
except Exception:  # pragma: no cover
    cv2 = None

import os  # noqa: F401
