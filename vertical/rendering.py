"""vertical.rendering — FFmpeg рендеринг, facecam layout, vertical conversion."""

import logging
import os
import subprocess
import time
import uuid

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from errors import VideoProcessingError, SubtitleError
from vertical.geometry import (
    _camera_rect_sanity_ok,
    _pseudo_face_box_from_camera_rect,
    _build_hard_side_camera_rect,
    _build_camera_crop,
    _camera_crop_sanity_ok,
    _build_content_crop,
    _build_split_filter,
)
from vertical.detection import (
    _normalize_subject_side,
    _normalize_facecam_backend,
    _normalize_facecam_fallback_mode,
    _normalize_facecam_anchor,
    _get_face_detectors,
    _detect_face_once,
    _probe_frames_with_indices_from_start,
)
from vertical.webcam import _detect_webcam_region, _save_facecam_debug_frames


def _run_ffmpeg(cmd, error_prefix, cwd=None):
    """Запускает ffmpeg-команду и пробрасывает понятную ошибку."""
    try:
        subprocess.run(cmd, check=True, cwd=cwd)
    except subprocess.CalledProcessError as exc:
        raise VideoProcessingError(f"{error_prefix}: {exc}") from exc


def _encoding_params(encode_preset):
    """Единые параметры кодирования видео с приоритетом скорости."""
    preset = encode_preset or "veryfast"
    return ["-c:v", "libx264", "-preset", preset, "-crf", "23"]


def _probe_video_metadata(input_clip):
    """Возвращает ширину, высоту и длительность видео через ffprobe."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration", "-of", "json", input_clip]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        payload = __import__("json").loads(result.stdout or "{}")
    except Exception as exc:
        raise VideoProcessingError(f"Не удалось получить метаданные видео: {exc}") from exc
    stream = (payload.get("streams") or [{}])[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    duration_raw = (payload.get("format") or {}).get("duration")
    try:
        duration = float(duration_raw) if duration_raw not in (None, "N/A", "") else 0.0
    except (TypeError, ValueError):
        duration = 0.0
    if width <= 0 or height <= 0:
        raise VideoProcessingError("ffprobe вернул некорректные размеры видео.")
    return width, height, max(0.0, duration)


def _run_standard_layout(input_clip, output_file, target_width, target_height, subs_file=None, encode_preset="veryfast", face_found_source="standard"):
    """Текущий базовый режим: масштаб + центральный crop до 9:16."""
    vf_str = f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase:flags=lanczos,crop={target_width}:{target_height},setsar=1,setdar={target_width}/{target_height}"
    subs_cwd = None
    if subs_file:
        subs_cwd = os.path.dirname(os.path.abspath(subs_file))
        subs_name = os.path.basename(subs_file)
        vf_str = f"{vf_str},subtitles={subs_name}"
    cmd = ["ffmpeg", "-i", input_clip, "-vf", vf_str, *_encoding_params(encode_preset), "-c:a", "copy", output_file, "-y"]
    started = time.perf_counter()
    _run_ffmpeg(cmd, "Ошибка вертикального ресайза видео", cwd=subs_cwd)
    ffmpeg_encode_ms = int((time.perf_counter() - started) * 1000)
    logging.info("Vertical encode done: mode=standard face_found_source=%s ffmpeg_encode_ms=%s", face_found_source, ffmpeg_encode_ms)


def _run_facecam_top_split_layout(input_clip, output_file, target_width, target_height, facecam_ratio=1/3, facecam_subject_side="left", facecam_detector_backend="yolo_window_v1", facecam_fallback_mode="hard_side", facecam_anchor="edge_middle", subs_file=None, encode_preset="veryfast"):
    """Режим: верх 1/3 facecam, низ 2/3 игровой контент."""
    source_w, source_h, duration = _probe_video_metadata(input_clip)
    normalized_side = _normalize_subject_side(facecam_subject_side)
    normalized_backend = _normalize_facecam_backend(facecam_detector_backend)
    normalized_fallback_mode = _normalize_facecam_fallback_mode(facecam_fallback_mode)
    normalized_anchor = _normalize_facecam_anchor(facecam_anchor)
    detect_started = time.perf_counter()
    probe_seconds = min(6.0, duration) if duration > 0 else 6.0
    probe_indexed = _probe_frames_with_indices_from_start(input_clip, max_probe_seconds=probe_seconds, sample_every_n_frames=3, max_samples=36)
    probe_frames = [frame for _, frame in probe_indexed]
    detectors = _get_face_detectors()
    top_height = max(2, min(target_height - 2, int(round(target_height * facecam_ratio))))
    bottom_height = target_height - top_height
    if bottom_height <= 1:
        raise VideoProcessingError("Некорректные размеры верхнего/нижнего блока.")
    top_aspect = target_width / top_height
    bottom_aspect = target_width / bottom_height
    webcam_rect, webcam_debug = _detect_webcam_region(source_w, source_h, probe_frames, detectors=detectors, subject_side=normalized_side, detector_backend=normalized_backend, anchor=normalized_anchor, min_score=0.30, return_debug=True)
    detect_once_ms = int((time.perf_counter() - detect_started) * 1000)
    selected_face_box = None
    camera_rect = None
    face_found_source = None
    fallback_reason = webcam_debug.get("fallback_reason")
    resolved_side = webcam_debug.get("preferred_side") or "left"
    if webcam_rect is not None and _camera_rect_sanity_ok(webcam_rect, source_w, source_h):
        camera_rect = webcam_rect
        selected_face_box = _pseudo_face_box_from_camera_rect(webcam_rect)
        face_found_source = f"webcam_region_{normalized_backend}"
        fallback_reason = None
        resolved_side = normalized_side if normalized_side in {"left", "right"} else (webcam_debug.get("preferred_side") or "left")
    else:
        if webcam_rect is not None and not _camera_rect_sanity_ok(webcam_rect, source_w, source_h):
            fallback_reason = "camera_rect_sanity_failed"
        if normalized_fallback_mode == "hard_side":
            if normalized_side in {"left", "right"}:
                resolved_side = normalized_side
            camera_rect = _build_hard_side_camera_rect(source_w, source_h, top_aspect, subject_side=resolved_side, anchor=normalized_anchor)
            if not _camera_rect_sanity_ok(camera_rect, source_w, source_h):
                raise VideoProcessingError("Hard-side fallback построил некорректный camera_rect.")
            selected_face_box = _pseudo_face_box_from_camera_rect(camera_rect)
            face_found_source = f"hard_side_{resolved_side}"
        else:
            detect_result = _detect_face_once(input_clip, max_probe_seconds=min(2.0, probe_seconds), sample_every_n_frames=3, max_side=640, subject_side=normalized_side, return_probe_frames=True, return_debug=True)
            face_box, probe_frames, detect_debug = detect_result
            candidate_boxes = []
            if face_box:
                candidate_boxes.append(tuple(face_box))
            for item in detect_debug.get("ranked_candidates") or []:
                ranked_box = item.get("face_box")
                if ranked_box:
                    ranked_box = tuple(ranked_box)
                    if ranked_box not in candidate_boxes:
                        candidate_boxes.append(ranked_box)
            for candidate_box in candidate_boxes[:2]:
                candidate_camera_rect = _build_camera_crop(candidate_box, source_w, source_h, top_aspect)
                if _camera_crop_sanity_ok(candidate_box, candidate_camera_rect):
                    selected_face_box = candidate_box
                    camera_rect = candidate_camera_rect
                    face_found_source = "legacy_face_fallback"
                    fallback_reason = None
                    break
            if selected_face_box is None:
                fallback_reason = fallback_reason or detect_debug.get("fallback_reason") or "legacy_face_fallback_failed"
    if selected_face_box is None or camera_rect is None:
        raise VideoProcessingError("Не удалось получить валидный camera_rect для facecam-режима.")
    if not _camera_rect_sanity_ok(camera_rect, source_w, source_h):
        raise VideoProcessingError("Итоговый camera_rect не прошел sanity-check.")
    _save_facecam_debug_frames(probe_indexed, camera_rect, output_file, max_frames=5)
    logging.info("facecam_detect_v3 backend=%s fallback_mode=%s anchor=%s subject_side=%s preferred_side=%s probe_frames=%s candidate_count=%s track_count=%s best_score=%s fallback_reason=%s final_source=%s final_rect=%s detect_once_ms=%s", normalized_backend, normalized_fallback_mode, normalized_anchor, normalized_side, resolved_side, len(probe_frames), webcam_debug.get("candidate_count"), webcam_debug.get("track_count"), webcam_debug.get("best_score"), fallback_reason, face_found_source, camera_rect, detect_once_ms)
    content_rect = _build_content_crop(selected_face_box, source_w, source_h, bottom_aspect)
    filter_graph = _build_split_filter(camera_rect, content_rect, target_width, top_height, bottom_height)
    output_label = "[v]"
    subs_cwd = None
    if subs_file:
        subs_cwd = os.path.dirname(os.path.abspath(subs_file))
        subs_name = os.path.basename(subs_file)
        output_label = "[vsub]"
        filter_graph = f"{filter_graph};[v]subtitles={subs_name}{output_label}"
    cmd = ["ffmpeg", "-i", input_clip, "-filter_complex", filter_graph, "-map", output_label, "-map", "0:a?", *_encoding_params(encode_preset), "-c:a", "copy", output_file, "-y"]
    encode_started = time.perf_counter()
    _run_ffmpeg(cmd, "Ошибка компоновки режима facecam_top_split", cwd=subs_cwd)
    ffmpeg_encode_ms = int((time.perf_counter() - encode_started) * 1000)
    logging.info("Facecam detect summary: detect_once_ms=%s face_found_source=%s ffmpeg_encode_ms=%s", detect_once_ms, face_found_source, ffmpeg_encode_ms)


def convert_to_vertical(input_clip, output_dir=None, target_width=1080, target_height=1920, layout_mode="standard", facecam_ratio=1/3, facecam_subject_side="left", facecam_detector_backend="yolo_window_v1", facecam_fallback_mode="hard_side", facecam_anchor="edge_middle", subs_file=None, encode_preset="veryfast"):
    """Приводит входной ролик к вертикальному формату."""
    vert_id = str(uuid.uuid4())
    output_file = os.path.join(output_dir, f"vertical_{vert_id}.mp4") if output_dir else f"vertical_{vert_id}.mp4"
    selected_layout = layout_mode if layout_mode in {"standard", "facecam_top_split"} else "standard"
    if selected_layout != layout_mode:
        logging.warning("Неизвестный layout_mode '%s', использую standard.", layout_mode)
    try:
        if selected_layout == "facecam_top_split":
            try:
                _run_facecam_top_split_layout(input_clip, output_file, target_width, target_height, facecam_ratio=facecam_ratio, facecam_subject_side=facecam_subject_side, facecam_detector_backend=facecam_detector_backend, facecam_fallback_mode=facecam_fallback_mode, facecam_anchor=facecam_anchor, subs_file=subs_file, encode_preset=encode_preset)
            except Exception as exc:
                logging.warning("Facecam-режим недоступен (%s). Fallback на standard 9:16.", exc)
                _run_standard_layout(input_clip, output_file, target_width, target_height, subs_file=subs_file, encode_preset=encode_preset, face_found_source="fallback_standard")
        else:
            _run_standard_layout(input_clip, output_file, target_width, target_height, subs_file=subs_file, encode_preset=encode_preset, face_found_source="standard")
        if os.path.exists(output_file):
            return output_file
        raise VideoProcessingError("FFmpeg не создал вертикальный файл.")
    except Exception as exc:
        raise VideoProcessingError(f"Ошибка вертикального ресайза видео: {exc}") from exc


def burn_subtitles(input_clip, subs_file, output_dir=None):
    """Прожигает субтитры (.srt) на видео с помощью FFmpeg."""
    sub_id = str(uuid.uuid4())
    output_file = os.path.join(output_dir, f"subs_{sub_id}.mp4") if output_dir else f"subs_{sub_id}.mp4"
    if not subs_file or not os.path.exists(subs_file):
        raise SubtitleError(f"Файл субтитров не найден: {subs_file}")
    subs_cwd = os.path.dirname(os.path.abspath(subs_file))
    subs_name = os.path.basename(subs_file)
    subtitles_filter = f"subtitles={subs_name}"
    cmd = ["ffmpeg", "-i", input_clip, "-vf", subtitles_filter, "-c:a", "copy", output_file, "-y"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=subs_cwd)
        if os.path.exists(output_file):
            return output_file
        else:
            raise SubtitleError("FFmpeg не создал итоговый файл с субтитрами.")
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        stdout = (e.stdout or "").strip()
        details = stderr[:500] if stderr else stdout[:500]
        raise SubtitleError(f"Ошибка при добавлении субтитров: {e}. {details}")
    except Exception as e:
        raise SubtitleError(f"Ошибка при добавлении субтитров: {e}")
