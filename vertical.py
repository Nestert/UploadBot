"""vertical.py — перевод клипа в вертикальный формат и добавление субтитров."""

import glob
import json
import logging
import os
import statistics
import subprocess
import time
import uuid
from collections import Counter, defaultdict

try:
    import cv2
except Exception:  # pragma: no cover - среда может быть без OpenCV
    cv2 = None

_FACE_DETECTORS = None

_DETECTOR_BONUS = {
    "haarcascade_frontalface_alt2.xml": 0.03,
    "haarcascade_frontalface_alt.xml": 0.025,
    "haarcascade_frontalface_default.xml": 0.02,
    "haarcascade_profileface.xml": 0.008,
}
_SUPPORTED_FACECAM_SIDES = {"left", "right", "auto"}


def _run_ffmpeg(cmd, error_prefix, cwd=None):
    """Запускает ffmpeg-команду и пробрасывает понятную ошибку."""
    try:
        subprocess.run(cmd, check=True, cwd=cwd)
    except subprocess.CalledProcessError as exc:
        raise Exception(f"{error_prefix}: {exc}") from exc


def _encoding_params(encode_preset):
    """Единые параметры кодирования видео с приоритетом скорости."""
    preset = encode_preset or "veryfast"
    return [
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        "23",
    ]


def _run_standard_layout(
    input_clip,
    output_file,
    target_width,
    target_height,
    subs_file=None,
    encode_preset="veryfast",
    face_found_source="standard",
):
    """Текущий базовый режим: масштаб + центральный crop до 9:16."""
    vf_str = (
        f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={target_width}:{target_height},"
        f"setsar=1,setdar={target_width}/{target_height}"
    )
    # На Windows используем basename + cwd, чтобы не экранировать путь в фильтре
    subs_cwd = None
    if subs_file:
        subs_cwd = os.path.dirname(os.path.abspath(subs_file))
        subs_name = os.path.basename(subs_file)
        vf_str = f"{vf_str},subtitles={subs_name}"

    cmd = [
        "ffmpeg",
        "-i",
        input_clip,
        "-vf",
        vf_str,
        *_encoding_params(encode_preset),
        "-c:a",
        "copy",
        output_file,
        "-y",
    ]
    started = time.perf_counter()
    _run_ffmpeg(cmd, "Ошибка вертикального ресайза видео", cwd=subs_cwd)
    ffmpeg_encode_ms = int((time.perf_counter() - started) * 1000)
    logging.info(
        "Vertical encode done: mode=standard face_found_source=%s ffmpeg_encode_ms=%s",
        face_found_source,
        ffmpeg_encode_ms,
    )


def _probe_video_metadata(input_clip):
    """Возвращает ширину, высоту и длительность видео через ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height:format=duration",
        "-of",
        "json",
        input_clip,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout or "{}")
    except Exception as exc:
        raise Exception(f"Не удалось получить метаданные видео: {exc}") from exc

    stream = (payload.get("streams") or [{}])[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    duration_raw = (payload.get("format") or {}).get("duration")
    try:
        duration = float(duration_raw) if duration_raw not in (None, "N/A", "") else 0.0
    except (TypeError, ValueError):
        duration = 0.0

    if width <= 0 or height <= 0:
        raise Exception("ffprobe вернул некорректные размеры видео.")
    return width, height, max(0.0, duration)


def _candidate_cascade_paths():
    """Ищет доступные Haar-cascade файлы для детекции лиц."""
    if cv2 is None:
        return []

    dirs = set()
    if hasattr(cv2, "data"):
        cascades_dir = getattr(cv2.data, "haarcascades", None)
        if cascades_dir:
            dirs.add(cascades_dir)

    cv2_root = os.path.dirname(cv2.__file__)
    dirs.update(
        {
            os.path.join(cv2_root, "data"),
            "/opt/homebrew/share/opencv4/haarcascades",
            "/usr/local/share/opencv4/haarcascades",
            "/usr/share/opencv4/haarcascades",
            "/usr/share/OpenCV/haarcascades",
        }
    )
    dirs.update(glob.glob("/opt/homebrew/Cellar/opencv/*/share/opencv4/haarcascades"))

    filenames = [
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_frontalface_alt.xml",
        "haarcascade_frontalface_default.xml",
        "haarcascade_profileface.xml",
    ]

    paths = []
    for filename in filenames:
        found = None
        for d in sorted(dirs):
            if not d:
                continue
            candidate = os.path.join(d, filename)
            if os.path.exists(candidate):
                found = candidate
                break
        if found:
            paths.append((filename, found))
    return paths


def _get_face_detectors():
    """Ленивая инициализация детекторов лиц."""
    global _FACE_DETECTORS
    if _FACE_DETECTORS is not None:
        return _FACE_DETECTORS

    if cv2 is None:
        logging.warning("OpenCV недоступен, facecam-режим будет с fallback на стандартный 9:16.")
        _FACE_DETECTORS = []
        return _FACE_DETECTORS

    detectors = []
    for name, path in _candidate_cascade_paths():
        detector = cv2.CascadeClassifier(path)
        if detector.empty():
            continue
        detectors.append({"name": name, "detector": detector})

    if not detectors:
        logging.warning("Не найдены Haar-cascade файлы для детекции лиц.")
    _FACE_DETECTORS = detectors
    return _FACE_DETECTORS


def _detect_largest_face(frame, detectors):
    """Возвращает самый крупный face-box (x, y, w, h) на кадре."""
    if cv2 is None or frame is None:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    min_side = max(20, int(min(gray.shape[:2]) * 0.04))

    best = None
    best_area = 0
    for detector_item in detectors:
        detector = detector_item["detector"] if isinstance(detector_item, dict) else detector_item
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(min_side, min_side),
        )
        for x, y, w, h in faces:
            area = int(w) * int(h)
            if area > best_area:
                best_area = area
                best = (int(x), int(y), int(w), int(h))
    return best


def _normalize_subject_side(subject_side):
    normalized = (subject_side or "").strip().lower()
    if normalized in _SUPPORTED_FACECAM_SIDES:
        return normalized
    return "left"


def _detector_bonus(source_name):
    return _DETECTOR_BONUS.get(source_name, 0.0)


def _box_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    a2x = ax + aw
    a2y = ay + ah
    b2x = bx + bw
    b2y = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(a2x, b2x)
    inter_y2 = min(a2y, b2y)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    union = (aw * ah) + (bw * bh) - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area) / float(union)


def _median_box(boxes):
    if not boxes:
        return None
    xs = [box[0] for box in boxes]
    ys = [box[1] for box in boxes]
    ws = [box[2] for box in boxes]
    hs = [box[3] for box in boxes]
    return (
        int(round(statistics.median(xs))),
        int(round(statistics.median(ys))),
        int(round(statistics.median(ws))),
        int(round(statistics.median(hs))),
    )


def _normalize_face_box(x, y, w, h, frame_w, frame_h):
    x = max(0, min(int(x), max(0, frame_w - 2)))
    y = max(0, min(int(y), max(0, frame_h - 2)))
    w = max(2, min(int(w), frame_w - x))
    h = max(2, min(int(h), frame_h - y))
    return (x, y, w, h)


def _is_candidate_face_box_valid(face_box, frame_w, frame_h):
    x, y, w, h = face_box
    area = w * h
    min_area = max(2500, int(frame_w * frame_h * 0.0025))
    if area < min_area:
        return False
    ratio = w / float(max(h, 1))
    if ratio < 0.65 or ratio > 1.45:
        return False

    edge_pad = max(4, int(round(min(frame_w, frame_h) * 0.01)))
    touching_edges = 0
    if x <= edge_pad:
        touching_edges += 1
    if y <= edge_pad:
        touching_edges += 1
    if (x + w) >= (frame_w - edge_pad):
        touching_edges += 1
    if (y + h) >= (frame_h - edge_pad):
        touching_edges += 1
    return touching_edges < 2


def _side_score(center_x, frame_w, subject_side):
    if frame_w <= 0:
        return 0.5
    normalized_side = _normalize_subject_side(subject_side)
    cx_norm = _clamp(center_x / float(frame_w), 0.0, 1.0)
    if normalized_side == "left":
        return 1.0 - cx_norm
    if normalized_side == "right":
        return cx_norm
    return 0.5


def _select_evenly_spaced(items, max_count):
    if max_count <= 0 or len(items) <= max_count:
        return list(items)
    if max_count == 1:
        return [items[len(items) // 2]]

    selected = []
    used_indexes = set()
    last_index = len(items) - 1
    for i in range(max_count):
        raw_idx = int(round((i * last_index) / float(max_count - 1)))
        idx = raw_idx
        while idx in used_indexes and idx < last_index:
            idx += 1
        while idx in used_indexes and idx > 0:
            idx -= 1
        used_indexes.add(idx)
        selected.append(items[idx])
    return selected


def _detect_faces_in_resized_frame(frame, detectors):
    if cv2 is None or frame is None:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    min_side = max(20, int(min(gray.shape[:2]) * 0.04))
    gray_width = int(gray.shape[1]) if len(gray.shape) > 1 else 0
    flipped = None
    detections = []

    for detector_item in detectors:
        if isinstance(detector_item, dict):
            detector = detector_item.get("detector")
            detector_name = detector_item.get("name", "unknown")
        else:
            detector = detector_item
            detector_name = "unknown"
        if detector is None:
            continue

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(min_side, min_side),
        )
        for x, y, w, h in faces:
            detections.append(
                {
                    "box": (int(x), int(y), int(w), int(h)),
                    "source": detector_name,
                    "mirrored": False,
                }
            )

        if "profileface" in detector_name and gray_width > 0:
            if flipped is None:
                flipped = cv2.flip(gray, 1)
            mirrored_faces = detector.detectMultiScale(
                flipped,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(min_side, min_side),
            )
            for x, y, w, h in mirrored_faces:
                mirrored_x = gray_width - (int(x) + int(w))
                detections.append(
                    {
                        "box": (int(mirrored_x), int(y), int(w), int(h)),
                        "source": detector_name,
                        "mirrored": True,
                    }
                )

    return detections


def _select_best_face_candidate(candidates, total_probe_frames, frame_w, frame_h, subject_side="left"):
    if not candidates or total_probe_frames <= 0:
        return {"best": None, "ranked": []}

    clusters = []
    for candidate in candidates:
        candidate_box = candidate["box"]
        best_cluster_idx = -1
        best_cluster_iou = 0.0
        for idx, cluster in enumerate(clusters):
            overlap = _box_iou(candidate_box, cluster["anchor_box"])
            if overlap >= 0.35 and overlap > best_cluster_iou:
                best_cluster_iou = overlap
                best_cluster_idx = idx

        if best_cluster_idx < 0:
            clusters.append({"anchor_box": candidate_box, "members": [candidate]})
            continue

        clusters[best_cluster_idx]["members"].append(candidate)
        clusters[best_cluster_idx]["anchor_box"] = _median_box(
            [item["box"] for item in clusters[best_cluster_idx]["members"]]
        )

    frame_candidates = defaultdict(list)
    for candidate in candidates:
        frame_candidates[candidate["probe_idx"]].append(candidate["box"])

    ranked = []
    for cluster in clusters:
        cluster_boxes = [item["box"] for item in cluster["members"]]
        cluster_box = _median_box(cluster_boxes)
        if cluster_box is None:
            continue
        bx, by, bw, bh = cluster_box
        center_x = bx + (bw / 2.0)
        center_y = by + (bh / 2.0)
        rel_area = (bw * bh) / float(max(1, frame_w * frame_h))

        temporal_hits = 0
        for probe_idx in range(total_probe_frames):
            overlap_hit = any(_box_iou(cluster_box, box) >= 0.35 for box in frame_candidates.get(probe_idx, []))
            if overlap_hit:
                temporal_hits += 1
        s_temporal = temporal_hits / float(total_probe_frames)
        s_area = _clamp(rel_area / 0.08, 0.0, 1.0)
        s_side = _side_score(center_x, frame_w, subject_side)
        center_y_norm = _clamp(center_y / float(max(frame_h, 1)), 0.0, 1.0)
        s_center_y = max(0.0, 1.0 - (abs(center_y_norm - 0.45) / 0.45))

        source_counts = Counter(item.get("source", "unknown") for item in cluster["members"])
        best_source, _ = source_counts.most_common(1)[0]
        bonus = _detector_bonus(best_source)
        score = (0.40 * s_temporal) + (0.30 * s_area) + (0.20 * s_side) + (0.10 * s_center_y) + bonus

        representative = max(cluster["members"], key=lambda item: item["box"][2] * item["box"][3])
        ranked.append(
            {
                "face_box": cluster_box,
                "score": float(score),
                "source": best_source,
                "frame_idx": representative.get("frame_idx", 0),
                "breakdown": {
                    "s_temporal": round(s_temporal, 4),
                    "s_area": round(s_area, 4),
                    "s_side": round(s_side, 4),
                    "s_center_y": round(s_center_y, 4),
                    "detector_bonus": round(bonus, 4),
                },
            }
        )

    ranked.sort(key=lambda item: (item["score"], item["breakdown"]["s_temporal"]), reverse=True)
    return {"best": ranked[0] if ranked else None, "ranked": ranked}


def _resize_for_detection(frame, max_side=640):
    """Уменьшает кадр для ускорения детекции и возвращает scale."""
    if cv2 is None or frame is None:
        return frame, 1.0
    h, w = frame.shape[:2]
    max_dim = max(h, w)
    if max_dim <= 0 or max_dim <= max_side:
        return frame, 1.0
    scale = max_side / float(max_dim)
    resized = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def _probe_frames_with_indices_from_start(
    input_clip,
    max_probe_seconds=5.0,
    sample_every_n_frames=6,
    max_samples=None,
):
    """Читает начало клипа и возвращает [(frame_idx, frame)] без random-seek."""
    if cv2 is None:
        return []

    capture = cv2.VideoCapture(input_clip)
    if not capture.isOpened():
        return []

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    max_probe_frames = max(1, int(max_probe_seconds * fps))
    sample_step = max(1, int(sample_every_n_frames))

    frames = []
    frame_idx = 0
    try:
        while frame_idx <= max_probe_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_idx % sample_step == 0:
                frames.append((frame_idx, frame))
            frame_idx += 1
    finally:
        capture.release()

    if max_samples is not None:
        frames = _select_evenly_spaced(frames, max(1, int(max_samples)))
    return frames


def _probe_frames_from_start(
    input_clip,
    max_probe_seconds=5.0,
    sample_every_n_frames=6,
    max_samples=None,
):
    """Читает только начало клипа и возвращает выборку кадров."""
    indexed = _probe_frames_with_indices_from_start(
        input_clip,
        max_probe_seconds=max_probe_seconds,
        sample_every_n_frames=sample_every_n_frames,
        max_samples=max_samples,
    )
    return [frame for _, frame in indexed]


def _detect_face_once(
    input_clip,
    max_probe_seconds=2.0,
    sample_every_n_frames=3,
    max_side=640,
    subject_side="left",
    min_best_score=0.55,
    max_probe_samples=24,
    return_probe_frames=False,
    return_debug=False,
):
    """
    Ищет устойчивый face-box в начале клипа по multi-frame scoring.
    """
    normalized_side = _normalize_subject_side(subject_side)
    detectors = _get_face_detectors()
    probe_indexed = _probe_frames_with_indices_from_start(
        input_clip,
        max_probe_seconds=max_probe_seconds,
        sample_every_n_frames=sample_every_n_frames,
        max_samples=max_probe_samples,
    )
    probe_frames = [frame for _, frame in probe_indexed]
    detect_debug = {
        "subject_side": normalized_side,
        "probe_frames": len(probe_frames),
        "detector_counts": {},
        "best_score": None,
        "best_source": None,
        "best_frame_idx": None,
        "ranked_candidates": [],
        "fallback_reason": None,
    }

    def _pack(face_box):
        if return_probe_frames and return_debug:
            return face_box, probe_frames, detect_debug
        if return_probe_frames:
            return face_box, probe_frames
        if return_debug:
            return face_box, detect_debug
        return face_box

    if not detectors:
        detect_debug["fallback_reason"] = "no_detectors"
        return _pack(None)
    if not probe_indexed:
        detect_debug["fallback_reason"] = "no_probe_frames"
        return _pack(None)

    detector_counts = Counter()
    all_candidates = []
    frame_w = 0
    frame_h = 0

    for probe_idx, (frame_idx, frame) in enumerate(probe_indexed):
        h, w = frame.shape[:2]
        frame_w = max(frame_w, int(w))
        frame_h = max(frame_h, int(h))
        resized, scale = _resize_for_detection(frame, max_side=max_side)
        detected_faces = _detect_faces_in_resized_frame(resized, detectors)

        for detected in detected_faces:
            source = detected.get("source", "unknown")
            detector_counts[source] += 1
            x, y, fw, fh = detected.get("box", (0, 0, 0, 0))
            if scale > 0 and scale != 1.0:
                inv = 1.0 / scale
                x = int(round(x * inv))
                y = int(round(y * inv))
                fw = int(round(fw * inv))
                fh = int(round(fh * inv))

            face_box = _normalize_face_box(x, y, fw, fh, w, h)
            if not _is_candidate_face_box_valid(face_box, w, h):
                continue

            all_candidates.append(
                {
                    "probe_idx": probe_idx,
                    "frame_idx": frame_idx,
                    "box": face_box,
                    "source": source,
                }
            )

    detect_debug["detector_counts"] = dict(detector_counts)
    if not all_candidates:
        detect_debug["fallback_reason"] = "no_valid_candidates"
        return _pack(None)

    selected = _select_best_face_candidate(
        all_candidates,
        total_probe_frames=len(probe_indexed),
        frame_w=frame_w,
        frame_h=frame_h,
        subject_side=normalized_side,
    )
    ranked = selected.get("ranked", [])
    best = selected.get("best")
    detect_debug["ranked_candidates"] = ranked
    if not best:
        detect_debug["fallback_reason"] = "no_ranked_candidates"
        return _pack(None)

    detect_debug["best_score"] = round(float(best.get("score", 0.0)), 4)
    detect_debug["best_source"] = best.get("source")
    detect_debug["best_frame_idx"] = best.get("frame_idx")

    if float(best.get("score", 0.0)) < float(min_best_score):
        detect_debug["fallback_reason"] = "low_score"
        top3 = [
            {
                "score": round(float(item.get("score", 0.0)), 4),
                "source": item.get("source"),
                "frame_idx": item.get("frame_idx"),
                "breakdown": item.get("breakdown"),
            }
            for item in ranked[:3]
        ]
        logging.debug("face_detect_v2 low_confidence top_candidates=%s", top3)
        return _pack(None)

    return _pack(best.get("face_box"))


def _clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def _fit_aspect_crop(center_x, center_y, frame_w, frame_h, target_aspect, preferred_h=None):
    """Строит crop прямоугольник нужного aspect ratio вокруг заданного центра."""
    if preferred_h is None:
        crop_w = float(frame_w)
        crop_h = crop_w / target_aspect
        if crop_h > frame_h:
            crop_h = float(frame_h)
            crop_w = crop_h * target_aspect
    else:
        crop_h = float(preferred_h)
        crop_w = crop_h * target_aspect
        if crop_w > frame_w:
            crop_w = float(frame_w)
            crop_h = crop_w / target_aspect
        if crop_h > frame_h:
            crop_h = float(frame_h)
            crop_w = crop_h * target_aspect

    crop_w = max(2, int(round(crop_w)))
    crop_h = max(2, int(round(crop_h)))

    if crop_w > frame_w:
        crop_w = frame_w
    if crop_h > frame_h:
        crop_h = frame_h

    x = int(round(center_x - crop_w / 2))
    y = int(round(center_y - crop_h / 2))

    x = _clamp(x, 0, max(0, frame_w - crop_w))
    y = _clamp(y, 0, max(0, frame_h - crop_h))

    return x, y, crop_w, crop_h


def _build_camera_crop(face_box, source_w, source_h, top_aspect):
    """Строит crop для верхнего блока (facecam)."""
    x, y, w, h = face_box
    face_cx = x + w / 2
    face_cy = y + h / 2
    face_cy_adj = face_cy - (0.08 * h)
    preferred_h = _clamp(h * 2.6, source_h * 0.20, source_h * 0.55)
    return _fit_aspect_crop(face_cx, face_cy_adj, source_w, source_h, top_aspect, preferred_h=preferred_h)


def _camera_crop_sanity_ok(face_box, camera_rect):
    """Проверяет, что лицо занимает адекватную долю в верхнем блоке."""
    if not face_box or not camera_rect:
        return False
    _, _, _, face_h = face_box
    _, _, _, crop_h = camera_rect
    if crop_h <= 0:
        return False
    face_ratio = face_h / float(crop_h)
    return 0.10 <= face_ratio <= 0.65


def _build_content_crop(face_box, source_w, source_h, bottom_aspect):
    """Строит crop для нижнего блока (игровой контент), смещая от facecam."""
    face_x, face_y, face_w, face_h = face_box
    face_cx = face_x + face_w / 2
    face_cy = face_y + face_h / 2

    if source_w / source_h >= bottom_aspect:
        base_h = float(source_h)
        base_w = base_h * bottom_aspect
    else:
        base_w = float(source_w)
        base_h = base_w / bottom_aspect

    center_x = source_w / 2
    center_y = source_h / 2
    # center_x += (base_w * 0.15) if face_cx < source_w / 2 else -(base_w * 0.15)
    center_y += (base_h * 0.08) if face_cy < source_h / 2 else -(base_h * 0.08)

    return _fit_aspect_crop(center_x, center_y, source_w, source_h, bottom_aspect, preferred_h=base_h)


def _build_split_filter(camera_rect, content_rect, target_width, top_height, bottom_height):
    """Собирает filter_complex для верхнего facecam + нижнего контента."""
    cam_x, cam_y, cam_w, cam_h = camera_rect
    cont_x, cont_y, cont_w, cont_h = content_rect
    return (
        f"[0:v]crop={cam_w}:{cam_h}:{cam_x}:{cam_y},"
        f"scale={target_width}:{top_height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={target_width}:{top_height}[top];"
        f"[0:v]crop={cont_w}:{cont_h}:{cont_x}:{cont_y},"
        f"scale={target_width}:{bottom_height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={target_width}:{bottom_height}[bottom];"
        f"[top][bottom]vstack=inputs=2,setsar=1,setdar={target_width}/{top_height + bottom_height}[v]"
    )


def _skin_ratio_bgr(frame):
    """Грубая оценка доли skin-tone пикселей на фрагменте."""
    if cv2 is None or frame is None or frame.size == 0:
        return 0.0
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    total = float(mask.size)
    if total <= 0:
        return 0.0
    return float(cv2.countNonZero(mask)) / total


def _edge_strength_at_boundary(gray_frame, x, y, w, h, thickness=3):
    """Измеряет силу границ (edges) на краях прямоугольника webcam-оверлея."""
    if cv2 is None or gray_frame is None:
        return 0.0
    fh, fw = gray_frame.shape[:2]
    if w < 6 or h < 6:
        return 0.0

    edges = cv2.Canny(gray_frame, 50, 150)
    t = max(1, thickness)
    boundary_pixels = []

    # Top edge
    y1, y2 = max(0, y - t), min(fh, y + t)
    x1, x2 = max(0, x), min(fw, x + w)
    if y2 > y1 and x2 > x1:
        boundary_pixels.append(edges[y1:y2, x1:x2])
    # Bottom edge
    by = y + h
    y1, y2 = max(0, by - t), min(fh, by + t)
    if y2 > y1 and x2 > x1:
        boundary_pixels.append(edges[y1:y2, x1:x2])
    # Left edge
    x1, x2 = max(0, x - t), min(fw, x + t)
    y1, y2 = max(0, y), min(fh, y + h)
    if y2 > y1 and x2 > x1:
        boundary_pixels.append(edges[y1:y2, x1:x2])
    # Right edge
    rx = x + w
    x1, x2 = max(0, rx - t), min(fw, rx + t)
    if y2 > y1 and x2 > x1:
        boundary_pixels.append(edges[y1:y2, x1:x2])

    if not boundary_pixels:
        return 0.0

    import numpy as np
    combined = np.concatenate([p.ravel() for p in boundary_pixels])
    if combined.size == 0:
        return 0.0
    return float(np.count_nonzero(combined)) / float(combined.size)


def _histogram_divergence(frame, x, y, w, h):
    """Разница гистограмм цвета между ROI и остальной частью кадра."""
    if cv2 is None or frame is None:
        return 0.0
    fh, fw = frame.shape[:2]
    roi = frame[max(0, y):min(fh, y + h), max(0, x):min(fw, x + w)]
    if roi is None or roi.size == 0:
        return 0.0

    import numpy as np
    # Маска для "вне ROI"
    mask_outside = np.ones((fh, fw), dtype=np.uint8) * 255
    mask_outside[max(0, y):min(fh, y + h), max(0, x):min(fw, x + w)] = 0

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist_roi = cv2.calcHist([hsv], [0, 1], None, [30, 32],
                            [0, 180, 0, 256])
    # Обнуляем и пересчитываем для outside
    hsv_roi = hsv[max(0, y):min(fh, y + h), max(0, x):min(fw, x + w)]
    hist_roi = cv2.calcHist([hsv_roi], [0, 1], None, [30, 32],
                            [0, 180, 0, 256])
    hist_outside = cv2.calcHist([hsv], [0, 1], mask_outside, [30, 32],
                                [0, 180, 0, 256])

    cv2.normalize(hist_roi, hist_roi)
    cv2.normalize(hist_outside, hist_outside)

    # Bhattacharyya distance: 0 = identical, 1 = completely different
    distance = cv2.compareHist(hist_roi, hist_outside, cv2.HISTCMP_BHATTACHARYYA)
    return float(distance)


def _temporal_activity_in_region(frames, x, y, w, h):
    """Средняя межкадровая дельта (активность) внутри указанного прямоугольника."""
    if cv2 is None or len(frames) < 2:
        return 0.0
    import numpy as np
    fh_max, fw_max = frames[0].shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(fw_max, x + w)
    y2 = min(fh_max, y + h)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    deltas = []
    prev_gray = cv2.cvtColor(frames[0][y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    for frame in frames[1:]:
        cur_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, cur_gray)
        deltas.append(float(np.mean(diff)) / 255.0)
        prev_gray = cur_gray

    return statistics.mean(deltas) if deltas else 0.0


def _boundary_consistency(frames, x, y, w, h, thickness=3):
    """
    Стабильность границы: насколько одинаковы edge-линии на границе
    прямоугольника по всем кадрам. Высокая стабильность = устойчивый оверлей.
    """
    if cv2 is None or len(frames) < 2:
        return 0.0
    import numpy as np
    edge_strengths = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        strength = _edge_strength_at_boundary(gray, x, y, w, h, thickness)
        edge_strengths.append(strength)

    if not edge_strengths:
        return 0.0
    mean_str = statistics.mean(edge_strengths)
    std_str = statistics.stdev(edge_strengths) if len(edge_strengths) > 1 else 0.0
    # Высокая средняя сила + низкое отклонение = устойчивая граница
    if mean_str <= 0:
        return 0.0
    consistency = mean_str * max(0.0, 1.0 - (std_str / max(mean_str, 0.001)))
    return _clamp(consistency, 0.0, 1.0)


def _detect_webcam_region(source_w, source_h, frames, detectors=None, subject_side="auto"):
    """
    Находит прямоугольный webcam-оверлей через контурный анализ:
    Canny edges → findContours → фильтр по размеру/прямоугольности/углу.
    Если передан detectors, проверяет наличие лица внутри кандидата (+score).
    Возвращает (x, y, w, h) webcam-оверлея или None.
    """
    if cv2 is None or not frames:
        return None
    import numpy as np

    normalized_side = _normalize_subject_side(subject_side)
    frame = frames[0]
    fh, fw = frame.shape[:2]

    frame_area = fw * fh
    min_area = int(frame_area * 0.003)   # от 0.3% кадра
    max_area = int(frame_area * 0.40)    # до 40% кадра

    # Несколько порогов Canny для надёжности
    all_rects = []
    for low_t, high_t in ((30, 100), (50, 150), (80, 200)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, low_t, high_t)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) < 4 or len(approx) > 6:
                continue

            bx, by, bw, bh = cv2.boundingRect(approx)
            rect_area = bw * bh
            if rect_area < min_area or bw < 30 or bh < 30:
                continue
            rectangularity = area / float(rect_area)
            if rectangularity < 0.60:
                continue
            aspect = bw / float(bh)
            if aspect < 0.3 or aspect > 5.0:
                continue

            # Должен касаться левого или правого края
            margin_x = fw * 0.02
            touches_left = bx <= margin_x
            touches_right = (bx + bw) >= fw - margin_x

            position = None
            if touches_left:
                position = "left_edge"
            elif touches_right:
                position = "right_edge"

            # Если не касается ни левого, ни правого края — пропускаем
            if position is None:
                continue

            all_rects.append({
                "rect": (bx, by, bw, bh),
                "position": position,
                "area": rect_area,
                "rectangularity": rectangularity,
            })

    if not all_rects:
        logging.debug("webcam_region: no rectangular contours found on left/right edges")
        return None

    # Убираем дубликаты (IoU > 0.5)
    unique = []
    for r in all_rects:
        is_dup = False
        for u in unique:
            iou = _box_iou(r["rect"], u["rect"])
            if iou > 0.5:
                is_dup = True
                if r["area"] > u["area"]:
                    unique.remove(u)
                    unique.append(r)
                break
        if not is_dup:
            unique.append(r)

    # Скоринг
    scored = []
    for cand in unique:
        cx, cy, cw, ch = cand["rect"]
        s_hist = _histogram_divergence(frames[0], cx, cy, cw, ch)
        
        # Активность (движение)
        raw_activity = _temporal_activity_in_region(frames, cx, cy, cw, ch)
        s_activity = _clamp(raw_activity / 0.05, 0.0, 1.0) if len(frames) >= 2 else 0.5
        if raw_activity < 0.005:
            s_activity = 0.0

        center_x = cx + cw / 2.0
        s_side = _side_score(center_x, source_w, normalized_side)
        s_rect = cand["rectangularity"]

        # Face check inside the candidate
        s_face = 0.0
        if detectors:
            # Crop to candidate
            crop = frame[max(0, cy):min(fh, cy + ch), max(0, cx):min(fw, cx + cw)]
            if crop.size > 0:
                face_res = _detect_largest_face(crop, detectors)
                if face_res:
                    # Found a face inside the webcam candidate!
                    # Check if face size is reasonable relative to webcam candidate
                    fx, fy, fw_f, fh_f = face_res
                    face_area_ratio = (fw_f * fh_f) / (cw * ch)
                    if face_area_ratio > 0.05: # Face should be at least 5% of webcam area
                        s_face = 1.0

        # Веса: Лицо 0.50 (супер-бонус), Активность 0.25, Гистограмма 0.15, Прям-ть 0.10
        score = (0.50 * s_face) + (0.25 * s_activity) + (0.15 * s_hist) + (0.10 * s_rect)

        # Если лица нет, но активность высокая — тоже шанс (backwards compat)
        if s_face == 0.0:
             # Fallback scoring w/o face
             score = (0.45 * s_activity) + (0.25 * s_hist) + (0.20 * s_rect) + (0.10 * s_side)
        
        scored.append({
            "score": score,
            "rect": cand["rect"],
            "position": cand["position"],
            "debug": {
                "s_face": s_face,
                "s_activity": round(s_activity, 4),
                "raw_act": round(raw_activity, 5),
                "s_hist": round(s_hist, 4),
            },
        })

    scored.sort(key=lambda item: item["score"], reverse=True)
    best = scored[0]

    # Если побеждает абсолютно статичный регион без лица — реджектим
    if best["debug"]["raw_act"] < 0.002 and best["debug"]["s_face"] == 0.0 and len(frames) >= 2:
         logging.debug("webcam_region: detected rect is static and no face, reject. debug=%s", best["debug"])
         return None

    if best["score"] < 0.12:
        logging.debug(
            "webcam_region: low score, best=%.4f rect=%s debug=%s",
            best["score"], best["rect"], best["debug"],
        )
        return None

    logging.info(
        "webcam_region: detected rect=%s pos=%s score=%.4f debug=%s top3=%s",
        best["rect"], best["position"], best["score"], best["debug"],
        [(s["rect"], round(s["score"], 3)) for s in scored[:3]],
    )
    return best["rect"]


def _heuristic_face_box_from_corners(source_w, source_h, frames, subject_side="auto"):
    """
    Резервная эвристика: находит webcam-оверлей через edge/activity/histogram,
    и возвращает pseudo-face-box в его центре.
    """
    webcam_rect = _detect_webcam_region(source_w, source_h, frames, subject_side)
    if webcam_rect is None:
        return None

    x, y, w, h = webcam_rect
    # Возвращаем pseudo-face-box внутри кандидата (центр webcam окна).
    fw = max(20, int(round(w * 0.4)))
    fh = max(20, int(round(h * 0.45)))
    fx = x + (w - fw) // 2
    fy = y + int(round((h - fh) * 0.35))
    return (fx, fy, fw, fh)


def _run_facecam_top_split_layout(
    input_clip,
    output_file,
    target_width,
    target_height,
    facecam_ratio=1 / 3,
    facecam_subject_side="left",
    subs_file=None,
    encode_preset="veryfast",
):
    """Режим: верх 1/3 facecam, низ 2/3 игровой контент."""
    source_w, source_h, duration = _probe_video_metadata(input_clip)
    normalized_side = _normalize_subject_side(facecam_subject_side)
    detect_started = time.perf_counter()

    # Шаг 1: пробуем найти webcam-оверлей по границам/активности/гистограмме
    #         (быстрее и надёжнее для стримов, чем детекция лиц)
    probe_seconds = min(2.0, duration) if duration > 0 else 2.0
    probe_indexed = _probe_frames_with_indices_from_start(
        input_clip,
        max_probe_seconds=probe_seconds,
        sample_every_n_frames=3,
        max_samples=24,
    )
    probe_frames = [frame for _, frame in probe_indexed]

    # Pre-load detectors for region validation
    detectors = _get_face_detectors()

    webcam_rect = _detect_webcam_region(
        source_w, source_h, probe_frames, detectors=detectors, subject_side=normalized_side,
    )
    face_box = None
    face_found_source = None
    fallback_reason = None
    detect_debug = {
        "subject_side": normalized_side,
        "probe_frames": len(probe_frames),
        "detector_counts": {},
        "best_score": None,
        "best_source": None,
        "best_frame_idx": None,
        "ranked_candidates": [],
        "fallback_reason": None,
    }

    if webcam_rect is not None:
        # Webcam найден — строим pseudo-face-box в его центре
        wx, wy, ww, wh = webcam_rect
        fw = max(20, int(round(ww * 0.4)))
        fh = max(20, int(round(wh * 0.45)))
        fx = wx + (ww - fw) // 2
        fy = wy + int(round((wh - fh) * 0.35))
        face_box = (fx, fy, fw, fh)
        face_found_source = "webcam_region"
        logging.info(
            "webcam_region detected rect=%s pseudo_face_box=%s",
            webcam_rect, face_box,
        )
    else:
        # Шаг 2: fallback на детекцию лиц (Haar cascades)
        detect_result = _detect_face_once(
            input_clip,
            max_probe_seconds=probe_seconds,
            sample_every_n_frames=3,
            max_side=640,
            subject_side=normalized_side,
            return_probe_frames=True,
            return_debug=True,
        )
        face_box, probe_frames, detect_debug = detect_result
        fallback_reason = detect_debug.get("fallback_reason")
        if face_box is not None:
            face_found_source = detect_debug.get("best_source") or "haar"
        else:
            # Шаг 3: heuristic fallback (тоже использует webcam_region внутри)
            face_box = _heuristic_face_box_from_corners(
                source_w,
                source_h,
                probe_frames,
                subject_side=normalized_side,
            )
            if face_box is not None:
                face_found_source = "heuristic"
                fallback_reason = None

    detect_once_ms = int((time.perf_counter() - detect_started) * 1000)

    top_height = max(2, min(target_height - 2, int(round(target_height * facecam_ratio))))
    bottom_height = target_height - top_height
    if bottom_height <= 1:
        raise Exception("Некорректные размеры верхнего/нижнего блока.")

    top_aspect = target_width / top_height
    bottom_aspect = target_width / bottom_height

    selected_face_box = None
    camera_rect = None

    if webcam_rect is not None:
        camera_rect = webcam_rect
        selected_face_box = face_box
        logging.info(
            "webcam_region: exact webcam rect as camera_rect=%s",
            camera_rect,
        )
    else:
        # Face-box путь (Haar / heuristic) — стандартный crop вокруг лица
        ranked_candidates = detect_debug.get("ranked_candidates") or []
        candidate_boxes = []
        if face_box:
            candidate_boxes.append(tuple(face_box))
        for item in ranked_candidates:
            ranked_box = item.get("face_box")
            if not ranked_box:
                continue
            ranked_box = tuple(ranked_box)
            if ranked_box not in candidate_boxes:
                candidate_boxes.append(ranked_box)

        for idx, candidate_box in enumerate(candidate_boxes[:2]):
            candidate_camera_rect = _build_camera_crop(candidate_box, source_w, source_h, top_aspect)
            if _camera_crop_sanity_ok(candidate_box, candidate_camera_rect):
                selected_face_box = candidate_box
                camera_rect = candidate_camera_rect
                if idx > 0 and face_found_source:
                    face_found_source = f"{face_found_source}_candidate_{idx + 1}"
                break

    if selected_face_box is None:
        fallback_reason = fallback_reason or "camera_crop_sanity_failed"


    logging.info(
        "face_detect_v2 subject_side=%s probe_frames=%s detector_counts=%s best_score=%s best_source=%s best_frame_idx=%s fallback_reason=%s detect_once_ms=%s",
        normalized_side,
        len(probe_frames),
        detect_debug.get("detector_counts") or {},
        detect_debug.get("best_score"),
        detect_debug.get("best_source"),
        detect_debug.get("best_frame_idx"),
        fallback_reason,
        detect_once_ms,
    )

    if selected_face_box is None:
        logging.info(
            "Facecam detect summary: detect_once_ms=%s face_found_source=fallback_standard ffmpeg_encode_ms=0",
            detect_once_ms,
        )
        raise Exception("Не удалось обнаружить лицо для facecam-режима.")

    content_rect = _build_content_crop(selected_face_box, source_w, source_h, bottom_aspect)
    filter_graph = _build_split_filter(camera_rect, content_rect, target_width, top_height, bottom_height)
    output_label = "[v]"
    # На Windows используем basename + cwd, чтобы не экранировать путь в фильтре
    subs_cwd = None
    if subs_file:
        subs_cwd = os.path.dirname(os.path.abspath(subs_file))
        subs_name = os.path.basename(subs_file)
        output_label = "[vsub]"
        filter_graph = f"{filter_graph};[v]subtitles={subs_name}{output_label}"

    cmd = [
        "ffmpeg",
        "-i",
        input_clip,
        "-filter_complex",
        filter_graph,
        "-map",
        output_label,
        "-map",
        "0:a?",
        *_encoding_params(encode_preset),
        "-c:a",
        "copy",
        output_file,
        "-y",
    ]
    encode_started = time.perf_counter()
    _run_ffmpeg(cmd, "Ошибка компоновки режима facecam_top_split", cwd=subs_cwd)
    ffmpeg_encode_ms = int((time.perf_counter() - encode_started) * 1000)
    logging.info(
        "Facecam detect summary: detect_once_ms=%s face_found_source=%s ffmpeg_encode_ms=%s",
        detect_once_ms,
        face_found_source,
        ffmpeg_encode_ms,
    )


def convert_to_vertical(
    input_clip,
    output_dir=None,
    target_width=1080,
    target_height=1920,
    layout_mode="standard",
    facecam_ratio=1 / 3,
    facecam_subject_side="left",
    subs_file=None,
    encode_preset="veryfast",
):
    """
    Приводит входной ролик к вертикальному формату с нужными размерами.
    Поддерживаемые layout_mode: standard, facecam_top_split.
    """
    vert_id = str(uuid.uuid4())
    if output_dir:
        output_file = os.path.join(output_dir, f"vertical_{vert_id}.mp4")
    else:
        output_file = f"vertical_{vert_id}.mp4"

    selected_layout = layout_mode if layout_mode in {"standard", "facecam_top_split"} else "standard"
    if selected_layout != layout_mode:
        logging.warning("Неизвестный layout_mode '%s', использую standard.", layout_mode)

    try:
        if selected_layout == "facecam_top_split":
            try:
                _run_facecam_top_split_layout(
                    input_clip,
                    output_file,
                    target_width,
                    target_height,
                    facecam_ratio=facecam_ratio,
                    facecam_subject_side=facecam_subject_side,
                    subs_file=subs_file,
                    encode_preset=encode_preset,
                )
            except Exception as exc:
                logging.warning("Facecam-режим недоступен (%s). Fallback на standard 9:16.", exc)
                _run_standard_layout(
                    input_clip,
                    output_file,
                    target_width,
                    target_height,
                    subs_file=subs_file,
                    encode_preset=encode_preset,
                    face_found_source="fallback_standard",
                )
        else:
            _run_standard_layout(
                input_clip,
                output_file,
                target_width,
                target_height,
                subs_file=subs_file,
                encode_preset=encode_preset,
                face_found_source="standard",
            )

        if os.path.exists(output_file):
            return output_file
        raise Exception("FFmpeg не создал вертикальный файл.")
    except Exception as exc:
        raise Exception(f"Ошибка вертикального ресайза видео: {exc}") from exc

def burn_subtitles(input_clip, subs_file, output_dir=None):
    """
    Прожигает субтитры (.srt) на видео с помощью FFmpeg.
    Возвращает путь к итоговому файлу.
    """
    sub_id = str(uuid.uuid4())
    if output_dir:
        output_file = os.path.join(output_dir, f"subs_{sub_id}.mp4")
    else:
        output_file = f"subs_{sub_id}.mp4"
    if not subs_file or not os.path.exists(subs_file):
        raise Exception(f"Файл субтитров не найден: {subs_file}")

    # На Windows используем basename + cwd, чтобы не экранировать путь в фильтре
    subs_cwd = os.path.dirname(os.path.abspath(subs_file))
    subs_name = os.path.basename(subs_file)
    subtitles_filter = f"subtitles={subs_name}"

    # Прожиг субтитров через фильтр subtitles
    cmd = [
        "ffmpeg",
        "-i", input_clip,
        "-vf", subtitles_filter,
        "-c:a", "copy",
        output_file,
        "-y"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=subs_cwd)
        if os.path.exists(output_file):
            return output_file
        else:
            raise Exception("FFmpeg не создал итоговый файл с субтитрами.")
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        stdout = (e.stdout or "").strip()
        details = stderr[:500] if stderr else stdout[:500]
        raise Exception(f"Ошибка при добавлении субтитров: {e}. {details}")
    except Exception as e:
        raise Exception(f"Ошибка при добавлении субтитров: {e}")

# Пример:
# vert_clip = convert_to_vertical("autoedit_in.mp4")
# final_video = burn_subtitles(vert_clip, "subs.srt")
