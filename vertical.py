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
_SUPPORTED_FACECAM_BACKENDS = {"yolo_window_v1", "legacy"}
_SUPPORTED_FACECAM_FALLBACK_MODES = {"hard_side", "standard"}
_SUPPORTED_FACECAM_ANCHORS = {"edge_middle"}
_FACECAM_MODEL_CACHE_DIR = os.path.join("cache", "models", "facecam")
_FACECAM_MODEL_FILE = "yolov8n.pt"

_YOLO_FACECAM_MODEL = None
_YOLO_FACECAM_LOAD_FAILED = False


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


def _normalize_facecam_backend(backend):
    normalized = (backend or "").strip().lower()
    if normalized in _SUPPORTED_FACECAM_BACKENDS:
        return normalized
    return "yolo_window_v1"


def _normalize_facecam_fallback_mode(fallback_mode):
    normalized = (fallback_mode or "").strip().lower()
    if normalized in _SUPPORTED_FACECAM_FALLBACK_MODES:
        return normalized
    return "hard_side"


def _normalize_facecam_anchor(anchor):
    normalized = (anchor or "").strip().lower()
    if normalized in _SUPPORTED_FACECAM_ANCHORS:
        return normalized
    return "edge_middle"


def _normalize_detector_list(detectors):
    if not isinstance(detectors, list):
        return []
    normalized = []
    for item in detectors:
        if isinstance(item, dict):
            detector = item.get("detector")
            if detector is not None and hasattr(detector, "detectMultiScale"):
                normalized.append(item)
            continue
        if item is not None and hasattr(item, "detectMultiScale"):
            normalized.append(item)
    return normalized


def _get_facecam_yolo_model(model_dir=None):
    global _YOLO_FACECAM_MODEL, _YOLO_FACECAM_LOAD_FAILED

    if _YOLO_FACECAM_MODEL is not None:
        return _YOLO_FACECAM_MODEL
    if _YOLO_FACECAM_LOAD_FAILED:
        return None

    try:
        from ultralytics import YOLO
    except Exception as exc:
        _YOLO_FACECAM_LOAD_FAILED = True
        logging.warning("ultralytics недоступен для facecam-detector (%s).", exc)
        return None

    cache_dir = os.path.abspath(model_dir or _FACECAM_MODEL_CACHE_DIR)
    model_path = os.path.join(cache_dir, _FACECAM_MODEL_FILE)
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception as exc:
        _YOLO_FACECAM_LOAD_FAILED = True
        logging.warning("Не удалось создать кеш-директорию для YOLO (%s).", exc)
        return None

    try:
        _YOLO_FACECAM_MODEL = YOLO(model_path)
        return _YOLO_FACECAM_MODEL
    except Exception as exc:
        _YOLO_FACECAM_LOAD_FAILED = True
        logging.warning("Не удалось инициализировать YOLO facecam-detector (%s).", exc)
        return None


def _detect_person_boxes_with_yolo(frames, max_side=960):
    """
    Возвращает {probe_idx: [{"box": (x, y, w, h), "confidence": float}, ...]}.
    Используется как мягкий сигнал для webcam-окна, а не как единственный источник.
    """
    if not frames:
        return {}

    yolo_model = _get_facecam_yolo_model()
    if yolo_model is None:
        return {}

    detected = {}
    try:
        for probe_idx, frame in enumerate(frames):
            if frame is None:
                continue
            resized, scale = _resize_for_detection(frame, max_side=max_side)
            result_items = yolo_model.predict(
                source=resized,
                classes=[0],  # person
                conf=0.25,
                verbose=False,
                imgsz=max(320, int(max_side)),
            )
            frame_boxes = []
            for item in result_items:
                boxes = getattr(item, "boxes", None)
                if boxes is None:
                    continue
                xyxy = getattr(boxes, "xyxy", None)
                confs = getattr(boxes, "conf", None)
                if xyxy is None:
                    continue

                for idx in range(len(xyxy)):
                    raw = xyxy[idx]
                    coords = raw.tolist() if hasattr(raw, "tolist") else raw
                    if not coords or len(coords) < 4:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in coords[:4]]
                    if scale > 0 and scale != 1.0:
                        inv = 1.0 / scale
                        x1 *= inv
                        y1 *= inv
                        x2 *= inv
                        y2 *= inv
                    x = int(round(x1))
                    y = int(round(y1))
                    w = int(round(x2 - x1))
                    h = int(round(y2 - y1))
                    box = _normalize_face_box(x, y, w, h, frame.shape[1], frame.shape[0])
                    score = 0.0
                    if confs is not None and idx < len(confs):
                        score_raw = confs[idx]
                        try:
                            score = float(score_raw.item())
                        except Exception:
                            score = float(score_raw)
                    frame_boxes.append({"box": box, "confidence": score})
            detected[probe_idx] = frame_boxes
    except Exception as exc:
        logging.warning("YOLO facecam detect failed, continue without DNN (%s).", exc)
        return {}
    return detected


def _intersection_area(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    return inter_w * inter_h


def _person_overlap_ratio(candidate_box, person_boxes):
    if not person_boxes:
        return 0.0
    cand_area = float(max(1, candidate_box[2] * candidate_box[3]))
    best = 0.0
    for person in person_boxes:
        person_box = person.get("box") if isinstance(person, dict) else person
        if not person_box:
            continue
        inter = _intersection_area(candidate_box, person_box)
        if inter <= 0:
            continue
        overlap = inter / cand_area
        if overlap > best:
            best = overlap
    return _clamp(best, 0.0, 1.0)


def _pseudo_face_box_from_camera_rect(camera_rect):
    x, y, w, h = camera_rect
    fw = max(20, int(round(w * 0.40)))
    fh = max(20, int(round(h * 0.45)))
    fx = x + (w - fw) // 2
    fy = y + int(round((h - fh) * 0.35))
    return (fx, fy, fw, fh)


def _camera_rect_sanity_ok(camera_rect, source_w, source_h):
    if not camera_rect:
        return False
    x, y, w, h = camera_rect
    if min(w, h) < 30:
        return False
    if x < 0 or y < 0:
        return False
    if (x + w) > source_w or (y + h) > source_h:
        return False

    area_ratio = (w * h) / float(max(1, source_w * source_h))
    if area_ratio < 0.01 or area_ratio > 0.65:
        return False

    ratio = w / float(max(h, 1))
    if ratio < 0.25 or ratio > 5.0:
        return False
    return True


def _build_hard_side_camera_rect(source_w, source_h, top_aspect, subject_side="left", anchor="edge_middle"):
    normalized_side = _normalize_subject_side(subject_side)
    normalized_anchor = _normalize_facecam_anchor(anchor)
    side = normalized_side if normalized_side in {"left", "right"} else "left"

    crop_w_float = _clamp(source_w * 0.34, source_w * 0.22, source_w * 0.50)
    crop_w = int(round(crop_w_float))
    crop_h = int(round(crop_w / float(max(top_aspect, 0.001))))

    if crop_h > source_h:
        crop_h = int(round(source_h * 0.90))
        crop_w = int(round(crop_h * float(max(top_aspect, 0.001))))

    crop_w = max(2, min(source_w, crop_w))
    crop_h = max(2, min(source_h, crop_h))

    x = 0 if side == "left" else max(0, source_w - crop_w)
    if normalized_anchor == "edge_middle":
        y = max(0, min(source_h - crop_h, int(round((source_h - crop_h) / 2.0))))
    else:
        y = 0
    return (x, y, crop_w, crop_h)


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


def _extract_rect_candidates_from_frame(frame, probe_idx):
    if cv2 is None or frame is None:
        return []

    fh, fw = frame.shape[:2]
    frame_area = float(max(1, fw * fh))
    min_area = frame_area * 0.0025
    max_area = frame_area * 0.45
    candidates = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    for low_t, high_t in ((25, 80), (40, 120), (60, 180)):
        edges = cv2.Canny(blurred, low_t, high_t)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area or area > max_area:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri <= 0:
                continue
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            if len(approx) < 4 or len(approx) > 8:
                continue

            bx, by, bw, bh = cv2.boundingRect(approx)
            if bw < 40 or bh < 40:
                continue
            rect_area = float(bw * bh)
            if rect_area <= 0:
                continue
            rectangularity = area / rect_area
            if rectangularity < 0.55:
                continue
            aspect = bw / float(max(bh, 1))
            if aspect < 0.28 or aspect > 5.0:
                continue

            center_x = bx + (bw / 2.0)
            center_x_norm = _clamp(center_x / float(max(fw, 1)), 0.0, 1.0)
            if center_x_norm <= 0.35:
                side = "left"
            elif center_x_norm >= 0.65:
                side = "right"
            else:
                continue

            center_y = by + (bh / 2.0)
            center_y_norm = _clamp(center_y / float(max(fh, 1)), 0.0, 1.0)
            y_middle_score = max(0.0, 1.0 - (abs(center_y_norm - 0.5) / 0.5))
            candidates.append(
                {
                    "probe_idx": probe_idx,
                    "rect": (int(bx), int(by), int(bw), int(bh)),
                    "rectangularity": float(rectangularity),
                    "side": side,
                    "y_middle": float(y_middle_score),
                }
            )
    return candidates


def _deduplicate_rect_candidates(candidates, iou_threshold=0.55):
    if not candidates:
        return []
    scored = sorted(
        candidates,
        key=lambda item: (item.get("rectangularity", 0.0), item["rect"][2] * item["rect"][3]),
        reverse=True,
    )
    deduped = []
    for item in scored:
        if any(_box_iou(item["rect"], existing["rect"]) >= iou_threshold for existing in deduped):
            continue
        deduped.append(item)
    return deduped


def _build_webcam_tracks(candidates):
    if not candidates:
        return []

    tracks = []
    ordered = sorted(candidates, key=lambda item: (item["probe_idx"], -(item["rect"][2] * item["rect"][3])))
    for candidate in ordered:
        best_track = None
        best_iou = 0.0
        for track in tracks:
            if track["side"] != candidate["side"]:
                continue
            overlap = _box_iou(candidate["rect"], track["last_rect"])
            if overlap >= 0.35 and overlap > best_iou:
                best_iou = overlap
                best_track = track
        if best_track is None:
            tracks.append(
                {
                    "side": candidate["side"],
                    "members": [candidate],
                    "last_rect": candidate["rect"],
                }
            )
            continue
        best_track["members"].append(candidate)
        best_track["last_rect"] = _median_box([member["rect"] for member in best_track["members"]])

    prepared = []
    for track in tracks:
        boxes = [member["rect"] for member in track["members"]]
        aggregate = _median_box(boxes)
        if not aggregate:
            continue
        prepared.append(
            {
                "side": track["side"],
                "members": track["members"],
                "rect": aggregate,
                "frame_hits": sorted(set(member["probe_idx"] for member in track["members"])),
                "mean_rectangularity": statistics.mean(
                    member.get("rectangularity", 0.0) for member in track["members"]
                ),
                "mean_middle": statistics.mean(member.get("y_middle", 0.5) for member in track["members"]),
            }
        )
    return prepared


def _score_webcam_track(track, frames, source_w, source_h, subject_side, person_boxes_by_frame, detectors=None):
    rect = track["rect"]
    x, y, w, h = rect
    total_frames = max(1, len(frames))
    temporal_hits = len(track["frame_hits"]) / float(total_frames)
    hist_score = _histogram_divergence(frames[0], x, y, w, h)

    raw_activity = _temporal_activity_in_region(frames, x, y, w, h)
    activity_score = _clamp(raw_activity / 0.045, 0.0, 1.0)
    if raw_activity < 0.004:
        activity_score = 0.0

    boundary_score = _boundary_consistency(frames, x, y, w, h)
    center_x = x + (w / 2.0)
    side_score = _side_score(center_x, source_w, subject_side)
    rect_score = _clamp(track["mean_rectangularity"], 0.0, 1.0)
    middle_score = _clamp(track.get("mean_middle", 0.5), 0.0, 1.0)

    person_hits = []
    for probe_idx in track["frame_hits"]:
        person_hits.append(_person_overlap_ratio(rect, person_boxes_by_frame.get(probe_idx, [])))
    person_score = statistics.mean(person_hits) if person_hits else 0.0

    safe_detectors = _normalize_detector_list(detectors)
    face_score = 0.0
    if safe_detectors:
        fh, fw = frames[0].shape[:2]
        crop = frames[0][max(0, y):min(fh, y + h), max(0, x):min(fw, x + w)]
        if crop is not None and crop.size > 0:
            face = _detect_largest_face(crop, safe_detectors)
            if face:
                fx, fy, fwf, fhf = face
                face_ratio = (fwf * fhf) / float(max(1, w * h))
                if face_ratio >= 0.02:
                    face_score = _clamp(face_ratio / 0.20, 0.0, 1.0)

    score = (
        (0.23 * temporal_hits)
        + (0.16 * boundary_score)
        + (0.15 * activity_score)
        + (0.10 * hist_score)
        + (0.09 * rect_score)
        + (0.08 * side_score)
        + (0.06 * middle_score)
        + (0.08 * person_score)
        + (0.05 * face_score)
    )
    return {
        "score": float(score),
        "rect": rect,
        "side": track["side"],
        "temporal_hits": temporal_hits,
        "debug": {
            "s_temporal": round(temporal_hits, 4),
            "s_boundary": round(boundary_score, 4),
            "s_activity": round(activity_score, 4),
            "raw_activity": round(raw_activity, 5),
            "s_hist": round(hist_score, 4),
            "s_rect": round(rect_score, 4),
            "s_side": round(side_score, 4),
            "s_middle": round(middle_score, 4),
            "s_person": round(person_score, 4),
            "s_face": round(face_score, 4),
        },
    }


def _detect_webcam_region(
    source_w,
    source_h,
    frames,
    detectors=None,
    subject_side="auto",
    detector_backend="yolo_window_v1",
    anchor="edge_middle",
    min_score=0.30,
    return_debug=False,
):
    """
    Находит webcam-окно по мультикадровому трекингу прямоугольников.
    Использует опциональный YOLO-сигнал (person внутри региона) и face-check.
    """
    debug = {
        "detector_backend": _normalize_facecam_backend(detector_backend),
        "anchor": _normalize_facecam_anchor(anchor),
        "candidate_count": 0,
        "track_count": 0,
        "best_score": None,
        "best_rect": None,
        "preferred_side": "left",
        "fallback_reason": None,
        "top3": [],
    }
    if cv2 is None or not frames:
        debug["fallback_reason"] = "no_frames_or_cv2"
        if return_debug:
            return None, debug
        return None

    normalized_side = _normalize_subject_side(subject_side)
    normalized_backend = _normalize_facecam_backend(detector_backend)
    safe_detectors = _normalize_detector_list(detectors)

    all_candidates = []
    for probe_idx, frame in enumerate(frames):
        frame_candidates = _extract_rect_candidates_from_frame(frame, probe_idx)
        frame_candidates = _deduplicate_rect_candidates(frame_candidates)
        all_candidates.extend(frame_candidates)

    debug["candidate_count"] = len(all_candidates)
    if not all_candidates:
        debug["fallback_reason"] = "no_rect_candidates"
        if return_debug:
            return None, debug
        return None

    tracks = _build_webcam_tracks(all_candidates)
    debug["track_count"] = len(tracks)
    if not tracks:
        debug["fallback_reason"] = "no_tracks"
        if return_debug:
            return None, debug
        return None

    person_boxes_by_frame = {}
    if normalized_backend == "yolo_window_v1":
        person_boxes_by_frame = _detect_person_boxes_with_yolo(frames)

    scored = []
    side_votes = {"left": 0.0, "right": 0.0}
    for track in tracks:
        ranked = _score_webcam_track(
            track,
            frames,
            source_w=source_w,
            source_h=source_h,
            subject_side=normalized_side,
            person_boxes_by_frame=person_boxes_by_frame,
            detectors=safe_detectors,
        )
        if normalized_side in {"left", "right"} and ranked["side"] == normalized_side:
            ranked["score"] += 0.02
        side_votes[ranked["side"]] += max(0.0, ranked["score"])
        scored.append(ranked)

    debug["preferred_side"] = "left" if side_votes["left"] >= side_votes["right"] else "right"
    scored.sort(key=lambda item: (item["score"], item["temporal_hits"]), reverse=True)
    best = scored[0]
    debug["best_score"] = round(float(best["score"]), 4)
    debug["best_rect"] = best["rect"]
    debug["top3"] = [
        {
            "rect": item["rect"],
            "side": item["side"],
            "score": round(float(item["score"]), 4),
            "temporal_hits": round(float(item["temporal_hits"]), 4),
            "debug": item["debug"],
        }
        for item in scored[:3]
    ]
    debug["preferred_side"] = best["side"] if normalized_side == "auto" else normalized_side

    if float(best["score"]) < float(min_score):
        debug["fallback_reason"] = "low_score"
        if return_debug:
            return None, debug
        return None
    if not _camera_rect_sanity_ok(best["rect"], source_w, source_h):
        debug["fallback_reason"] = "camera_rect_sanity_failed"
        if return_debug:
            return None, debug
        return None

    logging.info(
        "webcam_region_v3 backend=%s candidates=%s tracks=%s best_rect=%s best_score=%.4f top3=%s",
        normalized_backend,
        debug["candidate_count"],
        debug["track_count"],
        best["rect"],
        best["score"],
        [(item["rect"], round(item["score"], 3)) for item in scored[:3]],
    )
    if return_debug:
        return best["rect"], debug
    return best["rect"]


def _heuristic_face_box_from_corners(source_w, source_h, frames, subject_side="auto"):
    """
    Резервная эвристика: находит webcam-оверлей через edge/activity/histogram,
    и возвращает pseudo-face-box в его центре.
    """
    webcam_rect = _detect_webcam_region(
        source_w,
        source_h,
        frames,
        detectors=None,
        subject_side=subject_side,
    )
    if webcam_rect is None:
        return None

    return _pseudo_face_box_from_camera_rect(webcam_rect)


def _save_facecam_debug_frames(probe_indexed, camera_rect, output_file, max_frames=5):
    """
    Сохраняет несколько кадров с рамкой camera_rect, если включен env FACECAM_SAVE_DEBUG_FRAMES=1.
    """
    if cv2 is None:
        return []
    if not probe_indexed or not camera_rect:
        return []
    if os.getenv("FACECAM_SAVE_DEBUG_FRAMES", "0").strip() != "1":
        return []

    x, y, w, h = camera_rect
    out_dir = os.path.join(os.path.dirname(os.path.abspath(output_file)), "facecam_debug")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        return []

    saved = []
    for frame_idx, (source_frame_idx, frame) in enumerate(probe_indexed[: max(1, int(max_frames))]):
        if frame is None:
            continue
        canvas = frame.copy()
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            f"camera_rect={x},{y},{w},{h}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        filename = os.path.join(out_dir, f"{os.path.basename(output_file)}_probe_{frame_idx}_{source_frame_idx}.jpg")
        try:
            if cv2.imwrite(filename, canvas):
                saved.append(filename)
        except Exception:
            continue
    return saved


def _run_facecam_top_split_layout(
    input_clip,
    output_file,
    target_width,
    target_height,
    facecam_ratio=1 / 3,
    facecam_subject_side="left",
    facecam_detector_backend="yolo_window_v1",
    facecam_fallback_mode="hard_side",
    facecam_anchor="edge_middle",
    subs_file=None,
    encode_preset="veryfast",
):
    """Режим: верх 1/3 facecam, низ 2/3 игровой контент."""
    source_w, source_h, duration = _probe_video_metadata(input_clip)
    normalized_side = _normalize_subject_side(facecam_subject_side)
    normalized_backend = _normalize_facecam_backend(facecam_detector_backend)
    normalized_fallback_mode = _normalize_facecam_fallback_mode(facecam_fallback_mode)
    normalized_anchor = _normalize_facecam_anchor(facecam_anchor)
    detect_started = time.perf_counter()

    probe_seconds = min(6.0, duration) if duration > 0 else 6.0
    probe_indexed = _probe_frames_with_indices_from_start(
        input_clip,
        max_probe_seconds=probe_seconds,
        sample_every_n_frames=3,
        max_samples=36,
    )
    probe_frames = [frame for _, frame in probe_indexed]

    detectors = _get_face_detectors()
    top_height = max(2, min(target_height - 2, int(round(target_height * facecam_ratio))))
    bottom_height = target_height - top_height
    if bottom_height <= 1:
        raise Exception("Некорректные размеры верхнего/нижнего блока.")
    top_aspect = target_width / top_height
    bottom_aspect = target_width / bottom_height

    webcam_rect, webcam_debug = _detect_webcam_region(
        source_w,
        source_h,
        probe_frames,
        detectors=detectors,
        subject_side=normalized_side,
        detector_backend=normalized_backend,
        anchor=normalized_anchor,
        min_score=0.30,
        return_debug=True,
    )

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
            camera_rect = _build_hard_side_camera_rect(
                source_w,
                source_h,
                top_aspect,
                subject_side=resolved_side,
                anchor=normalized_anchor,
            )
            if not _camera_rect_sanity_ok(camera_rect, source_w, source_h):
                raise Exception("Hard-side fallback построил некорректный camera_rect.")
            selected_face_box = _pseudo_face_box_from_camera_rect(camera_rect)
            face_found_source = f"hard_side_{resolved_side}"
        else:
            # Совместимый путь: fallback на face detection.
            detect_result = _detect_face_once(
                input_clip,
                max_probe_seconds=min(2.0, probe_seconds),
                sample_every_n_frames=3,
                max_side=640,
                subject_side=normalized_side,
                return_probe_frames=True,
                return_debug=True,
            )
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
        raise Exception("Не удалось получить валидный camera_rect для facecam-режима.")
    if not _camera_rect_sanity_ok(camera_rect, source_w, source_h):
        raise Exception("Итоговый camera_rect не прошел sanity-check.")

    _save_facecam_debug_frames(probe_indexed, camera_rect, output_file, max_frames=5)
    logging.info(
        "facecam_detect_v3 backend=%s fallback_mode=%s anchor=%s subject_side=%s preferred_side=%s probe_frames=%s candidate_count=%s track_count=%s best_score=%s fallback_reason=%s final_source=%s final_rect=%s detect_once_ms=%s",
        normalized_backend,
        normalized_fallback_mode,
        normalized_anchor,
        normalized_side,
        resolved_side,
        len(probe_frames),
        webcam_debug.get("candidate_count"),
        webcam_debug.get("track_count"),
        webcam_debug.get("best_score"),
        fallback_reason,
        face_found_source,
        camera_rect,
        detect_once_ms,
    )

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
    facecam_detector_backend="yolo_window_v1",
    facecam_fallback_mode="hard_side",
    facecam_anchor="edge_middle",
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
                    facecam_detector_backend=facecam_detector_backend,
                    facecam_fallback_mode=facecam_fallback_mode,
                    facecam_anchor=facecam_anchor,
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
