"""vertical.detection — Haar/YOLO детекция лиц, face scoring, frame probing."""

import glob
import logging
import os
from collections import Counter, defaultdict

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from vertical.geometry import (
    _clamp,
    _box_iou,
    _median_box,
    _normalize_face_box,
    _is_candidate_face_box_valid,
    _side_score,
    _select_evenly_spaced,
)

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
    dirs.update({
        os.path.join(cv2_root, "data"),
        "/opt/homebrew/share/opencv4/haarcascades",
        "/usr/local/share/opencv4/haarcascades",
        "/usr/share/opencv4/haarcascades",
        "/usr/share/OpenCV/haarcascades",
    })
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
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(min_side, min_side))
        for x, y, w, h in faces:
            area = int(w) * int(h)
            if area > best_area:
                best_area = area
                best = (int(x), int(y), int(w), int(h))
    return best


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
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(min_side, min_side))
        for x, y, w, h in faces:
            detections.append({"box": (int(x), int(y), int(w), int(h)), "source": detector_name, "mirrored": False})
        if "profileface" in detector_name and gray_width > 0:
            if flipped is None:
                flipped = cv2.flip(gray, 1)
            mirrored_faces = detector.detectMultiScale(flipped, scaleFactor=1.1, minNeighbors=4, minSize=(min_side, min_side))
            for x, y, w, h in mirrored_faces:
                mirrored_x = gray_width - (int(x) + int(w))
                detections.append({"box": (int(mirrored_x), int(y), int(w), int(h)), "source": detector_name, "mirrored": True})
    return detections


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
    """Возвращает {probe_idx: [{"box": ..., "confidence": float}, ...]}."""
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
            result_items = yolo_model.predict(source=resized, classes=[0], conf=0.25, verbose=False, imgsz=max(320, int(max_side)))
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
                        x1 *= inv; y1 *= inv; x2 *= inv; y2 *= inv
                    x, y = int(round(x1)), int(round(y1))
                    w, h = int(round(x2 - x1)), int(round(y2 - y1))
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
        clusters[best_cluster_idx]["anchor_box"] = _median_box([item["box"] for item in clusters[best_cluster_idx]["members"]])
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
            if any(_box_iou(cluster_box, box) >= 0.35 for box in frame_candidates.get(probe_idx, [])):
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
        ranked.append({
            "face_box": cluster_box, "score": float(score), "source": best_source,
            "frame_idx": representative.get("frame_idx", 0),
            "breakdown": {"s_temporal": round(s_temporal, 4), "s_area": round(s_area, 4),
                          "s_side": round(s_side, 4), "s_center_y": round(s_center_y, 4),
                          "detector_bonus": round(bonus, 4)},
        })
    ranked.sort(key=lambda item: (item["score"], item["breakdown"]["s_temporal"]), reverse=True)
    return {"best": ranked[0] if ranked else None, "ranked": ranked}


def _probe_frames_with_indices_from_start(input_clip, max_probe_seconds=5.0, sample_every_n_frames=6, max_samples=None):
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


def _probe_frames_from_start(input_clip, max_probe_seconds=5.0, sample_every_n_frames=6, max_samples=None):
    """Читает только начало клипа и возвращает выборку кадров."""
    indexed = _probe_frames_with_indices_from_start(input_clip, max_probe_seconds=max_probe_seconds, sample_every_n_frames=sample_every_n_frames, max_samples=max_samples)
    return [frame for _, frame in indexed]


def _detect_face_once(input_clip, max_probe_seconds=2.0, sample_every_n_frames=3, max_side=640, subject_side="left", min_best_score=0.55, max_probe_samples=24, return_probe_frames=False, return_debug=False):
    """Ищет устойчивый face-box в начале клипа по multi-frame scoring."""
    normalized_side = _normalize_subject_side(subject_side)
    detectors = _get_face_detectors()
    probe_indexed = _probe_frames_with_indices_from_start(input_clip, max_probe_seconds=max_probe_seconds, sample_every_n_frames=sample_every_n_frames, max_samples=max_probe_samples)
    probe_frames = [frame for _, frame in probe_indexed]
    detect_debug = {"subject_side": normalized_side, "probe_frames": len(probe_frames), "detector_counts": {}, "best_score": None, "best_source": None, "best_frame_idx": None, "ranked_candidates": [], "fallback_reason": None}

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
                x = int(round(x * inv)); y = int(round(y * inv))
                fw = int(round(fw * inv)); fh = int(round(fh * inv))
            face_box = _normalize_face_box(x, y, fw, fh, w, h)
            if not _is_candidate_face_box_valid(face_box, w, h):
                continue
            all_candidates.append({"probe_idx": probe_idx, "frame_idx": frame_idx, "box": face_box, "source": source})
    detect_debug["detector_counts"] = dict(detector_counts)
    if not all_candidates:
        detect_debug["fallback_reason"] = "no_valid_candidates"
        return _pack(None)
    selected = _select_best_face_candidate(all_candidates, total_probe_frames=len(probe_indexed), frame_w=frame_w, frame_h=frame_h, subject_side=normalized_side)
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
        top3 = [{"score": round(float(item.get("score", 0.0)), 4), "source": item.get("source"), "frame_idx": item.get("frame_idx"), "breakdown": item.get("breakdown")} for item in ranked[:3]]
        logging.debug("face_detect_v2 low_confidence top_candidates=%s", top3)
        return _pack(None)
    return _pack(best.get("face_box"))
