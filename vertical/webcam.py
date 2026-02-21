"""vertical.webcam — webcam overlay detection via multi-frame tracking."""

import logging
import os
import statistics

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from vertical.geometry import (
    _clamp,
    _box_iou,
    _median_box,
    _side_score,
    _person_overlap_ratio,
    _pseudo_face_box_from_camera_rect,
    _camera_rect_sanity_ok,
)
from vertical.detection import (
    _normalize_subject_side,
    _normalize_facecam_backend,
    _normalize_facecam_anchor,
    _normalize_detector_list,
    _detect_largest_face,
    _detect_person_boxes_with_yolo,
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
    """Измеряет силу границ на краях прямоугольника webcam-оверлея."""
    if cv2 is None or gray_frame is None:
        return 0.0
    fh, fw = gray_frame.shape[:2]
    if w < 6 or h < 6:
        return 0.0
    edges = cv2.Canny(gray_frame, 50, 150)
    t = max(1, thickness)
    boundary_pixels = []
    y1, y2 = max(0, y - t), min(fh, y + t)
    x1, x2 = max(0, x), min(fw, x + w)
    if y2 > y1 and x2 > x1:
        boundary_pixels.append(edges[y1:y2, x1:x2])
    by = y + h
    y1, y2 = max(0, by - t), min(fh, by + t)
    if y2 > y1 and x2 > x1:
        boundary_pixels.append(edges[y1:y2, x1:x2])
    x1, x2 = max(0, x - t), min(fw, x + t)
    y1, y2 = max(0, y), min(fh, y + h)
    if y2 > y1 and x2 > x1:
        boundary_pixels.append(edges[y1:y2, x1:x2])
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
    mask_outside = np.ones((fh, fw), dtype=np.uint8) * 255
    mask_outside[max(0, y):min(fh, y + h), max(0, x):min(fw, x + w)] = 0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_roi = hsv[max(0, y):min(fh, y + h), max(0, x):min(fw, x + w)]
    hist_roi = cv2.calcHist([hsv_roi], [0, 1], None, [30, 32], [0, 180, 0, 256])
    hist_outside = cv2.calcHist([hsv], [0, 1], mask_outside, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist_roi, hist_roi)
    cv2.normalize(hist_outside, hist_outside)
    distance = cv2.compareHist(hist_roi, hist_outside, cv2.HISTCMP_BHATTACHARYYA)
    return float(distance)


def _temporal_activity_in_region(frames, x, y, w, h):
    """Средняя межкадровая дельта внутри указанного прямоугольника."""
    if cv2 is None or len(frames) < 2:
        return 0.0
    import numpy as np
    fh_max, fw_max = frames[0].shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw_max, x + w), min(fh_max, y + h)
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
    """Стабильность границы webcam-оверлея по всем кадрам."""
    if cv2 is None or len(frames) < 2:
        return 0.0
    edge_strengths = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        strength = _edge_strength_at_boundary(gray, x, y, w, h, thickness)
        edge_strengths.append(strength)
    if not edge_strengths:
        return 0.0
    mean_str = statistics.mean(edge_strengths)
    std_str = statistics.stdev(edge_strengths) if len(edge_strengths) > 1 else 0.0
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
            candidates.append({"probe_idx": probe_idx, "rect": (int(bx), int(by), int(bw), int(bh)), "rectangularity": float(rectangularity), "side": side, "y_middle": float(y_middle_score)})
    return candidates


def _deduplicate_rect_candidates(candidates, iou_threshold=0.55):
    if not candidates:
        return []
    scored = sorted(candidates, key=lambda item: (item.get("rectangularity", 0.0), item["rect"][2] * item["rect"][3]), reverse=True)
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
            tracks.append({"side": candidate["side"], "members": [candidate], "last_rect": candidate["rect"]})
            continue
        best_track["members"].append(candidate)
        best_track["last_rect"] = _median_box([m["rect"] for m in best_track["members"]])
    prepared = []
    for track in tracks:
        boxes = [m["rect"] for m in track["members"]]
        aggregate = _median_box(boxes)
        if not aggregate:
            continue
        prepared.append({
            "side": track["side"], "members": track["members"], "rect": aggregate,
            "frame_hits": sorted(set(m["probe_idx"] for m in track["members"])),
            "mean_rectangularity": statistics.mean(m.get("rectangularity", 0.0) for m in track["members"]),
            "mean_middle": statistics.mean(m.get("y_middle", 0.5) for m in track["members"]),
        })
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
    side_score_val = _side_score(center_x, source_w, subject_side)
    rect_score = _clamp(track["mean_rectangularity"], 0.0, 1.0)
    middle_score = _clamp(track.get("mean_middle", 0.5), 0.0, 1.0)
    person_hits = [_person_overlap_ratio(rect, person_boxes_by_frame.get(pi, [])) for pi in track["frame_hits"]]
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
    score = (0.23 * temporal_hits) + (0.16 * boundary_score) + (0.15 * activity_score) + (0.10 * hist_score) + (0.09 * rect_score) + (0.08 * side_score_val) + (0.06 * middle_score) + (0.08 * person_score) + (0.05 * face_score)
    return {"score": float(score), "rect": rect, "side": track["side"], "temporal_hits": temporal_hits,
            "debug": {"s_temporal": round(temporal_hits, 4), "s_boundary": round(boundary_score, 4), "s_activity": round(activity_score, 4), "raw_activity": round(raw_activity, 5), "s_hist": round(hist_score, 4), "s_rect": round(rect_score, 4), "s_side": round(side_score_val, 4), "s_middle": round(middle_score, 4), "s_person": round(person_score, 4), "s_face": round(face_score, 4)}}


def _detect_webcam_region(source_w, source_h, frames, detectors=None, subject_side="auto", detector_backend="yolo_window_v1", anchor="edge_middle", min_score=0.30, return_debug=False):
    """Находит webcam-окно по мультикадровому трекингу прямоугольников."""
    debug = {"detector_backend": _normalize_facecam_backend(detector_backend), "anchor": _normalize_facecam_anchor(anchor), "candidate_count": 0, "track_count": 0, "best_score": None, "best_rect": None, "preferred_side": "left", "fallback_reason": None, "top3": []}
    if cv2 is None or not frames:
        debug["fallback_reason"] = "no_frames_or_cv2"
        return (None, debug) if return_debug else None
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
        return (None, debug) if return_debug else None
    tracks = _build_webcam_tracks(all_candidates)
    debug["track_count"] = len(tracks)
    if not tracks:
        debug["fallback_reason"] = "no_tracks"
        return (None, debug) if return_debug else None
    person_boxes_by_frame = {}
    if normalized_backend == "yolo_window_v1":
        person_boxes_by_frame = _detect_person_boxes_with_yolo(frames)
    scored = []
    side_votes = {"left": 0.0, "right": 0.0}
    for track in tracks:
        ranked = _score_webcam_track(track, frames, source_w=source_w, source_h=source_h, subject_side=normalized_side, person_boxes_by_frame=person_boxes_by_frame, detectors=safe_detectors)
        if normalized_side in {"left", "right"} and ranked["side"] == normalized_side:
            ranked["score"] += 0.02
        side_votes[ranked["side"]] += max(0.0, ranked["score"])
        scored.append(ranked)
    debug["preferred_side"] = "left" if side_votes["left"] >= side_votes["right"] else "right"
    scored.sort(key=lambda item: (item["score"], item["temporal_hits"]), reverse=True)
    best = scored[0]
    debug["best_score"] = round(float(best["score"]), 4)
    debug["best_rect"] = best["rect"]
    debug["top3"] = [{"rect": item["rect"], "side": item["side"], "score": round(float(item["score"]), 4), "temporal_hits": round(float(item["temporal_hits"]), 4), "debug": item["debug"]} for item in scored[:3]]
    debug["preferred_side"] = best["side"] if normalized_side == "auto" else normalized_side
    if float(best["score"]) < float(min_score):
        debug["fallback_reason"] = "low_score"
        return (None, debug) if return_debug else None
    if not _camera_rect_sanity_ok(best["rect"], source_w, source_h):
        debug["fallback_reason"] = "camera_rect_sanity_failed"
        return (None, debug) if return_debug else None
    logging.info("webcam_region_v3 backend=%s candidates=%s tracks=%s best_rect=%s best_score=%.4f top3=%s", normalized_backend, debug["candidate_count"], debug["track_count"], best["rect"], best["score"], [(item["rect"], round(item["score"], 3)) for item in scored[:3]])
    return (best["rect"], debug) if return_debug else best["rect"]


def _heuristic_face_box_from_corners(source_w, source_h, frames, subject_side="auto"):
    """Резервная эвристика через webcam-оверлей."""
    webcam_rect = _detect_webcam_region(source_w, source_h, frames, detectors=None, subject_side=subject_side)
    if webcam_rect is None:
        return None
    return _pseudo_face_box_from_camera_rect(webcam_rect)


def _save_facecam_debug_frames(probe_indexed, camera_rect, output_file, max_frames=5):
    """Сохраняет кадры с рамкой camera_rect, если включен env FACECAM_SAVE_DEBUG_FRAMES=1."""
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
    for frame_idx, (source_frame_idx, frame) in enumerate(probe_indexed[:max(1, int(max_frames))]):
        if frame is None:
            continue
        canvas = frame.copy()
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(canvas, f"camera_rect={x},{y},{w},{h}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        filename = os.path.join(out_dir, f"{os.path.basename(output_file)}_probe_{frame_idx}_{source_frame_idx}.jpg")
        try:
            if cv2.imwrite(filename, canvas):
                saved.append(filename)
        except Exception:
            continue
    return saved
