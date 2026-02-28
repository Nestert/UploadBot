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


def _classify_side(center_x, fw):
    """Возвращает 'left'/'right'/None для x-центра прямоугольника."""
    center_x_norm = _clamp(center_x / float(max(fw, 1)), 0.0, 1.0)
    if center_x_norm <= 0.35:
        return "left"
    if center_x_norm >= 0.65:
        return "right"
    return None


def _candidate_from_rect(bx, by, bw, bh, rectangularity, probe_idx, fw, fh):
    """Упакованный кандидат либо None, если не проходит базовые проверки."""
    if bw < 40 or bh < 40:
        return None
    aspect = bw / float(max(bh, 1))
    if aspect < 0.28 or aspect > 5.0:
        return None
    side = _classify_side(bx + bw / 2.0, fw)
    if side is None:
        return None
    center_y_norm = _clamp((by + bh / 2.0) / float(max(fh, 1)), 0.0, 1.0)
    # Webcam overlays sit in corners/edges, not center. Reward edge proximity.
    y_edge_score = _clamp(abs(center_y_norm - 0.5) * 2.0, 0.0, 1.0)
    return {
        "probe_idx": probe_idx,
        "rect": (int(bx), int(by), int(bw), int(bh)),
        "rectangularity": float(rectangularity),
        "side": side,
        "y_middle": float(y_edge_score),
    }


def _union_rect(rects):
    """Минимальный охватывающий bbox для списка (x,y,w,h)."""
    x1 = min(r[0] for r in rects)
    y1 = min(r[1] for r in rects)
    x2 = max(r[0] + r[2] for r in rects)
    y2 = max(r[1] + r[3] for r in rects)
    return (x1, y1, x2 - x1, y2 - y1)


def _merge_partial_candidates(candidates, frame_area, fw, fh, probe_idx):
    """Пытается слить близкие/перекрывающиеся кандидаты одной стороны в один union-bbox.

    Это нужно для случая, когда рамка вебкамеры разбивается Canny на 2-3 частичных
    прямоугольника. Union таких частей часто даёт правильный итоговый bbox.
    """
    merged = []
    for side in ("left", "right"):
        group = [c for c in candidates if c["side"] == side]
        if len(group) < 2:
            continue
        # Объединяем все пары/тройки, у которых рёбра находятся близко
        used = [False] * len(group)
        for i in range(len(group)):
            if used[i]:
                continue
            cluster = [group[i]["rect"]]
            for j in range(i + 1, len(group)):
                if used[j]:
                    continue
                ri, rj = group[i]["rect"], group[j]["rect"]
                iou = _box_iou(ri, rj)
                # Считаем «близкими» если IoU > 0 или их расстояние по любой оси < min(w,h) * 0.5
                ri_x2, ri_y2 = ri[0] + ri[2], ri[1] + ri[3]
                rj_x2, rj_y2 = rj[0] + rj[2], rj[1] + rj[3]
                gap_x = max(0, max(ri[0], rj[0]) - min(ri_x2, rj_x2))
                gap_y = max(0, max(ri[1], rj[1]) - min(ri_y2, rj_y2))
                proximity_thresh = min(ri[2], ri[3], rj[2], rj[3]) * 0.6
                if iou > 0 or (gap_x < proximity_thresh and gap_y < proximity_thresh):
                    cluster.append(rj)
                    used[j] = True
            if len(cluster) < 2:
                continue
            ux, uy, uw, uh = _union_rect(cluster)
            u_area = float(uw * uh)
            if u_area < frame_area * 0.0025 or u_area > frame_area * 0.20:
                continue
            # Rectangularity union-кандидата: считаем как соотношение суммы площадей к bbox
            sum_area = sum(r[2] * r[3] for r in cluster)
            rectangularity = min(1.0, sum_area / max(1.0, u_area))
            cand = _candidate_from_rect(ux, uy, uw, uh, rectangularity, probe_idx, fw, fh)
            if cand:
                merged.append(cand)
    return merged


def _extract_rect_candidates_from_frame(frame, probe_idx):
    if cv2 is None or frame is None:
        return []
    fh, fw = frame.shape[:2]
    frame_area = float(max(1, fw * fh))
    min_area = frame_area * 0.0025
    max_area = frame_area * 0.20
    candidates = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Три прохода Canny с разными порогами + один проход с крупным MORPH_CLOSE
    # чтобы соединить разрывы в рамке вебкамеры и получить единый контур
    edge_passes = [
        (blurred, (25, 80)),
        (blurred, (40, 120)),
        (blurred, (60, 180)),
    ]
    # Дополнительный проход: морфологическое закрытие с крупным ядром перед Canny
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    gray_closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, close_kernel)
    edge_passes.append((gray_closed, (30, 100)))

    for source_gray, (low_t, high_t) in edge_passes:
        edges = cv2.Canny(source_gray, low_t, high_t)
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
            if len(approx) < 4 or len(approx) > 12:   # было 8
                continue
            bx, by, bw, bh = cv2.boundingRect(approx)
            rect_area = float(bw * bh)
            if rect_area <= 0:
                continue
            rectangularity = area / rect_area
            if rectangularity < 0.45:   # было 0.55
                continue
            cand = _candidate_from_rect(bx, by, bw, bh, rectangularity, probe_idx, fw, fh)
            if cand:
                candidates.append(cand)

    # Слить частичные кандидаты одной стороны в union-bbox
    merged = _merge_partial_candidates(candidates, frame_area, fw, fh, probe_idx)
    candidates.extend(merged)

    # Дотянуть до края кадра те кандидаты, что почти касаются границы
    snapped = _snap_to_frame_edges(candidates, fw, fh, snap_px=20)
    candidates.extend(snapped)

    # Угловые кандидаты добавляются НЕ здесь, а один раз в _detect_webcam_region
    # чтобы не раздувать temporal_hits до 1.0 для каждого размера
    return candidates


def _snap_to_frame_edges(candidates, fw, fh, snap_px=20):
    """Растягивает кандидатов, касающихся края кадра, до самой границы.

    Вебкамера стримера почти всегда расположена в углу и прилегает к 1-2 краям.
    Canny не видит крайнее ребро (оно обрезано границей кадра), поэтому
    прямоугольник не замыкается полностью. Этот pass дотягивает его до края.
    """
    snapped = []
    for c in candidates:
        bx, by, bw, bh = c["rect"]
        x1, y1, x2, y2 = bx, by, bx + bw, by + bh
        changed = False
        if x1 <= snap_px:
            x1, changed = 0, True
        if y1 <= snap_px:
            y1, changed = 0, True
        if fw - x2 <= snap_px:
            x2, changed = fw, True
        if fh - y2 <= snap_px:
            y2, changed = fh, True
        if changed:
            new_w, new_h = x2 - x1, y2 - y1
            if new_w >= 40 and new_h >= 40:
                cand = _candidate_from_rect(x1, y1, new_w, new_h,
                                            c["rectangularity"], c["probe_idx"], fw, fh)
                if cand:
                    snapped.append(cand)
    return snapped


def _generate_corner_candidates(fw, fh, probe_idx):
    """Генерирует синтетических кандидатов покрывающих 4 угла кадра.

    Стримерские вебкамеры практически всегда размещены в одном из 4 углов.
    Canny плохо справляется с краями прямоугольника, вплотную прилегающего
    к границе кадра. Этот набор кандидатов участвует в скоринге наравне
    с Canny-кандидатами и выигрывает если в углу есть лицо/движение/гистограмма.
    """
    frame_area = float(fw * fh)
    candidates = []
    # Типичные размеры вебкамеры: 10-35% ширины, 10-30% высоты
    for w_frac in (0.12, 0.18, 0.25, 0.32):
        for h_frac in (0.10, 0.15, 0.22, 0.30):
            cw = int(round(fw * w_frac))
            ch = int(round(fh * h_frac))
            area = float(cw * ch)
            if area < frame_area * 0.005 or area > frame_area * 0.20:
                continue
            corners = [
                (0, 0),           # top-left
                (fw - cw, 0),     # top-right
                (0, fh - ch),     # bottom-left
                (fw - cw, fh - ch),  # bottom-right
            ]
            for cx, cy in corners:
                cand = _candidate_from_rect(cx, cy, cw, ch, 1.0, probe_idx, fw, fh)
                if cand:
                    # Помечаем как «corner» чтобы дедупликация не выбросила сразу
                    cand["corner"] = True
                    candidates.append(cand)
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
    edge_score = _clamp(track.get("mean_middle", 0.5), 0.0, 1.0)
    person_hits = [_person_overlap_ratio(rect, person_boxes_by_frame.get(pi, [])) for pi in track["frame_hits"]]
    person_score = statistics.mean(person_hits) if person_hits else 0.0
    safe_detectors = _normalize_detector_list(detectors)
    face_score = 0.0
    if safe_detectors:
        # Проверяем несколько кадров вместо только frames[0] — лицо может быть
        # повёрнуто или за кадром в первом пробном фрейме
        probe_frame_indices = sorted(track["frame_hits"])[:5]  # первые 5 кадров трека
        best_face_ratio = 0.0
        for fi in probe_frame_indices:
            if fi >= len(frames):
                continue
            frame_i = frames[fi]
            fi_h, fi_w = frame_i.shape[:2]
            crop = frame_i[max(0, y):min(fi_h, y + h), max(0, x):min(fi_w, x + w)]
            if crop is None or crop.size == 0:
                continue
            face = _detect_largest_face(crop, safe_detectors)
            if face:
                fx, fy, fwf, fhf = face
                face_ratio = (fwf * fhf) / float(max(1, w * h))
                if face_ratio > best_face_ratio:
                    best_face_ratio = face_ratio
        if best_face_ratio >= 0.02:
            face_score = _clamp(best_face_ratio / 0.20, 0.0, 1.0)
            # Штраф за слишком маленький rect: если rect < 1.5% кадра, вероятно
            # это частичная детекция где лицо заполняет почти весь crop.  
            # При area_ratio >= 1.5% штраф нулевой, ниже — линейно убывает.
            rect_area_ratio = (w * h) / float(max(1, source_w * source_h))
            size_factor = _clamp(rect_area_ratio / 0.015, 0.0, 1.0)
            face_score *= size_factor
    # Веса: face_score — лучший дискриминатор (0.80+ для вебкамеры с лицом).
    # edge_score награждает кандидатов у краёв кадра (вебкамеры всегда в углах).
    raw_score = (0.16 * temporal_hits) + (0.10 * boundary_score) + (0.14 * activity_score) + (0.10 * hist_score) + (0.07 * rect_score) + (0.07 * side_score_val) + (0.08 * edge_score) + (0.05 * person_score) + (0.23 * face_score)
    # Penalise large rects (>15% of frame area) — they are likely game UI, not a webcam overlay
    area_ratio = (w * h) / float(max(1, source_w * source_h))
    if area_ratio > 0.15:
        size_penalty = max(0.4, 1.0 - (area_ratio - 0.15) * 3.0)
        raw_score *= size_penalty
    score = raw_score
    return {"score": float(score), "rect": rect, "side": track["side"], "temporal_hits": temporal_hits,
            "debug": {"s_temporal": round(temporal_hits, 4), "s_boundary": round(boundary_score, 4), "s_activity": round(activity_score, 4), "raw_activity": round(raw_activity, 5), "s_hist": round(hist_score, 4), "s_rect": round(rect_score, 4), "s_side": round(side_score_val, 4), "s_middle": round(edge_score, 4), "s_person": round(person_score, 4), "s_face": round(face_score, 4)}}


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

    # Corner candidates: generate across multiple frames for fair temporal scoring.
    # Previously only added once, giving temporal_hits ≈ 1/N which made them
    # uncompetitive against even small Canny rects with high temporal_hits.
    # Also generate even when Canny found 0 candidates (was blocked by
    # 'if all_candidates' condition — bug).
    if frames:
        fw_c, fh_c = frames[0].shape[1], frames[0].shape[0]
        corner_step = max(1, len(frames) // 8)
        for ci in range(0, len(frames), corner_step):
            corner_cands = _generate_corner_candidates(fw_c, fh_c, ci)
            all_candidates.extend(corner_cands)

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
        logging.info(
            "webcam_region_v3 FAILED low_score=%.4f < %.4f candidates=%s tracks=%s top3=%s",
            best["score"], min_score, debug["candidate_count"], debug["track_count"],
            [(item["rect"], round(item["score"], 3), item["debug"]) for item in scored[:3]],
        )
        return (None, debug) if return_debug else None

    # Перебираем кандидатов по убыванию score до первого прошедшего sanity.
    # Ключевой fix: #1 часто является частичным rect (крошечным) с высокой face_score;
    # #2 или #3 — полный прямоугольник вебкамеры.
    chosen = None
    expansion_candidates = []  # collect small face-rects for corner expansion
    for candidate in scored:
        if float(candidate["score"]) < float(min_score):
            break  # отсортированы по убыванию, дальше только хуже
        bx, by, bw, bh = candidate["rect"]
        bx = max(0, bx)
        by = max(0, by)
        bw = min(bw, source_w - bx)
        bh = min(bh, source_h - by)
        clamped = (bx, by, bw, bh)
        if _camera_rect_sanity_ok(clamped, source_w, source_h):
            chosen = {**candidate, "rect": clamped}
            break
        else:
            # Collect for corner expansion attempt
            face_s = candidate.get("debug", {}).get("s_face", 0)
            if face_s > 0:
                expansion_candidates.append((candidate, clamped, face_s))
            area_ratio = (bw * bh) / float(max(1, source_w * source_h))
            logging.debug(
                "webcam_region_v3 skip rect=%s score=%.3f area_ratio=%.4f (sanity fail)",
                clamped, candidate["score"], area_ratio,
            )

    # Corner expansion fallback: try to expand small face-containing rects
    # to plausible corner-anchored webcam rects
    if chosen is None and expansion_candidates:
        for candidate, clamped, face_s in expansion_candidates[:5]:
            expanded_options = _try_expand_to_corner(clamped, source_w, source_h, face_score=face_s)
            if expanded_options:
                # Pick the smallest expanded rect that passes sanity
                expanded_options.sort(key=lambda item: item[0][2] * item[0][3])
                expanded_rect, corner_name = expanded_options[0]
                chosen = {**candidate, "rect": expanded_rect}
                chosen["score"] *= 0.93  # small penalty for expansion
                logging.info(
                    "webcam_region_v3 expanded rect=%s -> %s corner=%s score=%.3f face_s=%.2f",
                    clamped, expanded_rect, corner_name, chosen["score"], face_s,
                )
                break

    if chosen is None:
        area_ratio = (best["rect"][2] * best["rect"][3]) / float(max(1, source_w * source_h))
        w_h_ratio = best["rect"][2] / float(max(1, best["rect"][3]))
        logging.info(
            "webcam_region_v3 FAILED sanity all_candidates_rejected best_rect=%s area_ratio=%.4f source=%dx%d top3=%s",
            best["rect"], area_ratio, source_w, source_h,
            [(item["rect"], round(item["score"], 3), item["debug"]) for item in scored[:3]],
        )
        debug["fallback_reason"] = "camera_rect_sanity_failed"
        return (None, debug) if return_debug else None

    debug["best_rect"] = chosen["rect"]
    debug["best_score"] = round(float(chosen["score"]), 4)
    logging.info(
        "webcam_region_v3 backend=%s candidates=%s tracks=%s best_rect=%s best_score=%.4f top3=%s",
        normalized_backend, debug["candidate_count"], debug["track_count"],
        chosen["rect"], chosen["score"],
        [(item["rect"], round(item["score"], 3), item["debug"]) for item in scored[:3]],
    )
    return (chosen["rect"], debug) if return_debug else chosen["rect"]


def _try_expand_to_corner(rect, source_w, source_h, face_score=0.0):
    """Expand a small face-containing rect near a frame edge to a plausible webcam rect.

    Webcam overlays are always anchored to a corner. When Canny detects only a
    partial border fragment (small rect with a face), this function expands it
    to a reasonable webcam-sized rect anchored at the nearest corner.

    Returns a list of (expanded_rect, corner_name) tuples that pass sanity, or [].
    """
    x, y, w, h = rect
    area_ratio = (w * h) / float(max(1, source_w * source_h))
    if area_ratio >= 0.01:
        return []  # already big enough, no need to expand
    if face_score < 0.15:
        return []  # no face evidence, don't expand blindly

    # Check proximity to frame edges
    snap = max(20, int(min(source_w, source_h) * 0.02))
    near_left = x <= snap
    near_right = (x + w) >= (source_w - snap)
    near_top = y <= snap
    near_bottom = (y + h) >= (source_h - snap)

    if not (near_left or near_right) and not (near_top or near_bottom):
        return []  # not near any edge, can't determine corner

    # Try multiple corner-anchored sizes that contain the original rect
    results = []
    for w_frac in (0.13, 0.18, 0.24, 0.32):
        for h_frac in (0.10, 0.15, 0.20, 0.28):
            cw = int(round(source_w * w_frac))
            ch = int(round(source_h * h_frac))

            corners = []
            if near_top or (not near_bottom and y < source_h * 0.35):
                if near_left or (not near_right and x < source_w * 0.5):
                    corners.append((0, 0, "top-left"))
                if near_right or (not near_left and (x + w) > source_w * 0.5):
                    corners.append((source_w - cw, 0, "top-right"))
            if near_bottom or (not near_top and (y + h) > source_h * 0.65):
                if near_left or (not near_right and x < source_w * 0.5):
                    corners.append((0, source_h - ch, "bottom-left"))
                if near_right or (not near_left and (x + w) > source_w * 0.5):
                    corners.append((source_w - cw, source_h - ch, "bottom-right"))

            for cx, cy, corner_name in corners:
                # Check that the original rect is contained in this corner rect
                if x >= cx and y >= cy and (x + w) <= (cx + cw) and (y + h) <= (cy + ch):
                    expanded = (max(0, cx), max(0, cy),
                                min(cw, source_w - max(0, cx)),
                                min(ch, source_h - max(0, cy)))
                    if _camera_rect_sanity_ok(expanded, source_w, source_h):
                        # Avoid duplicates
                        if expanded not in [r[0] for r in results]:
                            results.append((expanded, corner_name))
    return results


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
