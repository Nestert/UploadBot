"""vertical.geometry — операции над прямоугольниками и геометрия crop."""

import statistics


def _clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


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
    from vertical.detection import _normalize_subject_side
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
    from vertical.detection import _normalize_subject_side, _normalize_facecam_anchor
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
