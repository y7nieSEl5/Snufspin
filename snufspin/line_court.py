"""Fit the metric court to LETR segments by scoring template homographies."""

from itertools import combinations, islice

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .court import DOUBLES_WIDTH, LANDMARKS, LENGTH, LINES

_PAINT = np.array([[LANDMARKS[name][:2] for name in ends] for ends in LINES.values()])
_CORNERS = np.array([[0, 0, 1], [0, DOUBLES_WIDTH, 1], [LENGTH, 0, 1], [LENGTH, DOUBLES_WIDTH, 1]])
_CROSS = ("baseline_a", "service_a", "service_b", "baseline_b")
_ALONG = ("doubles_high", "singles_high", "center_service", "singles_low", "doubles_low")


def _line_axes(name):
    a, b = _PAINT[list(LINES).index(name)]
    return a if name in _CROSS else b


def _model_rectangles():
    """Map each metric line rectangle to a unit square, ordered near-left first."""
    transforms = []
    for near, far in combinations(_CROSS, 2):
        x0, x1 = _line_axes(near)[0], _line_axes(far)[0]
        for left, right in combinations(_ALONG, 2):
            y0, y1 = _line_axes(left)[1], _line_axes(right)[1]
            dx, dy = x1 - x0, y1 - y0
            transforms.append([[0, 1 / dy, -y0 / dy], [1 / dx, 0, -x0 / dx], [0, 0, 1]])
    return np.array(transforms)


_MODEL_RECTS = _model_rectangles()
_SQUARE = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])


def _intersect(a, b):
    point = np.cross(a, b)
    return point[:2] / point[2] if abs(point[2]) > 1e-7 else np.full(2, np.nan)


def _rectangles(longitudinal, transverse, equations, segments, width, height):
    for near, far in combinations(transverse, 2):
        if segments[near, :, 1].mean() < segments[far, :, 1].mean():
            near, far = far, near
        for left, right in combinations(longitudinal, 2):
            uv = np.array([_intersect(equations[i], equations[j])
                           for i, j in ((near, left), (near, right), (far, left), (far, right))])
            if not np.isfinite(uv).all() or np.any(uv < [-width, -height]) or np.any(uv > [2*width, 2*height]):
                continue
            if uv[0, 0] > uv[1, 0]:
                uv = uv[[1, 0, 3, 2]]
            if uv[3, 0] - uv[2, 0] < width / 40 or uv[1, 0] - uv[0, 0] < width / 20:
                continue
            if uv[:2, 1].mean() - uv[2:, 1].mean() < height / 60:
                continue
            if cv2.isContourConvex(uv[[0, 1, 3, 2]].astype(np.float32)):
                yield cv2.getPerspectiveTransform(_SQUARE, uv.astype(np.float32))


def _segment_scores(predicted, segments, tolerance):
    """Score overlapping evidence along each projected template segment."""
    start = predicted[..., 0, :]
    direction = predicted[..., 1, :] - start
    length = np.linalg.norm(direction, axis=-1)
    tangent = direction / np.maximum(length[..., None], 1e-9)
    normal = tangent[..., ::-1] * [-1, 1]
    relative = segments[None, None, :, :, :] - start[..., None, None, :]
    offset = np.einsum("bkneq,bkq->bkne", relative, normal)
    along = np.einsum("bkneq,bkq->bkne", relative, tangent)
    lo, hi = along.min(axis=-1), along.max(axis=-1)
    overlap = np.maximum(0, np.minimum(hi, length[..., None]) - np.maximum(lo, 0))
    observed_length = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=-1)
    fraction = overlap / np.maximum(observed_length, 1e-9)
    # Reward support for short visible fragments while preferring longer coverage.
    coverage = (overlap / np.maximum(np.minimum(length[..., None], 350), 1e-9)).clip(0, 1)
    alignment = np.exp(-np.mean(offset**2, axis=-1) / tolerance**2)
    return alignment * fraction * (0.5 + 0.5 * coverage)


def refine_segments(image, segments):
    """Center near-court segment proposals on bright paint in source pixels."""
    if not len(segments):
        return segments
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    radius = max(5, round(width / 100))
    offsets = np.arange(-radius - 6, radius + 7, dtype=float)
    refined = segments.copy()
    for index, (a, b) in enumerate(segments):
        delta = b - a
        length = np.linalg.norm(delta)
        if length < min(width, height) * 0.06:
            continue
        tangent = delta / length
        normal = tangent[::-1] * [-1, 1]
        centers = np.linspace(a, b, 81)
        grid = centers[:, None] + offsets[None, :, None] * normal
        valid = np.all((grid >= 0) & (grid < [width - 1, height - 1]), axis=(1, 2))
        if valid.sum() < 20:
            continue
        grid = grid[valid].astype(np.float32)
        values = cv2.remap(gray, grid[..., 0], grid[..., 1], cv2.INTER_LINEAR).astype(float)
        # Compare each candidate to the surface on either side of the paint.
        smoothed = (values[:, 5:-7] + 2 * values[:, 6:-6] + values[:, 7:-5]) / 4
        prominence = smoothed - np.maximum(values[:, :-12], values[:, 12:])
        best = (prominence - np.abs(offsets[6:-6]) * 0.35).argmax(axis=1)
        rows = np.arange(len(values))
        good = prominence[rows, best] >= 12
        if good.sum() < max(14, len(rows) * 0.35):
            continue
        observed = grid[rows[good], best[good] + 6]
        if np.ptp(observed @ tangent) < length * 0.50:
            continue
        vx, vy, x, y = cv2.fitLine(observed, cv2.DIST_WELSCH, 0, 0.01, 0.01).ravel()
        axis = np.array([vx, vy])
        origin = np.array([x, y])
        endpoints = origin + ((np.array([a, b]) - origin) @ axis)[:, None] * axis
        if np.linalg.norm(endpoints - [a, b], axis=1).max() <= radius + 3:
            refined[index] = endpoints
    return refined


def fit_lines(segments, width, height):
    """Return named intersections measured from the matched LETR segments."""
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    order = np.argsort(-lengths, kind="stable")
    order = order[lengths[order] >= min(width, height) * 0.06]
    segments = segments[order]
    if len(segments) < 4:
        return {}
    length = lengths[order]
    angles = np.abs(segments[:, 1, 1] - segments[:, 0, 1]) / length
    transverse = np.flatnonzero(angles <= 0.38)[:12]
    longitudinal = np.flatnonzero(angles > 0.38)[:12]
    equations = np.cross(*np.moveaxis(np.dstack((segments, np.ones(segments.shape[:2]))), 1, 0))
    equations /= length[:, None]
    rects = _rectangles(longitudinal, transverse, equations, segments, width, height)

    tolerance = max(4.0, width / 120)
    paint = np.concatenate((_PAINT, np.ones((*_PAINT.shape[:2], 1))), axis=-1)
    best = None
    best_score = 0.0
    while batch := list(islice(rects, 24)):
        H = (np.asarray(batch)[:, None] @ _MODEL_RECTS).reshape(-1, 3, 3)
        depth = H[:, 2] @ _CORNERS.T
        H = H[np.all(depth > 1e-7, axis=1) | np.all(depth < -1e-7, axis=1)]
        if not len(H):
            continue
        projected = np.einsum("bij,kej->bkei", H, paint)
        projected = projected[..., :2] / projected[..., 2:]
        scores = _segment_scores(projected, segments, tolerance)
        # Greedy upper bound followed by one-to-one scoring for competitive hypotheses.
        upper = scores.max(axis=2).sum(axis=1)
        for i in np.argsort(-upper):
            if upper[i] <= best_score:
                break
            model_ids, image_ids = linear_sum_assignment(scores[i], maximize=True)
            weights = scores[i, model_ids, image_ids]
            supported = weights >= 0.35
            total = weights.sum()
            if total > best_score and np.count_nonzero(supported) >= 4:
                best_score = total
                best = dict(zip(model_ids[supported], image_ids[supported]))

    if best is None:
        return {}
    matches = {list(LINES)[model_id]: image_id for model_id, image_id in best.items()}
    landmarks = {}
    for name in LANDMARKS:
        line_kind, end_id, side = name.split("_", 2)
        end = f"{line_kind}_{end_id}"
        side = "center_service" if side == "center" else side
        if end not in matches or side not in matches:
            continue
        i, j = matches[end], matches[side]
        point = _intersect(equations[i], equations[j])
        if not np.all((point >= 0) & (point < [width, height])):
            continue
        # Keep only crossings close to both reported segment extents.
        measured = segments[[i, j]]
        delta = measured[:, 1] - measured[:, 0]
        lengths = length[[i, j]]
        distance = np.einsum("ij,ij->i", point - measured[:, 0], delta) / lengths
        margin = max(12.0, width / 40)
        if np.all((distance >= -margin) & (distance <= lengths + margin)):
            landmarks[name] = point.tolist()
    return landmarks
