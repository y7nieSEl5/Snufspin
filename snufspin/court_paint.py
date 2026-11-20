"""Fixed-camera court selection and refinement from source-resolution paint."""

from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import CourtMapping
from .court import LANDMARKS, LINES
from .court_review import markings_agree
from .line_court import line_hypotheses, matched_landmarks

_SEGMENTS = {line: np.array([LANDMARKS[n][:2] for n in ends]) for line, ends in LINES.items()}
_ANCHORS = np.array(list(LANDMARKS.values()))[:, :2]
_ALONG = {name for name in LINES if "baseline" not in name and not name.startswith("service")}


@dataclass
class PaintCandidate:
    mapping: CourtMapping
    score: float
    frames: dict


def _sample_segment(points, width, height, step=2):
    """Clip a projected segment to the image, then sample at fixed pixel spacing."""
    start, end = points
    vector = end - start
    length = np.linalg.norm(vector)
    if length < 1e-6:
        return np.empty((0, 2)), np.zeros(2), 0.0
    first, last = 0.0, 1.0
    for origin, delta, bound in zip(start, vector, (width - 1, height - 1)):
        if abs(delta) < 1e-8:
            if not 0 <= origin <= bound:
                return np.empty((0, 2)), np.zeros(2), 0.0
            continue
        a, b = sorted((-origin / delta, (bound - origin) / delta))
        first, last = max(first, a), min(last, b)
    visible = max(0.0, last - first) * length
    count = int(np.ceil(visible / step))
    t = first + (np.arange(count) + 0.5) / max(count, 1) * (last - first)
    return start + t[:, None] * vector, vector[::-1] * [-1, 1] / length, visible


def _ridge_strength(gray, samples, normal, radius):
    # Search offsets around each projected point, comparing the center with both flanks.
    offsets = np.arange(-radius - 10, radius + 11)
    coords = samples[:, None, :] + offsets[None, :, None] * normal
    columns, rows = np.moveaxis(coords.astype(np.float32), -1, 0)
    values = cv2.remap(gray, columns, rows, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(float)
    center = (values[:, 9:-11] + 2 * values[:, 10:-10] + values[:, 11:-9]) / 4
    strength = np.maximum.reduce([
        center - np.maximum(values[:, 10 - flank:-10 - flank],
                            values[:, 10 + flank:values.shape[1] - 10 + flank])
        for flank in (3, 6, 10)
    ])
    return strength


def score_mapping(mapping, gray_frames):
    """Sum Farin's +1/-0.5/0 rule using direction-aware local paint support."""
    height, width = next(iter(gray_frames.values())).shape
    radius = max(1, round(width / 960))
    lines = {}
    for name, xy in _SEGMENTS.items():
        lines[name] = _sample_segment(mapping.project(xy), width, height)
    scores = {}
    for index, gray in gray_frames.items():
        measurements = {}
        for name, (points, normal, visible) in lines.items():
            hits = int((_ridge_strength(gray, points, normal, radius).max(axis=1) >= 12).sum()) if len(points) else 0
            measurements[name] = {
                "visible_px": float(visible), "samples": len(points), "paint_hits": hits,
                "support": hits / len(points) if len(points) else None,
            }
        supported = {name: line["support"] for name, line in measurements.items() if line["samples"] >= 12}
        along = sum(measurements[name]["visible_px"] for name in supported.keys() & _ALONG
                    if supported[name] >= 0.5)
        across = sum(measurements[name]["visible_px"] for name in supported.keys() - _ALONG
                     if supported[name] >= 0.5)
        scores[index] = {
            "markings": measurements,
            "score": sum(1.5 * line["paint_hits"] - 0.5 * line["samples"] for line in measurements.values()),
            "supported": markings_agree(supported) and min(along, across) >= width * 0.18,
        }
    return PaintCandidate(mapping, sum(frame["score"] for frame in scores.values()), scores)


def select_paint_mapping(frames, segments):
    """Rerank distinct LETR hypotheses with a single mapping scored across the clip."""
    gray = {index: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for index, image in frames.items()}
    height, width = next(iter(gray.values())).shape
    projections = []
    ranked = []
    for frame, lines in segments.items():
        for H in line_hypotheses(lines, width, height):
            mapping = CourtMapping(H)
            points = mapping.project(_ANCHORS)
            if any(np.max(np.linalg.norm(points - earlier, axis=1)) < 3 for earlier in projections):
                continue
            projections.append(points)
            candidate = score_mapping(mapping, gray)
            if sum(info["supported"] for info in candidate.frames.values()) >= max(2, len(frames) // 2 + 1):
                ranked.append((candidate, points, frame))
    ranked.sort(key=lambda row: -row[0].score)
    diagnostics = {"scored_hypotheses": len(projections), "eligible_hypotheses": len(ranked)}
    if not ranked:
        diagnostics["status"] = "no_shared_paint_mapping"
        return None, gray, diagnostics
    winner, points, source_frame = ranked[0]
    diagnostics.update({"status": "selected", "winning_frame": source_frame, "score": winner.score,
                        "winning_H": winner.mapping.H.tolist(), "frames": winner.frames})
    for runner, other, frame in ranked[1:]:
        spread = np.quantile(np.linalg.norm(points - other, axis=1), 0.9)
        if spread > max(6.0, width / 120):
            margin = winner.score - runner.score
            ambiguous = margin <= max(2 * len(frames), 0.02 * winner.score)
            diagnostics["alternative"] = {"frame": frame, "score": runner.score, "margin": margin,
                                          "p90_separation_px": float(spread), "ambiguous": ambiguous,
                                          "H": runner.mapping.H.tolist()}
            if ambiguous:
                diagnostics["status"] = "ambiguous_paint_mapping"
                return None, gray, diagnostics
            break
    return winner, gray, diagnostics


def measured_landmarks(candidate, segments, width, height):
    """Return only observed LETR intersections close to the selected court."""
    predicted = dict(zip(LANDMARKS, candidate.mapping.project(_ANCHORS)))
    return {index: {name: uv for name, uv in matched_landmarks(lines, candidate.mapping, width, height).items()
                    if np.linalg.norm(uv - predicted[name]) <= max(6.0, width / 120)}
            for index, lines in segments.items()}


def _paint_observations(candidate, gray_frames):
    """Collect ridge pixels and associate each with its nearest finite court marking."""
    height, width = next(iter(gray_frames.values())).shape
    radius = max(4, round(width / 240))
    ids, pixels = [], []
    for line_id, (name, xy) in enumerate(_SEGMENTS.items()):
        projected, normal, _ = _sample_segment(candidate.mapping.project(xy), width, height, step=4)
        if len(projected) < 12:
            continue
        offsets = np.arange(-radius, radius + 1)
        for index, image in gray_frames.items():
            evidence = candidate.frames[index]["markings"][name]["support"]
            if evidence is None or evidence < 0.35:
                continue
            strength = _ridge_strength(image, projected, normal, radius)
            strongest = (strength - 0.5 * np.abs(offsets)).argmax(axis=1)
            keep = strength[np.arange(len(projected)), strongest] >= 12
            points = projected[keep] + offsets[strongest[keep], None] * normal
            inside = np.all((points >= 0) & (points < [width, height]), axis=1)
            pixels.extend(points[inside])
            ids.extend([line_id] * int(inside.sum()))
    pixels = np.array(pixels).reshape(-1, 2)
    homogeneous = np.column_stack((pixels, np.ones(len(pixels))))
    world = homogeneous @ np.linalg.inv(candidate.mapping.H).T
    xy = world[:, :2] / world[:, 2:]
    paint = np.array(list(_SEGMENTS.values()))
    tangents = paint[:, 1] - paint[:, 0]
    lengths = np.linalg.norm(tangents, axis=1)
    tangents /= lengths[:, None]
    offset = xy[:, None] - paint[:, 0]
    length_along = np.einsum("nkq,kq->nk", offset, tangents).clip(0, lengths)
    distances = np.linalg.norm(offset - length_along[..., None] * tangents[None], axis=2)
    ids = np.array(ids, dtype=int)
    keep = (distances.argmin(axis=1) == ids) & (distances[np.arange(len(ids)), ids] < 0.45)
    normals = tangents[:, ::-1] * [-1, 1]
    lines = np.column_stack((normals, -np.einsum("kq,kq->k", normals, paint[:, 0])))
    return pixels[keep], lines[ids[keep]]


def refine_paint_mapping(candidate, gray_frames):
    """Fit one inverse homography and accept only an improvement on withheld frames."""
    from scipy.optimize import least_squares

    indexes = sorted(gray_frames)
    train, held = indexes[::2], indexes[1::2]
    diagnostics = {
        "refinement_frames": train, "held_out_frames": held,
        "held_out_score_before": sum(candidate.frames[index]["score"] for index in held),
    }
    if not held:
        diagnostics["refinement_status"] = "insufficient_frames"
        return candidate, diagnostics
    pixels, lines = _paint_observations(candidate, {i: gray_frames[i] for i in train})
    diagnostics["paint_observations"] = len(pixels)
    if len(pixels) < 60 or len(np.unique(lines, axis=0)) < 4:
        diagnostics["refinement_status"] = "insufficient_paint_observations"
        return candidate, diagnostics

    height, width = next(iter(gray_frames.values())).shape
    normalize = np.array([[1 / width, 0, -0.5], [0, 1 / height, -0.5], [0, 0, 1]])
    normalized_pixels = np.column_stack((pixels, np.ones(len(pixels)))) @ normalize.T
    inverse = np.linalg.inv(candidate.mapping.H) @ np.linalg.inv(normalize)
    pivot = 6 + int(np.abs(inverse[2]).argmax())
    inverse /= inverse.flat[pivot]
    free = np.arange(9) != pivot

    def assemble(parameters):
        matrix = inverse.copy()
        matrix.flat[free] = parameters
        return matrix

    def residuals(parameters):
        coords = normalized_pixels @ assemble(parameters).T
        denominator = np.where(np.abs(coords[:, 2]) < 1e-7, 1e-7, coords[:, 2])
        return np.einsum("ij,ij->i", coords, lines) / denominator

    result = least_squares(residuals, inverse.ravel()[free], method="lm", max_nfev=80)
    inverse_fit = assemble(result.x) @ normalize
    if not np.isfinite(inverse_fit).all() or np.linalg.matrix_rank(inverse_fit) != 3:
        diagnostics["refinement_status"] = "singular_mapping"
        return candidate, diagnostics
    H = np.linalg.inv(inverse_fit)
    depth = _ANCHORS @ H[2, :2] + H[2, 2]
    if not (np.all(depth > 0) or np.all(depth < 0)):
        diagnostics["refinement_status"] = "court_crosses_horizon"
        return candidate, diagnostics
    proposed = CourtMapping(H)
    displacement = np.linalg.norm(proposed.project(_ANCHORS) - candidate.mapping.project(_ANCHORS), axis=1)
    diagnostics["maximum_landmark_shift_px"] = float(displacement.max())
    if displacement.max() > width / 80:
        diagnostics["refinement_status"] = "outside_local_neighborhood"
        return candidate, diagnostics

    refined = score_mapping(proposed, gray_frames)
    held_after = sum(refined.frames[index]["score"] for index in held)
    diagnostics["held_out_score_after"] = held_after
    diagnostics["proposed_score"] = refined.score
    if (held_after <= diagnostics["held_out_score_before"]
            or sum(info["supported"] for info in refined.frames.values()) < max(2, len(indexes) // 2 + 1)
            or any(candidate.frames[index]["supported"] and not refined.frames[index]["supported"] for index in held)):
        diagnostics["refinement_status"] = "withheld_frames_did_not_improve"
        return candidate, diagnostics
    diagnostics["refinement_status"] = "accepted"
    return refined, diagnostics
