"""Fixed-camera court selection and refinement from source-resolution paint."""

from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import CourtMapping
from .court import LANDMARKS, LINES
from .court_review import markings_agree
from .line_court import line_hypotheses

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
                        "frames": winner.frames})
    for runner, other, frame in ranked[1:]:
        spread = np.quantile(np.linalg.norm(points - other, axis=1), 0.9)
        if spread > max(6.0, width / 120):
            margin = winner.score - runner.score
            ambiguous = margin <= max(2 * len(frames), 0.02 * winner.score)
            diagnostics["alternative"] = {"frame": frame, "score": runner.score, "margin": margin,
                                          "p90_separation_px": float(spread), "ambiguous": ambiguous}
            if ambiguous:
                diagnostics["status"] = "ambiguous_paint_mapping"
                return None, gray, diagnostics
            break
    return winner, gray, diagnostics
