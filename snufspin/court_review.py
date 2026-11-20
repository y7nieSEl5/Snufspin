"""Overlay and image evidence for projected painted court markings."""

from pathlib import Path

import cv2
import numpy as np

from .court import LANDMARKS, LINES

LINE_POINTS = {
    name: np.linspace(*(LANDMARKS[point][:2] for point in ends), 101)
    for name, ends in LINES.items()
}


def marking_support(frame, mapping):
    """Measure bright ridges along the projected segments in the decoded frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    radius = max(3, round(width / 240))
    offsets = np.arange(-2 * radius, 2 * radius + 1)
    support = {}
    for name, xy in LINE_POINTS.items():
        path = mapping.project(xy)
        tangent = path[2:] - path[:-2]
        normal = tangent[:, ::-1] * [-1, 1]
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        grid = path[1:-1, None, :] + normal[:, None, :] * offsets[None, :, None]
        visible = (
            (grid[..., 0] >= 0) & (grid[..., 0] < width - 1)
            & (grid[..., 1] >= 0) & (grid[..., 1] < height - 1)
        ).all(axis=1)
        if visible.sum() < 12:
            continue
        grid = grid[visible].astype(np.float32)
        values = cv2.remap(gray, grid[..., 0], grid[..., 1], cv2.INTER_LINEAR).astype(float)
        # The line must be brighter than the nearby surface on both sides.
        peak = values[:, radius:-radius].max(axis=1)
        flanks = np.maximum(np.median(values[:, :radius], axis=1),
                            np.median(values[:, -radius:], axis=1))
        support[name] = round(float(np.mean(peak - flanks >= 12)), 3)
    return support


def markings_agree(support):
    along = ("doubles_low", "singles_low", "singles_high", "doubles_high", "center_service")
    across = ("baseline_a", "baseline_b", "service_a", "service_b")
    supported = {name for name, fraction in support.items() if fraction >= 0.50}
    return len(supported) >= 3 and bool(supported.intersection(along)) and bool(supported.intersection(across))


def write_overlay(path: Path, frame, mapping, points=None, camera=None):
    overlay = frame.copy()
    for geometry, color in ((mapping, (255, 220, 0)), (camera, (60, 140, 255))):
        if geometry is None:
            continue
        for xy in LINE_POINTS.values():
            coords = (geometry.project(xy) if geometry is mapping else
                      geometry.project(np.column_stack((xy, np.zeros(len(xy))))))
            cv2.polylines(overlay, [np.rint(coords).astype(np.int32)], False, color, 1, cv2.LINE_AA)
    for name, point in (points or {}).items():
        u, v = np.rint(point).astype(int)
        cv2.drawMarker(overlay, (u, v), (120, 255, 0), cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)
        cv2.putText(overlay, name, (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 255, 0),
                    1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), overlay):
        raise ValueError(f"could not write court overlay {path}")
