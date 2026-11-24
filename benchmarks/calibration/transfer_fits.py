"""Measure whether fixed CalTennis court fits transfer between sampled frames."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from snufspin.calibration import CourtMapping
from snufspin.court_review import LINE_POINTS
from snufspin.court_fit import _physical_court

from .estimate import CLIPS
from .replay_segments import MANIFEST, OUTPUT as REPLAY

OUTPUT = Path("outputs/caltennis-fit-transfer")
SEEDS = {
    "Mar 4 SW": ((0, "vp-crossings"), (0, "vp-segments")),
    "Mar 9 SW": ((2, "vp-crossings"), (2, "vp-segments")),
    "Oct 2 N": ((0, "vp-segments"), (2, "vp-segments")),
}
SEARCH_PX = 32
MIN_PROMINENCE = 16
MIN_SAMPLES = 12


def load_replay(path):
    """Load the blind fits and verify all source pixels against the frame manifest."""
    fits = json.loads((path / "fits.json").read_text())
    frames = {}
    for label, manifest in json.loads(MANIFEST.read_text()).items():
        frames[label] = []
        for fit, sample in zip(fits[label], manifest, strict=True):
            index = sample["index"]
            frame = cv2.imread(str(path / f"{CLIPS[label][:5]}-{index}.png"))
            if (frame is None or hashlib.sha256(frame).hexdigest() != sample["bgr_sha256"]
                    or any(fit[key] != value for key, value in sample.items())):
                raise ValueError(f"{label} {index} differs from the original replay")
            frames[label].append(frame)
    return fits, frames


def line_ridges(mapping, gray_frames):
    """Locate candidate paint ridges on the same normal scans in every frame."""
    height, width = gray_frames[0].shape
    offsets = np.arange(-SEARCH_PX - 8, SEARCH_PX + 9)
    for name, xy in LINE_POINTS.items():
        points = mapping.project(xy[10:-10])
        tangent = points[-1] - points[0]
        normal = tangent[::-1] * [-1, 1] / np.linalg.norm(tangent)
        grid = points[:, None] + offsets[None, :, None] * normal
        visible = ((grid >= 0) & (grid < [width - 1, height - 1])).all(axis=(1, 2))
        grid = grid[visible].astype(np.float32)
        if len(grid) < MIN_SAMPLES:
            yield name, points[visible], normal, [np.empty((0, 2))] * len(gray_frames)
            continue

        profiles = []
        for gray in gray_frames:
            values = cv2.remap(gray.astype(np.float32), grid[..., 0], grid[..., 1], cv2.INTER_LINEAR)
            # Compare smoothed centers with their flanks at two paint widths.
            smooth = 0.25 * (values[:, 7:-9] + 2 * values[:, 8:-8] + values[:, 9:-7])
            narrow = smooth - np.maximum(values[:, 4:-12], values[:, 12:-4])
            wide = smooth - np.maximum(values[:, :-16], values[:, 16:])
            profiles.append(np.maximum(narrow, wide))
        yield name, points[visible], normal, profiles


def source_ridge(profile):
    """Find locally prominent peaks that follow one line across the source image."""
    offsets = np.arange(-SEARCH_PX, SEARCH_PX + 1)
    # A line offset may vary along the marking when the seed fit has angular error.
    # Estimate its local offset from an overlapping window of 21 normal scans.
    ridge = np.full(len(profile), np.nan)
    for i, row in enumerate(profile):
        window = profile[max(0, i - 10):i + 11]
        consensus = np.median(window, axis=0) - 0.35 * abs(offsets)
        anchor = offsets[consensus.argmax()]
        nearby = abs(offsets - anchor) <= 4
        peak = np.argmax(np.where(nearby, row - 0.5 * abs(offsets - anchor), -np.inf))
        if row[peak] >= MIN_PROMINENCE and abs(offsets[peak]) < SEARCH_PX:
            ridge[i] = offsets[peak]
    return ridge


def target_ridge(profile, source):
    """Locate a ridge on the same scans, using the source offsets as anchors."""
    offsets = np.arange(-SEARCH_PX, SEARCH_PX + 1)
    distance = abs(offsets - source[:, None])
    peak = np.argmax(np.where(distance <= 20, profile - 1.5 * distance, -np.inf), axis=1)
    valid = (np.isfinite(source) & (profile[np.arange(len(profile)), peak] >= MIN_PROMINENCE)
             & (abs(offsets[peak]) < SEARCH_PX) & (abs(offsets[peak] - source) < 20))
    return np.where(valid, offsets[peak], np.nan)


def transfer(mapping, gray_frames, source_frame):
    lines = [{} for _ in gray_frames]
    traces = [{} for _ in gray_frames]
    for name, points, normal, profiles in line_ridges(mapping, gray_frames):
        if len(points) < MIN_SAMPLES:
            continue
        source = source_ridge(profiles[source_frame])
        for i, profile in enumerate(profiles):
            target = source if i == source_frame else target_ridge(profile, source)
            paired = np.isfinite(target)
            deltas = target[paired] - source[paired]
            offset = target[paired]
            src_offset = source[paired]
            count = int(paired.sum())
            stats = {
                "scan_count": len(points),
                "source_detected": int(np.isfinite(source).sum()),
                "paired": count,
            }
            if count >= MIN_SAMPLES:
                stats.update(
                    source_abs_p90_px=float(np.quantile(abs(src_offset), .90)),
                    target_abs_p90_px=float(np.quantile(abs(offset), .90)),
                    displacement_median_px=float(np.median(deltas)),
                    displacement_abs_p90_px=float(np.quantile(abs(deltas), .90)),
                )
            lines[i][name] = stats
            traces[i][name] = points[paired] + offset[:, None] * normal
    return lines, traces


def register_scene(source, target):
    """Estimate small coherent image motion independently of the court template."""
    points = cv2.goodFeaturesToTrack(source, 1200, .012, 10)
    if points is None or len(points) < MIN_SAMPLES:
        return None
    moved, forward, _ = cv2.calcOpticalFlowPyrLK(source, target, points, None)
    back, backward, _ = cv2.calcOpticalFlowPyrLK(target, source, moved, None)
    keep = (forward.ravel().astype(bool) & backward.ravel().astype(bool)
            & (np.linalg.norm((points - back).reshape(-1, 2), axis=1) < .8))
    if np.count_nonzero(keep) < MIN_SAMPLES:
        return None
    source_points = points.reshape(-1, 2)[keep]
    target_points = moved.reshape(-1, 2)[keep]
    affine, inliers = cv2.estimateAffine2D(source_points, target_points, ransacReprojThreshold=1.5)
    if affine is None or inliers.sum() < MIN_SAMPLES:
        return None
    return affine, int(inliers.sum())


def scene_motion(mapping, registration, shape):
    if registration is None:
        return None
    affine, inliers = registration
    height, width = shape
    points = np.concatenate([mapping.project(xy) for xy in LINE_POINTS.values()])
    points = points[((points >= 0) & (points < [width, height])).all(axis=1)]
    moved = points @ affine[:, :2].T + affine[:, 2]
    displacement = np.linalg.norm(moved - points, axis=1)
    return {
        "affine": affine.tolist(),
        "feature_inliers": inliers,
        "visible_court_p90_px": float(np.quantile(displacement, .90)),
    }


def overlay(frame, mapping, traces, title, path):
    image = frame.copy()

    def text(label, point):
        cv2.putText(image, label, point, cv2.FONT_HERSHEY_SIMPLEX, .58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, label, point, cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 1, cv2.LINE_AA)

    for name, xy in LINE_POINTS.items():
        points = mapping.project(xy)
        uv = np.rint(points).astype(np.int32)
        cv2.polylines(image, [uv], False, (255, 220, 0), 2, cv2.LINE_AA)
        visible = ((points > [45, 55]) & (points < [frame.shape[1] - 115, frame.shape[0] - 15])).all(axis=1)
        if np.count_nonzero(visible) > 2:
            mids = uv[visible]
            label_point = mids[len(mids) // 2]
            text(name, tuple(label_point + [6, -7]))
        for point in np.rint(traces.get(name, np.empty((0, 2)))[::3]).astype(int):
            cv2.circle(image, tuple(point), 2, (35, 90, 255), -1, cv2.LINE_AA)
    text(title, (20, 30))
    text("Cyan: fixed projection | orange: paired ridge measurements", (20, 55))
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=REPLAY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cv2.setRNGSeed(0)
    fits, frames = load_replay(args.replay)
    report = {}
    for label, seeds in SEEDS.items():
        prefix = CLIPS[label][:5]
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames[label]]
        registrations = {
            (source, i): register_scene(gray_frames[source], frame)
            for source in sorted({source for source, _ in seeds})
            for i, frame in enumerate(gray_frames) if i != source
        }
        report[label] = []
        for source, mode in seeds:
            source_index = fits[label][source]["index"]
            mapping = CourtMapping(fits[label][source]["modes"][mode]["H"])
            if not _physical_court(mapping):
                raise ValueError(f"invalid seed mapping for {label} {source_index}")
            lines, traces = transfer(mapping, gray_frames, source)
            targets = []
            for i, frame in enumerate(frames[label]):
                index = fits[label][i]["index"]
                seconds = fits[label][i]["time_s"]
                name = f"{prefix}-{source_index}-{mode}-to-{index}.png"
                title = f"{label} | {fits[label][source]['time_s']:.0f} s {mode} -> {seconds:.0f} s"
                overlay(frame, mapping, traces[i], title, args.output / name)
                motion = scene_motion(mapping, registrations.get((source, i)), frame.shape[:2])
                targets.append({"index": index, "time_s": seconds, "overlay": name,
                                "scene_motion": motion, "markings": lines[i]})
            report[label].append({"source_index": source_index, "mode": mode, "targets": targets})
    (args.output / "transfer.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
