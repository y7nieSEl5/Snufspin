"""Reproduce the CalTennis court review from the repository root."""

import argparse
from bisect import bisect_left
import json
from pathlib import Path

import cv2
import numpy as np

from snufspin.court import LANDMARKS
from snufspin.court_commands import import_caltennis
from snufspin.court_fit import fit_court
from snufspin.court_review import marking_support, write_overlay
from snufspin.video import probe_video, read_sample_frames


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate(video, calibration_path, annotations, destination):
    indexes = {bisect_left(video.timestamps_s, float(time)): points for time, points in annotations.items()}
    frames = dict(read_sample_frames(video, list(indexes)))
    cameras = {
        transform: import_caltennis(video, calibration_path, transform)
        for transform in ("identity", "resize", "center-padding", "top-padding")
    }
    result = {"frames": {}, "comparison": {}}
    for transform, calibration in cameras.items():
        errors = []
        for index, points in indexes.items():
            predicted = calibration.camera.project([LANDMARKS[name] for name in points])
            delta = predicted - np.array(list(points.values()))
            errors.extend(delta)
            entry = result["frames"].setdefault(str(index), {
                "time_s": video.time_at(index),
                "landmarks": points,
            })
            entry.setdefault("landmark_error_px", {})[transform] = {
                name: np.round(error, 3).tolist() for name, error in zip(points, delta)
            }
        errors = np.array(errors)
        result["comparison"][transform] = {
            "landmark_count": len(errors),
            "median_euclidean_px": float(np.median(np.linalg.norm(errors, axis=1))),
            "rms_vertical_px": float(np.sqrt(np.mean(errors[:, 1]**2))),
        }
    # Vertical resize is the working re-encode hypothesis. Keep the transform in provenance.
    calibration = cameras["resize"]
    report = calibration.to_report(video)
    report["source"]["pixel_transform_evidence"] = "inferred; see benchmarks/calibration/README.md"
    dump(destination / "camera.json", report)
    result["camera_center_m"] = calibration.camera.center.tolist()
    result["min_court_depth_m"] = float(min(calibration.camera.depth(list(LANDMARKS.values()))))

    for index, frame in frames.items():
        entry = result["frames"][str(index)]
        entry["line_support_resize"] = marking_support(frame, calibration.mapping)
        luma = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
        entry["top_row_luma_std"] = np.std(luma[:8], axis=1).round(3).tolist()
        entry["bottom_row_luma_std"] = np.std(luma[-8:], axis=1).round(3).tolist()
        entry["mean_luma_change_across_row_1080"] = float(np.abs(luma[1080] - luma[1079]).mean())
        write_overlay(destination / f"frame-{index:06}.png", frame, calibration.mapping, indexes[index])

    source = {"method": "CalTennis benchmark reviewed landmarks"}
    result["manual_fit"] = {}
    for method, intrinsics in (("supplied", calibration.camera.intrinsics), ("estimated", None)):
        refit, diagnostics = fit_court(frames, indexes, {**source, "intrinsics_method": method}, intrinsics)
        run = result["manual_fit"][method] = {
            "status": diagnostics["status"],
            "pose_status": diagnostics.get("pose_status"),
            "pose_spread": diagnostics.get("pose_spread"),
        }
        if refit is not None:
            dump(destination / f"manual-refit-{method}.json", refit.to_report(video))
        if refit is not None and refit.camera is not None:
            run["center_delta_m"] = float(np.linalg.norm(refit.camera.center - calibration.camera.center))
            run["focal_px"] = float(refit.camera.intrinsics.K[0, 0])
    dump(destination / "manual-landmarks.json", indexes)
    dump(destination / "landmark-comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/caltennis"))
    parser.add_argument("--output", type=Path, default=Path("outputs/court-calibration/caltennis"))
    parser.add_argument("--clip", help="run only the matching clip name prefix")
    args = parser.parse_args()
    annotations = json.loads(Path(__file__).with_name("caltennis-landmarks.json").read_text())
    for stem, points in annotations.items():
        if args.clip and not stem.startswith(args.clip):
            continue
        video = probe_video(args.data / "originals" / f"{stem}.mp4")
        destination = args.output / stem
        destination.mkdir(parents=True, exist_ok=True)
        result = validate(video, args.data / "calib" / f"{stem}.json", points, destination)
        print(stem, json.dumps(result["comparison"]), result["manual_fit"], flush=True)


if __name__ == "__main__":
    main()
