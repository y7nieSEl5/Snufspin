"""Replay cached LETR segments to compare fixed-camera court fitting methods."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from snufspin.calibration import Calibration, CourtMapping
from snufspin.court import LANDMARKS, LINES
from snufspin.court_commands import read_json, sample_indices
from snufspin.court_fit import fit_court
from snufspin.court_paint import measured_landmarks, score_mapping, select_paint_mapping
from snufspin.line_court import fit_lines, refine_segments
from snufspin.video import probe_video, read_sample_frames

_ANCHORS = np.array(list(LANDMARKS.values()))[:, :2]


def dump(path, result):
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def mapping_errors(mapping, reviewed):
    per_marking = {}
    errors = []
    for row in reviewed.values():
        for name, point in row.items():
            error = float(np.linalg.norm(mapping.project(LANDMARKS[name][:2]) - point))
            per_marking.setdefault(name, []).append(error)
            errors.append(error)
    return {
        "count": len(errors), "median_px": float(np.median(errors)), "p90_px": float(np.quantile(errors, 0.9)),
        "by_landmark_px": per_marking,
    } if errors else {}


def write_paint_overlay(path, image, candidate, frame_info, labels):
    overlay = image.copy()
    height, width = image.shape[:2]
    for name, ends in LINES.items():
        xy = [LANDMARKS[n][:2] for n in ends]
        start, end = np.rint(candidate.mapping.project(xy)).astype(int)
        visible, start, end = cv2.clipLine((0, 0, width, height), start, end)
        if not visible:
            continue
        support = frame_info["markings"][name]["support"]
        if support is not None and support >= 0.50:
            cv2.line(overlay, start, end, (255, 230, 0), 2, cv2.LINE_AA)
        else:
            delta = np.subtract(end, start)
            length = np.linalg.norm(delta)
            for offset in np.arange(0, length, 18):
                a = np.rint(start + delta * offset / length).astype(int)
                b = np.rint(start + delta * min(offset + 10, length) / length).astype(int)
                cv2.line(overlay, a, b, (100, 140, 255), 2, cv2.LINE_AA)
    for point in labels.values():
        uv = np.rint(point).astype(int)
        cv2.drawMarker(overlay, uv, (60, 255, 80), cv2.MARKER_CROSS, 14, 1)
    if not cv2.imwrite(str(path), overlay):
        raise ValueError(f"could not write overlay to {path}")


def emit_result(destination, video, frames, gray, reviewed, calibration, diagnostics):
    destination.mkdir(parents=True, exist_ok=True)
    if calibration is not None:
        scored = score_mapping(calibration.mapping, gray)
        diagnostics["paint_score"] = scored.score
        diagnostics["paint_frames"] = scored.frames
        diagnostics["labeled_projection_errors"] = mapping_errors(calibration.mapping, reviewed)
        dump(destination / "camera.json", calibration.to_report(video))
        for index, image in frames.items():
            frame_info = scored.frames.get(index)
            if frame_info is None:
                frame_info = score_mapping(calibration.mapping, {index: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)}).frames[index]
            write_paint_overlay(destination / f"frame-{index:06}.jpg", image, scored,
                                frame_info, reviewed.get(index, {}))
    dump(destination / "diagnostics.json", diagnostics)


def check_camera(candidate, frames, refined_segments, video, source):
    observations = measured_landmarks(candidate, refined_segments, video.width, video.height)
    calibration, diagnostics = fit_court(frames, observations, source)
    camera = None if calibration is None else calibration.camera
    if camera is not None:
        pose_mapping = CourtMapping.from_camera(camera)
        spread = np.quantile(np.linalg.norm(pose_mapping.project(_ANCHORS) -
                                            candidate.mapping.project(_ANCHORS), axis=1), 0.9)
        diagnostics["p90_from_paint_mapping_px"] = float(spread)
        if spread > max(6, video.width / 120):
            diagnostics["pose_status"] = "camera_disagrees_with_paint_mapping"
            camera = None
    return Calibration(candidate.mapping, source, camera), diagnostics


def run(video, segments_path, label_path, output, selection_only):
    samples = sample_indices(video, 7)
    raw_segments = read_json(segments_path)
    reviewed = {int(i): row for i, row in read_json(label_path).items()}
    frames = dict(read_sample_frames(video, sorted(set(samples) | reviewed.keys())))
    sampled = {index: frames[index] for index in samples}
    segments = {index: refine_segments(image, np.array(raw_segments[str(index)]).reshape(-1, 2, 2))
                for index, image in sampled.items()}
    start = perf_counter()
    winner, gray, selection = select_paint_mapping(sampled, segments)
    selection["selection_s"] = perf_counter() - start
    source = {"method": "TEXflip LETR, cached segments", "cache": str(segments_path)}
    baseline, current = fit_court(sampled,
                                  {i: fit_lines(rows, video.width, video.height) for i, rows in segments.items()},
                                  {**source, "fit": "segment alignment"})
    emit_result(output / "current", video, frames, gray, reviewed, baseline, current)
    if winner is None and "winning_H" in selection:
        selection["provisional_winner_errors"] = mapping_errors(CourtMapping(selection["winning_H"]), reviewed)
        selection["provisional_runner_errors"] = mapping_errors(CourtMapping(selection["alternative"]["H"]), reviewed)
    variants = {"current": current, "paint": selection}
    candidates = {"paint": winner}
    if not selection_only:
        from snufspin.court_paint import refine_paint_mapping

        refined, refinement = refine_paint_mapping(winner, gray) if winner else (None, {"status": "no_seed"})
        if winner is None and baseline is not None:
            control, validation = refine_paint_mapping(score_mapping(baseline.mapping, gray), gray)
            validation["projection_errors"] = mapping_errors(control.mapping, reviewed)
            refinement["existing_mapping_control"] = validation
        variants["refined"] = refinement
        candidates["refined"] = refined
    for name, candidate in candidates.items():
        diagnostics = variants[name]
        calibration = None
        if candidate is not None:
            calibration, pose = check_camera(candidate, sampled, segments, video, {**source, "fit": name})
            diagnostics["camera_checks"] = pose
            diagnostics["status"] = "camera" if calibration.camera else "court_mapping_only"
        emit_result(output / name, video, frames, gray, reviewed, calibration, diagnostics)

    for name, result in variants.items():
        print(video.path.stem, name, result["status"], result.get("paint_score"),
              result.get("labeled_projection_errors", {}).get("median_px"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--cache", type=Path, default=Path("outputs/texflip-court-detection/refined"))
    parser.add_argument("--labels", type=Path, default=Path("outputs/texflip-court-detection/labels"))
    parser.add_argument("--output", type=Path, default=Path("outputs/farin-court-paint"))
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()
    for path in args.videos:
        run(probe_video(path), args.cache / path.stem / "texflip/segments.json",
            args.labels / f"{path.stem}.landmarks.json", args.output / path.stem, args.selection_only)


if __name__ == "__main__":
    main()
