"""Compare court detectors on identical samples and independently reviewed landmarks."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from snufspin.court_commands import DETECTOR, read_json, sample_indices
from snufspin.court_detector import TennisCourtDetector
from snufspin.court_fit import fit_court
from snufspin.court_review import markings_agree, write_overlay
from snufspin.letr import LETR
from snufspin.line_court import fit_lines, refine_segments
from snufspin.video import probe_video, read_sample_frames


def compare(video, frames, observations, reviewed, source, samples, output):
    output.mkdir(parents=True, exist_ok=True)
    calibration, diagnostics = fit_court(
        {index: frames[index] for index in samples},
        {index: observations[index] for index in samples},
        source,
    )
    errors = {}
    for index, labels in reviewed.items():
        errors[index] = {
            name: float(np.linalg.norm(np.subtract(point, observations[index][name])))
            for name, point in labels.items() if name in observations[index]
        }
    flat = [distance for row in errors.values() for distance in row.values()]
    diagnostics["observations"] = observations
    diagnostics["reviewed_errors_px"] = errors
    diagnostics["reviewed_count"] = sum(map(len, reviewed.values()))
    diagnostics["matched_count"] = len(flat)
    diagnostics["within_12_px"] = sum(distance <= 12 for distance in flat)
    diagnostics["matched_median_px"] = float(np.median(flat)) if flat else None
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    if calibration is not None:
        (output / "camera.json").write_text(json.dumps(calibration.to_report(video), indent=2) + "\n")
        for index in samples:
            write_overlay(output / f"court-{index:06}.jpg", frames[index], calibration.mapping,
                          observations[index], calibration.camera)
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--labels", type=Path, help="directory with <video stem>.landmarks.json files")
    parser.add_argument("--output", type=Path, default=Path("outputs/texflip-court-detection"))
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--raw-lines", action="store_true", help="compare without local paint refinement")
    args = parser.parse_args()

    heatmap = TennisCourtDetector(DETECTOR / "TennisCourtDetector/tracknet.py",
                                  DETECTOR / "model.pth", "cpu", args.threads)
    letr = LETR(DETECTOR / "texflip", DETECTOR / "letr_best_checkpoint.pth", "cpu", args.threads)
    summary = {}
    for path in args.videos:
        video = probe_video(path)
        label_path = args.labels / f"{path.stem}.landmarks.json" if args.labels else None
        reviewed = {int(i): row for i, row in read_json(label_path).items()} if label_path else {}
        samples = sample_indices(video, args.samples)
        frames = dict(read_sample_frames(video, sorted(set(samples) | reviewed.keys())))
        results = summary[path.stem] = {}
        for name, detector in (("heatmap", heatmap), ("texflip", letr)):
            out = args.output / path.stem / name
            out.mkdir(parents=True, exist_ok=True)
            observations = {}
            timings = []
            segments = {}
            for index, frame in frames.items():
                start = perf_counter()
                evidence = detector(frame)
                if name == "texflip":
                    segments[index] = evidence.tolist()
                    if not args.raw_lines:
                        evidence = refine_segments(frame, evidence)
                    evidence_for_fit = fit_lines(evidence, video.width, video.height)
                else:
                    evidence_for_fit = evidence
                observations[index] = evidence_for_fit
                timings.append(perf_counter() - start)
                if index in reviewed:
                    canvas = frame.copy()
                    if name == "texflip":
                        for a, b in np.rint(evidence).astype(int):
                            cv2.line(canvas, a, b, (230, 190, 20), 1, cv2.LINE_AA)
                    for points, color in ((reviewed[index], (80, 245, 65)),
                                          (observations[index], (55, 128, 255))):
                        for point in points.values():
                            cv2.drawMarker(canvas, np.rint(point).astype(int), color, cv2.MARKER_CROSS, 15, 2)
                    cv2.imwrite(str(out / f"labels-{index:06}.jpg"), canvas)
            if segments:
                (out / "segments.json").write_text(json.dumps(segments, indent=2) + "\n")
            diagnostics = compare(video, frames, observations, reviewed, detector.source, samples, out)
            results[name] = {key: diagnostics[key] for key in (
                "status", "reviewed_count", "matched_count", "within_12_px", "matched_median_px",
            )}
            results[name]["observed_counts"] = [len(observations[index]) for index in samples]
            results[name]["marking_supported_frames"] = [
                int(index) for index, frame in diagnostics["frames"].items()
                if "line_support" in frame and markings_agree(frame["line_support"])
            ]
            results[name]["median_detection_time_s"] = float(np.median(timings))
            print(path.stem, name, results[name], flush=True)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
