"""Inspect candidate extraction on reviewed frames using the existing checkpoint."""

import argparse
from collections import deque
from importlib.util import module_from_spec, spec_from_file_location
import json
from math import dist
from pathlib import Path

from .evaluation import evaluate_review
from .tracking import file_sha256
from .video import probe_video


def diagnose(video, report, review, model_source, weights):
    import cv2
    import numpy as np
    import torch

    evaluate_review(video, report, review)
    spec = spec_from_file_location("tennis_vision_tracknet", model_source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.BallTrackerNet(input_channels=9, out_channels=256)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()
    torch.set_num_threads(4)
    labels = {label["frame"]: label for label in review["corrections"]}
    points = {point["frame"]: point for point in report["ball_track"]}
    scale = np.array([video.width / 640, video.height / 360])
    results = []
    window = deque(maxlen=3)
    camera = cv2.VideoCapture(str(video.path))
    # Match Snufspin's encoded-pixel coordinate convention.
    camera.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    try:
        for frame in range(max(labels) + 1):
            ok, image = camera.read()
            if not ok:
                raise ValueError(f"could not decode frame {frame}")
            window.appendleft(cv2.resize(image, (640, 360)))
            if frame not in labels:
                continue
            label = labels[frame]
            row = {"frame": frame, "label": label["status"]}
            if frame < 2:
                results.append(row | {"cause": "two_frame_warmup"})
                continue
            stack = np.concatenate(tuple(window), axis=2).astype(np.float32) / 255
            tensor = torch.from_numpy(stack.transpose(2, 0, 1)).unsqueeze(0)
            with torch.inference_mode():
                heat = model(tensor).argmax(dim=1).reshape(360, 640).numpy()
            # Reproduce the pinned tennis-vision notebook's conversion exactly.
            mask = cv2.threshold((heat * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)[1]
            circles = cv2.HoughCircles(
                mask, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50,
                param2=2, minRadius=2, maxRadius=7,
            )
            candidates = [] if circles is None else (circles[0, :, :2] * scale).tolist()
            previous = points.get(frame - 1)
            accepted = candidates
            if previous and previous["x_px"]:
                previous_xy = [previous["x_px"], previous["y_px"]]
                maximum_step = 80 * float(max(scale)) / 2
                accepted = [xy for xy in candidates if dist(xy, previous_xy) < maximum_step]
            predicted = accepted[0] if accepted else None
            recorded = points.get(frame)
            recorded_xy = [recorded["x_px"], recorded["y_px"]] if recorded else None
            if (predicted is None) != (recorded_xy is None) or (
                predicted is not None and not np.allclose(predicted, recorded_xy)
            ):
                raise ValueError(f"checkpoint/model did not reproduce baseline at frame {frame}")

            count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
            components = [
                (centroids[index] * scale).tolist()
                for index in range(1, count)
                if 1 <= stats[index, cv2.CC_STAT_AREA] <= 200
            ]
            cause = "candidate_selected"
            if not candidates:
                cause = "no_hough_candidate"
            elif predicted is None:
                cause = "distance_gate_rejected_all"
            row.update(cause=cause, circles=candidates, component_candidates=components)

            if label["status"] == "visible":
                truth = [label["x_px"], label["y_px"]]
                row["recorded_error_px"] = dist(predicted, truth) if predicted else None
                row["closest_circle_error_px"] = min(
                    (dist(xy, truth) for xy in candidates), default=None,
                )
                earlier = sorted(index for index in points if index < frame)[-2:]
                if components and earlier:
                    last = points[earlier[-1]]
                    expected = [last["x_px"], last["y_px"]]
                    if len(earlier) == 2:
                        before = points[earlier[0]]
                        interval = video.time_at(last["frame"]) - video.time_at(before["frame"])
                        factor = (video.time_at(frame) - video.time_at(last["frame"])) / interval
                        expected = [last[key] + factor * (last[key] - before[key]) for key in ("x_px", "y_px")]
                    chosen = min(components, key=lambda xy: dist(xy, expected))
                    row["component_selected_error_px"] = dist(chosen, truth)
            results.append(row)
    finally:
        camera.release()
    return {
        "video_sha256": report["video_sha256"],
        "weights_sha256": file_sha256(weights),
        "model_sha256": file_sha256(model_source),
        "scope": "isolated_reviewed_frames_with_cached_baseline_history",
        "frames": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--ball-track", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(
        probe_video(args.video), json.loads(args.ball_track.read_text()),
        json.loads(args.review.read_text()), args.model_source, args.weights,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
