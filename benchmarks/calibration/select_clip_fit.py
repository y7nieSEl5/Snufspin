"""Test selecting one unchanged CalTennis homography from several frames."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from snufspin.calibration import CourtMapping
from snufspin.court import DOUBLES_WIDTH, LANDMARKS, LENGTH
from snufspin.court_commands import import_caltennis
from snufspin.court_paint import score_mapping
from snufspin.video import Video

from .estimate import CLIPS
from .transfer_fits import OUTPUT, REPLAY, load_replay

ANCHORS = np.array(list(LANDMARKS.values()))[:, :2]


def candidates(fits, source_indices):
    """Yield each distinct existing hypothesis, preserving its generating frame."""
    seen = []
    for index in source_indices:
        for mode, fit in fits[index]["modes"].items():
            if fit["H"] is None:
                continue
            mapping = CourtMapping(fit["H"])
            projected = mapping.project(ANCHORS)
            if any(np.max(np.linalg.norm(projected - earlier, axis=1)) < .01 for earlier in seen):
                continue
            seen.append(projected)
            yield index, mode, mapping


def rank(fits, gray_frames, train):
    """Rank by the existing paint score, requiring support in a majority of frames."""
    frames = {index: gray_frames[index] for index in train}
    required = max(2, len(train) // 2 + 1)
    entries = []
    for source, mode, mapping in candidates(fits, train):
        paint = score_mapping(mapping, frames)
        supported_frames = sum(row["supported"] for row in paint.frames.values())
        entries.append({
            "source": source,
            "mode": mode,
            "H": mapping.H.tolist(),
            "paint_score": paint.score,
            "supported_frames": supported_frames,
            "eligible": supported_frames >= required,
            "training_paint": paint.frames,
        })
    return sorted(entries, key=lambda row: (not row["eligible"], -row["paint_score"]))


def ambiguity(ranking, frame_count, width):
    """Apply the production paint selector's margin to the experimental candidates."""
    eligible = [entry for entry in ranking if entry["eligible"]]
    if not eligible:
        return None
    top, *others = eligible
    projected = CourtMapping(top["H"]).project(ANCHORS)
    for other in others:
        alternative = CourtMapping(other["H"]).project(ANCHORS)
        spread = np.quantile(np.linalg.norm(projected - alternative, axis=1), .9)
        if spread > max(6.0, width / 120):
            margin = top["paint_score"] - other["paint_score"]
            return {
                "source": other["source"],
                "mode": other["mode"],
                "spread_p90_px": float(spread),
                "score_margin": margin,
                "ambiguous": margin <= max(2 * frame_count, .02 * top["paint_score"]),
            }
    return None


def assess(result, target, gray_frames, reference):
    winner = next((entry for entry in result["ranking"] if entry["eligible"]), None)
    if winner is None:
        return None

    mapping = CourtMapping(winner["H"])
    projected = mapping.project(ANCHORS)
    discrepancy = min(
        np.quantile(np.linalg.norm(projected - reference.project(xy), axis=1), .9)
        for xy in (ANCHORS, [LENGTH, DOUBLES_WIDTH] - ANCHORS)
    )
    return {
        "source": winner["source"],
        "mode": winner["mode"],
        "reference_p90_px": float(discrepancy),
        "accepted_by_margin": not (result["alternative"] and result["alternative"]["ambiguous"]),
        "held_out_paint": score_mapping(mapping, {target: gray_frames[target]}).frames[target]
        if target is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=REPLAY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--references", type=Path, default=Path("data/caltennis/calib"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fits, frames = load_replay(args.replay)
    gray_frames = {
        label: [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in images]
        for label, images in frames.items()
    }

    report = {}
    for label, gray in gray_frames.items():
        all_frames = tuple(range(len(gray)))
        report[label] = {}
        for held_out in (None, *all_frames):
            train = [i for i in all_frames if i != held_out]
            key = "all" if held_out is None else str(fits[label][held_out]["index"])
            ranking = rank(fits[label], gray, train)
            report[label][key] = {
                "held_out": held_out,
                "ranking": ranking,
                "alternative": ambiguity(ranking, len(train), gray[0].shape[1]),
            }
    (args.output / "selection-blind.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    for label, splits in report.items():
        stem = CLIPS[label]
        height, width = gray_frames[label][0].shape
        reference = import_caltennis(
            Video(Path(stem + ".mp4"), width, height, ()),
            args.references / (stem + ".json"), "resize",
        ).mapping
        for result in splits.values():
            result["selected"] = assess(result, result["held_out"], gray_frames[label], reference)
    (args.output / "selection.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
