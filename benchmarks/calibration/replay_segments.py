"""Replay the CalTennis line matcher and evaluate two geometry changes."""

import argparse
from bisect import bisect_left
import hashlib
from itertools import combinations
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from snufspin.calibration import CourtMapping
from snufspin.court import DOUBLES_WIDTH, LANDMARKS, LENGTH, LINES
from snufspin.court_commands import DETECTOR, import_caltennis
from snufspin.court_fit import _fit_mapping, _physical_court
from snufspin.court_review import marking_support, markings_agree
from snufspin.letr import LETR
from snufspin.line_court import (
    _CROSS, _PAINT, _intersections, _line_hypotheses, _prepare_segments, refine_segments,
)
from snufspin.video import Video, read_sample_frames

from .estimate import CLIPS, TIMES_S, probe_sample_video

OUTPUT = Path("outputs/caltennis-segment-replay")
MANIFEST = Path(__file__).with_name("replay-frames.json")
VP_TOLERANCE = np.sin(np.deg2rad(4))
VP_PARTITIONS = 12


def replay(videos, output):
    detector = LETR(DETECTOR / "texflip", DETECTOR / "letr_best_checkpoint.pth", "cpu", 4)
    torch.manual_seed(0)
    cv2.setRNGSeed(0)
    manifest = json.loads(MANIFEST.read_text())
    observations = {}
    for label, stem in CLIPS.items():
        video = probe_sample_video(videos / f"{stem}.mp4")
        indexes = [bisect_left(video.timestamps_s, t) for t in TIMES_S]
        observations[label] = []
        for (index, frame), sample in zip(read_sample_frames(video, indexes), manifest[label], strict=True):
            digest = hashlib.sha256(frame).hexdigest()
            if (index, video.time_at(index), digest) != (sample["index"], sample["time_s"], sample["bgr_sha256"]):
                raise ValueError(f"{label} {index} differs from the original benchmark frame")
            raw = detector(frame)
            refined = refine_segments(frame, raw)
            path = output / f"{stem[:5]}-{index}.png"
            if not cv2.imwrite(str(path), frame):
                raise OSError(f"could not save {path}")
            observations[label].append(sample | {"raw": raw.tolist(), "refined": refined.tolist()})
            print(f"{label}, frame {index}: {len(raw)} proposals", flush=True)
    return observations


def vp_partitions(segments, lengths, equations):
    """Propose two pencils of lines by angular agreement with homogeneous VPs."""
    if len(segments) < 4:
        return
    points = np.cross(*np.moveaxis(equations[np.array(list(combinations(range(len(segments)), 2)))], 1, 0))
    # Center and scale the projective coordinates before measuring angular errors.
    center = segments.mean(axis=(0, 1))
    scale = np.ptp(segments.reshape(-1, 2), axis=0).max()
    if scale <= 0:
        return
    points[:, :2] = (points[:, :2] - points[:, 2:] * center) / scale
    points = points[np.linalg.norm(points, axis=1) > 1e-9]
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    midpoints = (segments.mean(axis=1) - center) / scale
    vectors = points[:, None, :2] - points[:, None, 2:] * midpoints
    sine_error = np.abs(np.einsum("pnd,nd->pn", vectors, equations[:, :2]))
    magnitude = np.linalg.norm(vectors, axis=2)
    sine_error = np.divide(sine_error, magnitude, out=np.ones_like(sine_error), where=magnitude > 1e-9)

    partitions = {}
    indexes = np.arange(len(segments))
    for a, b in combinations(range(len(points)), 2):
        first = sine_error[a] <= sine_error[b]
        best_error = np.minimum(sine_error[a], sine_error[b])
        included = best_error <= VP_TOLERANCE
        groups = (tuple(indexes[first & included][:12]), tuple(indexes[~first & included][:12]))
        if min(map(len, groups)) < 2:
            continue
        # Require that each assigned pencil has support distinct from the other pencil.
        difference = np.abs(sine_error[a] - sine_error[b])
        if min(np.count_nonzero(difference[list(group)] >= VP_TOLERANCE / 2) for group in groups) < 2:
            continue
        score = float(np.sum(np.sqrt(lengths) * np.maximum(0, 1 - best_error / VP_TOLERANCE)))
        key = tuple(sorted(groups))
        previous = partitions.get(key)
        if previous is None or score > previous[0]:
            vps = points[[a, b]].copy()
            vps[:, :2] = vps[:, :2] * scale + vps[:, 2:] * center
            partitions[key] = score, groups, vps

    for _, groups, vps in sorted(partitions.values(), key=lambda x: -x[0])[:VP_PARTITIONS]:
        yield np.array(groups[0]), np.array(groups[1]), vps
        yield np.array(groups[1]), np.array(groups[0]), vps[::-1]


def direct_mapping(matches, equations, width, height):
    """Fit H from endpoint-on-supporting-line constraints in normalized coordinates."""
    model_ids = list(matches)
    across = sum(list(LINES)[i] in _CROSS for i in model_ids)
    if min(across, len(matches) - across) < 2:
        return None

    image_normalizer = np.array([[1 / width, 0, -0.5], [0, 1 / width, -height / (2 * width)], [0, 0, 1]])
    court_scale = np.ptp(_PAINT[:, :, 0])
    center = (_PAINT.max(axis=(0, 1)) + _PAINT.min(axis=(0, 1))) / 2
    court_normalizer = np.eye(3)
    court_normalizer[:2, :2] /= court_scale
    court_normalizer[:2, 2] = -center / court_scale
    image_to_pixels = np.linalg.inv(image_normalizer)
    lines = equations[list(matches.values())] @ image_to_pixels
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    model = np.concatenate((_PAINT[model_ids], np.ones((len(matches), 2, 1))), axis=-1)
    model = model @ court_normalizer.T
    constraints = np.einsum("ni,nkj->nkij", lines, model).reshape(-1, 9)
    _, singular_values, basis = np.linalg.svd(constraints)
    if singular_values[7] < 1e-8 * singular_values[0]:
        return None
    H = image_to_pixels @ basis[-1].reshape(3, 3) @ court_normalizer
    if np.linalg.matrix_rank(H) != 3:
        return None
    mapping = CourtMapping(H)
    return mapping if _physical_court(mapping) else None


def fit_candidates(frame, prepared):
    height, width = frame.shape[:2]
    segments, lengths, equations, steep, shallow = prepared
    slope = (steep, shallow, None)
    vp = None
    for groups in vp_partitions(segments, lengths, equations):
        along, across, _ = groups
        candidates = _line_hypotheses(segments, equations, along, across, width, height, 1)
        if candidates and (vp is None or candidates[0][0] > vp[0][0]):
            vp = (candidates[0], groups)

    primary = _line_hypotheses(segments, equations, steep, shallow, width, height, 1)
    return {"slope": (primary[0] if primary else None, slope),
            "vp": vp or (None, (np.array([], dtype=int), np.array([], dtype=int), None))}


def fit_modes(frame, prepared, candidates, ids):
    height, width = frame.shape[:2]
    segments, lengths, equations, _, _ = prepared
    modes = {}
    for grouping, (candidate, groups) in candidates.items():
        if candidate is None:
            points, matches = {}, {}
        else:
            _, _, matches, _ = candidate
            points = _intersections(segments, lengths, equations, matches, width, height)
        xyz = np.array([LANDMARKS[name] for name in points], dtype=float).reshape(-1, 3)
        crossing = _fit_mapping(xyz, np.array(list(points.values())), None, max(3.0, width / 240))
        direct = direct_mapping(matches, equations, width, height)
        for label, mapping in (("crossings", crossing[0] if crossing else None), ("segments", direct)):
            support = marking_support(frame, mapping) if mapping else {}
            modes[f"{grouping}-{label}"] = {
                "matches": {list(LINES)[m]: int(ids[s]) for m, s in matches.items()},
                "crossings": points,
                "H": None if mapping is None else mapping.H.tolist(),
                "paint_gate": markings_agree(support),
                "paint_support": support,
            }
    return modes


def describe(frame, observation):
    height, width = frame.shape[:2]
    refined = np.asarray(observation["refined"]).reshape(-1, 2, 2)
    raw = np.asarray(observation["raw"]).reshape(-1, 2, 2)
    lengths = np.linalg.norm(refined[:, 1] - refined[:, 0], axis=1)
    order = np.argsort(-lengths, kind="stable")
    survives = lengths >= min(width, height) * 0.06
    ids = order[survives[order]] + 1
    prepared = _prepare_segments(refined, width, height)
    candidates = fit_candidates(frame, prepared)
    groups = {}
    for name, (_, (along, across, vps)) in candidates.items():
        groups[name] = {"first": ids[along].tolist(), "second": ids[across].tolist(),
                        "vanishing_points": None if vps is None else vps.tolist()}
    return {k: observation[k] for k in ("index", "time_s", "bgr_sha256")} | {
        "groups": groups,
        "segments": [
            {"id": i + 1, "raw": raw[i].tolist(), "refined": refined[i].tolist(),
             "length_px": float(length), "survives_length": bool(survives[i]),
             "angle_deg": float(np.degrees(np.arcsin(np.clip(
                 abs(refined[i, 1, 1] - refined[i, 0, 1]) / max(length, 1e-12), 0, 1))))}
            for i, length in enumerate(lengths)
        ],
        "modes": fit_modes(frame, prepared, candidates, ids),
    }


def score(results, references):
    """Measure projected court locations against the supplied camera after fitting."""
    xy = np.array(list(LANDMARKS.values()))[:, :2]
    for label, frames in results.items():
        stem = CLIPS[label]
        reference = import_caltennis(Video(Path(stem + ".mp4"), 1920, 1088, ()),
                                     references / (stem + ".json"), "resize").mapping
        references_by_orientation = [reference.project(xy), reference.project([LENGTH, DOUBLES_WIDTH] - xy)]
        for frame in frames:
            for mode in frame["modes"].values():
                if mode["H"] is None:
                    continue
                predicted = CourtMapping(mode["H"]).project(xy)
                scores = [
                    np.quantile(np.linalg.norm(predicted - ref, axis=1), [.5, .9])
                    for ref in references_by_orientation
                ]
                flipped = bool(scores[1][1] < scores[0][1])
                mode["reference_median_px"], mode["reference_p90_px"] = scores[int(flipped)].tolist()
                mode["reference_flipped_180"] = flipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=Path("data/caltennis/originals"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--references", type=Path, default=Path("data/caltennis/calib"))
    parser.add_argument("--frames-dir", type=Path,
                        help="reuse cached segments.json in --output and frame PNGs from this directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "segments.json"
    if args.frames_dir is None:
        observations = replay(args.videos, args.output)
        cache.write_text(json.dumps(observations, indent=2, allow_nan=False) + "\n")
        frames_dir = args.output
    else:
        observations = json.loads(cache.read_text())
        frames_dir = args.frames_dir

    cv2.setRNGSeed(0)
    manifest = json.loads(MANIFEST.read_text())
    results = {}
    for label, expected_frames in manifest.items():
        results[label] = []
        for observation, expected in zip(observations[label], expected_frames, strict=True):
            index = observation["index"]
            stem = CLIPS[label]
            frame = cv2.imread(str(frames_dir / f"{stem[:5]}-{index}.png"))
            if (any(observation[key] != value for key, value in expected.items()) or frame is None
                    or hashlib.sha256(frame).hexdigest() != expected["bgr_sha256"]):
                raise ValueError(f"{label} {index} differs from the original benchmark frame")
            results[label].append(describe(frame, observation))
    (args.output / "fits.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    score(results, args.references)
    (args.output / "replay.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
