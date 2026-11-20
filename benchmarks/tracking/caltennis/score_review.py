"""Score frozen WASB candidates using human CalTennis review labels."""
import argparse
from collections import Counter
import csv
import json
from math import hypot
from pathlib import Path


def score(log, labels):
    frames = {row["frame"]: row for row in log["frames"]}
    counts = {limit: Counter() for limit in (5, 10)}
    visibility = Counter()
    hidden_with_position = Counter()
    details = []
    hand_frames = set()
    seen = set()
    for label in labels:
        status = label["status"].strip()
        hand = label["winner_on_hand"].strip().lower()
        if not status and not hand:
            continue
        if status not in {"", "visible", "occluded", "out_of_frame", "uncertain"} or hand not in {"", "yes", "no"}:
            raise ValueError("unknown visibility or hand label")
        frame = int(label["frame"])
        if frame in seen or label["video_sha256"] != log["video_sha256"]:
            raise ValueError("duplicate frame or label for a different clip")
        seen.add(frame)
        rec = frames[frame]
        if abs(float(label["timestamp_s"]) - rec["timestamp_s"]) > 1e-6:
            raise ValueError(f"label timestamp differs at frame {frame}")
        selected = rec["selected_index"]
        candidates = rec["candidates"]
        if hand == "yes":
            if selected is None:
                raise ValueError(f"frame {frame} has no winner to mark as hand")
            hand_frames.add(frame)
        row = {"frame": frame, "status": status, "winner_on_hand": hand, "selected_index": selected}
        if status:
            visibility[status] += 1
        if status in {"occluded", "out_of_frame"}:
            hidden_with_position[status] += selected is not None
        if status == "visible":
            x, y = float(label["x_px"]), float(label["y_px"])
            if not (0 <= x < log["width"] and 0 <= y < log["height"]):
                raise ValueError(f"ball center outside the image at frame {frame}")
            distances = [hypot(c["x_px"] - x, c["y_px"] - y) for c in candidates]
            row["selected_error_px"] = distances[selected] if selected is not None else None
            row["thresholds"] = {}
            for limit, tally in counts.items():
                near = [i for i, distance in enumerate(distances) if distance <= limit]
                allowed = [i for i in near if candidates[i]["passed_gate"]]
                if selected in near:
                    outcome = "correct_candidate_won"
                elif allowed:
                    outcome = "correct_candidate_lost_on_score"
                elif near:
                    outcome = "correct_candidate_gated_out"
                else:
                    outcome = "no_correct_candidate"
                tally[outcome] += 1
                row["thresholds"][limit] = {
                    "outcome": outcome, "near": near, "passed_gate": allowed,
                }
        details.append(row)

    episodes = []
    for frame in sorted(hand_frames):
        if episodes and frame == episodes[-1]["last_frame"] + 1:
            episodes[-1]["last_frame"] = frame
        else:
            episodes.append({"first_frame": frame, "last_frame": frame})
    return {
        "video_sha256": log["video_sha256"],
        "visibility": visibility,
        "hidden_with_position": hidden_with_position,
        "visible_candidate_outcomes": counts,
        "confirmed_hand_frames": len(hand_frames),
        "contiguous_reviewed_hand_episodes": episodes,
        "frames": details,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("labels", type=Path)
    args = parser.parse_args()
    log = json.loads(args.candidates.read_text(encoding="utf-8"))
    with args.labels.open(newline="", encoding="utf-8-sig") as stream:
        result = score(log, csv.DictReader(stream))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
