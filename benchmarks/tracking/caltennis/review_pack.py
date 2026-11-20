"""Prepare clean and candidate-overlay review reels from a frozen WASB run."""
import argparse
from contextlib import closing
import csv
import json
from math import dist
from pathlib import Path
import subprocess

from snufspin.tracking import read_ball_track
from snufspin.video import decode_bgr_frames, probe_video


def mine_windows(video, points, controls):
    """Rank jumps and slowly moving runs for review, without labeling them."""
    points = {point.frame: (point.x_px, point.y_px) for point in points}
    diagonal = dist((0, 0), (video.width, video.height))
    candidates = []
    for frame, xy in points.items():
        previous = points.get(frame - 1)
        if previous:
            jump = dist(xy, previous) / diagonal
            if jump >= 0.025:
                candidates.append((jump, frame, "jump"))
    for frame, xy in points.items():
        tail = [points.get(index) for index in range(frame + 1, frame + 16)]
        if all(point is not None and dist(point, xy) <= 0.01 * diagonal for point in tail):
            candidates.append((0.03, frame + 8, "slow_run"))

    occupied = {index for first, last, _ in controls for index in range(first, last + 1)}
    result = []
    remaining = {"jump": 2, "slow_run": 2}
    for _, frame, cue in sorted(candidates, key=lambda candidate: (-candidate[0], candidate[1])):
        if not remaining[cue]:
            continue
        first, last = max(0, frame - 20), min(len(video.timestamps_s) - 1, frame + 20)
        indices = set(range(first, last + 1))
        if indices.isdisjoint(occupied):
            occupied.update(indices)
            result.append((first, last, cue))
            remaining[cue] -= 1
            if not any(remaining.values()):
                break
    return result


def make_reels(clip, track, log, directory, first_source_frame, controls):
    import cv2
    import numpy as np

    video = probe_video(clip)
    points = read_ball_track(video, track)
    if log["video_sha256"] != track["video_sha256"] or log["timestamps_s"] != track["timestamps_s"]:
        raise ValueError("candidate log does not describe the supplied clip")
    records = {event["frame"]: event for event in log["frames"]}
    if set(records) != set(range(len(video.timestamps_s))):
        raise ValueError("review requires a candidate log for every clip frame")
    controls = sorted(controls)
    if any(not 0 <= first <= last < len(video.timestamps_s) for first, last, _ in controls):
        raise ValueError("a control window is outside the clip")
    if any(first <= prev_last for (_, prev_last, _), (first, _, _) in zip(controls, controls[1:])):
        raise ValueError("control windows must not overlap")
    windows = sorted(controls + mine_windows(video, points, controls))
    frames = {frame: tag for first, last, tag in windows for frame in range(first, last + 1)}
    directory.mkdir(parents=True, exist_ok=True)
    plain = directory / "review-clean.mp4"
    marked = directory / "review-candidates.mp4"

    def encoder(output):
        return subprocess.Popen([
            "ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{video.width}x{video.height + 48}", "-framerate", "15", "-i", "-", "-an",
            "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output),
        ], stdin=subprocess.PIPE)

    with encoder(plain) as clean_video, encoder(marked) as candidate_video:
        with closing(decode_bgr_frames(video)) as decoded:
            for frame, pixels in enumerate(decoded):
                if frame not in frames:
                    continue
                image = np.frombuffer(pixels, dtype=np.uint8).reshape(video.height, video.width, 3)
                visible = cv2.copyMakeBorder(image, 0, 48, 0, 0, cv2.BORDER_CONSTANT)
                caption = f"Clip frame {frame}  |  Source frame {first_source_frame + frame}  |  {video.time_at(frame):.3f} s"
                cv2.putText(visible, caption, (12, video.height + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                clean_video.stdin.write(visible.tobytes())
                rec = records[frame]
                for index, candidate in enumerate(rec["candidates"]):
                    center = (round(candidate["x_px"]), round(candidate["y_px"]))
                    selected = index == rec["selected_index"]
                    color = (0, 240, 255) if selected else (
                        (255, 245, 25) if candidate["passed_gate"] else (220, 75, 245)
                    )
                    cv2.circle(visible, center, 10 if selected else 6, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.circle(visible, center, 10 if selected else 6, color, 2, cv2.LINE_AA)
                    offset = (center[0] + 12, center[1] - 15 + 15 * (index % 4))
                    cv2.putText(visible, str(index), offset, cv2.FONT_HERSHEY_SIMPLEX,
                                0.51, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(visible, str(index), offset, cv2.FONT_HERSHEY_SIMPLEX,
                                0.51, color, 1, cv2.LINE_AA)
                candidate_video.stdin.write(visible.tobytes())
        clean_video.stdin.close()
        candidate_video.stdin.close()
        if any(code != 0 for code in (clean_video.wait(), candidate_video.wait())):
            raise RuntimeError("failed to encode review reels")
    if any(len(probe_video(reel).timestamps_s) != len(frames) for reel in (plain, marked)):
        raise ValueError("review reel frame count changed")

    with (directory / "labels.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["video_sha256", "frame", "source_frame", "timestamp_s", "window",
                         "status", "x_px", "y_px", "winner_on_hand", "notes"])
        for frame, tag in frames.items():
            writer.writerow([log["video_sha256"], frame, first_source_frame + frame,
                             video.time_at(frame), tag, "", "", "", "", ""])
    (directory / "windows.json").write_text(json.dumps(windows, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path)
    parser.add_argument("track", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--first-source-frame", required=True, type=int)
    parser.add_argument("--control", metavar=("FIRST", "LAST"), nargs=2, type=int, action="append", required=True)
    args = parser.parse_args()
    track = json.loads(args.track.read_text(encoding="utf-8"))
    log = json.loads(args.candidates.read_text(encoding="utf-8"))
    controls = [(first, last, "control") for first, last in args.control]
    make_reels(args.clip, track, log, args.directory, args.first_source_frame, controls)


if __name__ == "__main__":
    main()
