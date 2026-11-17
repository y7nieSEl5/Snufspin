import argparse
import json
import subprocess
from pathlib import Path

from .overlay import render_overlay
from .pipeline import MetricPoint, MetricTrack, Shot, ShotLabels, analyze
from .tennis_vision import import_ball_track
from .tracking import BallPoint, read_ball_track
from .video import probe_video


def read_shot(data: dict) -> Shot:
    fields = dict(data)
    labels = fields.pop("labels", None)
    if labels is not None:
        labels = dict(labels)
        spin = labels.get("spin")
        if spin is not None:
            if not isinstance(spin, list) or not all(isinstance(tag, str) for tag in spin):
                raise ValueError("spin must be an array of labels or null")
            labels["spin"] = tuple(spin)
        labels = ShotLabels(**labels)

    track = fields.pop("metric_track", None)
    if track is not None:
        track = MetricTrack(track["method"], tuple(MetricPoint(**point) for point in track["samples"]))
    return Shot(**fields, labels=labels, metric_track=track)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python3 -m snufspin")
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe", help="show frame indexes and timestamps")
    probe.add_argument("video", type=Path)
    review = commands.add_parser("analyze", help="produce per-shot labels and speed summaries")
    review.add_argument("video", type=Path)
    review.add_argument("--observations", type=Path, required=True)
    review.add_argument("--ball-track", type=Path, help="use a cached ball track with these shot observations")
    overlay = commands.add_parser("render", help="draw a cached ball track onto the video")
    overlay.add_argument("video", type=Path)
    overlay.add_argument("--ball-track", type=Path, required=True)
    overlay.add_argument("--output", type=Path, required=True)
    upstream = commands.add_parser("import-tennis-vision", help="convert a tennis-vision upstream.npz ball track")
    upstream.add_argument("video", type=Path)
    upstream.add_argument("artifact", type=Path)
    args = parser.parse_args()

    try:
        video = probe_video(args.video)
        if args.command == "import-tennis-vision":
            print(json.dumps(import_ball_track(video, args.artifact), indent=2, allow_nan=False))
            return
        if args.command == "render":
            report = json.loads(args.ball_track.read_text(encoding="utf-8"))
            render_overlay(video, read_ball_track(video, report), args.output)
            return
        output = {
            "video": str(args.video),
            "width": video.width,
            "height": video.height,
            "frame_count": len(video.timestamps_s),
        }
        if args.command == "probe":
            output["timestamps_s"] = video.timestamps_s
        else:
            observations = json.loads(args.observations.read_text(encoding="utf-8"))
            if args.ball_track is None:
                points = tuple(BallPoint(**point) for point in observations.get("ball_track", []))
            else:
                report = json.loads(args.ball_track.read_text(encoding="utf-8"))
                points = read_ball_track(video, report)
            shots = tuple(read_shot(shot) for shot in observations["shots"])
            output["source"] = observations.get("source")
            output["shots"] = analyze(video, points, shots)
        print(json.dumps(output, indent=2, allow_nan=False))
    except subprocess.CalledProcessError as error:
        parser.exit(1, f"{error.cmd[0]} could not process {args.video}: {error.stderr.strip()}\n")
    except (OSError, AttributeError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f"snufspin: {error}\n")


if __name__ == "__main__":
    main()
