import argparse
import json
import subprocess
from pathlib import Path

from .evaluation import evaluate_review
from .overlay import render_overlay
from .pipeline import MetricPoint, MetricTrack, Shot, ShotLabels, analyze
from .repair import interpolate_short_gaps
from .tennis_vision import import_ball_track
from .tracking import BallPoint, read_ball_track
from .tracknet import run_tracknet
from .wasb import WasbCandidateLog, run_wasb
from .wasb_motion import MotionSettings, replay_wasb
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
    evaluation = commands.add_parser("evaluate-track", help="evaluate ball positions on reviewed frames")
    evaluation.add_argument("video", type=Path)
    evaluation.add_argument("--ball-track", type=Path, required=True)
    evaluation.add_argument("--review", type=Path, required=True)
    repair = commands.add_parser("interpolate-track", help="experimentally fill short bracketed gaps")
    repair.add_argument("video", type=Path)
    repair.add_argument("--ball-track", type=Path, required=True)
    repair.add_argument("--max-gap-s", type=float, default=0.15)
    tracking = commands.add_parser("track", help="run the ball detector locally from Python")
    tracking.add_argument("video", type=Path)
    tracking.add_argument("--model-source", type=Path, required=True)
    tracking.add_argument("--weights", type=Path, required=True)
    tracking.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    tracking.add_argument("--threads", type=int, default=4)
    wasb = commands.add_parser("track-wasb", help="run the official WASB Tennis checkpoint")
    wasb.add_argument("video", type=Path)
    wasb.add_argument("--upstream", type=Path, required=True)
    wasb.add_argument("--weights", type=Path, required=True)
    wasb.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    wasb.add_argument("--threads", type=int, default=4)
    wasb.add_argument("--stride", choices=(1, 3), type=int, default=1)
    wasb.add_argument("--candidate-log", type=Path, help="write diagnostic candidate JSON to a new file")
    wasb.add_argument("--candidate-frames", nargs=2, type=int, metavar=("FIRST", "LAST"),
                      help="inclusive source frame range for --candidate-log")
    replay = commands.add_parser("replay-wasb", help="check cached WASB candidates against surrounding motion")
    replay.add_argument("video", type=Path)
    replay.add_argument("--ball-track", type=Path, required=True)
    replay.add_argument("--candidate-log", type=Path, required=True)
    replay.add_argument("--mode", choices=("reject", "rerank"), required=True)
    replay.add_argument("--mask-radius", type=int, default=MotionSettings.mask_radius)
    replay.add_argument("--side-frames", type=int, default=MotionSettings.side_frames)
    replay.add_argument("--tolerance-heatmap-px", type=float, default=MotionSettings.tolerance_heatmap_px)
    replay.add_argument("--max-extrapolation-s", type=float, default=MotionSettings.max_extrapolation_s)
    replay.add_argument("--decision-log", type=Path, help="write per-frame decisions to a new JSON file")
    args = parser.parse_args()

    try:
        video = probe_video(args.video)
        if args.command == "track":
            report = run_tracknet(video, args.model_source, args.weights, args.device, args.threads)
            print(json.dumps(report, indent=2, allow_nan=False))
            return
        if args.command == "track-wasb":
            if (args.candidate_log is None) != (args.candidate_frames is None):
                parser.error("--candidate-log and --candidate-frames must be provided together")
            diagnostic = None if args.candidate_log is None else WasbCandidateLog(*args.candidate_frames)
            if diagnostic is not None and args.candidate_log.exists():
                raise ValueError(f"candidate log already exists: {args.candidate_log}")
            report = run_wasb(
                video, args.upstream, args.weights, args.device, args.threads, args.stride,
                candidate_log=diagnostic,
            )
            if diagnostic is not None:
                log = {key: value for key, value in report.items() if key != "ball_track"}
                log["frames"] = [
                    {**frame, "timestamp_s": video.time_at(frame["frame"])} for frame in diagnostic.frames
                ]
                with args.candidate_log.open("x", encoding="utf-8") as stream:
                    json.dump(log, stream, indent=2, allow_nan=False)
                    stream.write("\n")
            print(json.dumps(report, indent=2, allow_nan=False))
            return
        if args.command == "interpolate-track":
            report = json.loads(args.ball_track.read_text(encoding="utf-8"))
            result = interpolate_short_gaps(video, report, args.max_gap_s)
            print(json.dumps(result, indent=2, allow_nan=False))
            return
        if args.command == "replay-wasb":
            baseline = json.loads(args.ball_track.read_text(encoding="utf-8"))
            candidates = json.loads(args.candidate_log.read_text(encoding="utf-8"))
            settings = MotionSettings(
                args.mask_radius, args.side_frames, args.tolerance_heatmap_px, args.max_extrapolation_s,
            )
            report, decisions = replay_wasb(video, baseline, candidates, args.mode, settings)
            if args.decision_log is not None:
                with args.decision_log.open("x", encoding="utf-8") as stream:
                    json.dump({
                        "video_sha256": report["video_sha256"],
                        "source": report["source"],
                        "frames": decisions,
                    }, stream, indent=2, allow_nan=False)
                    stream.write("\n")
            print(json.dumps(report, indent=2, allow_nan=False))
            return
        if args.command == "evaluate-track":
            report = json.loads(args.ball_track.read_text(encoding="utf-8"))
            review = json.loads(args.review.read_text(encoding="utf-8"))
            print(json.dumps(evaluate_review(video, report, review), indent=2, allow_nan=False))
            return
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
    except (OSError, AttributeError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.exit(1, f"snufspin: {error}\n")


if __name__ == "__main__":
    main()
