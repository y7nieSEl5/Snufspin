"""Extract a lossless CalTennis excerpt and record its frame mapping."""
import argparse
import json
from math import isfinite
from pathlib import Path
import subprocess

from snufspin.tracking import file_sha256
from snufspin.video import probe_video


def read_source_prefix(source, end_frame):
    # FFprobe counts packets here. A short cushion allows for decoder reordering.
    result = subprocess.run([
        "ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "V:0",
        "-read_intervals", f"%+#{end_frame + 16}", "-show_entries",
        "stream=width,height:frame=best_effort_timestamp_time", "-of", "json", str(source.resolve()),
    ], capture_output=True, text=True, check=True)
    metadata = json.loads(result.stdout)
    times = [float(frame["best_effort_timestamp_time"]) for frame in metadata["frames"][:end_frame]]
    if len(times) != end_frame or not all(map(isfinite, times)) or any(
        b <= a for a, b in zip(times, times[1:])
    ):
        raise ValueError("could not read the requested presentation frame timestamps")
    stream = metadata["streams"][0]
    return stream["width"], stream["height"], [time - times[0] for time in times]


def prepare(source, clip, first, count, expected_sha256):
    source_hash = file_sha256(source)
    if source_hash != expected_sha256:
        raise ValueError(f"source hash differs from the pinned dataset file: {source}")
    if first < 0 or count < 1:
        raise ValueError("requested frame interval must have a nonnegative start and a positive count")
    if clip.exists() or clip.suffix.lower() != ".mp4":
        raise ValueError("clip must be a new .mp4")
    clip.parent.mkdir(parents=True, exist_ok=True)
    width, height, timestamps = read_source_prefix(source, first + count)
    staged = clip.with_name(clip.stem + ".partial.mp4")
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-n", "-noautorotate", "-i", str(source),
        "-map", "0:V:0", "-an", "-vf",
        f"trim=start_frame={first}:end_frame={first + count},setpts=PTS-STARTPTS",
        "-fps_mode", "passthrough", "-enc_time_base", "filter",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0", "-threads", "4",
        "-movflags", "+faststart", str(staged),
    ], check=True)
    result = probe_video(staged)
    source_times = timestamps[first:]
    if (result.width, result.height, len(result.timestamps_s)) != (width, height, count):
        raise ValueError("excerpt changed source dimensions or frame count")
    if any(abs((src - source_times[0]) - dst) > 2e-6
           for src, dst in zip(source_times, result.timestamps_s)):
        raise ValueError("excerpt changed source frame timing")
    clip_hash = file_sha256(staged)
    staged.rename(clip)
    return {
        "source_sha256": source_hash,
        "clip_sha256": clip_hash,
        "width": width,
        "height": height,
        "first_source_frame": first,
        "last_source_frame": first + count - 1,
        "source_timestamps_s": source_times,
        "clip_timestamps_s": result.timestamps_s,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("clip", type=Path)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.clip, args.first, args.count, args.source_sha256),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
