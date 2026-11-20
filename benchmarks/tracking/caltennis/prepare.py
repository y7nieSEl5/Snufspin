"""Extract a lossless CalTennis excerpt and record its frame mapping."""
import argparse
import json
from pathlib import Path
import subprocess

from snufspin.tracking import file_sha256
from snufspin.video import probe_video


def prepare(source, clip, first, count, expected_sha256):
    source_hash = file_sha256(source)
    if source_hash != expected_sha256:
        raise ValueError(f"source hash differs from the pinned dataset file: {source}")
    video = probe_video(source)
    if first < 0 or count < 1 or first + count > len(video.timestamps_s):
        raise ValueError("requested frame interval lies outside the source")
    if clip.exists() or clip.suffix.lower() != ".mp4":
        raise ValueError("clip must be a new .mp4")
    clip.parent.mkdir(parents=True, exist_ok=True)
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
    source_times = video.timestamps_s[first:first + count]
    if (result.width, result.height, len(result.timestamps_s)) != (video.width, video.height, count):
        raise ValueError("excerpt changed source dimensions or frame count")
    if any(abs((src - source_times[0]) - dst) > 2e-6
           for src, dst in zip(source_times, result.timestamps_s)):
        raise ValueError("excerpt changed source frame timing")
    clip_hash = file_sha256(staged)
    staged.rename(clip)
    return {
        "source_sha256": source_hash,
        "clip_sha256": clip_hash,
        "width": video.width,
        "height": video.height,
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
