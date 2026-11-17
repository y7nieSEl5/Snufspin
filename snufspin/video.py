import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Video:
    path: Path
    width: int
    height: int
    timestamps_s: tuple[float, ...]

    def time_at(self, frame: int) -> float:
        if type(frame) is not int or not 0 <= frame < len(self.timestamps_s):
            raise ValueError(f"invalid video frame index: {frame!r}")
        return self.timestamps_s[frame]


def probe_video(path: Path) -> Video:
    path = path.resolve(strict=True)
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-protocol_whitelist", "file",
            "-select_streams", "V:0", "-show_entries",
            "stream=width,height:frame=best_effort_timestamp_time", "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    metadata = json.loads(result.stdout)
    if not metadata.get("streams") or not metadata.get("frames"):
        raise ValueError(f"no decodable video frames in {path}")

    stream = metadata["streams"][0]
    try:
        timestamps = [float(frame["best_effort_timestamp_time"]) for frame in metadata["frames"]]
    except (KeyError, ValueError) as error:
        raise ValueError("video has frames without usable timestamps") from error
    if not all(math.isfinite(t) for t in timestamps) or any(
        b <= a for a, b in zip(timestamps, timestamps[1:])
    ):
        raise ValueError("video frame timestamps must be finite and strictly increasing")

    start = timestamps[0]
    return Video(path, stream["width"], stream["height"], tuple(t - start for t in timestamps))
