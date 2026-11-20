import json
import math
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Video:
    path: Path
    width: int
    height: int
    timestamps_s: tuple[float, ...]
    first_pts_s: float = 0.0

    def time_at(self, frame: int) -> float:
        if type(frame) is not int or not 0 <= frame < len(self.timestamps_s):
            raise ValueError(f"invalid video frame index: {frame!r}")
        return self.timestamps_s[frame]


def probe_video(path: Path) -> Video:
    path = path.resolve(strict=True)
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-threads", "4", "-protocol_whitelist", "file",
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
    return Video(path, stream["width"], stream["height"], tuple(t - start for t in timestamps), start)


def decode_bgr_frames(video: Video) -> Iterator[bytes]:
    """Stream one encoded-size BGR image for each probed presentation frame."""
    import cv2

    camera = cv2.VideoCapture(str(video.path))
    camera.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    try:
        for frame_index in range(len(video.timestamps_s)):
            ok, frame = camera.read()
            if not ok:
                raise ValueError(f"video ended before probed frame {frame_index} could be decoded")
            if frame.shape != (video.height, video.width, 3):
                raise ValueError("decoded frame dimensions do not match ffprobe metadata")
            yield frame.tobytes()
        if camera.read()[0]:
            raise ValueError("decoded more video frames than were found by ffprobe")
    finally:
        camera.release()


def read_sample_frames(video: Video, indexes: list[int]):
    """Read selected presentation frames using ffmpeg timestamp seeks."""
    import numpy as np

    for index in sorted(set(indexes)):
        timestamp = video.time_at(index)
        # Seek between this and the preceding frame to avoid rounding past its PTS.
        seek = video.first_pts_s + (video.time_at(index - 1) + timestamp) / 2 if index else None
        result = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-nostdin", "-noautorotate",
                "-protocol_whitelist", "file",
                *([] if seek is None else ["-seek_timestamp", "1", "-ss", f"{seek:.9f}"]),
                "-i", str(video.path), "-map", "0:V:0", "-frames:v", "1",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-threads", "1", "-",
            ],
            capture_output=True,
            check=True,
        )
        if len(result.stdout) != video.width * video.height * 3:
            raise ValueError(f"could not decode sample frame {index} at the probed dimensions")
        yield index, np.frombuffer(result.stdout, dtype=np.uint8).reshape(video.height, video.width, 3)
