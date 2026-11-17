"""Import the ball observations from a tennis-vision Phase 1 export."""

from pathlib import Path
from zipfile import BadZipFile

from .tracking import BallPoint, file_sha256, track_report
from .video import Video


def import_ball_track(video: Video, artifact: Path) -> dict:
    try:
        import numpy as np
    except ImportError as error:
        raise ValueError("the tennis-vision importer needs NumPy: python3 -m pip install numpy") from error

    try:
        with artifact.open("rb") as stream:
            data = np.load(stream, allow_pickle=False)
            if not isinstance(data, np.lib.npyio.NpzFile):
                raise ValueError("expected a tennis-vision upstream.npz archive")
            with data:
                video_sha256 = data["video_sha256"].item()
                frame_count = data["frame_count"].item()
                coordinates = data["ball_track"]
    except (BadZipFile, EOFError, KeyError, ValueError) as error:
        raise ValueError(f"invalid tennis-vision artifact {artifact}: {error}") from error

    expected_frames = len(video.timestamps_s)
    if type(frame_count) is not int or frame_count != expected_frames:
        raise ValueError("tennis-vision frame count does not match the input video")
    if coordinates.shape != (expected_frames, 2) or coordinates.dtype.kind != "f":
        raise ValueError("tennis-vision ball_track must contain one float (x, y) pair per frame")

    missing = np.isnan(coordinates).all(axis=1)
    if not np.all(missing | np.isfinite(coordinates).all(axis=1)):
        raise ValueError("tennis-vision ball coordinates must be finite pairs or both NaN")
    points = tuple(
        BallPoint(int(frame), float(x), float(y), "measured")
        for frame, (x, y) in zip(np.flatnonzero(~missing), coordinates[~missing])
    )
    report = track_report(video, points, {
        "engine": "philippdubach/tennis-vision",
        "artifact_sha256": file_sha256(artifact),
    })
    if video_sha256 != report["video_sha256"]:
        raise ValueError("tennis-vision artifact was generated for a different video")
    return report
