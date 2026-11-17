import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from .tracking import BallPoint, index_ball_points
from .video import Video


def render_overlay(video: Video, points: tuple[BallPoint, ...], output: Path) -> None:
    output = output.resolve()
    if output == video.path or output.suffix.lower() != ".mp4":
        raise ValueError("overlay output must be a separate .mp4 file")

    by_frame = index_ball_points(video, points)
    commands = []
    hidden = (-24, -24)
    previous_position = hidden
    previous_color = "lime"
    for frame, time_s in enumerate(video.timestamps_s):
        point = by_frame.get(frame)
        position = hidden if point is None else (round(point.x_px) - 5, round(point.y_px) - 5)
        color = "gold" if point is not None and point.provenance == "interpolated" else "lime"
        if position == previous_position and color == previous_color:
            continue
        # Schedule between frame times to avoid timestamp rounding at a boundary.
        start = (video.time_at(frame - 1) + time_s) / 2 if frame else 0
        x, y = position
        commands.append(f"{start:.6f} drawbox@ball x {x}, drawbox@ball y {y}, drawbox@ball color {color};")
        previous_position = position
        previous_color = color

    filters = ["setpts=PTS-STARTPTS"]
    if commands:
        filters.extend([
            "sendcmd=f=markers.txt",
            "drawbox@ball=x=-24:y=-24:w=11:h=11:t=2:color=lime",
        ])
    # yuv420p requires even dimensions. Pad on the right/bottom to keep coordinates.
    filters.append("pad=ceil(iw/2)*2:ceil(ih/2)*2")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="snufspin-overlay-") as directory:
        Path(directory, "markers.txt").write_text("\n".join(commands), encoding="ascii")
        with NamedTemporaryFile(dir=output.parent, suffix=".mp4", delete=False) as temporary:
            staged = Path(temporary.name)
        try:
            subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-v", "error", "-y", "-protocol_whitelist", "file",
                    "-noautorotate", "-i", str(video.path), "-map", "0:V:0",
                    "-vf", ",".join(filters), "-fps_mode", "passthrough", "-enc_time_base", "filter",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(staged),
                ],
                cwd=directory,
                capture_output=True,
                text=True,
                check=True,
            )
            staged.replace(output)
        finally:
            staged.unlink(missing_ok=True)
