# Snufspin

Snufspin is an offline tennis shot analysis framework. The current CLI takes a local video and a JSON sidecar of observations, then produces a timestamped shot report. The sidecar supplies ball tracks, shot boundaries, and any labels or metric trajectories you have. This gives manual annotations and future models the same path through the pipeline.

## Run the example

From the repository root, use Python 3.10+ and FFmpeg with `ffprobe` on your `PATH`. The Python code uses the standard library.

```sh
mkdir -p data outputs
ffmpeg -hide_banner -loglevel error -f lavfi \
  -i color=size=320x180:rate=30:duration=4 -c:v mpeg4 data/synthetic.mp4
python3 -m snufspin probe data/synthetic.mp4 > outputs/frames.json
python3 -m snufspin analyze data/synthetic.mp4 \
  --observations examples/synthetic-observations.json > outputs/shots.json
```

The clip is blank. Its synthetic annotations exercise the pipeline and produce a forehand with two spin tags and a 36 km/h average speed, followed by a backhand drop shot with unknown spin and unavailable speed.

## Use your own footage

Run `probe` on a local clip before annotating it. Use zero-based frame indexes in presentation order and pixel coordinates in the encoded image, before any display rotation or resizing. `probe` returns one timestamp in seconds per frame, relative to the first frame. Keep the same clip and frame order when exporting model observations.

Copy [the example sidecar](examples/synthetic-observations.json) as a starting point. The only required top-level field is `shots`. For an initial dataset, annotate a few shot windows per clip and fill in labels you can judge from the video. Use `source` to keep the original URL or recording ID and permission notes with each imported clip. Local footage and generated reports are ignored under `data/` and `outputs/`.

| Field | Meaning |
| --- | --- |
| `shots` | Ordered shot windows with `contact_frame`, `end_frame`, and an optional `end_event` such as `bounce`, `contact`, or `net`. Windows may share endpoints. |
| `ball_track` | Optional 2D ball centers. Mark each as `manual`, `measured`, `interpolated`, or `predicted` and include `confidence` from 0 to 1 when available. |
| `labels` | Optional `stroke`, `intent`, `spin`, and `source` for a shot. For example, `forehand`, `drop_shot`, and `["sidespin"]` can coexist. Use `null` for an unknown field. |
| `metric_track` | Optional time-ordered 3D samples in meters and a `method` describing how they were obtained. |

Average speed is the summed 3D sample-to-sample distance divided by elapsed video time. The report includes the sampled time window and whether it covers the whole shot. With fewer than two metric samples, the speed is unavailable. This calculation gives average speed over the samples. Estimating racket-exit speed requires a trajectory model that can estimate velocity at contact.

## Add models

[`analyze`](snufspin/pipeline.py) accepts independent `classify` and `reconstruct` callables. Both receive the shot window, video path and timestamps, and that shot's 2D ball observations. The classifier returns `ShotLabels`, and the reconstructor returns a `MetricTrack`. Either can return `None` when the observations are insufficient. A ball tracker or contact detector can start by writing the sidecar format, so you can reuse its output while iterating on stroke recognition or reconstruction.
