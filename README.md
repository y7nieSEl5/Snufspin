# Snufspin

Snufspin is an offline tennis shot analysis framework. Its Python ball tracker follows the reviewed tennis-vision baseline, with short-gap interpolation and overlays for review. The shot analysis command also accepts manual observations, with optional shot labels and metric trajectories.

## Set up the local tracker

Use Python 3.10+ and FFmpeg with `ffprobe` and `libx264` on your `PATH`. Install the runtime dependencies and obtain the pinned external model architecture and pretrained ball checkpoint once.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt gdown
mkdir -p data/tracknet
git clone https://github.com/yastrebksv/TennisProject.git data/TennisProject
git -C data/TennisProject checkout b7552e955253c008c7612298cb40694682a74ef9
gdown 1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl -O data/tracknet/model_best.pt
```

The runtime loads `BallTrackerNet` from that checkout and records hashes of the model file and checkpoint. The detector implementation in `snufspin/tracknet.py` preserves the reviewed baseline's three-frame BGR input, coordinate scaling, mask conversion, and distance gate. The ball model and checkpoint are credited to yastrebksv's TrackNet and TennisProject work. [Model and checkpoint source](https://github.com/yastrebksv/TrackNet#pretrained-model)

## Track, interpolate, and render

Start with a short, upright MP4. Use a clip with orientation baked into the pixels so the detector and overlay see the same image coordinates.

```sh
mkdir -p outputs
python -m snufspin track data/clip.mp4 \
  --model-source data/TennisProject/tracknet.py \
  --weights data/tracknet/model_best.pt > outputs/raw-track.json
python -m snufspin interpolate-track data/clip.mp4 \
  --ball-track outputs/raw-track.json --max-gap-s 0.15 > outputs/ball-track.json
python -m snufspin render data/clip.mp4 --ball-track outputs/ball-track.json \
  --output outputs/overlay.mp4
```

Tracking runs on CPU with four threads by default. Use `--device cuda` or `--device mps` with a supported PyTorch installation, and `--threads` to set the CPU thread count. The tracker streams decoded frames with a three-frame input window. The reviewed baseline's first two frames have no detection. Short gaps between detections can be filled by the interpolation step.

Track files record original frame timestamps, source-pixel positions, provenance, and the source video's SHA-256 hash. Green boxes show detector observations and gold boxes show interpolated positions, following CourtVision's distinction between measured and inferred tracks.

The same cached track can be passed to `analyze --ball-track outputs/ball-track.json` alongside a shot observation file when you begin annotating flight windows.

## Import existing tennis-vision results

For an existing `upstream.npz`, run `python -m snufspin import-tennis-vision data/clip.mp4 data/upstream.npz > outputs/raw-track.json`, then use the interpolation and rendering commands above. The importer reads numeric ball tracks and clip identity with pickle disabled.

## Evaluate reviewed tracking errors

Manual review sets and the original cached predictions live under `benchmarks/tracking/`. The video hash binds each review to its source clip. Use the exact local source MP4 for the evaluation command.

```sh
python -m snufspin evaluate-track outputs/tennis-vision-baseline/source.mp4 \
  --ball-track benchmarks/tracking/grass/baseline-track.json \
  --review benchmarks/tracking/grass/review.json
python -m unittest discover -s tests -v
```

The report counts missing positions on visible frames, positions on hidden frames, and localization within 5, 10, and 20 source pixels. It includes the provenance and localization error for each reviewed frame. These reviews were selected by watching for errors. They provide regression cases and diagnostic evidence. Whole-clip precision and recall require a representative annotated sample.

## Tracking experiments

The tracking baseline is pinned as `tracking-baseline-v1` after visual acceptance of three fresh clips. It uses the upstream detector plus short, bracketed interpolations. The next development stage is 3D reconstruction. Hidden frames provide no visible center for assessing localization error. [Experiment findings](benchmarks/tracking/README.md)

```sh
python -m snufspin interpolate-track outputs/tennis-vision-baseline/source.mp4 \
  --ball-track benchmarks/tracking/grass/baseline-track.json \
  --max-gap-s 0.15 > outputs/grass-interpolated.json
python -m snufspin evaluate-track outputs/tennis-vision-baseline/source.mp4 \
  --ball-track outputs/grass-interpolated.json --review benchmarks/tracking/grass/review.json
```

Added positions have `interpolated` provenance. The command preserves existing positions, skips unbracketed and longer gaps, and leaves its input untouched. Its gap limit measures the entire time between the endpoint detections.

When rendering an interpolated track, green boxes show detector observations and gold boxes show interpolated positions. Frames with no position have no box.

## Run the example

This example uses the Python standard library and FFmpeg with `ffprobe` on your `PATH`.

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

## Credits

Snufspin builds on Philipp Dubach's tennis-vision pipeline, yastrebksv's TennisProject and TrackNet model, and tracking design ideas from kzheng18's CourtVision. GridTrackNet's sample collection supplied the footage used for our tracking reviews. [Project credits and upstream notices](CREDITS.md)
