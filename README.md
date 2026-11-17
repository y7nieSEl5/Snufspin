# Snufspin

Snufspin is an offline tennis shot analysis framework. The first tracking backend is tennis-vision. Snufspin imports its 2D ball detections and renders an overlay for review. The shot analysis command also accepts manual observations, with optional shot labels and metric trajectories.

## Run the tennis-vision baseline

Start with a short, upright, fixed-camera MP4. Use a clip with the orientation baked into the pixels so the detector and overlay see the same image coordinates.

1. Open the Phase 1 notebook in Colab and choose a GPU runtime. In the **Upload input video** section, set `input_video` to your clip’s path on Drive or use the browser upload option. Run the notebook and download its `upstream.npz`. [tennis-vision Phase 1 notebook](https://colab.research.google.com/github/philippdubach/tennis-vision/blob/a501953194f5207534273c7b2b59c0b642df7a05/scripts/colab_run.ipynb)
2. Put the artifact and the exact MP4 processed by the notebook under `data/`. From the Snufspin repository root, run the importer and renderer locally with Python 3.10+ and FFmpeg (`ffprobe` and `libx264` included).

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy
mkdir -p outputs
python -m snufspin import-tennis-vision data/clip.mp4 data/upstream.npz > outputs/ball-track.json
python -m snufspin render data/clip.mp4 --ball-track outputs/ball-track.json \
  --output outputs/overlay.mp4
```

`ball-track.json` records frame timestamps from the original video, detections in source pixels, and SHA-256 hashes for the video and artifact. The green box appears on frames with a detection. Missed frames stay empty, following CourtVision’s distinction between observations and filled gaps. Snufspin reads the numeric ball track and clip identity from `upstream.npz` with pickle disabled. Keep the upstream artifact if you need its court, player, or bounce outputs later.

The same cached track can be passed to `analyze --ball-track outputs/ball-track.json` alongside a shot observation file when you begin annotating flight windows.

## Evaluate reviewed tracking errors

Manual review sets and the original cached predictions live under `benchmarks/tracking/`. The video hash binds each review to its source clip. Use the exact local source MP4 for the evaluation command.

```sh
python -m snufspin evaluate-track outputs/tennis-vision-baseline/source.mp4 \
  --ball-track benchmarks/tracking/grass/baseline-track.json \
  --review benchmarks/tracking/grass/review.json
python -m unittest discover -s tests -v
```

The report counts missing positions on visible frames, positions on hidden frames, and localization within 5, 10, and 20 source pixels. It includes the provenance and localization error for each reviewed frame. These reviews were selected by watching for errors. They provide regression cases and diagnostic evidence. Whole-clip precision and recall require a representative annotated sample.

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
