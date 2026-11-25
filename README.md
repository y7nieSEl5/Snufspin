# Snufspin

<img src="assets/snufspin-app-icon-sage.png" alt="Snufspin logo with a sage traveler’s hat above a spinning tennis ball" width="180">

Snufspin is an offline tennis shot analysis framework. Its default ball tracker is the official WASB Tennis model at stride 1, with measured positions in source pixels. The shot analysis command also accepts manual observations, with optional shot labels and metric trajectories.

See the [architecture](docs/architecture.md) for the pipeline, data contracts and algorithms on `main`. The [trajectory decision log](docs/trajectory-decisions.md) records benchmark outcomes and experiments still to try.

## Set up the local tracker

Use Python 3.10+ and FFmpeg with `ffprobe` and `libx264` on your `PATH`. Install the tracking dependencies and download the pinned WASB code and Tennis checkpoint once.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-wasb.txt
mkdir -p data/wasb
git clone https://github.com/nttcom/WASB-SBDT.git data/wasb/WASB-SBDT
git -C data/wasb/WASB-SBDT checkout --detach 923462cacdeb3353b84ddebdedb3f4b7a8553b0f
curl -fL 'https://drive.usercontent.google.com/download?id=14AeyIOCQ2UaQmbZLNQJa1H_eSwxUXk7z&export=download&confirm=t' \
  -o data/wasb/wasb_tennis_best.pth.tar
shasum -a 256 data/wasb/wasb_tennis_best.pth.tar
```

The expected SHA-256 is `9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb`. [WASB model zoo](https://github.com/nttcom/WASB-SBDT/blob/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/MODEL_ZOO.md)

## Track and render

Use an MP4 with orientation baked into its pixels.

```sh
mkdir -p outputs
python -m snufspin track data/clip.mp4 > outputs/ball-track.json
python -m snufspin render data/clip.mp4 --ball-track outputs/ball-track.json \
  --output outputs/overlay.mp4
```

The adapter uses WASB's RGB affine warp and ImageNet normalization, with 512×288 inputs and heatmaps. It runs overlapping three-frame windows at stride 1. For each original frame, it sends the combined candidates to WASB's online tracker, using the official 0.5 heatmap threshold, weighted connected components, and 300-source-pixel displacement gate. The adapter defaults to CPU with four threads and one window per forward pass. [Official stride-1 Tennis configuration](https://github.com/nttcom/WASB-SBDT/blob/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/GET_STARTED.md#wasb-ours-step1), [WASB inference method](https://arxiv.org/html/2311.05237#S3.SS3)

The output uses Snufspin's track JSON, with source-video SHA-256 and timestamps for every frame. Selected in-bounds detections use the inverse affine transform and carry `measured` provenance. Frames without a selected in-bounds detection have no point in `ball_track`. Clips shorter than three frames have no complete window.

Use `--upstream` and `--weights` to override the setup paths, `--device mps` or `--device cuda` to use a supported GPU, and `--threads` to set the CPU thread count. The same cached track can be passed to `analyze --ball-track outputs/ball-track.json` with a shot observation file.

## Import existing tennis-vision results

For an existing `upstream.npz`, run `python -m snufspin import-tennis-vision data/clip.mp4 data/upstream.npz > outputs/raw-track.json`, then render or analyze the imported track. The importer reads numeric ball tracks and clip identity with pickle disabled.

## Evaluate reviewed tracking errors

Historical TrackNet review sets and their cached predictions live under `benchmarks/tracking/`. The video hash binds each review to its source clip. Use the exact local source MP4 for the evaluation command.

```sh
python -m snufspin evaluate-track outputs/tennis-vision-baseline/source.mp4 \
  --ball-track benchmarks/tracking/grass/baseline-track.json \
  --review benchmarks/tracking/grass/review.json
python -m unittest discover -s tests -v
```

The report counts missing positions on visible frames, positions on hidden frames, and localization within 5, 10, and 20 source pixels. It includes the provenance and localization error for each reviewed frame. These reviews were selected by watching for errors. They provide regression cases and diagnostic evidence. Whole-clip precision and recall require a representative annotated sample.

## Historical TrackNet baseline

The `tracking-baseline-v1` tag preserves the original TrackNet setup, frozen results, and 0.15-second short-gap interpolation workflow. Existing `track --model-source PATH --weights PATH` invocations can still reproduce TrackNet outputs. The `interpolate-track` command remains available for explicitly filling short gaps in a cached track. [TrackNet benchmark archive](benchmarks/tracking/README.md)

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

Snufspin uses NTT Communications' WASB Tennis model and builds on Philipp Dubach's tennis-vision pipeline, yastrebksv's TennisProject and TrackNet model, and tracking design ideas from kzheng18's CourtVision. GridTrackNet's sample collection supplied the footage used for our tracking reviews. [Project credits and upstream notices](CREDITS.md)

## Fixed-camera court calibration

`calibrate-court` defaults to TEXflip's tennis-trained LETR detector. Install its optional
dependencies and the tested source and checkpoint before running it.

```bash
python -m pip install -r requirements-texflip.txt
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/TEXflip/tennis-court-detection.git data/tennis-court/texflip
git -C data/tennis-court/texflip checkout 49bae72d9f50d022be7d1aeeac4681f053a59188
git -C data/tennis-court/texflip submodule update --init LETR
```

Save the [published checkpoint](https://drive.google.com/file/d/1kgPfxeYMSED6bu0xWP-KOSgLDukNPj5t/view)
as `data/tennis-court/letr_best_checkpoint.pth`. Its SHA-256 is
`0f088c6f4dfef766bd10adc401e5012d255c07d699d9afae98da5976f821d829`.

```bash
python -m snufspin calibrate-court input.mp4 --review-dir outputs/court-review > input.camera.json
```

For fixed-camera footage, pass `--paint-fallback` to try Farin-style shared paint selection
when TEXflip's individually supported frame fits disagree. The fallback keeps the original
decision for successful fits and other rejection categories. It writes a court-plane mapping
when a clear winner has paint support in both directions across multiple frames. A tied
selection leaves the clip unresolved. The `review-dir` diagnostics record which path fired.

For a fixed pinhole camera, opt into the vanishing-direction segment solver:

```bash
python -m snufspin calibrate-court input.mp4 --solver vp-segments \
  --intrinsics input.intrinsics.json --review-dir outputs/court-review > input.camera.json
```

Omit `--intrinsics` to use the median GeoCalib pinhole estimate across the sampled frames.
Install the optional [GeoCalib inference package](https://github.com/cvg/GeoCalib#setup-and-demo)
for this mode. GeoCalib can download its official pinhole weights on first use, or
`--geocalib-weights PATH` can point to a local checkpoint. The solver groups TEXflip
segments by vanishing direction, fits pose with fixed intrinsics, and requires support
from matched segments and visible paint in both court directions across multiple frames.
The selected camera uses the matcher's court labeling, which can differ by 180° from
an externally labeled court. The default `--solver crossings` remains available.

`--detector heatmap` selects the older TennisCourtDetector, using its
[published source and checkpoint](https://github.com/yastrebksv/TennisCourtDetector#pretrained-model).
`--landmarks` supplies manual source-pixel points as a JSON object keyed by zero-based frame
index and [court landmark name](snufspin/court.py). `--intrinsics` accepts decoded `width`,
`height`, `K` and optional OpenCV `distortion`. For a supplied CalTennis camera, use
`import-caltennis VIDEO CALIBRATION` with an explicit `--pixel-transform` for a size
mismatch. Use `review-court VIDEO SIDECAR --review-dir DIR` to inspect the court overlay.

The JSON sidecar is tied to the exact video. It contains the metric court-plane mapping
and, when available, the full camera `K, R, t`. A plane-only sidecar supports court-contact
locations via `pixel_to_court`. A full camera also supports world-space rays and projection
above the court.

## Gravity-only flight fit

Choose a single airborne interval bounded by zero-based presentation frames. Use a ball
track and full calibration sidecar for that same video.

```bash
python -m snufspin fit-flight input.mp4 --ball-track input.track.json --calibration input.camera.json \
  --start-frame 2966 --end-frame 3021 > input.flight.json
```

The fit models constant horizontal velocity and vertical acceleration of
`-9.80665 m/s²`. It uses the actual timestamps, seeds the trajectory from calibrated rays,
and refines in source pixels. Only manual and measured detections provide evidence.
The output includes the inlier and excluded frames, reprojection errors, velocity at
the first fitted sample, and a `metric_track` compatible with the shot observation schema.
The returned trajectory spans the supporting measurements within the requested interval.
A single fit covers one flight segment between contacts or bounces. Drag, spin and speed
validation against independent 3D measurements are future steps.
