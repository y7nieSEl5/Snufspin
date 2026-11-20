# Snufspin

Snufspin is an offline tennis shot analysis framework. Its default ball tracker is the official WASB Tennis model at stride 1, with measured positions in source pixels. The shot analysis command also accepts manual observations, with optional shot labels and metric trajectories.

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

## Court calibration

The metric court uses metres with the origin at the `baseline_a_doubles_low` intersection, +X toward baseline B, +Y toward the high sidelines, and the surface at `z = 0`. See [the named intersections and painted lines](snufspin/court.py). Automatic detection assigns baseline A to the near end in the detector's broadcast convention. Use manual names or an existing calibration when the physical orientation must be consistent across cameras.

Set up the [published TennisCourtDetector checkpoint](https://github.com/yastrebksv/TennisCourtDetector#pretrained-model) as `data/tennis-court/model.pth`. The tested source revision and checkpoint digest are in [CREDITS.md](CREDITS.md). Then run:

```bash
git clone https://github.com/yastrebksv/TennisCourtDetector.git data/tennis-court/TennisCourtDetector
git -C data/tennis-court/TennisCourtDetector checkout e5cd4f1ce26b15361700d3d89e068cbf0e82749e

python -m snufspin calibrate-court input.mp4 --review-dir outputs/court-review > input.camera.json
```

The CLI samples seven presentation frames by default. It fits a RANSAC homography to named intersections in decoded pixels and checks the nearby brightness ridges along the projected painted lines. The diagnostics record the sampled landmarks, inliers, marking support, and pose checks. The fixed-camera solver requires marking agreement on a majority of the samples, with at least two frames. An inconsistent court or a failed marking check exits nonzero and writes diagnostics with `--review-dir`.

- Supply known intrinsics with `--intrinsics intrinsics.json`. The file has fields `width`, `height`, `K` and optional `distortion` (OpenCV pinhole coefficients). `K` must already use the decoded pixel grid.
- Supply manual points with `--landmarks landmarks.json` to use the same solver. The file maps zero-based presentation-frame indices to dictionaries of named `[u, v]` observations, for example `{"120": {"service_a_center": [616, 422]}}`. Label at least four noncollinear points in each of at least two frames for a full-camera solve.
- With no supplied intrinsics, the solver uses perpendicular court directions to estimate a focal length in pixels with square pixels, a centered principal point, and zero assumed distortion. It keeps the court homography when the pose or focal length fails its checks. The sidecar then has `camera: null` and its mapping can locate a visible bounce on the surface.
- For the three checked dataset videos, see the [CalTennis review](benchmarks/calibration/README.md). `import-caltennis` takes the original JSON and, for a size mismatch, requires an explicitly chosen pixel transform. `review-court` renders any sidecar on its matching video and accepts repeatable `--frame` arguments.

Use `read_calibration(video, report)` from `snufspin.calibration` to verify a sidecar against the video digest and dimensions. `calibration.camera.project(xyz)` maps visible 3D points to source pixels. `calibration.camera.pixel_ray(uv)` returns the world-space camera center and a unit direction. `calibration.mapping.pixel_to_court(uv)` intersects a surface observation with the metric court. Camera coordinates follow `X_cam = R @ X_world + t`. The homography maps world XY to ideal pixels when supplied distortion is present, and the mapping methods apply the distortion model at the source-pixel boundary.

### TEXflip line detector experiment

`calibrate-court --detector texflip` uses the published tennis-trained LETR checkpoint. Install the optional dependencies and check out the tested revision.

```bash
python -m pip install -r requirements-texflip.txt
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/TEXflip/tennis-court-detection.git data/tennis-court/texflip
git -C data/tennis-court/texflip checkout 49bae72d9f50d022be7d1aeeac4681f053a59188
git -C data/tennis-court/texflip submodule update --init LETR
```

Save the [published checkpoint](https://drive.google.com/file/d/1kgPfxeYMSED6bu0xWP-KOSgLDukNPj5t/view) as `data/tennis-court/letr_best_checkpoint.pth`. The expected SHA-256 is `0f088c6f4dfef766bd10adc401e5012d255c07d699d9afae98da5976f821d829`.

```bash
python -m snufspin calibrate-court input.mp4 --detector texflip \
  --review-dir outputs/texflip-review > input.camera.json
```

The line fitter targets rear views. It uses intersections of the detected segments to propose metric court homographies and chooses an assignment based on projected line overlap and alignment. Local image refinement centers the predicted segments on bright paint where it finds sufficient support. The resulting measured intersections go into the existing calibration solver. The heatmap detector remains the default. See the [matched comparison on the three amateur clips](benchmarks/calibration/texflip.md).
