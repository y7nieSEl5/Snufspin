# Tracking review and first post-processing experiment

These records describe the historical TrackNet baseline at `tracking-baseline-v1`. The current default tracker is documented in the [Snufspin README](../../README.md).

## Decision

The user accepted the three fresh demos as sufficient for starting monocular 3D reconstruction. The pinned baseline, `tracking-baseline-v1`, uses the upstream detector with short-gap interpolation at a maximum endpoint separation of 0.15 seconds. One isolated hardcourt misidentification near a player's body was reported and accepted as a known limitation. Its frame number has not been identified. The next development stage is 3D reconstruction, including robust treatment of isolated bad observations.

The original 45 reviewed frames include 41 visible-ball labels and four hidden labels. The baseline has 30 visible-frame misses and positions on two hidden frames. Among the 11 visible labels with a detection, errors range from 6.0 to 37.6 source pixels at 1280×720.

These frames were selected because the user observed tracking failures. Their counts describe this failure set. Whole-video accuracy remains unmeasured. Hidden means that the annotator could not locate the ball in the frame. A box on such a frame cannot be verified against a visible ball center.

## Baseline and gap-filling results

Short-gap interpolation was tested with a maximum endpoint separation of 0.15 seconds, chosen as an initial experiment setting. It uses actual frame timestamps. The comparison includes all positions in each output, including interpolated positions.

| Clip | Visible labels | Baseline misses | Interpolated misses | Baseline within 10 px | Interpolated within 10 px | Hidden labels with boxes, baseline → interpolated |
| --- | --- | --- | --- | --- | --- | --- |
| Grass | 14 | 6 | 0 | 1 | 2 | 2 → 3 |
| Hardcourt | 14 | 13 | 2 | 0 | 11 | 0 → 0 |
| Clay | 13 | 11 | 10 | 0 | 1 | 0 → 0 |

Interpolation is promising for short hardcourt gaps. It adds an inferred position on hidden grass frame 36 and carries some incorrect endpoints through other frames. A hidden label provides no visible center for validating an inferred position. The results support keeping interpolation experimental while candidate selection and image-supported gap recovery undergo further testing.

## Candidate-extraction diagnosis

The diagnostic reruns the checkpoint on the reviewed frames and verifies that its Hough-circle and distance-gate decisions reproduce the cached baseline. Four visible misses are the first two frames of the hardcourt and clay clips, which the upstream detector deliberately skips for its three-frame input. Of the remaining 26 visible misses, 25 have no Hough-circle candidate and one has candidates rejected by the distance gate. Grass frame 75 has a circle 9.49 pixels from the manual label, but the upstream displacement gate rejects all candidates.

A second diagnostic extracts connected-component centroids from the same binary heatmaps. Components are limited to 1–200 network pixels for this initial probe. It selects the centroid nearest a constant-velocity extrapolation from the last two cached baseline observations. This is an isolated candidate-selection experiment using cached history. A deployed tracker would feed its own previous outputs back into the selection.

The centroid probe helps some reviewed frames, including clay 32 and 33 and hardcourt 44. It still follows the hand in much of the grass interval, and most missing frames have no usable component. Its use of baseline history and the selected review set limits conclusions about end-to-end improvement. Per-frame candidate coordinates and errors are in each clip's `candidate-diagnosis.json`.

Image-supported recovery remains a possible future tracking improvement. All 12 control frames in `control-selection.json` have now been labeled. These were sampled without replacement with a fixed seed and exclude the original review frames.

The four grass control frames all have measured detections. Three are within 5 source pixels of the manual center. Frame 74 differs by 16.49 pixels. Interpolation yields the same errors on this sample because it fills missing positions only. The control sample is small, and the additional localization error supports checking image evidence before trusting nearby detections as gap endpoints. The grass control labels and evaluations are saved in `grass/control-review.json` and `grass/control-results.json`.

The user corrected hardcourt control frame 33 from visible to hidden after identifying the original annotation as a highlight spot. The previous 32.98-pixel disagreement with that annotation is excluded from localization metrics. The interpolated position at (770, 300) was judged plausible by the user. Its localization accuracy remains unmeasured because the ball center is hidden.

Across all three corrected control samples there are 11 visible labels and one hidden label. The baseline detects all 11 visible balls, with eight positions within 5 source pixels and nine within 10 pixels. Interpolation produces the same visible-frame results and adds an inferred position at hidden frame 33. The `hidden_with_position` count records such positions without assigning them a localization error.

The isolated centroid selection probe also detects all 11 visible control balls, with nine positions within 5 pixels and nine within 10 pixels. Grass frame 74 worsens from 16.49 to 19.38 pixels. Hardcourt frame 33 has neither a Hough-circle nor a connected-component candidate. The remaining visible-frame errors and grass hand-latching cases support testing image-supported recovery. The candidate probe continues to use cached baseline history, so it needs a subsequent full-sequence comparison using its own previous outputs. Each clip's `control-candidate-diagnosis.json` records the rerun and errors.

## Three fresh review clips

`fresh-review/manifest.json` records the candidate settings, source URLs, source-video Git blob hashes, exact excerpt hashes, and inference checkpoint identity. The three new source videos are professional hardcourt, grass, and clay matches from the GridTrackNet sample collection. Four-second rally excerpts were selected by watching the original footage before running the detector. This review tests additional camera views in professional footage. The earlier recreational cases remain part of the baseline record.

Each surface directory under `fresh-review/` stores the raw detector track and the interpolated track for comparison. Local sources and overlays live under `outputs/fresh-tracking-review/`. Green boxes mark detector observations and gold boxes mark interpolated positions. Detection coverage counts positions produced and has no manual accuracy interpretation. All three demos were visually accepted on November 17, 2026.

All three excerpts have 100 frames at 25 FPS. Hardcourt has 94 detector positions and two interpolated positions, grass has 82 detector positions and six interpolated positions, and clay has 95 detector positions and three interpolated positions. Each clip retains the detector's two startup misses. Hardcourt also ends with two misses. Grass retains a ten-frame gap from frame 71 through 80, at overlay times 2.84–3.20 seconds, for visual review. Every overlay decodes completely and preserves the source's relative frame timestamps.

## Reproduction

`baseline-track.json` contains the frozen upstream predictions and metadata. `review.json` contains the user's labels. `experiment-results.json` records the baseline and interpolation evaluation results. Videos remain local under `outputs/tennis-vision-baseline`, `outputs/tennis-vision-hardcourt-review`, and `outputs/tennis-vision-clay-review`.

The baseline used tennis-vision revision `a501953194f5207534273c7b2b59c0b642df7a05` and its pinned TennisProject revision `b7552e955253c008c7612298cb40694682a74ef9`. Each candidate diagnostic records SHA-256 hashes of the model source and checkpoint. The checkpoint is the ball model linked by yastrebksv/TrackNet. Snufspin's local `track` command reproduces the reviewed detector from Python. [Model and checkpoint source](https://github.com/yastrebksv/TrackNet#pretrained-model)

The local Python runner was checked on the full 100-frame fresh hardcourt excerpt. Every raw detector position matched the frozen baseline exactly (94 positions), and every position after interpolation matched the frozen candidate (96 positions). The overlay preserved all 100 source frame timestamps. CPU inference was used for this comparison. It uses OpenCV decoding to reproduce the reviewed detector's input pixels.

With NumPy, PyTorch, and OpenCV available, run the candidate diagnostic using the local TennisProject `tracknet.py` and pretrained checkpoint from the setup instructions at `tracking-baseline-v1`.

```sh
python -m snufspin.diagnostics outputs/tennis-vision-baseline/source.mp4 \
  --ball-track benchmarks/tracking/grass/baseline-track.json \
  --review benchmarks/tracking/grass/review.json \
  --model-source /path/to/TennisProject/tracknet.py \
  --weights /path/to/model_best.pt > outputs/grass-candidate-diagnosis.json
```

The model-source argument executes the supplied local Python module. Use the inspected upstream revision for reproduction.
