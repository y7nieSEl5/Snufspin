# Tracking review and first post-processing experiment

## Decision

Improve tracking before relying on its output for monocular 3D reconstruction. The 45 reviewed frames include 41 visible-ball labels and four hidden labels. The baseline has 30 visible-frame misses and positions on two hidden frames. Among the 11 visible labels with a detection, errors range from 6.0 to 37.6 source pixels at 1280×720.

These frames were selected because the user observed tracking failures. Their counts describe this failure set. Whole-video accuracy remains unmeasured. Hidden means that the annotator could not locate the ball in the frame. A box on such a frame cannot be verified against a visible ball center.

## Baseline and gap-filling results

Short-gap interpolation was tested with a maximum endpoint separation of 0.15 seconds, chosen as an initial experiment setting. It uses actual frame timestamps. The comparison includes all positions in each output, including interpolated positions.

| Clip | Visible labels | Baseline misses | Interpolated misses | Baseline within 10 px | Interpolated within 10 px | Hidden labels with boxes, baseline → interpolated |
| --- | --- | --- | --- | --- | --- | --- |
| Grass | 14 | 6 | 0 | 1 | 2 | 2 → 3 |
| Hardcourt | 14 | 13 | 2 | 0 | 11 | 0 → 0 |
| Clay | 13 | 11 | 10 | 0 | 1 | 0 → 0 |

Interpolation is promising for short hardcourt gaps. It adds a box on hidden grass frame 36 and carries some incorrect endpoints through other frames. The results support keeping it experimental. Correct candidate selection and image-supported gap recovery need further testing before enabling a replacement tracker.

## Candidate-extraction diagnosis

The diagnostic reruns the checkpoint on the reviewed frames and verifies that its Hough-circle and distance-gate decisions reproduce the cached baseline. Four visible misses are the first two frames of the hardcourt and clay clips, which the upstream detector deliberately skips for its three-frame input. Of the remaining 26 visible misses, 25 have no Hough-circle candidate and one has candidates rejected by the distance gate. Grass frame 75 has a circle 9.49 pixels from the manual label, but the upstream displacement gate rejects all candidates.

A second diagnostic extracts connected-component centroids from the same binary heatmaps. Components are limited to 1–200 network pixels for this initial probe. It selects the centroid nearest a constant-velocity extrapolation from the last two cached baseline observations. This is an isolated candidate-selection experiment using cached history. A deployed tracker would feed its own previous outputs back into the selection.

The centroid probe helps some reviewed frames, including clay 32 and 33 and hardcourt 44. It still follows the hand in much of the grass interval, and most missing frames have no usable component. Its use of baseline history and the selected review set limits conclusions about end-to-end improvement. Per-frame candidate coordinates and errors are in each clip's `candidate-diagnosis.json`.

The next comparison should test image-supported recovery with the same weights, retain raw observations, and keep all inferred points identifiable. Before adopting changed candidate selection, label the 12 previously unreviewed control frames in `control-selection.json`. These were sampled without replacement with a fixed seed and exclude the original review frames. Grass controls have been labeled. The eight hardcourt and clay control labels remain pending.

The four grass control frames all have measured detections. Three are within 5 source pixels of the manual center. Frame 74 differs by 16.49 pixels. Interpolation yields the same errors on this sample because it fills missing positions only. The control sample is small, and the additional localization error supports checking image evidence before trusting nearby detections as gap endpoints. The grass control labels and evaluations are saved in `grass/control-review.json` and `grass/control-results.json`.

## Reproduction

`baseline-track.json` contains the frozen upstream predictions and metadata. `review.json` contains the user's labels. `experiment-results.json` records the baseline and interpolation evaluation results. Videos remain local under `outputs/tennis-vision-baseline`, `outputs/tennis-vision-hardcourt-review`, and `outputs/tennis-vision-clay-review`.

The baseline used tennis-vision revision `a501953194f5207534273c7b2b59c0b642df7a05` and its pinned TennisProject revision `b7552e955253c008c7612298cb40694682a74ef9`. Each candidate diagnostic records SHA-256 hashes of the model source and checkpoint. The checkpoint is the ball model linked by the upstream notebook. [Upstream notebook](https://github.com/philippdubach/tennis-vision/blob/a501953194f5207534273c7b2b59c0b642df7a05/scripts/colab_run.ipynb)

With NumPy, PyTorch, and OpenCV available, run the candidate diagnostic using the local TennisProject `tracknet.py` and checkpoint downloaded by the notebook.

```sh
python -m snufspin.diagnostics outputs/tennis-vision-baseline/source.mp4 \
  --ball-track benchmarks/tracking/grass/baseline-track.json \
  --review benchmarks/tracking/grass/review.json \
  --model-source /path/to/TennisProject/tracknet.py \
  --weights /path/to/model_best.pt > outputs/grass-candidate-diagnosis.json
```

The model-source argument executes the supplied local Python module. Use the inspected upstream revision for reproduction.
