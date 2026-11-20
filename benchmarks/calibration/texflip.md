# TEXflip on the three amateur clips

This experiment branches from the completed calibration work at `e9ccaf8`. Both detectors use the same metric court, RANSAC, frame-agreement and marking checks, and camera pose solver. The TEXflip backend uses the published tennis-trained LETR checkpoint and its pinned upstream model. Its line-fitting step is adapted for the existing rear-view clips. It builds homography hypotheses from intersections of segment pairs in the two court directions, scores the projected template by segment alignment and overlap, and passes intersections of matched observed segments to the solver. It also tries to center each segment on nearby paint before matching. This addresses clipped endpoints and uses the existing metric template. [Upstream TEXflip method](https://github.com/TEXflip/tennis-court-detection/tree/49bae72d9f50d022be7d1aeeac4681f053a59188)

## Matched run on the reviewed excerpts

Both methods received the same seven evenly spaced calibration frames per clip. The 48 manually marked intersections on seven review frames were used only for scoring, including extra detector runs when a review frame fell outside the seven calibration samples. The ≤12 px columns count predictions within 12 decoded pixels of the reviewed point with the same name, divided by all reviewed points. The median is measured on same-name pairs, so misses are accounted for by the count, with the number of pairs shown in parentheses.

| Clip | Heatmap result | TEXflip result | Heatmap ≤12 px | TEXflip ≤12 px | Heatmap median | TEXflip median |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Grass | No mapping | Court mapping | 4/20 | 15/20 | 284.2 px (13) | 2.4 px (15) |
| Hardcourt | No mapping | Rejected across frames | 0/15 | 11/15 | 44.0 px (6) | 3.8 px (11) |
| Clay | No mapping | Court mapping | 0/13 | 6/13 | 433.4 px (11) | 3.9 px (8) |
| **Total** | **0/3 plane mappings** | **2/3 plane mappings** | **4/48** | **32/48** | | |

Neither backend produced a full camera on these excerpts. The final grass mapping aligned with the brightness checks on all seven sampled frames. Compared with all 20 reviewed grass landmarks, including the five it did not detect individually, its median projection error was 2.3 px and its 90th-percentile error was 4.3 px. Clay also passed the shared marking check on seven frames, though just one frame gave an individually supported homography. Its median and 90th-percentile errors against all 13 reviewed landmarks were 2.7 and 15.5 px.

Hardcourt had marking-supported per-frame fits on all seven samples. Their worst pair disagreed by 52.9 px at the 90th percentile, exceeding the solver's 16 px limit. In a separate diagnostic, pooling all observations into one RANSAC fit gave 42/59 inliers and supported markings on seven frames. The production solver rejected the shared mapping, so the diagnostic does not produce a court sidecar.

With raw LETR segments, before centering on nearby paint, the detector was within 12 px on 14/48 reviewed points and produced zero accepted mappings. Both the raw and refined treatments used the same model and acceptance checks. This is an exploratory comparison on the three available clips. The local feature refinement was added after inspecting the raw results.

## Original GridTrackNet files

A second seven-frame run on each untrimmed original gave the same clip-level decisions. The heatmap detector produced zero mappings. TEXflip produced plane mappings on grass and clay, and the frame-agreement check rejected hardcourt. The manually reviewed scoring above uses the labeled excerpts.

## Reproduce

Install both court detector checkpoints and the optional TEXflip dependencies as described in the repository README. Save the review exports as `<video stem>.landmarks.json` under the label directory.

```bash
python -m benchmarks.calibration.compare_detectors data/labeling/*.mp4 \
  --labels outputs/texflip-court-detection/labels \
  --output outputs/texflip-court-detection/refined
```

The output contains a JSON summary, per-frame scoring diagnostics, line and landmark overlays, and camera-format sidecars when a mapping passes. Green crosses show reviewed points in the comparison overlays. Orange crosses show predicted points, and cyan shows refined LETR segments. Add `--raw-lines` and use a separate output directory to reproduce the unrefined variant.

To run on the original files:

```bash
python -m benchmarks.calibration.compare_detectors data/gridtracknet-samples/Amateur*.mp4 \
  --output outputs/texflip-court-detection/originals
```

See the [fixed-camera paint experiment](farin-paint.md) for the later three-way comparison using these cached segments.
