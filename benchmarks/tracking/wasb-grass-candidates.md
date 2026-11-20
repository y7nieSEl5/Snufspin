# WASB candidate diagnosis on Amateur Grass

The analysis covers source frames 25–39 of the hash-matched GridTrackNet excerpt, using the official WASB Tennis checkpoint at stride 1. The candidate log captures the combined detections from the overlapping windows, their heatmap-blob scores, the upstream distance-gate result, and the zero-based index of the selected candidate. All 22 candidates passed the distance gate in this interval.

| Visible frame | Selected error (px) | Nearest candidate (px) | Result at both 5 and 10 px |
| ---: | ---: | ---: | --- |
| 26 | No selection | No candidates | No nearby candidate |
| 27 | No selection | No candidates | No nearby candidate |
| 28 | 19.1 | 19.1 | No nearby candidate |
| 29 | 25.7 | 25.7 | No nearby candidate |
| 30 | No selection | No candidates | No nearby candidate |
| 33 | 39.6 | 38.4 | No nearby candidate |
| 35 | 32.6 | 0.6 | Two nearby candidates passed and lost on score |
| 37 | No selection | No candidates | No nearby candidate |

Frames 25, 34, 38, and 39 were within 5 pixels. Frames 31, 32, and 36 have hidden-ball labels and no manual center for distance scoring.

The zoomed frames place WASB's selections on the far player's apparent hand or forearm. The footage is clearest on frames 33 and 35, where the bright hand is separated from the torso. On frame 35, the selected blob scored 4.37. The two candidates within 5 pixels of the label scored 3.36 and 3.28. A selector could recover frame 35 using the available candidates. Investigating frames 26–30, 33, and 37 would require looking upstream of selection, such as checking whether the heatmaps contain a weaker peak near the manual center.

The selected track JSON matched byte for byte across the logged MPS run, the plain MPS rerun, and the saved WASB benchmark run.

```sh
mkdir -p outputs/wasb-candidate-diagnostics
python -m snufspin track-wasb outputs/tennis-vision-baseline/source.mp4 \
  --upstream data/wasb/WASB-SBDT --weights data/wasb/wasb_tennis_best.pth.tar \
  --device mps --candidate-frames 25 39 \
  --candidate-log outputs/wasb-candidate-diagnostics/candidates.json \
  > outputs/wasb-candidate-diagnostics/logged-track.json
```

The local diagnostic directory also contains `assessment.json` with per-frame distances and threshold outcomes, `grass-candidates.mp4` with all 15 annotated source frames shown at 3 FPS, and `grass-wrong-points.png` with clean and annotated crops of the four mislocalized frames.
