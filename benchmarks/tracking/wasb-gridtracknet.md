# WASB Tennis versus the frozen TrackNet baseline

Both trackers used the same six GridTrackNet excerpts, totaling 669 frames. Each local excerpt matched its frozen TrackNet track's video SHA-256 and timestamps. The six upstream videos were downloaded at GridTrackNet revision `0764162b73fb64d440fd9e6c363d592965400799` and checked against their Git blobs. Decoded thumbnails matched the excerpts to the source frames below. [GridTrackNet samples](https://github.com/VKorpelshoek/GridTrackNet/tree/0764162b73fb64d440fd9e6c363d592965400799/Sample%20Video%27s)

WASB used the [pinned Tennis checkpoint and settings](../../README.md#set-up-the-official-wasb-tennis-tracker) at both strides on Apple MPS. The TrackNet columns use the existing measured outputs and the same outputs with the baseline's 0.15-second interpolation. WASB outputs contain measured detections only.

## Localization on reviewed amateur frames

| Review set | Metric | TrackNet | TrackNet + interpolation | WASB stride 1 | WASB stride 3 |
| --- | --- | ---: | ---: | ---: | ---: |
| Failure review | Within 5 px / 41 visible | 0 | 12 | 24 | 15 |
| Failure review | Within 10 px / 41 visible | 1 | 14 | 25 | 16 |
| Failure review | Within 20 px / 41 visible | 6 | 21 | 26 | 19 |
| Failure review | Visible frames without a position | 30 | 12 | 12 | 19 |
| Failure review | Position on a hidden frame / 4 hidden | 2 | 3 | 3 | 0 |
| Control | Within 5 px / 11 visible | 8 | 8 | 8 | 8 |
| Control | Within 10 px / 11 visible | 9 | 9 | 8 | 8 |
| Control | Within 20 px / 11 visible | 11 | 11 | 10 | 10 |
| Control | Visible frames without a position | 0 | 0 | 1 | 1 |
| Control | Position on a hidden frame / 1 hidden | 0 | 1 | 1 | 1 |

WASB stride 1 improved localization on all three failure-review excerpts. Its within-10-pixel counts were 6/14 on grass, 11/14 on hardcourt, and 8/13 on clay. The failure review was chosen around known TrackNet errors, and the randomly selected control contains only 12 frames including one hidden label. These samples support keeping stride 1 for further comparison. Overall accuracy remains unmeasured.

## Coverage

| GridTrackNet sample | Original frame range, inclusive | Excerpt frames | TrackNet | TrackNet + interpolation | WASB stride 1 | WASB stride 3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Amateur Clay | 148–226 | 79 | 61 | 66 | 59 | 52 |
| Amateur Grass | 0–163 | 164 | 98 | 113 | 112 | 103 |
| Amateur Hardcourt | 69–194 | 126 | 111 | 124 | 123 | 108 |
| Professional Clay | 50–149 | 100 | 95 | 98 | 100 | 99 |
| Professional Grass | 50–149 | 100 | 82 | 88 | 96 | 90 |
| Professional Hardcourt | 25–124 | 100 | 94 | 96 | 99 | 95 |
| Total | | 669 | 541 | 585 | 589 | 547 |

Coverage counts frames with positions. The professional clips have no frame-level ball labels. Stride 3 leaves incomplete terminal windows without detections and took 49 seconds across the six clips, compared with 100 seconds for stride 1. These are local wall-clock totals including process startup and video decoding.

## Reproducibility checks

The CPU and MPS stride-1 runs on Amateur Grass selected the same 112 frames, with a maximum coordinate difference below 0.0002 source pixels. A TrackNet CPU rerun using the pinned architecture and checkpoint in the current environment, with PyTorch 2.15.0 and OpenCV 5.1.0, returned 96 positions versus 98 in the frozen output. It matched the frozen result on both grass samples at 10 pixels. The tables use the frozen tracks throughout.

Run `track-wasb` with `--stride 1` and `--stride 3` on each excerpt to regenerate the two WASB tracks. On the amateur clips, use `evaluate-track` with each surface's `review.json` and `control-review.json`. The local run saved the tracks, per-frame scores, and timestamp-checked side-by-side raw TrackNet and WASB stride-1 overlays under `outputs/wasb-comparison-mps/`.
