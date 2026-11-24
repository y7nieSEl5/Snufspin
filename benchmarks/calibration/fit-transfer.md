# CalTennis fit transfer across time

Each row of the overlays uses one unchanged homography in all three columns. The selected frames are Mar 4 at 25 s, Mar 9 at 700 s, and Oct 2 at both 25 s and 700 s. Because the visual review identified frames without specifying a fit mode, both available vanishing-direction fits are shown for the two March views. Oct 2 has one distinct segment fit at each time.

| View | Unchanged projections at 25, 300, and 700 s |
| --- | --- |
| Mar 4 SW | [Crossing fit and segment fit from 25 s](../../outputs/caltennis-fit-transfer/03_04-transfer.jpg) |
| Mar 9 SW | [Crossing fit and segment fit from 700 s](../../outputs/caltennis-fit-transfer/03_09-transfer.jpg) |
| Oct 2 N | [Segment fits from 25 s and 700 s](../../outputs/caltennis-fit-transfer/10_02-transfer.jpg) |

## Named markings

Cyan shows the template projection and orange dots show paired ridge measurements. For each template marking, the benchmark samples the same perpendicular image scans in both frames. It locates a bright ridge in the seed frame and searches for corresponding paint in the target frame. The displacement statistics use only locations with ridge measurements in both frames, and the report includes the count for each line. A painted marking is identified by its template name and local image position. Large residuals, shadows, occlusions and adjacent edges can make the ridge correspondence ambiguous.

The table reports p90 absolute distance from the unchanged projection to the detected ridge, in decoded pixels. The same names are shown across each row. Larger entries can reflect occlusion or a different ridge being measured. The linked JSON also reports signed median and p90 absolute temporal displacement for each marking.

| Seed | Named marking | 25 s | 300 s | 700 s |
| --- | --- | ---: | ---: | ---: |
| Mar 4 25 s, crossings | `baseline_a` | 2.0 | 1.0 | 2.0 |
| | `singles_high` | 1.0 | 1.0 | 1.0 |
| | `service_a` | 1.0 | 2.0 | 5.0 |
| | `center_service` | 2.0 | 3.0 | 2.0 |
| Mar 9 700 s, segments | `baseline_a` | 3.0 | 1.0 | 1.0 |
| | `singles_high` | 2.0 | 1.0 | 1.0 |
| | `doubles_high` | 1.0 | 2.0 | 1.0 |
| | `service_a` | 10.0 | 11.8 | 0.0 |
| | `center_service` | 10.0 | 10.0 | 2.0 |
| Oct 2 25 s, segments | `baseline_a` | 3.0 | 3.0 | 3.0 |
| | `singles_high` | 2.0 | 8.0 | 2.5 |
| | `doubles_high` | 2.5 | 6.6 | 7.0 |
| | `center_service` | 2.0 | 9.2 | 3.0 |
| Oct 2 700 s, segments | `baseline_a` | 3.7 | 4.0 | 3.7 |
| | `singles_high` | 8.7 | 7.6 | 2.0 |
| | `doubles_high` | 10.2 | 5.0 | 3.0 |
| | `center_service` | 2.0 | 7.2 | 2.0 |

Independent scene registration used forward/backward feature tracks and a robust affine fit, with roughly 780–1,020 feature inliers per pair. Its p90 displacement evaluated over the visible court was **1.5–1.7 px on Mar 4**, **0.7–1.8 px on Mar 9**, and **1.3–3.2 px on Oct 2**. These small coherent shifts limit pixel-level agreement on an unchanged fit. They are much smaller than the large differences among framewise fits. The Oct 2 seed projections themselves differ by **36.1 px p90** over the court landmarks, while the independent scene motion between those two frames is **2.2 px p90** over the visible court. The cause of the small coherent shifts was unverified.

Mar 4's 25 s crossing fit transfers on the near baseline, singles sideline and center service line. The Mar 9 700 s segment fit transfers on the near baseline and measured sidelines, with weaker agreement at the service markings in the earlier frames. The Oct 2 cross-check transfers on the near baseline and center service line in both directions, with mixed sideline agreement. Several projected distant markings already have large ridge residuals in the seed images.

## One mapping per clip

The selection experiment reused the production Farin paint score and ambiguity margin on the earlier replay's candidate homographies. It ranked each candidate unchanged across the training frames. For leave-one-out, the target frame contributed neither candidates nor paint scores. The published camera was used afterwards to score p90 landmark projection discrepancy, allowing the court's 180° labeling symmetry.

| View | Top choice from all three frames | Reference p90 | Leave-one-out reference p90 for held-out 25 / 300 / 700 s |
| --- | --- | ---: | --- |
| Mar 4 SW | 700 s crossings, ambiguous with 25 s crossings | 34.4 px | 34.4 / 19.2 / 19.2 px |
| Mar 9 SW | 700 s segments, accepted | 8.5 px | 8.5 / 8.5 / 1,297.3 px |
| Oct 2 N | 25 s segments, accepted | 7.4 px | 38.8 / 7.4 / 7.4 px |

Mar 4's 25 s and 700 s crossing fits differ by 19.0 px p90 and their paint scores are 93 points apart, within the existing 2% ambiguity margin. On Mar 9, withholding the 700 s frame removes both of its candidates. The remaining winner still passes the paint gate on the held-out frame despite its large reference discrepancy. Multi-frame scoring can rank a transferable fit when the candidate pool contains one, and the held-out tests expose the limits of the paint check.

## Reproduce

```bash
python -m benchmarks.calibration.replay_segments
python -m benchmarks.calibration.transfer_fits
python -m benchmarks.calibration.select_clip_fit
```

[Named marking measurements and full-size overlay filenames](../../outputs/caltennis-fit-transfer/transfer.json) · [Clip selection results](../../outputs/caltennis-fit-transfer/selection.json)
