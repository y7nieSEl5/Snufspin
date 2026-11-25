# CalTennis pose with fixed GeoCalib intrinsics

The pose benchmark uses the same nine decoded frames as the [blind comparison](README.md) and [segment replay](segment-replay.md). For each view, it takes the elementwise median of GeoCalib's three pinhole intrinsic matrices and fixes that matrix throughout the pose fit. It fits only the three rotation and three translation parameters. The CalTennis camera files are opened after pose fitting and ranking are complete.

| View | Selected seed | Fixed `fx / fy` (px) | Supplied camera center (m) | Estimated camera center (m) | Center error (m) | Rotation error | Court-world ray median / p95 |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| Mar 4 SW | 25 s | 1748.0 / 1748.0 | (−6.257, 14.152, 1.738) | (−6.061, 14.082, 1.679) | 0.216 | 0.276° | 0.365° / 0.619° |
| Mar 9 SW | 700 s | 1703.0 / 1703.0 | (−6.248, 14.618, 1.714) | (−5.712, 14.332, 1.597) | 0.618 | 0.657° | 0.927° / 1.859° |
| Oct 2 N | 700 s | 1747.5 / 1747.5 | (30.127, 4.214, 2.211) | (30.024, 4.157, 2.158) | 0.129 | 0.250° | 0.232° / 0.345° |

The ray angles compare directions at the same 65 × 37 grid of decoded pixels after each camera rotates its rays into court coordinates. Position error is the Euclidean distance between ray origins. Rotation error is the geodesic angle between the two orientations. The supplied intrinsics use the same resize into the decoded 1920 × 1088 grid as the earlier benchmark. The four-pixel translation sensitivity check gives ray p95 values of 0.572°, 1.840°, and 0.279° respectively, with the same center errors.

A painted court admits two equivalent labelings related by a 180° turn. For scoring, both labelings are evaluated and the one with lower projected landmark p90 is used. Oct 2 uses the flipped labeling. The reference is used for this evaluation alignment only, and the blind result retains its original court labeling.

## Fitting and ranking

- Initialize from each frame's **vanishing-direction grouping plus direct segment homography**. Optimize signed distances from the observed fragment endpoints to their matched projected court lines, with a robust loss. This avoids requiring detected crossings or matching endpoints to particular places on the template. Require a finite pose with the camera above the court and all four corners at positive depth.
- Score each resulting pose unchanged across all three frames. For each frame, assign length-filtered fragments to the pose's two vanishing directions within 4°, then score one-to-one matches to the projected markings by distance and overlap. Require both the existing paint check and at least two segment matches in each direction on a majority of frames. Rank by median segment score, with paint score breaking ties.
- The selected poses passed both support checks on 3/3 Mar 4 frames, 2/3 Mar 9 frames and 2/3 Oct 2 frames. The Mar 9 25 s candidate passed the paint-only check across all three frames despite a 30.37 m center error. The combined segment-and-paint check rejected it. On Oct 2, the 25 s candidate also passed and had lower reference errors, 0.025 m center and 0.177° ray p95. The blind cross-frame scorer ranked 700 s first.

## Reproduce

From the repository root, with the model dependencies and local videos in place:

```bash
python -m benchmarks.calibration.estimate geocalib
python -m benchmarks.calibration.replay_segments
python -m benchmarks.calibration.geocalib_court_pose
```

The output includes [blind poses and per-frame support](../../outputs/caltennis-geocalib-pose/blind.json), [all scored candidates](../../outputs/caltennis-geocalib-pose/scores.json) and overlays of each selected pose on all three sampled frames.
