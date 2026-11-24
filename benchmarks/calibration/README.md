# Blind CalTennis camera comparison

## Results

The default automatic court pipeline returned **zero cameras from three views**. Both learned methods returned intrinsics on all nine frames. GeoCalib had lower median camera-frame ray error on all three views and lower p95 on Mar 4 and Oct 2. AnyCalib had a slightly lower p95 on Mar 9. Both methods underestimated Mar 9 focal length by roughly 5–7%, and that view had ray errors of about 2° near the tail.

| View | Method | Estimated `fx / fy` (px) | Signed `fx / fy` error | Camera ray median / p95 |
| --- | --- | ---: | ---: | ---: |
| Mar 4 SW | Supplied | 1773.6 / 1786.7 | | |
| | Automatic court | — | — | — |
| | GeoCalib | 1748.0 / 1748.0 | −1.44% / −2.17% | 0.255° / 0.488° |
| | AnyCalib | 1810.6 / 1785.6 | +2.08% / −0.06% | 0.299° / 0.602° |
| | Court with manual segments | 1737.2 / 1737.2 | −2.05% / −2.77% | 0.415° / 0.550° |
| Mar 9 SW | Supplied | 1794.3 / 1807.6 | | |
| | Automatic court | — | — | — |
| | GeoCalib | 1703.0 / 1703.0 | −5.09% / −5.79% | 0.937° / 1.857° |
| | AnyCalib | 1691.4 / 1682.7 | −5.73% / −6.91% | 1.154° / 1.774° |
| | Court with manual segments | — | — | — |
| Oct 2 N | Supplied | 1750.4 / 1763.4 | | |
| | Automatic court | — | — | — |
| | GeoCalib | 1747.5 / 1747.5 | −0.17% / −0.90% | 0.101° / 0.421° |
| | AnyCalib | 1747.9 / 1731.2 | −0.14% / −1.83% | 0.285° / 0.599° |
| | Court with manual segments | — | — | — |

TEXflip and court matching produced zero named crossings on the six SW frames and 4, 1, and 2 on the N frames. No frame passed the marking-supported homography check. With manually seeded paint segments, the same solver recovered the Mar 4 camera at **0.419° median and 0.624° p95 court-world ray error**, with a **0.24 m** camera-center error. Mar 9 failed the cross-frame mapping agreement check, and Oct 2 yielded a plane mapping with insufficient independent pose estimates.

The alternative four-pixel reference translation preserves the ray-error rankings for both learned methods. The primary comparison supports GeoCalib as the stronger of the two single-image intrinsics estimates on this sample. Full court-world calibration remains unverified for both learned methods, and the default court pipeline has no full-camera coverage on the sampled frames.

## Protocol

- Use the first presentation frame at or after 25, 300, and 700 seconds from each of the three local CalTennis views. Each run records the decoded BGR hash, frame index, presentation time, and image size. Scoring checks that these match across all methods. [CalTennis calibration](https://arxiv.org/html/2606.20542v1#S4)
- The automatic court baseline uses current Snufspin through TEXflip LETR, line refinement, metric court matching, and the multi-frame court solver. It uses the default centered, square-pixel pinhole assumptions and acceptance checks. GeoCalib and AnyCalib run independently on each of the same full frames using their official pinhole checkpoints. [GeoCalib](https://github.com/cvg/GeoCalib) · [AnyCalib](https://github.com/javrtg/AnyCalib)
- The supplementary manual run uses the same court solver. Its segment seeds were marked from the image at about 25 seconds, on a 960 × 544 preview. Each sampled frame independently refines those seeds on the visible paint in decoded pixels. The estimate commands read video pixels and, for this supplementary run, the blind paint seeds. Published intrinsics, poses, and the prior reference-seeded landmarks were withheld from these commands.
- The supplied camera uses a 1920 × 1080 grid and the video decodes to 1920 × 1088. The scoring stage converts the supplied intrinsics with the previously checked pixel-center resize. It also scores with a four-pixel vertical translation as a sensitivity check for the inferred re-encode mapping. [Earlier pixel-registration review](https://github.com/y7nieSEl5/Snufspin/tree/court-calibration/benchmarks/calibration)
- For each pixel on a 65 × 37 uniform full-image grid including its boundary, compute the unit viewing ray in the OpenCV camera frame. The angle between estimated and reference directions captures both focal length and principal point error. For a full court camera, also rotate both rays into court coordinates and report camera-center error separately. GeoCalib estimates gravity, which leaves court yaw and position unspecified. AnyCalib's fitted pinhole model supplies intrinsics.
- Report focal length and signed error separately for each axis. Each learned row uses the median focal length of its three single-frame estimates. Its ray percentiles pool the angles over those frames and grid points. Court-fit rows use the accepted shared camera over one grid. Rejections and plane-only fits have no camera score.

## Reproduce

Use the three CalTennis MP4s in `data/caltennis/originals` and the matching supplied JSONs in `data/caltennis/calib`. Set up TEXflip and its checkpoint using the [main README](../../README.md#fixed-camera-court-calibration). Run the rest from the repository root in the project Python environment.

```bash
python -m pip install kornia matplotlib
mkdir -p data/calibration-models/hub/{geocalib,anycalib}
git clone https://github.com/cvg/GeoCalib.git data/calibration-models/GeoCalib
git -C data/calibration-models/GeoCalib checkout 1dedb29f94fc11b896c5f572005978aeebccd67f
git clone https://github.com/javrtg/AnyCalib.git data/calibration-models/AnyCalib
git -C data/calibration-models/AnyCalib checkout 027a8497d893f4b2596f23d6324c05e4b81064ed
curl -fL https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar \
  -o data/calibration-models/hub/geocalib/pinhole.tar
curl -fL https://github.com/javrtg/AnyCalib/releases/download/v1.0.0/anycalib_pinhole.pt \
  -o data/calibration-models/hub/anycalib/anycalib_pinhole.pt

python -m benchmarks.calibration.estimate court
python -m benchmarks.calibration.estimate geocalib
python -m benchmarks.calibration.estimate anycalib
python -m benchmarks.calibration.estimate court-manual
python -m benchmarks.calibration.score --include-manual
```

The run used CPU inference with four Torch threads, Python 3.14.7, Torch 2.15.0, TorchVision 0.29.2 and OpenCV 5.1.0. The estimates and fitter diagnostics are written to `outputs/caltennis-blind`. The [checked-in scores](scores.json) include upstream revisions, checkpoint hashes, matching frame hashes and per-frame metrics.
