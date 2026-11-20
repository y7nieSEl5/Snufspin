# CalTennis court calibration review

The three locally downloaded videos have matching CalTennis calibration JSONs. The published world-to-camera poses project the court with its origin at the low-side corner of baseline A, +X toward baseline B, +Y across the court, and +Z up. This matches `tennis-metric-v1` directly. The reference fit retains the supplied extrinsics and the supplied distortion field, which is null on these three clips.

## Decoded dimensions

The paper reports 1920×1080 iPhone recordings, and each calibration JSON says 1920×1080. FFprobe reports **1920×1088 for both the visible and coded sizes** of all three published MP4s. Their H.264 sequence headers have 68 macroblock rows and `frame_cropping_flag = 0`. The files carry FFmpeg/libx264 re-encode tags. In all nine checked frames, the top and bottom eight rows show scene detail, with variation both across each row and between rows, including across the 1079→1080 boundary. This rules out hidden codec rows and ordinary constant or repeated-edge padding.

A vertical resize during re-encoding is the working explanation. For example, ImageIO's FFmpeg writer explicitly scales 1080 up to 1088 with its default 16-pixel alignment. The dataset does not publish the re-encode command or matching pre-encode frames. The court measurements distinguish a downwards pixel adjustment from leaving the intrinsics as-is, but the supplied poses and line measurements have enough error that they do not uniquely identify resize versus a roughly four-pixel vertical offset. The offset scores below describe an alternative pixel mapping, even though the image evidence rules out ordinary centered padding. [CalTennis recording and calibration](https://arxiv.org/html/2606.20542v1#S4) · [ImageIO FFmpeg writer](https://github.com/imageio/imageio-ffmpeg/blob/main/imageio_ffmpeg/_io.py)

The reference sidecars therefore record an **inferred** transform. They apply `K_decoded = A @ K_supplied`, with `u' = u` and `v' = (v + 0.5) * (1088 / 1080) - 0.5`. This scales `fy` and moves `cy` to about 544.004. The importer requires an explicit choice when the dimensions differ.

## Landmark and solver checks

I sampled the first frame at or after 25, 300, and 700 seconds in each clip. The [reviewed source-pixel measurements](caltennis-landmarks.json) are intersections of painted-line ridges fitted near the original calibration projections and visually screened for occlusion, the net, and nearby markings. These measurements check the registration of the on-court markings. They use the published projection as a search seed and are approximate, especially through the net and at shallow line intersections.

| Clip | Measurements | Identity vertical RMS | Resize vertical RMS | +4 px vertical RMS | +8 px vertical RMS | Resize median 2D error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mar 4, court 1 SW | 24 | 4.63 px | 2.33 px | 1.68 px | 4.02 px | 3.50 px |
| Mar 9, court 3 SW | 19 | 3.97 px | 0.94 px | 0.63 px | 4.12 px | 2.03 px |
| Oct 2, court 5 N | 17 | 4.11 px | 2.58 px | 2.85 px | 5.60 px | 3.27 px |

All reference cameras are above the surface and see the court at positive depth. The manual solver recovered a full camera on all three clips with the resized supplied intrinsics. The refit camera centers were 4–10 cm from the published centers. With no intrinsics, the two oblique SW views passed the self-calibration checks, while the N view kept the court mapping only.

The pretrained model was tested separately on clips from the local GridTrackNet sample set. The professional hardcourt clip passed the multi-frame marking check and retained its plane mapping. Its focal/pose solve failed. The professional clay and grass samples had inconsistent fits across frames. The three amateur samples did not yield marking-supported homographies.

## Reproduce

From the repository root, run the validator with the three matching files in `data/caltennis/originals` and `data/caltennis/calib`.

```bash
python -m benchmarks.calibration.validate_caltennis
```

The per-clip folders under `outputs/court-calibration/caltennis` contain the video-bound camera sidecars, PNG overlays, per-landmark residuals for all four transforms, and manual fits with and without supplied intrinsics. Cyan traces are the resized reference projections. Green crosses mark the reviewed observations. Use `--clip 03_04`, `--clip 03_09`, or `--clip 10_02` to run one clip. The Mar 4 camera is the reference for an initial fitter test because its view shows both court directions and its manual refit is closest to the published center.

To inspect the encoded dimensions and crop metadata for a clip:

```bash
ffprobe -v error -select_streams V:0 \
  -show_entries stream=width,height,coded_width,coded_height,sample_aspect_ratio:stream_tags=encoder \
  -of json "$video"
ffmpeg -hide_banner -loglevel info -i "$video" -frames:v 1 -c:v copy -bsf:v trace_headers -f null - 2>&1 \
  | rg 'pic_height_in_map_units_minus1|frame_cropping_flag|aspect_ratio_info_present_flag'
```
