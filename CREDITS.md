# Credits and upstream notices

## Projects used by Snufspin

| Project | Contribution to Snufspin |
| --- | --- |
| nttcom/WASB-SBDT | Default WASB Tennis model and official preprocessing, heatmap decoding, and candidate selection. |
| philippdubach/tennis-vision | Legacy TrackNet pipeline reference and cached detection artifact format. |
| yastrebksv/TennisProject | External `BallTrackerNet` architecture for legacy runs, pinned to `b7552e955253c008c7612298cb40694682a74ef9`. |
| yastrebksv/TrackNet | TrackNet implementation and the pretrained ball checkpoint linked by tennis-vision and TennisProject. |
| kzheng18/court-vision | Design reference for keeping measured, interpolated, and predicted positions distinguishable and evaluating tracking before downstream analytics. |
| VKorpelshoek/GridTrackNet | Public amateur and professional sample videos used for local tracking review. The source-video Git blob hashes and excerpt ranges are recorded in the benchmark manifests. |

Snufspin loads the WASB model and tracker from the external checkout pinned in the setup instructions.

- [WASB-SBDT by NTT Communications](https://github.com/nttcom/WASB-SBDT/tree/923462cacdeb3353b84ddebdedb3f4b7a8553b0f)
- [tennis-vision by Philipp Dubach](https://github.com/philippdubach/tennis-vision)
- [TennisProject by yastrebksv](https://github.com/yastrebksv/TennisProject)
- [TrackNet by yastrebksv](https://github.com/yastrebksv/TrackNet)
- [CourtVision by kzheng18](https://github.com/kzheng18/court-vision)
- [GridTrackNet by VKorpelshoek](https://github.com/VKorpelshoek/GridTrackNet)

## Runtime and tooling

PyTorch runs the neural model, OpenCV resizes frames and extracts ball candidates, NumPy handles image arrays, and FFmpeg/ffprobe decode video, provide timestamps, and render overlays. The optional gdown setup tool downloads the upstream checkpoint.

- [PyTorch](https://github.com/pytorch/pytorch)
- [OpenCV](https://github.com/opencv/opencv)
- [NumPy](https://github.com/numpy/numpy)
- [FFmpeg](https://ffmpeg.org/)
- [gdown](https://github.com/wkentaro/gdown)

## Model and data provenance

The default WASB Tennis checkpoint has SHA-256 `9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb`. The pinned upstream source carries an MIT license. [WASB model zoo](https://github.com/nttcom/WASB-SBDT/blob/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/MODEL_ZOO.md) · [Upstream license](https://github.com/nttcom/WASB-SBDT/blob/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/LICENSE.md)

The historical TrackNet checkpoint has SHA-256 `c735bc1a1b13a35f179c6492f778ef4ebb9bffd512a96f4d970b32e076653076`. It is the ball model linked by yastrebksv/TrackNet. The pinned TennisProject source has no top-level license file in the inspected revision. Checkpoint hashes and footage origins are recorded in the benchmark manifests.

## tennis-vision MIT notice

The local ball detector adapts behavior from tennis-vision revision `a501953194f5207534273c7b2b59c0b642df7a05`. Its MIT notice follows.

MIT License

Copyright (c) 2026 Philipp Dubach

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### Court calibration

- TEXflip provides the tennis-trained LETR model and line-fitting approach. The tested source uses LETR submodule revision `814e4e2c538de36ccca45abc68cb1e1cfaaf98a4`. [TEXflip tennis-court-detection](https://github.com/TEXflip/tennis-court-detection/tree/49bae72d9f50d022be7d1aeeac4681f053a59188)
- Farin et al. describe the paint-based template scoring rule adapted for fixed-camera fallback. [Robust Camera Calibration for Sport Videos using Court Models](https://www.dirk-farin.net/publications/data/Farin2004b.pdf#page=9)
- TennisCourtDetector provides the optional heatmap detector. [Pretrained model](https://github.com/yastrebksv/TennisCourtDetector#pretrained-model)
- CalTennis provides the reference camera conventions and calibrations. [CalTennis calibration method](https://arxiv.org/html/2606.20542v1#S4)
