# Credits and upstream notices

## Projects used by Snufspin

| Project | Contribution to Snufspin |
| --- | --- |
| philippdubach/tennis-vision | Baseline pipeline reference, cached detection artifact format, and the ball detector's input ordering, per-axis coordinate scaling, mask conversion, and candidate distance gate. |
| yastrebksv/TennisProject | External `BallTrackerNet` architecture loaded by the local Python tracker, pinned to `b7552e955253c008c7612298cb40694682a74ef9`. |
| yastrebksv/TrackNet | TrackNet implementation and the pretrained ball checkpoint linked by tennis-vision and TennisProject. |
| kzheng18/court-vision | Design reference for keeping measured, interpolated, and predicted positions distinguishable and evaluating tracking before downstream analytics. |
| VKorpelshoek/GridTrackNet | Public amateur and professional sample videos used for local tracking review. The source-video Git blob hashes and excerpt ranges are recorded in the benchmark manifests. |

Project sources are linked below. The current runner uses the external TennisProject model architecture with Snufspin's Python inference wrapper and interpolation. CourtVision and GridTrackNet contributed design references and review footage respectively.

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

The reviewed checkpoint has SHA-256 `c735bc1a1b13a35f179c6492f778ef4ebb9bffd512a96f4d970b32e076653076`. It is the ball model linked by yastrebksv/TrackNet. The pinned TennisProject source has no top-level license file in the inspected revision. Snufspin loads that source from an external checkout. Checkpoint hashes and footage origins are recorded in the benchmark manifests.

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
