# Fixed-camera paint selection and refinement

This experiment replays the cached LETR segments from the [TEXflip benchmark](texflip.md). It uses the same seven calibration frames per clip, existing segment refinement, and metric court template. The manual points are used only to score the resulting court projections. The experiment evaluates an adaptation of Farin's image-based model scoring for a fixed camera. [Farin et al., §§3.3 and 4.2](https://www.dirk-farin.net/publications/data/Farin2004b.pdf#page=9)

## Results

Errors are in source pixels, reported as **median / 90th percentile** over all reviewed intersections, including points available only through the projected template.

| Clip | Current TEXflip | Paint selection | Paint selection + refinement |
| --- | --- | --- | --- |
| Grass (20 points) | **2.3 / 4.3** | Ambiguous | No accepted seed |
| Hardcourt (15 points) | Rejected across frames | **3.0 / 6.1** | **3.0 / 6.1**, retained the seed |
| Clay (13 points) | 2.7 / 15.5 | **2.4 / 17.0** | 3.7 / **14.9** |

- **Hardcourt.** Full-template scoring recovered a shared plane even though independent segment fits failed the across-frame check. Whole-court refinement lost paint support on the withheld frames, so the result retained the selected mapping.
- **Grass.** Two paint-supported mappings differed by 20.1 px at the 90th percentile across template intersections and were separated by less than 1% in total paint score. The provisional winner had 2.9 / 5.0 px error on the reviewed points. In a separate control, using the accepted current Grass mapping as the refinement seed produced no held-out gain. The control retained the 2.3 / 4.3 px fit.
- **Clay.** Refinement raised the withheld-frame paint score from 8,919 to 9,537. It improved the tail of the landmark errors while increasing the median. Mean far-baseline support moved from 62.6% in current TEXflip to 61.6% after selection and 63.4% after refinement. Both service lines stayed near their original support. The two largest refined errors were on the **near** baseline.
- **Full camera.** The existing camera checks, run with measured LETR intersections associated to each selected mapping, accepted zero full cameras. The hardcourt and clay sidecars contain plane mappings for surface points.

## Method

Each sampled frame contributes up to 24 distinct homographies from the existing line-pair search. We combine the shortlists and score each candidate on all seven frames. Projected lines are clipped to the image and sampled about every two source pixels. Local brightness relative to both flanks supplies the paint evidence. A supported sample contributes +1, a visible miss −0.5, and an off-screen point zero. Candidates need support in both court directions across a majority of frames. A close score between materially different fits flags ambiguity.

Refinement collects paint ridges from a narrow neighborhood of the selected lines. It back-projects the ridge locations, associates them with the nearest finite court marking, and minimizes the squared perpendicular distances in court coordinates. Ordered frames alternate between refinement and validation. The held-out frames participate in initial selection and are withheld from the refinement objective. A candidate update must preserve valid geometry, stay within a 24 px landmark displacement at 1920 px width, and improve the held-out paint score. This is the fixed-camera adaptation and reuses LETR hypotheses. Farin's Hough-based detection and moving-camera prediction are outside this experiment.

## Reproduce

```bash
python -m benchmarks.calibration.compare_paint data/labeling/*.mp4 \
  --output outputs/farin-court-paint/three-way
```

Defaults read cached segments under `outputs/texflip-court-detection/refined` and review labels under `outputs/texflip-court-detection/labels`. Regenerate the cache using the TEXflip benchmark command if needed. Use `--selection-only` for the first two variants. The output includes per-marking visible length and paint support, label residuals, camera checks, sidecars, and overlays. Cyan solid lines have at least half their visible samples supported. Orange dashed lines have weaker support or extend beyond the available paint evidence. Green crosses mark the independent review points.
