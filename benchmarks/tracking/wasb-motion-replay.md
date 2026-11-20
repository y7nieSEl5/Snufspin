# WASB motion replay pilot

Replay reads the frozen winners and decoded candidates. Each target uses up to six observed frame slots on either side of a seven-frame mask. A line needs three supporting winners and must extrapolate at most 100 ms to the target. A candidate must be within 3 heatmap pixels of both predictions when they agree. For a bend, the fits must approach within tolerance at a shared time inside the mask, within the extrapolation cap on both sides. The candidate is checked against its side of that contact.

Rerank keeps any supported baseline winner. When replacing or adding a point, it considers the decoded candidates, including those rejected by the old distance gate. Candidates within 0.1 heatmap pixel of the smallest motion error tie on motion, and the higher WASB score takes priority. Inconclusive checks keep the baseline. Both variants use the original winners for all fits, so choices made earlier in replay cannot become fit evidence.

## CalTennis at approximately 60 FPS

The defaults give a source-pixel tolerance of 11.25 px on the three 1920×1088 clips. The table reports changes against the frozen track, before interpolation. CalTennis has no reviewer labels in this run. Counts describe proposed edits and support, with no accuracy or hand-latch interpretation.

| Session | Logged frames with a supported check | Reject removes | Rerank removes | Rerank replaces | Rerank adds | Rerank picks a candidate rejected by the old gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| October 2 | 926 / 3,596 | 190 | 171 | 19 | 0 | 0 |
| March 4 | 505 / 3,596 | 108 | 89 | 19 | 0 | 3 |
| March 9, held out | 639 / 3,596 | 214 | 201 | 13 | 2 | 4 |
| Total | 2,070 / 10,788 | 512 | 461 | 51 | 2 | 7 |

## Effect of short-gap interpolation

With the existing 0.15-second gap filler, 305 of the 512 reject-only removals and 274 of the 461 rerank removals received an interpolated estimate. These positions keep interpolated provenance. Their accuracy still needs human review.

| Session | Baseline fills | Reject fills | Reject fills at removed frames | Rerank fills | Rerank fills at removed frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| October 2 | 254 | 368 | 116 | 358 | 106 |
| March 4 | 424 | 482 | 58 | 464 | 40 |
| March 9, held out | 564 | 689 | 131 | 685 | 128 |

## Grass regression

The existing candidate log covers frames 25–39 at 29.97 FPS. With the default three-frame mask, the closest possible evidence is four frames away, about 133 ms, so all 15 checks are inconclusive. A separate replay with a one-frame mask supports only one check and still changes zero positions. Frame 35 therefore retains the known wrong selection even though a decoded alternate is within 1 px of the manual center. Interpolation brings the labeled subset from 4/12 to 5/12 within 10 px for the frozen track and both replay variants.

## Reproduce and review

```sh
python -m snufspin replay-wasb "$video" --ball-track "$baseline" --candidate-log "$candidates" \
  --mode rerank --decision-log "$output/rerank-decisions.json" > "$output/rerank.json"
```

Run with `--mode reject` for the other variant. Use `interpolate-track` on each saved track for the separate gap-filling comparison. The local `outputs/wasb-motion-replay/` directory contains tracks, decision logs, exact changed frame numbers, and three-panel reels. The reels show frozen, reject-only, and rerank outputs in that order, before interpolation. They include the original review windows and examples of changes outside those windows. Candidate indices in the reels and decision logs refer to the original WASB candidate log.
