# CalTennis segment replay

The same nine decoded frame hashes match the [blind calibration run](README.md). All 82 LETR proposals passed the 65.28 px length cutoff after refinement. The steep group is `abs(dy) / length > 0.38` and each group retains at most its 12 longest segments.

| View | Time (s) | Length pass / proposals | Slope steep / shallow | VP along / across | Matched lines with VP | Crossings with VP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mar 4 SW | 25 | 11/11 | 1 / 10 | 7 / 4 | 8 | 5 |
| | 300 | 13/13 | 1 / 12 | 8 / 5 | 8 | 7 |
| | 700 | 12/12 | 1 / 11 | 8 / 4 | 8 | 4 |
| Mar 9 SW | 25 | 6/6 | 1 / 5 | 2 / 3 | 5 | 1 |
| | 300 | 5/5 | 1 / 4 | 2 / 3 | 4 | 1 |
| | 700 | 6/6 | 1 / 5 | 3 / 3 | 5 | 4 |
| Oct 2 N | 25 | 9/9 | 5 / 4 | 6 / 3 | 7 | 4 |
| | 300 | 10/10 | 6 / 4 | 8 / 2 | 5 | 1 |
| | 700 | 10/10 | 3 / 7 | 6 / 3 | 6 | 2 |

## Per-frame homographies

Numbers are p90 projection errors in decoded pixels across the 14 template landmarks against the supplied camera. The score takes the smaller p90 under the two 180° court labelings. The supplied cameras were opened only after fitting. Each nonempty cell passed the existing bright-ridge paint check on its own frame. These are per-frame court-plane fits.

| View | Time (s) | Slope + crossings | Slope + segments | VP + crossings | VP + segments |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mar 4 SW | 25 | — | — | 19.2 | 34.0 |
| | 300 | — | — | 25.1 | 31.7 |
| | 700 | — | — | 34.4 | 19.1 |
| Mar 9 SW | 25 | — | — | — | 1340.3 |
| | 300 | — | — | — | 1297.3 |
| | 700 | — | — | 38.8 | 8.5 |
| Oct 2 N | 25 | — | 7.4 | — | 7.4 |
| | 300 | — | 219.4 | — | 219.4 |
| | 700 | — | 38.8 | — | 38.8 |

On Mar 4, VP grouping produces a homography in all three frames through both fitting paths. The largest between-frame p90 spread over the 14 template landmarks is 19.0 px with crossings and 50.4 px with matched segments. The same spread for VP + segments reaches 1343.0 px on Mar 9. For Oct 2, the slope and VP paths produce identical matched-segment fits, with 217.8 px maximum between-frame spread. The per-frame paint check passes even on some overlays with large reference discrepancies.

## Reproduce

```bash
python -m benchmarks.calibration.replay_segments
```

The script reuses the production length filter, template hypothesis scoring, crossing fit, and per-frame paint check. The VP alternative proposes vanishing points from pairs of supporting image lines and considers two-group assignments within 4° of either direction. For each of the top 12 distinct partitions, it tests both group labels and retains the highest existing template score. Grouping is used to generate candidates. The existing line assignment scores those candidates against the full length-filtered segment set.

The direct fit uses those same named matches. It constrains the projected endpoints of each template marking to lie on its matched image line and uses a normalized linear homography solve. This step does not require the two observed fragments to reach a crossing. It requires two matched lines per direction and an identifiable homography.

The output under `outputs/caltennis-segment-replay` includes the exact frame PNGs, the detector's raw and refined segments, pre-scoring fits, and the full [replay metrics](../../outputs/caltennis-segment-replay/replay.json). Segment IDs are detector proposal indices starting at one, local to a frame and consistent across the four comparisons.
