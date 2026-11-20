# CalTennis candidate review pilot

The pilot uses three one-minute excerpts at native resolution and timestamps. The camera can be placed wherever the ball is useful to review. The [pinned plan](pilot.json) records the CalTennis revision, original file paths and hashes, exact frame ranges, and control windows chosen from clean footage before inference. The [excerpt manifest](excerpts.json) records the generated clip hashes and timing. October 2 and March 4 are development sessions. March 9 is held out for evaluation. [CalTennis dataset](https://huggingface.co/datasets/demalenk/caltennis)

## Review order

1. Start with `review-clean.mp4` to judge visibility and, where possible, record a ball center in `labels.csv`. The bottom bar gives the clip frame, original source frame, and clip time. The reel shows each selected source frame at 15 FPS for inspection. Use `clean.mp4` for motion at the original timing.
2. Use `review-candidates.mp4` to inspect the same frames and compare the alternatives. Yellow marks the selected candidate, cyan marks other candidates that passed the gate, and magenta marks rejected candidates. Numbers are zero-based indices into that frame's candidate list. The JSON contains the scores and source-pixel coordinates. `baseline-overlay.mp4` shows the complete minute with the frozen winner.
3. Enter one of `visible`, `occluded`, `out_of_frame`, or `uncertain` under `status`. For `visible`, enter the center in native pixels under `x_px` and `y_px`. If you confirm a false selection on a hand, mark `winner_on_hand=yes`; use `notes` for a candidate index, shot event, or latch boundaries. Leave unreviewed cells blank. The `control` windows were chosen using clean footage. `jump` and `slow_run` are mechanical review cues.

Each clip directory under `outputs/caltennis-pilot/` contains the clean source excerpt, `provenance.json` with both timestamp arrays, `track.json`, the full candidate log, and the review files.

## Reproduce a clip

Run from the repository root. Use the pinned source file, SHA-256, start frame, and count in `pilot.json`. Use a fresh output directory for each run.

```sh
mkdir -p "$dir"
PYTHONPATH=. python benchmarks/tracking/caltennis/prepare.py "$original" "$dir/clean.mp4" \
  --first "$first" --count "$count" --source-sha256 "$sha256" > "$dir/provenance.json"
python -m snufspin track-wasb "$dir/clean.mp4" \
  --upstream data/wasb/WASB-SBDT --weights data/wasb/wasb_tennis_best.pth.tar \
  --device mps --stride 1 --candidate-frames 0 "$((count - 1))" \
  --candidate-log "$dir/candidates.json" > "$dir/track.json"
```

Pass `--first-source-frame "$first"` and repeat `--control FIRST LAST` for the windows in the plan when calling `review_pack.py` with the clean clip, track JSON, candidate JSON, and output directory. For the full baseline overlay, run `python -m snufspin render "$dir/clean.mp4" --ball-track "$dir/track.json" --output "$dir/baseline-overlay.mp4"`.

## Label-driven scoring

```sh
python benchmarks/tracking/caltennis/score_review.py \
  "$dir/candidates.json" "$dir/labels.csv" > "$dir/assessment.json"
```

The scorer reports the existing candidate pool's 5- and 10-pixel outcomes, hidden-frame emissions by visibility status, and contiguous runs of reviewer-confirmed hand positions. A review window can end mid-latch. Use reviewer notes to identify boundaries outside the sampled frames. The saved [grass diagnosis](../wasb-grass-candidates.md) remains the regression case.
