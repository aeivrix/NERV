# Replication on ds008836 (frozen before the run)

[OpenNeuro ds008836](https://openneuro.org/datasets/ds008836) (Cannard, Reggente et al., CC0):

- experienced Vipassana meditators, a different lab, different amplifiers (BrainAmp,
  500 Hz);
- spoken prompts asking for meditation depth since the last report (1–5).

## What carries over unchanged

- **Preprocessing:** `clean()` from stage 2 (0.1–40 Hz, LOF bad channels, ICA, average
  reference). The recording is resampled to 256 Hz, and the online reference FCz is
  added back.
- **Epochs and flags:** `epoch()` from stage 2 (pre window, probe window, Q&A-crowding
  rule).
- **Features:** `features()` from stage 3. **Models, evaluation and statistics:**
  `dataset()` and `evaluate()` from stage 3 (shrinkage LDA, LOSO, within-subject AUC,
  1,000 within-subject permutations, bootstrap, Wilcoxon C vs A).

## What had to change

- **Probes:** `prompt_window_begin` in the two prompted blocks (`silent_probe`,
  `sapo_probe`), about 9 per block, one of each block per person.
- **Recording unit:** one block = one recording for the z-scoring.
- **People:** the authors' own `in_analysis == yes` (37 of 39).
- **Labels:**
  - shallow (depth 1–2) = 1, the "wandering" end;
  - deep (depth 4–5) = 0;
  - depth 3, and invalid clicks (0, 6, 7, 8), are excluded.
- **Ratings cover more time:** each one is the deepest level *since the last report*
  (~3 min), not the moment of the probe, so the match to a 10 s window is looser than
  in ds001787.
- **Voice delay:** measured label-blind from the grand-average vertex N1 across all
  recordings, as delay = N1 − 0.1 s. A 2-subject pilot put the N1 at ~258 ms
  (delay ~0.16 s). The full-sample value is the one used.
- **Q&A end:** the confidence rating's timestamp.

## Predictions

- **R1:** model A (pre-prompt band power) separates shallow from deep above chance
  (permutation p < 0.05). This replicates ds001787's A.
- **R2:** model B (reaction to the prompt) is at chance, like ds001787. This is
  descriptive, since we expect a null.
- **R3:** C does not beat A.

## Seen before the freeze

- The event structure.
- Rating distribution (1: 48, 2: 139, 3: 238, 4: 176, 5: 58, invalid: 39).
- The 2-subject label-blind ERP.
- Per-block clean-epoch counts.

## Results (1,000 permutations each)

- **Voice delay:** the grand-average vertex N1 lands at 270 ms, so the voice arrives
  0.17 s after the prompt marker. This was set before any label was read.
- **Data:** 73 blocks, 72 usable. 375 probes from 37 people: 30 people have both
  classes, and 13 have at least 3 of each.

| Model | Within-subject AUC | 95% CI | p (perm) | p, C vs A |
|---|---|---|---|---|
| A: before the prompt | 0.536 | 0.473–0.597 | 0.258 | |
| B: reaction to the prompt | 0.486 | 0.405–0.572 | 0.614 | |
| C: both | 0.516 | 0.432–0.603 | 0.406 | 0.79 |

- **R1 (A above chance): not replicated.** The interval still overlaps ds001787's
  0.60 at its top, so the two results don't contradict each other, but this dataset
  alone shows nothing.
- **R2 (B at chance): as predicted.** The reaction to the probe carries no state
  information in a second lab either.
- **R3 (C doesn't beat A): as predicted.**

## Why A might not travel

- **The question differs:** ds008836 asks for the deepest level *since the last
  report*, a rating of the past ~3 min. ds001787 asks about the moment.
- **Less to learn from:** 375 probes, and only 13 people with at least 3 of each
  class (ds001787: 490 and 17).
- **Muscle-heavy recordings:** ICA removed a median of 12 of 25 components (ds001787:
  7). These recordings carry more neck and jaw muscle activity, which band power soaks
  up.
