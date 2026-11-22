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
