# Side angle: expert vs novice, or men vs women?

Most ML papers on ds001787 classify experts vs novices. Here, group and sex are
almost the same thing.

| | Female | Male |
|---|---|---|
| Expert | 2 | 9 |
| Novice | 9 | 2 |

These are the 22 usable people. "Male → expert" alone is right 18 times out of 22
(AUC 0.82). An EEG model can reach high accuracy just by learning sex, and EEG does
differ by sex (skull thickness, alpha frequency). Epoch-level splits make this worse:
the same person ends up in both training and testing.

## Design (frozen before the run; `src/probe_eeg/expertise.py`)

- **One row per person:** the mean of each stage 3 feature over that person's clean
  probes. The features aren't z-scored, because a trait is a difference between
  people. No probe ratings are used.
- **Feature sets:** `pre` (192 band powers), `probe` (320 reaction features) and
  `both`.
- **Model:** standardise, then shrinkage LDA (equal priors), scored
  leave-one-person-out (22 folds).
- **Targets:** expert, and sex as a positive control.
- **Tests:** 1,000 permutations each.
  - **vs chance:** labels shuffled across all people.
  - **beyond sex (the real test):** expert labels shuffled only within men and within
    women. This null keeps the full sex–group link, so it scores as well as sex can;
    a model only beats it by seeing expertise itself.
- **Tie-breakers:** the 4 people whose sex points the wrong way (sub-003 and sub-005
  are female experts; sub-016 and sub-023 are male novices). A model reading sex gets
  all four wrong.

## What was seen before the freeze

- The Med-EEG project (a different pipeline on the same data) found subject-level EEG
  AUC 0.83 for expertise, against 0.82 for a sex-only baseline.
- The group × sex table above.

No result from this pipeline had been seen.
