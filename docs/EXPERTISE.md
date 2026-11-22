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

## Results (1,000 permutations each)

| Target | Features | AUC | Accuracy | p vs chance | p beyond sex |
|---|---|---|---|---|---|
| expert | sex only | 0.82 | | | |
| expert | pre | 0.69 | 0.68 | 0.089 | 0.36 |
| expert | probe | 0.36 | 0.41 | 0.70 | 0.61 |
| expert | both | 0.59 | 0.45 | 0.21 | 0.37 |
| **sex** | **pre** | **0.92** | **0.86** | **0.003** | |
| sex | probe | 0.35 | 0.41 | 0.73 | |
| sex | both | 0.73 | 0.68 | 0.053 | |

**Tie-breakers** (pre-probe model; a positive score means "expert"):

| Person | Truth | Model says | Matches |
|---|---|---|---|
| sub-003 | female expert | expert (+18.6) | group |
| sub-005 | female expert | novice (−6.8) | sex |
| sub-016 | male novice | expert (+18.8) | sex |
| sub-023 | male novice | expert (+43.9) | sex |

## Reading it

- **The EEG reads sex, not expertise.**
  - Pre-probe band power tells men from women in unseen people with AUC 0.92
    (p = 0.003).
  - The same features score AUC 0.69 for expertise, not significant even against
    chance, and nowhere near significant beyond sex (p = 0.36).
  - Three of the four tie-breakers are classified by their sex.
- **The probe reaction carries neither.** Both its AUCs are below 0.5. That's the
  known downward bias of leave-one-out with 22 cases; the permutation null has the
  same bias, so the p-values hold.
- **Implication:** high expert-vs-novice accuracies on ds001787 deserve a sex
  baseline and a within-sex null before anyone believes them. The test is one
  function (`shuffle_within`) and works for any classifier.
- **Power:** 22 people, of whom only 4 break the confound. A modest real expertise
  effect could hide here. The claim is "not shown beyond sex", not "absent".
