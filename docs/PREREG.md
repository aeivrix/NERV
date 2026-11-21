# Analysis freeze (stage 3)

Written and committed before any model was run on real labels. The git commit that adds
this file is the timestamp. Copy it to OSF as is; list any later change under
*Deviations*.

## What was seen before the freeze

- The 2026-11-21 state-split ERP described in [PLAN.md](PLAN.md#polish).
- Label counts per subject, used to pick the eligibility rule below.
- Feature distributions without labels. These showed slow drift in single-trial ERPs,
  which led to the pre-voice baseline and the ±100 µV rule.

No model output, feature-by-label split or test statistic was seen.

## Data

- **Probes:** EEG triggers agree with the log, and the recording is usable (≤12 bad
  channels).
- **Clean on every window:**
  - pre window clean (stage 2);
  - probe window clean (stage 2);
  - every channel within ±100 µV of the pre-voice baseline during the ERP windows.
- **Labels:** mind wandering (MW) = Q2 ≥ 2 → 1; meditating (MED) = Q1 ≥ 2 and
  Q2 ≤ 1 → 0; all others are excluded.
- **Analysis set:** 490 probes from 22 subjects; 18 subjects have both classes and 17
  have at least 3 of each.
- **Same probes for all three models,** so A, B and C differ only in their features.

## Features (`src/probe_eeg/features.py`)

64 channels each. Times are relative to the probe trigger, and the voice arrives at
about 0.59 s.

| Set | Feature | Window |
|---|---|---|
| A (`pre.*`) | log10 Welch power, theta 4–8, alpha 8–13, beta 13–30 Hz | −10.05 to −0.05 s |
| B (`probe.*`) | alpha and theta log-power change, after voice − before trigger | 0.59–1.6 s vs −1.05 to −0.05 s |
| B | mean ERP amplitude, N1 / P2 / P3 | voice + 0.05–0.15 / 0.15–0.25 / 0.25–0.6 s |
| C | A + B | |

- **ERP baseline:** voice − 0.25 to voice − 0.05 s. No sound plays there yet, and
  it's close to the components, so drift is small.
- **Scaling:** each feature is z-scored within each recording, using all of that
  recording's clean probes, unlabelled ones included. No label touches the scaling.
  This removes between-person offsets, so the model sees only how each probe differs
  from that person's typical probe.

## Model and validation (`src/probe_eeg/models.py`)

- **Model:** linear discriminant analysis with automatic Ledoit–Wolf shrinkage and
  equal priors. There are no hyperparameters to tune, so nothing can be tuned on
  results.
- **Validation:** leave-one-subject-out. The test person's labels are never used.

## Outcomes and tests

- **Primary metric: within-subject AUC.** The probability that an MW probe scores
  higher than an MED probe from the same person, pooled over all same-person pairs.
  Cross-person pairs are ignored, so a model can't win by learning who wanders more.
- **H1:** model B's within-subject AUC is above chance.
  - Permutation test: 1,000 label shuffles within each subject, re-running the whole
    LOSO pipeline each time; one-sided, α = 0.05.
  - 95% confidence interval by bootstrapping over subjects (10,000 resamples).
- **H2:** per-subject AUC for C is higher than for A, among subjects with at least 3
  probes of each class. One-sided Wilcoxon signed-rank test, α = 0.05.
- **Also reported, not tested:**
  - A and C with the same permutation test;
  - pooled AUC and balanced accuracy (threshold 0);
  - a per-subject AUC figure.
- **Not in this stage:** H3 and the controls in [PLAN.md](PLAN.md#controls) (Q3, time
  on task, group/sex, artifact balance). These come in stage 4, with the same
  analysis set.

## Deviations from PLAN.md

- **Model:** shrinkage LDA instead of regularised logistic regression. Logistic
  regression needs a C value, and nested tuning × 1,000 permutations would take hours.
  A fixed C would be an unjustified choice. LDA chooses its shrinkage itself.
- **Scaling:** within-recording z-scoring. This uses the test person's unlabelled EEG,
  which PLAN.md excluded. It's label-free, and it stops the between-person base rate
  (novices wander more) from passing as a decoding result.
- **Probe features:** a P3 window was added, because attention capture by a probe is a
  P3 story.
- **Probe-window rule:** a ±100 µV drift rule was added for the ERP windows.

## Deviations after the freeze

None yet.
