# Research plan

## Question

When a voice probe interrupts meditation, does the EEG response in the first second
predict the rating the person gives about 3 s later? Does it add information beyond
the standard 10 s pre-probe window?

## Why it's new

- **Probe-locked decoding:** a literature search (about 45 queries across Europe PMC,
  OpenAlex and Semantic Scholar; Google Scholar not covered) found no study that
  time-locks to the thought probe itself and decodes its answer. The closest work uses
  separate task tones or oddball sounds, e.g.
  [Braboszcz & Delorme 2011](https://doi.org/10.1016/j.neuroimage.2010.10.008) and
  [Dong et al. 2021](https://doi.org/10.1371/journal.pone.0251490).
- **This dataset's ratings:** the ML papers on ds001787 that could be checked classify
  expert vs novice, mostly with epoch-level splits. Examples:
  [NeuroState 2026](https://doi.org/10.56578/ataiml050401) and
  [Islam & Tabassum 2024](https://doi.org/10.1016/j.health.2023.100286).
  None found uses the probe ratings as labels.
- **Labels:** no paper found mentions that `events.tsv` drops zero ratings (see
  [DATA.md](DATA.md)).

## Hypotheses

Fix these before looking at results; see *Preregistration* below.

- **H1:** the 0–1 s probe response predicts mind wandering vs meditation above chance
  in people the model has never seen.
- **H2:** pre-probe and probe-response features together beat pre-probe features
  alone.
- **H3:** higher mind-wandering ratings go with a weaker evoked response to the voice
  (smaller N1/P2) and less alpha suppression after the probe.

## Labels

Ratings come from the behavioural logs, not `events.tsv`.

- **Primary (binary):** mind wandering = Q2 ≥ 2; meditation = Q1 ≥ 2 and Q2 ≤ 1.
  Other probes are excluded.
- **Secondary:** Q2 as an ordinal 0–3 target.

## Methods

### 1. Label recovery and alignment
- Parse every log and match each one to its recording by probe timing.
- Drop sessions that don't align, and record why.
- Output: one table with one row per probe: subject, session, onset, Q1, Q2, Q3,
  response latency.

### 2. Preprocessing (MNE-Python)
- Common average reference, 1–40 Hz band-pass, 50 Hz notch.
- ICA to remove eye and muscle components; mark bad channels.
- Apply the same pipeline to every subject.

### 3. Epochs
- **Pre:** −10 to −0.05 s before probe onset, as in the original paper.
- **Probe:** −0.2 to 1.0 s around probe onset. This ends well before the median
  first key press at about 3.1 s.

### 4. Features
- **Pre:** log band power (theta, alpha, beta) per channel or region.
- **Probe:** ERP amplitudes (N1, P2) at central electrodes, and post-probe
  alpha/theta change relative to the pre-probe baseline.

### 5. Models
- **A:** pre-probe only, the standard approach.
- **B:** probe response only.
- **C:** A + B.
- Start with regularised logistic regression. Add a small CNN (EEGNet) only if the
  simple models work.

### 6. Evaluation
- Leave-one-subject-out cross-validation. Nothing from the test person is used for
  training, scaling or feature selection.
- Metrics: balanced accuracy and AUC, per subject and pooled.
- Chance level from permutation tests that shuffle labels within subjects.
- 95% confidence intervals by bootstrapping over subjects.
- H2: compare C and A on per-subject AUC with a paired test.

## Controls

These decide whether a result means anything.

1. **Pre-probe state:** C vs A tests whether the probe response adds more than what
   the brain was already doing.
2. **Drowsiness:** repeat with Q3 as a covariate. Also check whether the probe
   response predicts drowsiness better than mind wandering.
3. **Time on task:** add probe position in the session as a covariate, since mind
   wandering and drowsiness may rise over time.
4. **Group and sex:** include them as covariates, and report results within experts
   and within novices. Make no expert-vs-novice claims (see [DATA.md](DATA.md)).
5. **Artifacts:** check that eye and muscle components don't differ by class after ICA.
6. **Sanity check:** re-run the original paper's pre-probe theta/alpha group contrast.
   One public reanalysis reports it did not replicate
   ([evavillar/ds001787-meditation-reanalysis](https://github.com/evavillar/ds001787-meditation-reanalysis));
   not verified here.

## Polish

- **Preregistration:** post hypotheses, labels, features and the evaluation on OSF
  before running models. This turns "we found X" into "we predicted X".
- **Reproducibility:** pinned environment, fixed seeds, and one command per stage
  that regenerates every figure and table from `data/raw/`.
- **Data note:** write up the `events.tsv` zero-rating issue and the alignment
  problems, and offer them to the dataset authors as an OpenNeuro issue.
- **Honest reporting:** report null results with the same detail as positive ones,
  and include per-subject plots, not just pooled numbers.
- **Figures:**
  - grand-average probe ERP by rating;
  - per-subject AUC for models A, B and C;
  - alpha time course around the probe.

## Extension: replication on a second dataset

[OpenNeuro ds008836](https://openneuro.org/datasets/ds008836) has 39 Vipassana
meditators with probed depth reports on a 1–5 scale, 64-ch EEG and a different lab.
It probably has no subject overlap with ds001787.

- Test H1 and H3 for depth there.
- **Not yet checked:** whether its events mark exact probe onset times.
- Don't use ds003969 for this: same lab as ds001787, and subject overlap can't be
  ruled out.

## Risks

| Risk | Mitigation |
|---|---|
| Small sample (24 people, ~40 probes each), so effects may be small | Report confidence intervals and permutation tests; a well-powered null is still a result |
| The probe response may only reflect the pre-probe state | Model C vs A tests this directly |
| Some sessions may not align | Drop them and report how many |
| Few examples of one class for some subjects | Use balanced metrics and report per subject |

## Milestones

1. Label recovery and alignment, with an audit table.
2. Preprocessing and epochs; grand-average ERP figure.
3. Preregistration posted.
4. Models A, B and C with leave-one-subject-out evaluation; controls.
5. Write-up.
6. Optional: ds008836 replication.

## References

- Brandmeyer & Delorme 2018, *Exp Brain Res*.
  [doi:10.1007/s00221-016-4811-5](https://doi.org/10.1007/s00221-016-4811-5)
- Braboszcz & Delorme 2011, *NeuroImage*.
  [doi:10.1016/j.neuroimage.2010.10.008](https://doi.org/10.1016/j.neuroimage.2010.10.008)
- Dong et al. 2021, *PLOS ONE*.
  [doi:10.1371/journal.pone.0251490](https://doi.org/10.1371/journal.pone.0251490)
- Aviad et al. 2024, *Biol Psychiatry GOS*: meditation vs instructed mind wandering
  on different data.
  [doi:10.1016/j.bpsgos.2024.100423](https://doi.org/10.1016/j.bpsgos.2024.100423)
- Rodriguez-Larios et al. 2021, *NeuroImage*: aperiodic EEG, meditation vs mind
  wandering. [doi:10.1016/j.neuroimage.2021.118669](https://doi.org/10.1016/j.neuroimage.2021.118669)
