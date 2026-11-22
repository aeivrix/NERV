# Report: does the brain's reaction to a thought probe reveal the answer?

## Question

Meditation studies interrupt people with a voice ("were you mind wandering?") and
decode the answer from the EEG in the 10 s *before* the voice. Nobody had asked
whether the brain's *reaction* to the voice itself carries the answer, in the second
before the person replies.

## Data

- **Main dataset:** OpenNeuro ds001787 (Brandmeyer & Delorme 2018). 24 meditators,
  64-ch EEG, 938 probes rated for depth (Q1), mind wandering (Q2) and drowsiness (Q3).
- **Replication:** ds008836 (Cannard, Reggente et al.). 37 Vipassana meditators,
  spoken prompts, depth rated 1–5.

## What we did

1. **Recovered the true labels.** `events.tsv` silently drops every 0 rating. Ratings
   were taken from the behavioural logs instead, after aligning each log to its
   recording (99.5% trigger agreement). See [DATA.md](DATA.md).
2. **Measured when the voice reaches the ear** (0.59 s after the trigger, from the
   N1) and timed every reaction feature from there.
3. **Froze the analysis** ([PREREG.md](PREREG.md)), then ran it:
   - band power before the probe (A), the reaction (B), and both (C);
   - shrinkage LDA, leave-one-subject-out;
   - AUC within each person;
   - 1,000 permutations.
4. **Tested whether expert-vs-novice decoding is really sex decoding**
   ([EXPERTISE.md](EXPERTISE.md)).
5. **Followed up** with heavier models, drowsiness and time-on-task controls, and H3
   ([EXPLORE.md](EXPLORE.md)), all labelled exploratory.
6. **Replicated** on ds008836 with the same frozen pipeline
   ([REPLICATION.md](REPLICATION.md)).
7. **Built a meter:** model A swept continuously across a session, with an interactive
   replay, live at https://aeivrix.github.io/NERV/.

## Findings

| | ds001787 | ds008836 |
|---|---|---|
| A: 10 s before → state | **0.60** (p = 0.021) | 0.54 (p = 0.26) |
| B: reaction → state | 0.49 (p = 0.65) | 0.49 (p = 0.61) |
| C: both | 0.53 (p = 0.31) | 0.52 (p = 0.41) |

1. **The reaction to the probe doesn't reveal the answer.** This holds in two labs,
   with window means, Riemannian geometry and EEGNet, and in a within-person
   correlation with the rating (H3). It's the most consistent result here.
2. **The pre-probe signal is weak and fragile.**
   - It's significant in ds001787.
   - It shrinks to borderline once time on task is controlled.
   - It doesn't reach significance in ds008836, where the rating covers ~3 min and
     there's less data.
3. **Expert-vs-novice decoding on ds001787 mostly reads sex.** EEG decodes sex at AUC
   0.92, but expertise only at 0.69, which isn't significant beyond sex. Three of the
   four people whose sex and group disagree are classified by their sex.
4. **Data issues anyone using ds001787 should know** ([DATA_NOTE.md](DATA_NOTE.md)):
   - zero ratings are missing from `events.tsv`;
   - logs and recordings aren't one-to-one;
   - the voice arrives 0.59 s after its trigger.

## Limits

- **Sample size:** 22 people in the main analysis. A small reaction effect (AUC up to
  ~0.57) can't be ruled out.
- **Labels are self-reports:** mind wandering and meditation are what people said.
- **The preregistration is a git timestamp, not OSF.** One condition-split ERP was
  viewed before the freeze, and that is disclosed.

## Reproduce

`make setup data labels preprocess models expertise explore replicate demo`
(about 6 h on a laptop, mostly ICA and permutations).
