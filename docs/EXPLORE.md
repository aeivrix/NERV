# Stage 5: exploratory follow-ups (after the freeze, labelled as such)

None of this was in the [frozen plan](PREREG.md). Read it as hypothesis-generating.
Same 490 probes and the same leave-one-subject-out validation as stage 3.

## Is the reaction null just a weak model? (`deep.py`)

| Model | Window | Within-subject AUC | 95% CI | p (200 perms) |
|---|---|---|---|---|
| Riemannian: Xdawn covariances, tangent space, logistic | reaction | 0.500 | 0.436–0.552 | 0.46 |
| EEGNet (stock settings, 100 epochs, CPU) | reaction | 0.541 | 0.473–0.605 | not run (cost) |
| Riemannian: recentred covariances, tangent space, logistic | before | 0.527 | 0.464–0.586 | 0.34 |
| *Stage 3 band-power LDA (A, for reference)* | *before* | *0.597* | *0.516–0.661* | *0.021* |

- **The reaction null holds:** window means, covariance geometry and a CNN all fail on
  the reaction to the voice. EEGNet's interval includes chance.
- **For the pre-probe window, simple band power works best.** Covariance geometry
  scores lower, which fits a signal that lives in power, not in channel coupling.
- **Skipped:** EEGNet on the 10 s pre window (~8 h on a laptop CPU).

## Is model A a drowsiness meter? (`explore.py`, `results/tables/controls.csv`)

| Check | Result |
|---|---|
| A alone, within-person logistic | OR 1.40 per SD, p = 0.008 |
| A + drowsiness (Q3) + position in session | A: OR 1.29, p = 0.055 |
| Drowsiness rating alone (self-report, not EEG) | within-subject AUC 0.63 |
| Position in session alone | 0.57 |
| A vs Q3, within-person correlation | median ρ = 0.05 |
| A within experts / within novices | 0.63 / 0.57 |
| Artifact size (pre / probe peak-to-peak) | 0.50 / 0.49 |

- **A isn't reading drowsiness:** it hardly correlates with Q3.
- **Part of A is time on task:** with drowsiness and position in the model, A drops to
  borderline.
- **Artifacts:** amplitude is the same for both classes, so A isn't tracking muscle or
  movement.

## H3: weaker response when wandering? (`results/tables/h3.csv`)

Spearman of Q2 (0–3) with vertex amplitude, per person (20 people), then a Wilcoxon
test across people:

| Feature | Median ρ | p |
|---|---|---|
| N1 | 0.04 | 0.57 |
| P2 | 0.00 | 0.99 |
| P3 | 0.04 | 0.39 |
| Alpha change | 0.00 | 0.75 |

**Not supported.** The probe response doesn't scale with mind-wandering ratings either.
