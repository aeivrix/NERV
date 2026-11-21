# Does the brain's reaction to a question give away the answer?

During meditation, a recorded voice interrupts every ~30–90 s and asks people to rate
their meditation depth, mind wandering and drowsiness. Most EEG studies decode
mind wandering from the seconds *before* such a probe. This project tests whether the
brain's first second of reaction *to the probe itself* predicts the answer, and whether
it adds information beyond the standard pre-probe window.

- Research plan: [docs/PLAN.md](docs/PLAN.md)
- Known dataset issues: [docs/DATA.md](docs/DATA.md)

## Data

[OpenNeuro ds001787](https://openneuro.org/datasets/ds001787) v1.1.1 (CC0), from
Brandmeyer & Delorme, *Exp Brain Res* 2018,
[doi:10.1007/s00221-016-4811-5](https://doi.org/10.1007/s00221-016-4811-5).
The Zenodo record 2536267 is an older copy of the same recordings; don't combine the two.

Download (~6.1 GB) into `data/raw/` (git-ignored):

```bash
aws s3 sync --no-sign-request s3://openneuro.org/ds001787 data/raw/ds001787
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Layout

```
data/raw/         original BIDS dataset, never edited
data/interim/     aligned labels, epochs
data/processed/   feature tables used by models
notebooks/        exploration, numbered: 01-labels.ipynb, 02-epochs.ipynb, ...
src/probe_eeg/    reusable code imported by notebooks and scripts
results/          figures/ and tables/ that go into the write-up
docs/             plan and data notes
```

## License

Code: MIT. Data: CC0, from the original authors; please cite them.
