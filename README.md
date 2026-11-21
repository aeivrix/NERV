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

## Setup

Needs [uv](https://docs.astral.sh/uv/). Python 3.14 and exact package versions come
from `.python-version` and `uv.lock`.

```bash
make setup     # install the environment
make data      # download ds001787 (~6 GB) and the behavioural logs into data/raw/
make labels    # stage 1: true ratings from the logs, aligned to the EEG
make preprocess  # stage 2: filter, ICA, probe-locked epochs (~1 h)
make models    # stage 3: features, models A/B/C, permutation tests (~1 h)
make test
```

If the dataset already exists elsewhere, link it instead of downloading:
`ln -s /path/to/ds001787 data/raw/ds001787` and fetch only the logs (see the `data`
target), or pass `BIDS=/path/to/ds001787` to `make`.

## Layout

```
data/raw/         original BIDS dataset, never edited
data/interim/     aligned labels, epochs
data/processed/   feature tables used by models
notebooks/        exploration, numbered: 01-labels.ipynb, 02-epochs.ipynb, ...
src/probe_eeg/    pipeline code; each stage runs as `python -m probe_eeg.<stage>`
tests/            checks for the riskiest code (label recovery)
results/          figures/ and tables/ that go into the write-up
docs/             plan and data notes
```

## License

Code: MIT. Data: CC0, from the original authors; please cite them.
