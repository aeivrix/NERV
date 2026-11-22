# One command per pipeline stage; each regenerates its outputs from data/raw/.
# Point BIDS at another copy of the dataset with: make labels BIDS=/path/to/ds001787
BIDS ?= data/raw/ds001787
JOBS ?= 4

.PHONY: setup data labels preprocess features models expertise explore replicate demo figures test lint lab

setup:            ## install the pinned environment from uv.lock
	uv sync

data:             ## download ds001787 (~6 GB) and the behavioural logs
	uv run openneuro-py download --dataset ds001787 --target-dir $(BIDS)
	mkdir -p data/raw/behaviour
	curl -sL -o data/raw/behaviour/MW_Current_TextFileBIDS.zip \
		https://s3.amazonaws.com/openneuro.org/ds001787/code/MW_Current_TextFileBIDS.zip

labels:           ## stage 1: true ratings from the logs, aligned to EEG -> data/interim/
	uv run python -m probe_eeg.labels --bids $(BIDS)

preprocess: labels  ## stage 2: filter, ICA, probe-locked epochs -> data/interim/epochs/
	uv run python -m probe_eeg.preprocess --bids $(BIDS) --jobs $(JOBS)
	$(MAKE) figures

features:         ## stage 3a: band power + ERP windows per probe, label-blind -> data/processed/
	uv run python -m probe_eeg.features

models: features  ## stage 3b: models A/B/C, leave-one-subject-out + permutations -> results/tables/
	uv run python -m probe_eeg.models
	$(MAKE) figures

expertise: features  ## side angle: expert vs novice, tested beyond sex -> results/tables/
	uv run python -m probe_eeg.expertise

explore: models   ## stage 5 (exploratory): controls, H3, Riemannian + EEGNet -> results/tables/
	uv run python -m probe_eeg.explore
	uv run python -m probe_eeg.deep

replicate:        ## stage 6: same pipeline on ds008836 (needs data/raw/ds008836) -> results/
	uv run python -m probe_eeg.replicate

demo: features    ## the meter swept over every session -> results/demo/ (open index.html)
	uv run python -m probe_eeg.meter --replay

figures:          ## ERP sanity check and per-subject AUCs -> results/figures/
	uv run python -m probe_eeg.figures

test:
	uv run --with pytest pytest -q tests

lint:
	uv run --with ruff ruff format src tests
	uv run --with ruff ruff check --fix src tests

lab:
	uv run jupyter lab
