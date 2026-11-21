# One command per pipeline stage; each regenerates its outputs from data/raw/.
# Point BIDS at another copy of the dataset with: make labels BIDS=/path/to/ds001787
BIDS ?= data/raw/ds001787

.PHONY: setup data labels test lab

setup:            ## install the pinned environment from uv.lock
	uv sync

data:             ## download ds001787 (~6 GB) and the behavioural logs
	uv run openneuro-py download --dataset ds001787 --target-dir $(BIDS)
	mkdir -p data/raw/behaviour
	curl -sL -o data/raw/behaviour/MW_Current_TextFileBIDS.zip \
		https://s3.amazonaws.com/openneuro.org/ds001787/code/MW_Current_TextFileBIDS.zip

labels:           ## stage 1: true ratings from the logs, aligned to EEG -> data/interim/
	uv run python -m probe_eeg.labels --bids $(BIDS)

test:
	uv run --with pytest pytest -q tests

lab:
	uv run jupyter lab
