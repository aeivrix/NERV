"""Figures for the write-up: run with `python -m probe_eeg.figures`."""

import argparse
from pathlib import Path

import matplotlib
import mne
import numpy as np

from probe_eeg.preprocess import PROBE, SOUND_DELAY_S

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the backend, or macOS opens windows)

mne.set_log_level("error")
VERTEX = ["Cz", "FCz", "C1", "C2", "FC1", "FC2"]  # where hearing a voice shows up first


def erp_figure(epochs_dir: Path, fig_path: Path) -> float:
    """Grand-average voice ERP, labels unseen (no peeking pre-registration); returns N1 (s)."""
    per_subject = {}
    for f in sorted(epochs_dir.glob("*-epo.fif")):
        if ".part" in f.name:
            continue
        ep = mne.read_epochs(f).crop(*PROBE)
        ep = ep[ep.metadata.probe_clean.to_numpy()].apply_baseline((PROBE[0], 0))
        x = ep.copy().pick(VERTEX).get_data(copy=False).mean(1) * 1e6
        per_subject.setdefault(ep.metadata.subject.iloc[0], []).append(x)

    t = ep.times
    subs = np.array([np.concatenate(v).mean(0) for v in per_subject.values()])
    mean, sem = subs.mean(0), subs.std(0, ddof=1) / np.sqrt(len(subs))
    win = (t > SOUND_DELAY_S + 0.05) & (t < SOUND_DELAY_S + 0.25)  # N1 lives here
    n1 = float(t[win][np.argmin(mean[win])])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t * 1e3, mean, color="C0", label=f"all clean probes ({len(subs)} subjects)")
    ax.fill_between(t * 1e3, mean - sem, mean + sem, color="C0", alpha=0.25)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(SOUND_DELAY_S * 1e3, color="k", ls=":", lw=0.8, label="voice reaches the ear")
    ax.axvline(n1 * 1e3, color="grey", ls="--", lw=0.8, label=f"N1 at {n1 * 1e3:.0f} ms")
    ax.set(
        xlabel="ms after probe trigger",
        ylabel="µV (fronto-central mean)",
        title="The brain hears the voice (QC: labels not used)",
    )
    ax.legend(frameon=False)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    return n1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=Path, default=Path("data/interim/epochs"))
    ap.add_argument("--out", type=Path, default=Path("results/figures/erp_qc.png"))
    a = ap.parse_args()
    print(f"N1 at {erp_figure(a.epochs, a.out) * 1e3:.0f} ms after the trigger -> {a.out}")


if __name__ == "__main__":
    main()
