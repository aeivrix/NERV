"""Stage 3a: one row of numbers per probe -> data/processed/features.csv (labels not touched)."""

import argparse
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

from probe_eeg.preprocess import PRE, SOUND_DELAY_S

mne.set_log_level("error")

BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
ERP = {"n1": (0.05, 0.15), "p2": (0.15, 0.25), "p3": (0.25, 0.6)}  # s after the voice lands
BASE = (-1.05, -0.05)  # last quiet second before the trigger
MAX_ERP_UV = 100  # textbook +-100 uV; past that the "ERP" is a sweaty electrode


def band_power(x: np.ndarray, sf: float, nperseg: int) -> dict[str, np.ndarray]:
    """log10 Welch power per band, shape (epochs, channels)."""
    f, p = welch(x, sf, nperseg=nperseg, axis=-1)
    return {b: np.log10(p[..., (f >= lo) & (f < hi)].mean(-1)) for b, (lo, hi) in BANDS.items()}


def features(ep: mne.Epochs, delay: float = SOUND_DELAY_S) -> pd.DataFrame:
    """pre.* = what the mind was doing; probe.* = how it took the interruption.

    `delay` is when the voice actually reaches the ear (s after the trigger)."""
    x, t, sf = ep.get_data(copy=False), ep.times, ep.info["sfreq"]

    def win(lo, hi):
        return x[..., (t >= lo) & (t < hi)]

    out = {f"pre.{b}": v for b, v in band_power(win(*PRE), sf, 512).items()}
    base, heard = band_power(win(*BASE), sf, 256), band_power(win(delay, delay + 1.01), sf, 256)
    out |= {f"probe.{b}_change": heard[b] - base[b] for b in ("theta", "alpha")}
    erp = (x - win(delay - 0.25, delay - 0.05).mean(-1, keepdims=True)) * 1e6  # hush before it
    for k, (lo, hi) in ERP.items():
        w = (t >= delay + lo) & (t < delay + hi)
        out[f"probe.{k}"] = erp[..., w].mean(-1)  # window means: jitter-proof, unlike peaks
    in_erp = (t >= delay + ERP["n1"][0]) & (t < delay + ERP["p3"][1])

    cols = {f"{k}.{ch}": v[:, i] for k, v in out.items() for i, ch in enumerate(ep.ch_names)}
    md = ep.metadata.reset_index(drop=True)
    md["erp_dev_uv"] = np.abs(erp[..., in_erp]).max((1, 2))  # worst channel, worst moment
    return pd.concat([md, pd.DataFrame(cols)], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=Path, default=Path("data/interim/epochs"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/features.csv"))
    a = ap.parse_args()

    rows = []
    for f in sorted(a.epochs.glob("*-epo.fif")):
        if ".part" in f.name:
            continue
        qc = json.loads(f.with_name(f.name.replace("-epo.fif", "-qc.json")).read_text())
        if qc["usable"]:  # sub-010 stays benched
            rows.append(features(mne.read_epochs(f)))
    df = pd.concat(rows, ignore_index=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(
        f"{len(df)} probes x {df.columns.str.contains('.', regex=False).sum()} features -> {a.out}"
    )


if __name__ == "__main__":
    main()
