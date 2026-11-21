"""Stage 2: scrub each recording, then cut one epoch per probe -> data/interim/epochs/."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from probe_eeg.labels import read_events

mne.set_log_level("error")  # MNE narrates everything; we only want the plot twists

PRE = (-10.05, -0.05)  # the classic: 10 s of the mind minding its own business
SOUND_DELAY_S = 0.59  # trigger -> voice in the ear; N1 lands ~686 ms, MATLAB took its sweet time
PROBE = (-0.2, 1.6)  # ~1 s of listening after the voice; first key press is never before 2.3 s
MAX_BAD = 12  # 13+ dead channels (>20%) and the recording sits out
MAX_P2P_V = (
    150e-6  # louder than this is a jaw, not a thought (judged above 1 Hz; drift is no crime)
)
UNKNOWN_QA_S = 27.0  # probe with no log entry: assume a 99th-percentile chatterbox
N_ICA = 25
SEED = 42
VERSION = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:8]  # new code, new scrub


def load(path: Path) -> mne.io.Raw:
    raw = mne.io.read_raw_bdf(path, preload=True).pick(range(64))
    montage = mne.channels.make_standard_montage("biosemi64")
    raw.rename_channels(
        dict(zip(raw.ch_names, montage.ch_names, strict=True))
    )  # A1..B32 is biosemi64 order
    return raw.set_montage(montage)


def clean(raw: mne.io.Raw) -> tuple[mne.io.Raw, dict]:
    """0.1-40 Hz keeps the ERP; ICA trains on a 1 Hz copy because drift confuses it."""
    raw.filter(0.1, 40.0)
    fit = raw.copy().filter(1.0, None)
    bads = mne.preprocessing.find_bad_channels_lof(fit)
    raw.info["bads"] = fit.info["bads"] = list(bads)
    for r in (raw, fit):
        r.set_eeg_reference("average")

    ica = mne.preprocessing.ICA(
        N_ICA, method="infomax", fit_params={"extended": True}, rng=SEED, max_iter="auto"
    )
    ica.fit(fit, decim=2)
    blinks = ica.find_bads_eog(fit, ch_name=["Fp1", "Fp2"])[
        0
    ]  # no EOG lead; the forehead snitches, bad or not
    muscle = ica.find_bads_muscle(fit)[0]
    ica.exclude = sorted(set(blinks) | set(muscle))
    ica.apply(raw)

    raw.interpolate_bads()
    raw.set_eeg_reference("average")  # re-average now the ghosts are back
    return raw, {
        "bad_channels": ";".join(bads),
        "n_bad": len(bads),
        "n_ica_removed": len(ica.exclude),
    }


def previous_qa_end(onsets, markers, qa_by_onset) -> np.ndarray:
    """When the last chatter before each probe ended (-inf if nothing came before)."""
    ends = np.array([m + qa_by_onset.get(m, UNKNOWN_QA_S) for m in markers])
    return np.array([ends[markers < t].max(initial=-np.inf) for t in onsets])


def epoch(raw: mne.io.Raw, probes: pd.DataFrame, markers: np.ndarray) -> mne.Epochs:
    """One PRE..PROBE epoch per probe; windows get flagged, never silently dropped."""
    samples = raw.first_samp + np.round(probes.onset_s.to_numpy() * raw.info["sfreq"]).astype(int)
    kw = dict(
        events=np.c_[samples, np.zeros_like(samples), np.ones_like(samples)],
        tmin=PRE[0],
        tmax=PROBE[1],
        baseline=None,
        preload=True,
    )
    ep = mne.Epochs(raw, metadata=probes.reset_index(drop=True), **kw)
    hp = mne.Epochs(raw.copy().filter(1.0, None), **kw)  # filter first, cut second: no frayed edges

    x, t = hp.get_data(copy=False), ep.times

    def p2p(lo, hi):  # worst channel's swing in the window, per epoch
        return np.ptp(x[..., (t >= lo) & (t <= hi)], axis=-1).max(axis=1)

    md = ep.metadata.copy()
    qa = dict(zip(md.onset_s, md.qa_s.fillna(UNKNOWN_QA_S), strict=True))
    md["pre_p2p_uv"], md["probe_p2p_uv"] = p2p(*PRE) * 1e6, p2p(*PROBE) * 1e6
    md["pre_clean"] = (md.pre_p2p_uv < MAX_P2P_V * 1e6) & (
        md.onset_s + PRE[0] >= previous_qa_end(md.onset_s.to_numpy(), markers, qa)
    )
    md["probe_clean"] = md.probe_p2p_uv < MAX_P2P_V * 1e6
    ep.metadata = md
    return ep


def run(bids: Path, sub: str, ses: str, probes: pd.DataFrame, out: Path, force: bool) -> dict:
    name = f"{sub}_{ses}_task-meditation"
    target, qc_path = out / f"{name}-epo.fif", out / f"{name}-qc.json"
    if target.exists() and qc_path.exists() and not force:
        qc = json.loads(qc_path.read_text())
        if qc.get("version") == VERSION:
            return qc  # already scrubbed by this exact code, no double-dipping

    eeg = bids / sub / ses / "eeg" / name
    raw, qc = clean(load(eeg.with_name(name + "_eeg.bdf")))
    ep = epoch(raw, probes, read_events(eeg.with_name(name + "_events.tsv"))[0])

    tmp = out / f"{name}.part-epo.fif"
    ep.save(tmp, overwrite=True)
    os.replace(tmp, target)  # half-written files are a lie we refuse to tell
    md = ep.metadata
    qc |= {
        "version": VERSION,
        "usable": qc["n_bad"] <= MAX_BAD,
        "subject": sub,
        "session": ses,
        "n_probes": len(probes),
        "n_epochs": len(ep),
        "n_pre_clean": int(md.pre_clean.sum()),
        "n_probe_clean": int(md.probe_clean.sum()),
    }
    qc_path.write_text(json.dumps(qc))
    return qc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bids", type=Path, default=Path("data/raw/ds001787"))
    ap.add_argument("--probes", type=Path, default=Path("data/interim/probes.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/interim/epochs"))
    ap.add_argument("--jobs", type=int, default=4)  # ~2 GB RAM each, measured
    ap.add_argument("--only", nargs="*", help="e.g. sub-001 for a quick spin")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    probes = pd.read_csv(a.probes)
    probes = probes[probes.triggers_agree]  # only answers the EEG vouches for
    if a.only:
        probes = probes[probes.subject.isin(a.only)]
    a.out.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(a.jobs) as pool:
        jobs = [
            pool.submit(run, a.bids, sub, ses, g, a.out, a.force)
            for (sub, ses), g in probes.groupby(["subject", "session"])
        ]
        qc = pd.DataFrame([j.result() for j in jobs])
    qc.to_csv(a.out / "qc.csv", index=False)
    print(qc.to_string(index=False))
    print(
        f"\n{qc.n_epochs.sum()} epochs; clean pre {qc.n_pre_clean.sum()}, "
        f"clean probe {qc.n_probe_clean.sum()}"
    )


if __name__ == "__main__":
    main()
