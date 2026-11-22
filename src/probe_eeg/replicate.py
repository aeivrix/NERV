"""Stage 6: the same pipeline on ds008836 (Vipassana, depth 1-5, different lab and amps).

Does "the 10 s before the prompt tells the state, the reaction to it doesn't" survive
a new lab? Prompts here are spoken ("Please rate the depth of your meditation..."), and
the first click never comes before ~7 s, so the reaction window is press-free again.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from probe_eeg.features import features
from probe_eeg.figures import erp_figure
from probe_eeg.models import dataset, evaluate
from probe_eeg.preprocess import MAX_BAD, PRE, PROBE, UNKNOWN_QA_S, VERSION, clean, epoch

mne.set_log_level("error")
BLOCKS = ("silent_probe", "sapo_probe")  # prompted blocks only; "emerge" reports are self-timed
N1_AFTER_SOUND_S = 0.1  # the textbook N1 latency, used to turn the measured N1 into a delay


def prompts(events: Path) -> pd.DataFrame:
    """One row per prompt: onset, block, depth (NaN unless 1-5), and when the Q&A ended."""
    ev = pd.read_csv(events, sep="\t")
    rows = []
    for block in BLOCKS:
        b = ev[ev.block == block]
        starts = b[b.trial_type == "prompt_window_begin"].onset.to_numpy()
        for i, t in enumerate(starts):
            nxt = starts[i + 1] if i + 1 < len(starts) else np.inf
            after = b[(b.onset > t) & (b.onset < nxt)]
            depth = after[after.trial_type == "depth_rating"].depth.astype(float)
            conf = after[after.trial_type == "confidence_rating"].onset
            d = depth.iloc[0] if len(depth) else np.nan
            rows.append(
                {
                    "block": block,
                    "probe": i,
                    "onset_s": t,
                    "depth": d if 1 <= d <= 5 else np.nan,  # 0, 6, 8: clicker slips
                    "qa_s": conf.iloc[0] - t if len(conf) else UNKNOWN_QA_S,
                }
            )
    return pd.DataFrame(rows)


def run(bids: Path, sub: str, ses: str, out: Path) -> list[dict]:
    """Clean and epoch each prompted block in one session (cropped: 90 min -> ~35)."""
    name = f"{sub}_{ses}_task-meditation"
    eeg = bids / sub / ses / "eeg"
    table, qcs = prompts(eeg / f"{name}_events.tsv"), []
    for block, p in table.groupby("block"):
        stem = f"{sub}_{ses}_{block}"
        target, qc_path = out / f"{stem}-epo.fif", out / f"{stem}-qc.json"
        if target.exists() and qc_path.exists():
            qc = json.loads(qc_path.read_text())
            if qc.get("version") == VERSION:
                qcs.append(qc)
                continue
        start = max(p.onset_s.min() + PRE[0] - 30, 0)  # 30 s of runway for the filters
        stop = p.onset_s.max() + PROBE[1] + 30
        raw = mne.io.read_raw_eeglab(eeg / f"{name}_eeg.set", preload=False)
        chans = pd.read_csv(eeg / f"{name}_channels.tsv", sep="\t")
        raw = (
            raw.pick(chans.name[chans.type == "EEG"].tolist())
            .crop(start, min(stop, raw.times[-1]))
            .load_data()
        )
        raw = mne.add_reference_channels(raw.resample(256), "FCz")  # online ref, back in play
        raw.set_montage("standard_1005")
        raw, qc = clean(raw)
        rel = p.assign(onset_s=p.onset_s - start, subject=sub, session=ses)
        ep = epoch(raw, rel, rel.onset_s.to_numpy())

        tmp = out / f"{stem}.part-epo.fif"
        ep.save(tmp, overwrite=True)
        os.replace(tmp, target)
        qc |= {"version": VERSION, "usable": qc["n_bad"] <= MAX_BAD, "subject": sub}
        qc |= {"session": ses, "block": block, "n_epochs": len(ep)}
        qc |= {"n_pre_clean": int(ep.metadata.pre_clean.sum())}
        qc |= {"n_probe_clean": int(ep.metadata.probe_clean.sum())}
        qc_path.write_text(json.dumps(qc))
        qcs.append(qc)
    return qcs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bids", type=Path, default=Path("data/raw/ds008836"))
    ap.add_argument("--out", type=Path, default=Path("data/interim/ds008836"))
    ap.add_argument("--jobs", type=int, default=3)  # 16 GB laptop; each job holds a 35 min block
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()

    people = pd.read_csv(a.bids / "participants.tsv", sep="\t")
    subs = people[people.in_analysis == "yes"].participant_id  # the authors' own inclusion
    subs = [s for s in subs if not a.only or s in a.only]
    a.out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.jobs) as pool:
        jobs = [
            pool.submit(run, a.bids, s, ses.name, a.out)
            for s in subs
            for ses in sorted((a.bids / s).glob("ses-*"))
        ]
        qc = pd.DataFrame([q for j in jobs for q in j.result()])
    qc.to_csv(a.out / "qc.csv", index=False)
    print(qc.drop(columns="bad_channels").to_string(index=False))

    # Where does the prompt's voice land? Measured label-blind, like the 0.59 s in ds001787.
    fig = Path("results/figures/ds008836_erp_qc.png")
    n1 = erp_figure(a.out, fig, n1_search=(0.05, 1.2), delay=None)
    delay = round(n1 - N1_AFTER_SOUND_S, 2)
    assert delay + 1.01 <= PROBE[1], "voice lands too late for the saved probe window"
    print(f"N1 at {n1 * 1e3:.0f} ms -> voice lands ~{delay:.2f} s after the prompt marker")

    usable = set(qc[qc.usable].apply(lambda r: f"{r.subject}_{r.session}_{r.block}", axis=1))
    eps = [mne.read_epochs(a.out / f"{u}-epo.fif") for u in sorted(usable)]
    feats = pd.concat([features(ep, delay) for ep in eps], ignore_index=True)
    feats["session"] = feats.session + "_" + feats.block  # one recording = one block
    feats["y"] = feats.depth.map({1: 1, 2: 1, 4: 0, 5: 0})  # 1 = shallow, the "wandering" end
    scores, per, summary = evaluate(dataset(feats), a.perm)
    tables = Path("results/tables")
    summary.assign(delay_s=delay).to_csv(tables / "ds008836_summary.csv", index=False)
    per.to_csv(tables / "ds008836_subject_auc.csv", index_label="subject")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
