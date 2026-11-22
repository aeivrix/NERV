"""The demo engine: model A as a continuous mind-wandering meter over a whole session.

Every 2 s it reads the last 10 s of EEG and says how "wandering-like" that looks compared
with the rest of the session. The probes are the ground truth it gets checked against.

    python -m probe_eeg.meter --bdf any_biosemi64_recording.bdf      # bring your own
    python -m probe_eeg.meter --replay                                # all of ds001787
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import mne
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from probe_eeg.features import band_power
from probe_eeg.labels import state
from probe_eeg.models import dataset, within_auc
from probe_eeg.preprocess import MAX_P2P_V, PRE, UNKNOWN_QA_S, clean, load

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

mne.set_log_level("error")
WIN, STEP = PRE[1] - PRE[0], 2.0  # same 10 s the model was trained on, every 2 s


def train(features: Path, leave_out: str | None = None):
    """Model A on every labelled probe, minus one person if we're about to replay them."""
    df = dataset(pd.read_csv(features))
    df = df[df.subject != leave_out]
    cols = df.columns[df.columns.str.startswith("pre.")]
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])
    return lda.fit(df[cols].to_numpy(), df.y.to_numpy()), list(cols)


def sweep(raw: mne.io.Raw, model, cols, busy: list[tuple[float, float]]) -> pd.DataFrame:
    """Score every 10 s window; Q&A chatter and jaw clenches get flagged, not scored."""
    sf, x = raw.info["sfreq"], raw.get_data()
    hp = raw.copy().filter(1.0, None).get_data()
    n, starts = int(WIN * sf), np.arange(0, raw.times[-1] - WIN, STEP)
    idx = (starts * sf).astype(int)[:, None] + np.arange(n)
    parts, swing = [], []
    for i in range(0, len(idx), 100):  # 100 windows at a time; all at once is ~2 GB
        parts.append(band_power(x[:, idx[i : i + 100]].transpose(1, 0, 2), sf, 512))
        swing.append(np.ptp(hp[:, idx[i : i + 100]], axis=-1).max(0))
    bp = {b: np.concatenate([q[b] for q in parts]) for b in parts[0]}
    feats = pd.DataFrame(
        {f"pre.{b}.{ch}": v[:, i] for b, v in bp.items() for i, ch in enumerate(raw.ch_names)}
    )[cols]
    end = starts + WIN
    quiet = np.array(
        [not any(s < b1 and e > b0 for b0, b1 in busy) for s, e in zip(starts, end, strict=True)]
    )
    ok = quiet & (np.concatenate(swing) < MAX_P2P_V)
    z = (feats - feats[ok].mean()) / feats[ok].std()  # same per-recording scaling as training
    score = model.decision_function(z.fillna(0).to_numpy())
    return pd.DataFrame({"t": end, "score": np.where(ok, score, np.nan)})  # t = moment of reading


def replay(bids: Path, sub: str, ses: str, probes: pd.DataFrame, features: Path) -> dict:
    """One recording, scored by a model that never met this person."""
    model, cols = train(features, leave_out=sub)
    name = f"{sub}_{ses}_task-meditation"
    raw, _ = clean(load(bids / sub / ses / "eeg" / f"{name}_eeg.bdf"))
    p = probes[(probes.subject == sub) & (probes.session == ses) & probes.triggers_agree]
    busy = [
        (t, t + (q if q == q else UNKNOWN_QA_S)) for t, q in zip(p.onset_s, p.qa_s, strict=True)
    ]
    m = sweep(raw, model, cols, busy)
    at = np.searchsorted(m.t.to_numpy(), p.onset_s.to_numpy() + PRE[1], side="right") - 1
    p = p.assign(reading=m.score.to_numpy()[np.clip(at, 0, None)], state=state(p))
    return {
        "subject": sub,
        "session": ses,
        "group": p.group.iloc[0],
        "t": m.t.round(1).tolist(),
        "score": [None if np.isnan(s) else round(float(s), 3) for s in m.score],
        "probes": [
            {k: (None if v != v else v) for k, v in r.items()}  # NaN -> null for JSON
            for r in p[["onset_s", "q1", "q2", "q3", "state", "reading"]]
            .round(3)
            .to_dict("records")
        ],
    }


def scorecard(sessions: list[dict]) -> dict:
    """How well the meter's reading just before each probe matches the answer, per person."""
    rows = [
        (s["subject"], p["state"] == "mw", p["reading"])
        for s in sessions
        for p in s["probes"]
        if p["state"] in ("mw", "med") and p["reading"] is not None
    ]
    g, y, r = (np.array(c) for c in zip(*rows, strict=True))
    auc, per = within_auc(y.astype(int), r, g)
    return {
        "within_auc": round(auc, 3),
        "n_probes": len(rows),
        "per_subject": per.round(3).to_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bdf", type=Path, help="score your own BioSemi 64 recording")
    ap.add_argument("--replay", action="store_true", help="replay every ds001787 recording")
    ap.add_argument("--bids", type=Path, default=Path("data/raw/ds001787"))
    ap.add_argument("--features", type=Path, default=Path("data/processed/features.csv"))
    ap.add_argument("--probes", type=Path, default=Path("data/interim/probes.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/demo"))
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    if a.bdf:
        model, cols = train(a.features)
        m = sweep(clean(load(a.bdf))[0], model, cols, busy=[])
        m.to_csv(a.out / f"{a.bdf.stem}_meter.csv", index=False)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(m.t / 60, m.score.rolling(15, min_periods=5, center=True).mean(), color="C3")
        ax.axhline(0, color="k", lw=0.6)
        ax.set(xlabel="minutes", ylabel="< meditating · wandering >", title=a.bdf.stem)
        fig.savefig(a.out / f"{a.bdf.stem}_meter.png", dpi=150, bbox_inches="tight")
        print(f"{len(m)} readings -> {a.out}/{a.bdf.stem}_meter.csv / .png")
        return

    if a.replay:
        probes = pd.read_csv(a.probes)
        recs = pd.read_csv(a.features, usecols=["subject", "session"]).drop_duplicates()
        with ProcessPoolExecutor(a.jobs) as pool:
            jobs = [
                pool.submit(replay, a.bids, r.subject, r.session, probes, a.features)
                for r in recs.itertuples()
            ]
            out = [j.result() for j in jobs]
        (a.out / "replay.json").write_text(json.dumps({"sessions": out, "check": scorecard(out)}))
        print(f"{len(out)} sessions -> {a.out / 'replay.json'}", scorecard(out))


if __name__ == "__main__":
    main()
