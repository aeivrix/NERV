"""Stage 3b: models A (before), B (the reaction), C (both), leave-one-subject-out -> results/."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from probe_eeg.features import MAX_ERP_UV
from probe_eeg.labels import state

MODELS = {"A": ("pre.",), "B": ("probe.",), "C": ("pre.", "probe.")}
MIN_PER_CLASS = 3  # fewer of either and a per-subject AUC is a coin toss with extra steps
N_PERM, N_BOOT, SEED = 1000, 10_000, 42


def dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Same probes for A, B and C (every window clean); features z-scored per recording."""
    df = df[df.pre_clean & df.probe_clean & (df.erp_dev_uv < MAX_ERP_UV)].reset_index(drop=True)
    feats = df.columns[df.columns.str.startswith(("pre.", "probe."))]
    g = df.groupby(["subject", "session"])[feats]
    # Scaled on every clean probe, fence-sitters too: no label ever touches the scaling.
    z = ((df[feats] - g.transform("mean")) / g.transform("std")).fillna(0.0)
    df = df.drop(columns=feats).join(z)
    df["y"] = state(df).map({"mw": 1, "med": 0})
    return df.dropna(subset=["y"]).astype({"y": int}).reset_index(drop=True)


def loso(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Out-of-subject scores; >0 leans mind wandering. Shrinkage is auto, so no knob to peek at."""
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])
    return cross_val_predict(
        lda, X, y, groups=groups, cv=LeaveOneGroupOut(), method="decision_function"
    )


def within_auc(y, score, groups) -> tuple[float, pd.Series]:
    """P(an MW probe outscores an MED probe from the same person), plus per-subject AUCs.

    Only same-person pairs count, so "novices wander more" can't sneak in as a signal.
    """
    per, pairs = {}, {}
    for g, d in pd.DataFrame({"y": y, "s": score, "g": groups}).groupby("g"):
        n1 = d.y.sum()
        if 0 < n1 < len(d):
            per[g], pairs[g] = roc_auc_score(d.y, d.s), n1 * (len(d) - n1)
    per, pairs = pd.Series(per), pd.Series(pairs)
    return float((per * pairs).sum() / pairs.sum()), per


def shuffle_within(y: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    rng, y = np.random.default_rng([SEED, seed]), y.copy()
    for g in np.unique(groups):
        i = np.flatnonzero(groups == g)
        y[i] = rng.permutation(y[i])
    return y


def null_auc(X, y, groups, seed) -> float:
    yp = shuffle_within(y, groups, seed)
    return within_auc(yp, loso(X, yp, groups), groups)[0]


def bootstrap_ci(per: pd.Series, pairs: pd.Series) -> tuple[float, float]:
    """95% CI by resampling subjects, keeping each one's pair weight."""
    i = np.random.default_rng(SEED).integers(0, len(per), (N_BOOT, len(per)))
    a, w = per.to_numpy()[i], pairs.to_numpy()[i]
    return tuple(np.percentile((a * w).sum(1) / w.sum(1), [2.5, 97.5]))


def evaluate(df: pd.DataFrame, n_perm: int = N_PERM):
    """-> (per-probe scores, per-subject AUCs, one summary row per model)."""
    y, groups = df.y.to_numpy(), df.subject.to_numpy()
    counts = df.groupby("subject").y.agg(n_mw="sum", n=len)
    pairs = counts.n_mw * (counts.n - counts.n_mw)
    scores, per_subject, rows = df[["subject", "session", "probe", "y"]].copy(), {}, []
    for name, prefixes in MODELS.items():
        X = df.loc[:, df.columns.str.startswith(prefixes)].to_numpy()
        s = scores[name] = loso(X, y, groups)
        auc, per_subject[name] = within_auc(y, s, groups)
        lo, hi = bootstrap_ci(per_subject[name], pairs[per_subject[name].index])
        null = Parallel(n_jobs=-1)(delayed(null_auc)(X, y, groups, k) for k in range(n_perm))
        rows.append(
            {
                "model": name,
                "n_features": X.shape[1],
                "within_auc": auc,
                "ci_lo": lo,
                "ci_hi": hi,
                "p_perm": (1 + np.sum(np.array(null) >= auc)) / (1 + n_perm),
                "pooled_auc": roc_auc_score(y, s),
                "pooled_bal_acc": balanced_accuracy_score(y, s > 0),
            }
        )

    per = pd.DataFrame(per_subject).join(counts.assign(n_med=counts.n - counts.n_mw))
    solid = per[(per.n_mw >= MIN_PER_CLASS) & (per.n_med >= MIN_PER_CLASS)]
    summary = pd.DataFrame(rows).assign(n_probes=len(df), n_subjects=df.subject.nunique())
    summary["p_vs_A"] = [  # H2 (and B for the record): paired, one-sided, per subject
        np.nan if m == "A" else wilcoxon(solid[m], solid.A, alternative="greater").pvalue
        for m in summary.model
    ]
    per = per.drop(columns="n").assign(solid=per.index.isin(solid.index))
    return scores, per, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("data/processed/features.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/tables"))
    ap.add_argument("--perm", type=int, default=N_PERM)
    a = ap.parse_args()

    df = dataset(pd.read_csv(a.features))
    scores, per, summary = evaluate(df, a.perm)
    a.out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(a.out / "scores.csv", index=False)
    per.to_csv(a.out / "subject_auc.csv", index_label="subject")
    summary.to_csv(a.out / "summary.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
