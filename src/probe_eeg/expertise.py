"""Side angle: can EEG tell experts from novices, or is it just telling men from women?

Experts here are 9 men / 2 women, novices 9 women / 2 men, so "male -> expert" alone
scores 82%. The honest test shuffles expert labels *within* each sex: that null already
knows everything sex knows, so beating it means the EEG sees expertise itself.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from probe_eeg.features import MAX_ERP_UV

SETS = {"pre": ("pre.",), "probe": ("probe.",), "both": ("pre.", "probe.")}
N_PERM, SEED = 1000, 42


def people(df: pd.DataFrame) -> pd.DataFrame:
    """One row per person: mean of every feature over their clean probes. Raw, not z-scored:
    a trait lives in exactly the between-person differences stage 3 scaled away."""
    df = df[df.pre_clean & df.probe_clean & (df.erp_dev_uv < MAX_ERP_UV)]
    feats = df.columns[df.columns.str.startswith(("pre.", "probe."))]
    p = (
        df.groupby("subject")[list(feats)]
        .mean()
        .join(df.groupby("subject")[["group", "sex"]].first())
    )
    return p.assign(expert=(p.group == "expert").astype(int), male=(p.sex == "M").astype(int))


def lopo(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-one-person-out scores; >0 leans towards class 1."""
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])
    model = make_pipeline(StandardScaler(), lda)
    return cross_val_predict(model, X, y, cv=LeaveOneOut(), method="decision_function")


def shuffle_within(y: np.ndarray, strata: np.ndarray, seed: int) -> np.ndarray:
    rng, y = np.random.default_rng([SEED, seed]), y.copy()
    for s in np.unique(strata):
        i = np.flatnonzero(strata == s)
        y[i] = rng.permutation(y[i])
    return y


def p_value(X, y, strata, auc, n_perm) -> float:
    def null(k):
        yp = shuffle_within(y, strata, k)
        return roc_auc_score(yp, lopo(X, yp))

    with parallel_config("loky", inner_max_num_threads=1):
        nulls = np.array(Parallel(n_jobs=-1)(delayed(null)(k) for k in range(n_perm)))
    return (1 + np.sum(nulls >= auc)) / (1 + n_perm)


def evaluate(p: pd.DataFrame, n_perm: int = N_PERM) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (one row per feature set x target, one row per person with every score)."""
    rows, scores = [], p[["group", "sex"]].copy()
    anyone = np.zeros(len(p))  # one stratum = a plain shuffle across people
    rows.append(
        {"target": "expert", "features": "sex only", "auc": roc_auc_score(p.expert, p.male)}
    )
    for target in ("expert", "male"):
        y = p[target].to_numpy()
        for name, prefixes in SETS.items():
            X = p.loc[:, p.columns.str.startswith(prefixes)].to_numpy()
            s = scores[f"{target}_{name}"] = lopo(X, y)
            auc = roc_auc_score(y, s)
            row = {
                "target": target,
                "features": name,
                "auc": auc,
                "accuracy": np.mean((s > 0) == y),
            }
            row["p_vs_chance"] = p_value(X, y, anyone, auc, n_perm)
            if target == "expert":  # the one that matters
                row["p_beyond_sex"] = p_value(X, y, p.male.to_numpy(), auc, n_perm)
            rows.append(row)
    scores["odd_one_out"] = (p.expert == 1) != (p.male == 1)  # female experts, male novices
    return pd.DataFrame(rows), scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", type=Path, default=Path("data/processed/features.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/tables"))
    ap.add_argument("--perm", type=int, default=N_PERM)
    a = ap.parse_args()

    summary, scores = evaluate(people(pd.read_csv(a.features)), a.perm)
    a.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(a.out / "expertise_summary.csv", index=False)
    scores.to_csv(a.out / "expertise_scores.csv", index_label="subject")
    print(summary.round(3).to_string(index=False))
    print("\nThe tie-breakers (sex says one thing, group another):")
    print(scores[scores.odd_one_out].filter(regex="group|sex|^expert_").round(2).to_string())


if __name__ == "__main__":
    main()
