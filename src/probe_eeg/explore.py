"""Stage 5a (exploratory): is model A just a drowsiness meter, and does H3 hold?"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from statsmodels.discrete.conditional_models import ConditionalLogit

from probe_eeg.features import MAX_ERP_UV
from probe_eeg.figures import VERTEX
from probe_eeg.models import within_auc

KEY = ["subject", "session", "probe"]


def clogit(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Within-person logistic regression (one intercept per person, never estimated)."""
    X = df[cols].apply(lambda c: (c - c.mean()) / c.std())  # per-SD odds ratios
    fit = ConditionalLogit(df.y.to_numpy(), X.to_numpy(), groups=df.subject.to_numpy()).fit(
        disp=False
    )
    return pd.DataFrame({"odds_ratio_per_sd": np.exp(fit.params), "p": fit.pvalues}, index=cols)


def controls(scores: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Does A's score still predict mind wandering once drowsiness and time are in the room?"""
    d = scores.merge(meta, on=KEY)
    d["position"] = d.groupby(["subject", "session"]).probe.rank()
    rows = []
    for name in ("A", "B", "q3", "position", "pre_p2p_uv", "probe_p2p_uv"):
        sub = d.dropna(subset=[name])
        auc = within_auc(sub.y.to_numpy(), sub[name].to_numpy(), sub.subject.to_numpy())[0]
        rows.append({"test": f"{name} alone", "within_auc": auc, "n": len(sub)})
    for grp, g in d.groupby("group"):
        auc = within_auc(g.y.to_numpy(), g.A.to_numpy(), g.subject.to_numpy())[0]
        rows.append({"test": f"A within {grp}s", "within_auc": auc, "n": len(g)})
    d = d.dropna(subset=["q3"])
    alone = clogit(d, ["A"]).loc["A"].to_dict()
    rows.append({"test": "clogit: A alone", **alone, "n": len(d)})
    rho = d.groupby("subject").apply(lambda g: spearmanr(g.A, g.q3).statistic, include_groups=False)
    rows.append({"test": "median rho(A, q3) within person", "rho": rho.median(), "n": len(d)})
    full = clogit(d, ["A", "q3", "position"])
    rows += [
        {"test": f"clogit: {c} | A + q3 + position", **full.loc[c].to_dict(), "n": len(d)}
        for c in full.index
    ]
    return pd.DataFrame(rows)


def h3(features: pd.DataFrame) -> pd.DataFrame:
    """Per person: Spearman of Q2 (0-3) with the vertex N1, P2, P3 and alpha change."""
    d = features[features.pre_clean & features.probe_clean & (features.erp_dev_uv < MAX_ERP_UV)]
    d = d.dropna(subset=["q2"])
    rows = []
    for k in ("n1", "p2", "p3", "alpha_change"):
        x = d[[f"probe.{k}.{ch}" for ch in VERTEX]].mean(1)
        rho = {
            s: spearmanr(g.q2, x[g.index]).statistic
            for s, g in d.groupby("subject")
            if len(g) >= 5 and g.q2.nunique() > 1
        }
        r = pd.Series(rho).dropna()
        rows.append(
            {
                "feature": k,
                "median_rho": r.median(),
                "n_subjects": len(r),
                "n_positive": int((r > 0).sum()),
                "p_two_sided": wilcoxon(np.arctanh(r.clip(-0.99, 0.99))).pvalue,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("data/processed/features.csv"))
    ap.add_argument("--tables", type=Path, default=Path("results/tables"))
    a = ap.parse_args()

    feats = pd.read_csv(a.features)
    meta = feats[KEY + ["group", "q3", "pre_p2p_uv", "probe_p2p_uv"]]
    c, h = controls(pd.read_csv(a.tables / "scores.csv"), meta), h3(feats)
    c.to_csv(a.tables / "controls.csv", index=False)
    h.to_csv(a.tables / "h3.csv", index=False)
    print(c.round(3).to_string(index=False), "\n\n", h.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
