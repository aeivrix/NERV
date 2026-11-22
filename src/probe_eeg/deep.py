"""Stage 5b (exploratory): heavier machinery on the very same probes.

Riemannian geometry (covariance matrices, the BCI-competition workhorse) and EEGNet (a
compact CNN) get the raw signals instead of hand-made features. If the reaction to the
voice hides something the window means missed, these are the tools that would find it.
"""

import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed, parallel_config
from pyriemann.estimation import Covariances, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.mean import mean_riemann
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from torch import nn

from probe_eeg.features import MAX_ERP_UV
from probe_eeg.labels import state
from probe_eeg.models import SEED, bootstrap_ci, shuffle_within, within_auc
from probe_eeg.preprocess import PRE, PROBE, SOUND_DELAY_S

mne.set_log_level("error")
WINDOWS = {"pre": PRE, "probe": (SOUND_DELAY_S - 0.05, PROBE[1])}
DECIM = 2  # 256 -> 128 Hz; the 40 Hz low-pass already happened, so no aliasing
N_PERM = 200


def load(features: Path, epochs_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Signals for every clean probe (fence-sitters too, for scaling), z-scored per channel
    within each recording. Returns metadata with y (NaN for fence-sitters) and both windows."""
    f = pd.read_csv(features, usecols=lambda c: "." not in c)
    f = f[f.pre_clean & f.probe_clean & (f.erp_dev_uv < MAX_ERP_UV)]
    metas, xs = [], {w: [] for w in WINDOWS}
    for (sub, ses), g in f.groupby(["subject", "session"]):
        ep = mne.read_epochs(epochs_dir / f"{sub}_{ses}_task-meditation-epo.fif")
        keep = ep.metadata.probe.isin(g.probe).to_numpy()
        x, t = ep.get_data(copy=False)[keep][..., ::DECIM], ep.times[::DECIM]
        x = (x - x.mean((0, 2), keepdims=True)) / x.std((0, 2), keepdims=True)
        for w, (lo, hi) in WINDOWS.items():
            xs[w].append(x[..., (t >= lo) & (t < hi)].astype(np.float32))
        metas.append(ep.metadata[keep])
    meta = pd.concat(metas, ignore_index=True)
    meta["y"] = state(meta).map({"mw": 1, "med": 0})
    return meta, {w: np.concatenate(v) for w, v in xs.items()}


def recenter(covs: np.ndarray, recordings: np.ndarray) -> np.ndarray:
    """Riemannian cousin of z-scoring: each recording's mean covariance becomes identity."""
    out = covs.copy()
    for r in np.unique(recordings):
        i = recordings == r
        w = invsqrtm(mean_riemann(covs[i]))
        out[i] = w @ covs[i] @ w
    return out


def riemann_scores(X, y, groups, window) -> np.ndarray:
    lr = LogisticRegression(class_weight="balanced", max_iter=1000)
    if window == "pre":  # X is already recentred covariances
        model = make_pipeline(TangentSpace(), lr)
    else:  # Xdawn learns ERP prototypes inside each training fold
        model = make_pipeline(XdawnCovariances(4, estimator="oas"), TangentSpace(), lr)
    return cross_val_predict(
        model, X, y, groups=groups, cv=LeaveOneGroupOut(), method="decision_function"
    )


class EEGNet(nn.Module):
    """Lawhern et al. 2018, stock settings: temporal conv, depthwise spatial, separable."""

    def __init__(self, n_ch: int, k: int = 64, f1: int = 8, d: int = 2, f2: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, f1, (1, k), padding=(0, k // 2), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, (n_ch, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5),
            nn.Conv2d(f1 * d, f1 * d, (1, 16), padding=(0, 8), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, 1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.LazyLinear(1),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1)).squeeze(1)


def eegnet_scores(X, y, groups, epochs=100, device="cpu") -> np.ndarray:
    """LOSO with fixed stock hyperparameters and a fixed epoch count: nothing tuned on test."""
    out = np.zeros(len(y))
    for test in np.unique(groups):
        tr, te = groups != test, groups == test
        torch.manual_seed(SEED)
        net = EEGNet(X.shape[1]).to(device)
        net(torch.zeros(2, *X.shape[1:], device=device))  # materialise the lazy layer
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        pos_weight = torch.tensor(
            (y[tr] == 0).sum() / (y[tr] == 1).sum(), dtype=torch.float32, device=device
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        xt, yt = (
            torch.tensor(X[tr], device=device),
            torch.tensor(y[tr], dtype=torch.float32, device=device),
        )
        gen = torch.Generator().manual_seed(SEED)
        net.train()
        for _ in range(epochs):
            for b in torch.randperm(len(yt), generator=gen).split(32):
                opt.zero_grad()
                loss_fn(net(xt[b]), yt[b]).backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            out[te] = net(torch.tensor(X[te], device=device)).cpu().numpy()
    return out


def summarise(name, y, groups, s, null=None) -> dict:
    auc, per = within_auc(y, s, groups)
    d = pd.DataFrame({"y": y, "g": groups}).groupby("g").y.agg(["sum", "count"])
    lo, hi = bootstrap_ci(per, (d["sum"] * (d["count"] - d["sum"]))[per.index])
    p = np.nan if null is None else (1 + np.sum(np.array(null) >= auc)) / (1 + len(null))
    return {"model": name, "within_auc": auc, "ci_lo": lo, "ci_hi": hi, "p_perm": p}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", type=Path, default=Path("data/processed/features.csv"))
    ap.add_argument("--epochs", type=Path, default=Path("data/interim/epochs"))
    ap.add_argument("--out", type=Path, default=Path("results/tables/deep.csv"))
    ap.add_argument("--perm", type=int, default=N_PERM)
    ap.add_argument("--device", default="cpu")  # measured: MPS is 2x slower for this tiny net
    a = ap.parse_args()

    meta, X = load(a.features, a.epochs)
    rec = (meta.subject + meta.session).to_numpy()
    riem = {  # Riemann eats covariances (pre) or float64 signals (probe, for Xdawn)
        "pre": recenter(Covariances("oas").transform(X["pre"].astype(float)), rec),
        "probe": X["probe"].astype(float),
    }
    lab = meta.y.notna().to_numpy()
    y, groups = meta.y[lab].astype(int).to_numpy(), meta.subject[lab].to_numpy()

    rows = []
    for w in WINDOWS:
        Xw = riem[w][lab]
        s = riemann_scores(Xw, y, groups, w)

        def null(k, Xw=Xw, w=w):
            yp = shuffle_within(y, groups, k)
            return within_auc(yp, riemann_scores(Xw, yp, groups, w), groups)[0]

        with parallel_config("loky", inner_max_num_threads=1):
            nulls = Parallel(n_jobs=-1)(delayed(null)(k) for k in range(a.perm))
        rows.append(summarise(f"riemann_{w}", y, groups, s, nulls))
        print(rows[-1], flush=True)

    # ponytail: EEGNet on the 10 s pre window is ~8 h on a laptop CPU; A is settled without it
    s = eegnet_scores(X["probe"][lab], y, groups, device=a.device)
    rows.append(summarise("eegnet_probe", y, groups, s))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
