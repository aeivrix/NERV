"""The meter reads alpha, and keeps its mouth shut during Q&A and clenches."""

import mne
import numpy as np

from probe_eeg.meter import STEP, WIN, sweep


class AlphaFp1:  # a "model" that only watches alpha at Fp1
    def decision_function(self, X):
        return X[:, 64]


def test_meter_tracks_alpha_and_skips_busy_or_noisy_windows():
    sf, names = 256, mne.channels.make_standard_montage("biosemi64").ch_names
    t = np.arange(0, 300, 1 / sf)
    x = np.random.default_rng(0).normal(0, 5e-6, (64, len(t)))
    x += 20e-6 * np.sin(2 * np.pi * 10 * t) * (t > 150)  # alpha switches on halfway
    x[:, int(250 * sf)] += 400e-6  # a clench
    raw = mne.io.RawArray(x, mne.create_info(names, sf, "eeg"))

    cols = [f"pre.{b}.{ch}" for b in ("theta", "alpha", "beta") for ch in names]
    m = sweep(raw, AlphaFp1(), cols, busy=[(60.0, 80.0)])

    assert np.allclose(np.diff(m.t), STEP) and m.t.iloc[0] == WIN
    assert m.score[(m.t > 60) & (m.t < 90)].isna().all()  # Q&A overlaps these readings
    assert m.score[(m.t > 250) & (m.t < 260)].isna().all()  # clench
    early, late = m.score[(m.t > 100) & (m.t < 140)], m.score[(m.t > 170) & (m.t < 240)]
    assert late.mean() > early.mean()
