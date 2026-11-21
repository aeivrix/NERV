"""Checks for stage 3: the features land where they should and the models can't cheat."""

import mne
import numpy as np
import pandas as pd

from probe_eeg.features import features
from probe_eeg.models import dataset, evaluate, within_auc
from probe_eeg.preprocess import PRE, PROBE, SOUND_DELAY_S


def test_features_find_the_n1_and_the_alpha():
    sf, names = 256, mne.channels.make_standard_montage("biosemi64").ch_names
    t = np.arange(round(PRE[0] * sf), round(PROBE[1] * sf) + 1) / sf
    x = np.random.default_rng(0).normal(0, 1e-8, (2, 64, len(t)))  # a whisper: no log(0)
    x[:, names.index("Oz")] += 10e-6 * np.sin(2 * np.pi * 10 * t) * (t < 0)  # eyes-closed hum
    n1 = (t >= SOUND_DELAY_S + 0.05) & (t < SOUND_DELAY_S + 0.15)
    x[:, names.index("Cz"), n1] -= 5e-6  # the voice lands
    ep = mne.EpochsArray(
        x, mne.create_info(names, sf, "eeg"), tmin=t[0], metadata=pd.DataFrame({"probe": [0, 1]})
    )

    f = features(ep)
    assert np.allclose(f["probe.n1.Cz"], -5, atol=0.01)
    assert np.allclose(f["probe.n1.Fz"], 0, atol=0.01)
    assert (f["pre.alpha.Oz"] > f["pre.alpha.Fz"] + 3).all()  # 1000x the power, log10
    assert (f["probe.alpha_change.Oz"] < -3).all()  # alpha stops when the voice talks
    assert np.allclose(f.erp_dev_uv, 5, atol=0.05)


def test_within_auc_only_compares_people_to_themselves():
    y, s, g = [1, 0, 1, 1, 0, 1, 1], [1, 0, 0, 0, 1, 5, 5], list("aabbbcc")  # c: one class, out
    auc, per = within_auc(np.array(y), np.array(s), np.array(g))
    assert per.to_dict() == {"a": 1.0, "b": 0.0} and np.isclose(auc, 1 / 3)  # weighted 1:2 by pairs


def test_models_find_the_reaction_and_ignore_who_wanders_more():
    rng, rows = np.random.default_rng(0), []
    for k in range(8):
        rate = 0.2 + 0.08 * k  # later subjects wander more...
        for i in range(30):
            y = rng.random() < rate
            row = {"subject": f"s{k}", "session": "ses-01", "probe": i, "q1": 0 if y else 2}
            row |= {"q2": 2 if y else 0, "pre_clean": True, "probe_clean": i != 0, "erp_dev_uv": 50}
            row |= {f"pre.alpha.{c}": rng.normal(5 * rate) for c in range(10)}  # ...and look it
            row |= {f"probe.n1.{c}": rng.normal(1.5 * y * (c < 3)) for c in range(10)}
            rows.append(row)
    raw = pd.DataFrame(rows)
    raw.loc[1, ["q1", "q2"]] = 0  # a fence-sitter

    df = dataset(raw)
    assert len(df) == len(raw) - 8 - 1  # each probe 0 is unclean, plus one fence-sitter
    _, per, summary = evaluate(df, n_perm=19)
    s = summary.set_index("model")
    assert s.loc["B", "within_auc"] > 0.8 and s.loc["B", "p_perm"] == 0.05
    assert s.loc["C", "within_auc"] > 0.8
    # The subject-level "wanderers look different" signal is scaled away, within and pooled.
    assert abs(s.loc["A", "within_auc"] - 0.5) < 0.15 and abs(s.loc["A", "pooled_auc"] - 0.5) < 0.15
    assert per.solid.sum() >= 6
