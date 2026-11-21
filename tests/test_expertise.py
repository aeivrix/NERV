"""The side angle's one job: catch an EEG model that only learned sex."""

import numpy as np
import pandas as pd

from probe_eeg.expertise import evaluate


def fake_people(signal: str) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    expert = np.repeat([1, 0], 11)
    male = np.r_[np.ones(9), 0, 0, 0, 0, np.zeros(7), 1, 1].astype(int)  # 9M/2F vs 9F/2M
    cause = male if signal == "sex" else expert
    x = rng.normal(size=(22, 8)) + 3.0 * cause[:, None] * (np.arange(8) < 3)
    p = pd.DataFrame(x, columns=[f"pre.alpha.{i}" for i in range(8)])
    p[[f"probe.n1.{i}" for i in range(4)]] = rng.normal(size=(22, 4))
    return p.assign(expert=expert, male=male, group=np.where(expert, "expert", "novice"), sex="?")


def test_sex_only_brain_fools_the_naive_test_but_not_the_honest_one():
    s = evaluate(fake_people("sex"), n_perm=49)[0].set_index(["target", "features"])
    assert s.loc[("expert", "pre"), "p_vs_chance"] < 0.05  # "wow, it decodes expertise!"
    assert s.loc[("expert", "pre"), "p_beyond_sex"] > 0.2  # no: it decodes sex


def test_real_expertise_survives_the_honest_test():
    s, scores = evaluate(fake_people("expertise"), n_perm=49)
    s = s.set_index(["target", "features"])
    assert s.loc[("expert", "pre"), "p_beyond_sex"] < 0.05
    assert scores.odd_one_out.sum() == 4
