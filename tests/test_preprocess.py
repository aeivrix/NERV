"""Checks for stage 2: the window flags decide what the models ever get to see."""

import mne
import numpy as np
import pandas as pd

from probe_eeg.preprocess import PRE, epoch, previous_qa_end


def test_previous_qa_end_uses_log_or_assumes_the_worst():
    markers = np.array([100.0, 130.0, 200.0])
    qa = {100.0: 9.0}  # 130 s has no log entry: gets UNKNOWN_QA_S (27 s)
    assert previous_qa_end(np.array([100.0, 130.0, 200.0]), markers, qa).tolist() == [
        -np.inf,
        109.0,
        157.0,
    ]


def test_epoch_flags_noisy_and_crowded_windows_without_dropping_them():
    rng = np.random.default_rng(0)
    sf, n = 256, 256 * 120
    x = rng.normal(0, 5e-6, (64, n))
    x[:, int(60.5 * sf)] += 400e-6  # one jaw clench, 0.5 s after the second probe
    names = mne.channels.make_standard_montage("biosemi64").ch_names
    raw = mne.io.RawArray(x, mne.create_info(names, sf, "eeg"))

    probes = pd.DataFrame({"onset_s": [20.0, 60.0, 75.0], "qa_s": [8.0, 8.0, 8.0]})
    md = epoch(raw, probes, markers=probes.onset_s.to_numpy()).metadata

    assert md.probe_clean.tolist() == [True, False, True]  # the clench ruins probe 2 only
    # Probe 3's pre window (65-75 s) starts while probe 2's Q&A (60-68 s) is still talking.
    assert md.pre_clean.tolist() == [True, True, False]
    assert md.onset_s.tolist() == [20.0, 60.0, 75.0] and PRE[0] < 0
