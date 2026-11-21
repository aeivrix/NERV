"""Checks for label recovery: run with `make test`."""

import numpy as np
import pandas as pd

from probe_eeg.labels import align, parse_log, state

LOG = """\
MW question asked at time  54.1 second
Key 1 (code 1) pressed at time  58.4 seconds, status 1
Key 1 (code 1) pressed at time  62.5 seconds, status 2
Key 0 (code 0) pressed at time  66.3 seconds, status 3
Key 5 (code 5) pressed at time  80.0 seconds, status 0
MW question asked at time 162.5 second
Key 0 (code 0) pressed at time 165.7 seconds, status 1
Key 2 (code 2) pressed at time 168.3 seconds, status 2
Key c (code ) pressed at time 170.5 seconds, status 3
MW question asked at time 300.0 second
Key 3 (code 3) pressed at time 302.0 seconds, status 1
Key 2 (code 2) pressed at time 305.0 seconds, status 2
Key 1 (code 1) pressed at time 308.0 seconds, status 3
"""


def test_parse_log_keeps_zeros_and_ignores_stray_and_cancel_keys():
    log = parse_log(LOG)
    assert log[["q1", "q2"]].to_numpy().tolist() == [[1, 1], [0, 2], [3, 2]]
    assert log.q3.iloc[0] == 0 and np.isnan(log.q3.iloc[1])  # 'c' cancelled Q3
    assert abs(log.rt1_s.iloc[0] - 4.3) < 1e-9
    assert log.qa_s.round(1).tolist() == [12.2, 8.0, 8.0]  # the stray status-0 key doesn't count


def test_align_uses_answers_not_just_timing():
    log = parse_log(LOG)
    onsets = log.log_t.to_numpy() + 12.0  # EEG started 12 s before the script clock
    # Zeros send no trigger: probe 1 (1,1,0) -> [2, 2]; probe 2 (0,2,-) -> [4]; probe 3 -> [8, 4, 2]
    offset, pairs, n_agree = align(log, onsets, [[2, 2], [4], [8, 4, 2]])
    assert abs(offset - 12.0) < 1e-6 and pairs == {0: 0, 1: 1, 2: 2} and n_agree == 3
    # A log sharing the schedule but with different answers agrees nowhere.
    assert align(log, onsets, [[8], [8], [8]])[2] == 0


def test_state_rule_and_fence_sitters():
    d = pd.DataFrame({"q1": [3, 2, 0, 1, 2], "q2": [0, 3, 1, 1, 2]})
    assert state(d).fillna("-").tolist() == ["med", "mw", "-", "-", "mw"]  # MW beats MED on a tie
