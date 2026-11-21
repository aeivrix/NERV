"""Recover true probe ratings from the behavioural logs and align them to the EEG.

events.tsv can't be trusted for ratings: key k sends trigger 2**k, so key 0 sends
value 1, which never appears in the event files. Answers then shift position (see
docs/DATA.md). The logs record every key press with the question it answered (the
`status` field), so ratings come from there. Event files are used only to find
each probe's onset in EEG time.
"""

import argparse
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROBE_MARKER = 128
MATCH_TOLERANCE_S = 1.0  # log clock vs EEG clock, after removing the constant offset
MIN_MATCHED_FRACTION = 0.5  # fewer agreeing probes than this and the log is not assigned

PROBE_LINE = re.compile(r"MW question asked at time\s+([\d.]+)")
KEY_LINE = re.compile(r"Key (\S+) \(code .*?\) pressed at time\s+([\d.]+) seconds, status (\d)")
LOG_NAME = re.compile(r"sub(\d+)_+(?:(\d)_)?info(_old)?\.txt$")


def parse_log(text: str) -> pd.DataFrame:
    """One row per probe: log time, ratings q1-q3 (NaN if unanswered), Q1 latency,
    and qa_s, when the last key of this probe's Q&A landed.

    `status` is the question being answered (1-3) when the key is pressed; status 0
    means no question was pending. A non-digit key cancels the remaining questions.
    """
    probes = []
    for line in text.splitlines():
        if m := PROBE_LINE.search(line):
            probes.append(
                {
                    "log_t": float(m[1]),
                    "q1": np.nan,
                    "q2": np.nan,
                    "q3": np.nan,
                    "rt1_s": np.nan,
                    "qa_s": np.nan,
                }
            )
        elif (m := KEY_LINE.search(line)) and probes:
            key, t, status = m[1], float(m[2]), int(m[3])
            p = probes[-1]
            if 1 <= status <= 3:  # digit answers, or a cancel: both end the Q&A
                p["qa_s"] = t - p["log_t"]
            if 1 <= status <= 3 and key.isdigit() and np.isnan(p[f"q{status}"]):
                p[f"q{status}"] = int(key)
                if status == 1:
                    p["rt1_s"] = t - p["log_t"]
    return pd.DataFrame(probes, columns=["log_t", "q1", "q2", "q3", "rt1_s", "qa_s"])


def read_events(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    """Probe onsets (s) and, per probe, the response trigger values that follow it."""
    ev = pd.read_csv(path, sep="\t")
    onsets, responses = [], []
    for onset, value in zip(
        ev["onset"].astype(float), ev["value"].astype(float).astype(int), strict=True
    ):
        if value == PROBE_MARKER:
            onsets.append(onset)
            responses.append([])
        elif responses and value in (1, 2, 4, 8):
            responses[-1].append(value)
    return np.array(onsets), responses


def state(df: pd.DataFrame) -> pd.Series:
    """'mw' if Q2 >= 2, 'med' if Q1 >= 2 and Q2 <= 1, else NaN: the fence-sitters sit out."""
    return pd.Series(
        np.select([df.q2 >= 2, (df.q1 >= 2) & (df.q2 <= 1)], ["mw", "med"], None), index=df.index
    )


def triggers_agree(row: pd.Series, responses: list[int]) -> bool:
    """Do the EEG's response triggers equal the log's non-zero answers (2**k each)?"""
    expected = [
        2 ** int(row[q]) for q in ("q1", "q2", "q3") if not np.isnan(row[q]) and row[q] != 0
    ]
    return responses == expected


def match(
    log_t: np.ndarray, event_t: np.ndarray, offset: float, tol: float = MATCH_TOLERANCE_S
) -> dict[int, int]:
    """Pair each log probe with its nearest unused EEG marker within `tol` of log + offset."""
    pairs, used = {}, set()
    for i in np.argsort(log_t):
        d = np.abs(event_t - (log_t[i] + offset))
        d[list(used)] = np.inf
        j = int(np.argmin(d))
        if d[j] < tol:
            pairs[int(i)] = j
            used.add(j)
    return pairs


def align(log: pd.DataFrame, onsets: np.ndarray, responses: list[list[int]]):
    """Find the clock offset between a log and a recording.

    Every log/marker pairing proposes an offset. The winner has the most matched
    probes whose EEG triggers agree with the log's answers, so a log that only shares
    the timing schedule (schedules are seeded by subject name) loses to the real one.
    Returns (offset_s, {log_index: event_index}, n_agreeing).
    """
    log_t = log["log_t"].to_numpy()
    best = (np.nan, {}, -1)
    for offset in np.unique(np.round((onsets[None, :] - log_t[:, None]).ravel(), 1)):
        pairs = match(log_t, onsets, offset)
        n_agree = sum(triggers_agree(log.iloc[i], responses[j]) for i, j in pairs.items())
        if (n_agree, len(pairs)) > (best[2], len(best[1])):
            best = (offset, pairs, n_agree)
    offset, pairs, n_agree = best
    if pairs:  # refine the offset with the median residual and re-match
        offset = float(np.median([onsets[j] - log_t[i] for i, j in pairs.items()]))
        pairs = match(log_t, onsets, offset)
    return offset, pairs, n_agree


def build(bids_root: Path, logs_zip: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    participants = pd.read_csv(bids_root / "participants.tsv", sep="\t").set_index("participant_id")
    recordings = {}
    for path in sorted(bids_root.glob("sub-*/ses-*/eeg/*_events.tsv")):
        sub, ses = path.parts[-4], path.parts[-3]
        recordings.setdefault(sub, {})[ses] = read_events(path)

    candidates = []  # (log name, subject, session, offset, pairs, log df)
    with zipfile.ZipFile(logs_zip) as z:
        for name in sorted(z.namelist()):
            m = LOG_NAME.search(name)
            if not m or "__MACOSX" in name:
                continue
            sub = f"sub-{int(m[1]):03d}"
            log = parse_log(z.read(name).decode(errors="ignore"))
            best = None
            for ses, (onsets, responses) in recordings.get(sub, {}).items():
                offset, pairs, n_agree = align(log, onsets, responses)
                if best is None or n_agree > best[6]:
                    best = (Path(name).name, sub, ses, offset, pairs, log, n_agree)
            if best:
                candidates.append(best)

    # A recording claimed by two logs (e.g. sub23_info vs sub23_info_old) keeps the one
    # whose answers agree with the EEG triggers.
    chosen = {}
    for c in candidates:
        key = (c[1], c[2])
        if key not in chosen or c[6] > chosen[key][6]:
            chosen[key] = c

    rows, audit = [], []
    for name, sub, ses, offset, pairs, log, n_agree in candidates:
        n_events = len(recordings[sub][ses][0])
        assigned = chosen[(sub, ses)][0] == name and n_agree >= MIN_MATCHED_FRACTION * len(log)
        agree = []
        if assigned:
            onsets, responses = recordings[sub][ses]
            for i, j in sorted(pairs.items()):
                r = log.iloc[i]
                ok = triggers_agree(r, responses[j])
                agree.append(ok)
                rows.append(
                    {
                        "subject": sub,
                        "session": ses,
                        "group": participants.loc[sub, "group"],
                        "sex": participants.loc[sub, "gender"],
                        "probe": i,
                        "onset_s": onsets[j],
                        "q1": r.q1,
                        "q2": r.q2,
                        "q3": r.q3,
                        "rt1_s": r.rt1_s,
                        "qa_s": r.qa_s,
                        "triggers_agree": ok,
                    }
                )
        audit.append(
            {
                "log": name,
                "subject": sub,
                "session": ses if assigned else None,
                "assigned": assigned,
                "n_log_probes": len(log),
                "n_eeg_probes": n_events,
                "n_matched": len(pairs),
                "offset_s": round(offset, 2) if pairs else None,
                "trigger_agreement": round(float(np.mean(agree)), 3) if agree else None,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(audit)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bids", type=Path, default=Path("data/raw/ds001787"))
    ap.add_argument(
        "--logs", type=Path, default=Path("data/raw/behaviour/MW_Current_TextFileBIDS.zip")
    )
    ap.add_argument("--out", type=Path, default=Path("data/interim"))
    a = ap.parse_args()
    probes, audit = build(a.bids, a.logs)
    a.out.mkdir(parents=True, exist_ok=True)
    probes.to_csv(a.out / "probes.csv", index=False)
    audit.to_csv(a.out / "label_audit.csv", index=False)
    print(audit.to_string(index=False))
    print(
        f"\n{len(probes)} probes from {probes.groupby(['subject', 'session']).ngroups} recordings, "
        f"{probes.subject.nunique()} subjects; trigger agreement {probes.triggers_agree.mean():.1%}"
    )


if __name__ == "__main__":
    main()
