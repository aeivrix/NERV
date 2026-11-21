# Dataset notes: ds001787

Checked against OpenNeuro snapshot 1.1.1 (metadata, event files, behavioural logs and
BDF headers). The EEG signals themselves have not been inspected yet.

## Contents

- 24 participants (12 expert, 12 novice Himalayan Yoga meditators), 40 sessions of
  about 15–60 min. Roughly 27.6 h of recording in total.
- BioSemi 64-channel EEG, downsampled to 256 Hz; 80 channels per BDF file in all.
- Probe series: Q1 meditation depth, Q2 mind-wandering depth, Q3 drowsiness, each
  rated 0–3 (Brandmeyer & Delorme 2018, Methods).
- Probe onset is marked by trigger value 128 in `events.tsv`.

## Issue 1: `events.tsv` drops every 0 rating

Key `k` sends trigger value `2^k`, so key `0` sends value 1, which never appears in
the event files. A probe answered (1, 1, 0) shows only two responses, and a later 0
shifts the remaining answers into the wrong question.

Evidence: the behavioural logs contain 530 zero ratings for Q3 alone, and there is no
value 1 in any `events.tsv`.

**Fix:** take ratings from `code/MW_Current_TextFileBIDS.zip`, which logs every key
press with its time and question number (`status`).

## Issue 2: logs and recordings don't match one-to-one

`make labels` (`src/probe_eeg/labels.py`) matches each log to a recording by probe
timing under one clock offset. It keeps the offset and recording with the most probes
whose EEG answer triggers agree with the log. Results are in
`data/interim/label_audit.csv`:

- **Aligned:** 39 of 40 recordings, covering 938 probes from 23 subjects. For 99.5%
  of matched probes, the EEG triggers equal the log's non-zero answers.
- **sub-013 is lost:** its only log (`sub13_info.txt`) belongs to a different session.
  The probe gaps and answers don't match its recording, so sub-013 is excluded.
- **Unused logs:** `sub14_info.txt` and `sub23_info.txt`. Their recordings match
  `sub14_2_info.txt` and `sub23_info_old.txt` instead.
- **Shared schedules:** probe times come from a random sequence seeded by subject
  name, so two logs of one subject can share a schedule. That's why alignment checks
  answers, not just timing.
- **Late starts:** three recordings (`sub-003`, `sub-004` and `sub-022`, session 1 or
  2) have clock offsets of −1,200 to −1,700 s, meaning EEG started partway through the
  session. All of their probes still agree.
- **Unmatched markers:** 20 probe markers in aligned recordings have no log entry.
  Most recordings have one; `sub-021/ses-01` has 4, and `sub-023/ses-01` has 9, where
  only 10 of its 19 probes align.

## Issue 3: group is confounded with sex

- Experts: 10 male, 2 female.
- Novices: 2 male, 10 female.
- The paper reports 3 female experts, so the sources disagree slightly.

Expert vs novice contrasts can't be separated from sex differences with n = 24.

## Extra channels (sampled in 3 recordings)

| Channel | Status |
|---|---|
| EXG5 | Looks like ECG: periodic at 0.84–0.91 s (66–71 bpm) |
| GSR1 | Signal present |
| Resp | Signal present |
| Plet | Low variance; possibly unconnected |
| Temp, GSR2 | Flat |
| Other EXG | Low variance |

## Label counts (`data/interim/probes.csv`, agreeing probes only)

933 probes with Q1 and Q2. Using mind wandering = Q2 ≥ 2, and meditation = Q1 ≥ 2 with
Q2 ≤ 1:

| | Mind wandering | Meditation |
|---|---|---|
| Experts | 130 | 283 |
| Novices | 210 | 120 |
| Total | 340 | 403 |

- 190 probes fall in neither class and are excluded.
- 17 of 23 subjects have at least 5 probes of each class.
