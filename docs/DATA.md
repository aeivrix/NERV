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

- 42 session logs vs 40 recordings, for example `sub14_2_info.txt` (sub-014 has one
  recording) and `sub23_info_old.txt`.
- In `sub-001/ses-01`, the EEG has an extra probe marker with no responses after the
  first probe.
- A public reanalysis reports that alignment fails for 5 of 41 sessions
  ([ChristianV997/NeuroTopology-Sim](https://github.com/ChristianV997/NeuroTopology-Sim));
  not verified here.

**Fix:** align each log to its recording by probe timing; drop sessions that don't align.

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

## Label counts (from logs, approximate)

About 990 fully rated probes. Using mind wandering = Q2 ≥ 2, and meditation = Q1 ≥ 2
with Q2 ≤ 1:

- about 370 mind-wandering probes;
- about 425 meditation probes;
- about 18 of 24 people have at least 5 of each class.

These counts include the unmatched log files.
