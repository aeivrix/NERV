# Data note for ds001787 (draft OpenNeuro issue)

Ready to post at https://github.com/OpenNeuroDatasets/ds001787/issues. Nothing has
been posted yet.

---

**Title:** Zero ratings are missing from events.tsv; probe audio starts ~0.59 s after the trigger

Thanks for sharing this dataset. While building a probe-locked analysis we found three
things that may help other users. All checks were against snapshot 1.1.1. The code is
public (link to repo).

1. **`events.tsv` drops every 0 rating.** Key *k* sends trigger 2^k, so key 0 would
   send value 1, which never appears in any event file. A probe answered (1, 1, 0)
   shows only two responses, and a 0 in Q1 or Q2 shifts the later answers into the
   wrong question. The behavioural logs in `code/MW_Current_TextFileBIDS.zip` record
   every key press with its question (`status`), and they contain 530 zero ratings for
   Q3 alone. Reading ratings from the logs recovers them. When we aligned the logs to
   the recordings, the EEG triggers agreed with the logs' non-zero answers for 99.5%
   of 938 probes.
2. **Logs and recordings are not one-to-one.**
   - `sub13_info.txt` doesn't match sub-013's recording.
   - `sub14_info.txt` and `sub23_info.txt` match no recording; the recordings match
     `sub14_2_info.txt` and `sub23_info_old.txt`.
   - Three recordings start 20–28 min into their session.
   - Probe schedules are seeded by subject name, so two logs of one subject can share
     timings. Matching on timing alone can pick the wrong log.
3. **Probe audio arrives about 0.59 s after trigger 128.** The grand-average auditory
   N1 peaks at a median of 686 ms after the trigger (range 621–738 ms across 23
   subjects), where about 100 ms after sound onset would be expected. Probe-locked
   analyses that treat the trigger as sound onset will misplace every component.

We also noticed that group and sex largely coincide (experts 10M/2F, novices 2M/10F in
`participants.tsv`; the paper reports 3 female experts). In our analysis, EEG decoded
sex (AUC 0.92) better than expertise (0.69). That's worth a line in the README for
anyone classifying expert vs novice.
