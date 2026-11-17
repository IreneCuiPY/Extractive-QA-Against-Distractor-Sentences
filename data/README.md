# Data

Prepared SQuAD 1.1 and Adversarial SQuAD splits. `data.md` documents the
sources, file format, split construction, and adversarial pairing in full.

| File | Questions | Notes |
| --- | ---: | --- |
| `train/clean.jsonl` | 78,349 | Not stored in git; see below |
| `dev/clean.jsonl` | 9,250 | 44 held-out training articles |
| `test/clean.jsonl` | 10,570 | Official SQuAD 1.1 development set |
| `test/addsent.jsonl` | 2,560 rows (787 questions) | Up to five attacked candidates per question |
| `test/addonesent.jsonl` | 787 | One attacked candidate per question |
| `split_manifest.json` | | Article lists and adversarial cohort IDs |
| `validation.json` | | Row counts, hashes, and overlap checks |
| `source_manifest.json` | | Source URLs and SHA-256 hashes |
| `raw/` | | The four unmodified source downloads |

## Getting the training split

`train/clean.jsonl` (82 MB) is stored only inside the Milestone 1 archive.
From the repository root, run:

```sh
tar -xzf milestones/m1/data.tar.gz data/train/clean.jsonl
```

To rebuild it from `raw/` instead, run
`python3 code/data_prep/prepare_data.py`.

The source manifest still lists the raw files under `source_data/`, their
location when they were downloaded. They now live in `raw/`, with the same
hashes.
