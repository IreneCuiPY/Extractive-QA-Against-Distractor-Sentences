# Robustness of Extractive QA Against Distractor Sentences

CIS 5300 term project (Irene Cui, Crystal Jin, Rita Wang).

We study whether extractive QA models trained on SQuAD 1.1 are misled by
distractor sentences (Jia and Liang, 2017), and whether training on
distractor-augmented data with a hard-negative objective improves
robustness without hurting clean accuracy.

## Repository layout

```text
code/          evaluation, baselines, and data preparation (code/README.md)
  qa/          DistilBERT feature preparation, training, and prediction
  data_prep/   download, split, and package the data
data/          prepared train/dev/test splits and raw downloads (data/README.md)
output/        system predictions and how to score them (output/README.md)
milestones/    submitted reports and deliverables, one folder per milestone
references/    papers, source snapshots, and validation records
```

## Quick start

Run all commands from the repository root.

```sh
pip install -r requirements.txt
tar -xzf milestones/m1/data.tar.gz data/train/clean.jsonl   # training split

python3 code/simple-baseline.py --input data/test/clean.jsonl \
  --output output/predictions/simple-test.json
python3 code/score.py --gold data/test/clean.jsonl \
  --predictions output/predictions/simple-test.json
```

`code/README.md` lists the commands that reproduce every result.

## Milestone 2 deliverables

| Deliverable | Location |
| --- | --- |
| Report | `milestones/m2/milestone2.pdf` |
| Evaluation script and metric | `code/score.py`, `code/scoring.md` |
| Adversarial evaluation | `code/score-adversarial.py` (described in `code/scoring.md`) |
| Simple baseline | `code/simple-baseline.py`, `code/simple-baseline.md` |
| Strong baseline | `code/qa/`, `code/strong-baseline.md` |

## Results so far

EM / F1 (%). AddSent and AddOneSent use the 1,000-question adversarial
cohort, so their clean column is the clean score on those same 1,000 questions.

| Method | Dev (9,250) | Clean test (10,570) | Cohort clean (1,000) | AddSent | AddOneSent |
| --- | --- | --- | --- | --- | --- |
| Lexical overlap | 0.13 / 15.28 | 0.22 / 18.70 | 0.50 / 18.60 | 0.40 / 13.71 | 0.40 / 15.51 |
| DistilBERT | 67.05 / 79.63 | 76.26 / 84.70 | not run | not run | not run |

DistilBERT adversarial results need the trained checkpoint, which is not
stored in this repository.
