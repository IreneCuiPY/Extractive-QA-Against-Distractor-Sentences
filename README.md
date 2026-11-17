# Robustness of Extractive QA Against Distractor Sentences

CIS 5300 term project (Irene Cui, Crystal Jin, Rita Wang).

We study whether extractive QA models trained on SQuAD 1.1 are misled by
distractor sentences (Jia and Liang, 2017), and whether training on
distractor-augmented data with a hard-negative objective improves
robustness without hurting clean accuracy.

## Milestone 2 deliverables

| File | Purpose |
| --- | --- |
| `milestone2.pdf` | Milestone 2 report |
| `score.py`, `scoring.md` | SQuAD 1.1 EM / F1 evaluator and metric definition |
| `score-adversarial.py` | Grouped AddSent / AddOneSent evaluation (see `scoring.md`) |
| `simple-baseline.py`, `simple-baseline.md` | Lexical-overlap sentence selection |
| `prepare-features.py`, `train-strong.py`, `predict-strong.py`, `strong-baseline.md` | Fine-tuned DistilBERT |
| `check-tokenization.py` | Checks that every training answer aligns to tokens |
| `predictions/` | Baseline predictions |

Milestone 1 materials (report, `data.md`, data archive, preparation
scripts) are in `submission/`, `scripts/`, and `references/`.

## Setup

```sh
pip install -r requirements.txt
```

The training split (82 MB) is only stored inside the data archive.
Extract it before preparing features:

```sh
tar -xzf submission/data.tar.gz -C submission data/train/clean.jsonl
```

## Reproducing results

Run all commands from the repository root.

Simple baseline on clean test:

```sh
python3 simple-baseline.py --input submission/data/test/clean.jsonl \
  --output predictions/simple-test.json
python3 score.py --gold submission/data/test/clean.jsonl \
  --predictions predictions/simple-test.json
```

Strong baseline: see `strong-baseline.md` for feature preparation,
training, prediction, and evaluation.

Adversarial evaluation of any system's predictions:

```sh
python3 score-adversarial.py \
  --clean-predictions predictions/simple-test.json \
  --adv-gold submission/data/test/addsent.jsonl \
  --adv-predictions predictions/simple-addsent.json
```

## Results so far

EM / F1 (%). AddSent and AddOneSent use the 1,000-question adversarial
cohort, so their clean column is the clean score on those same 1,000 questions.

| Method | Dev (9,250) | Clean test (10,570) | Cohort clean (1,000) | AddSent | AddOneSent |
| --- | --- | --- | --- | --- | --- |
| Lexical overlap | 0.13 / 15.28 | 0.22 / 18.70 | 0.50 / 18.60 | 0.40 / 13.71 | 0.40 / 15.51 |
| DistilBERT | 67.05 / 79.63 | 76.26 / 84.70 | not run | not run | not run |

DistilBERT adversarial results need the trained checkpoint, which is not
stored in this repository.
