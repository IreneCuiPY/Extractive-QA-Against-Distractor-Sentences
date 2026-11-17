# Code

Run every command from the repository root, not from this folder.

| Path | Purpose |
| --- | --- |
| `score.py` | SQuAD 1.1 EM / F1 evaluator (`scoring.md`) |
| `score-adversarial.py` | Grouped AddSent / AddOneSent evaluation (`scoring.md`) |
| `simple-baseline.py` | Lexical-overlap sentence selection (`simple-baseline.md`) |
| `qa/prepare_features.py` | Tokenizes train/dev into DistilBERT windows |
| `qa/train.py` | Fine-tunes DistilBERT (`strong-baseline.md`) |
| `qa/predict.py` | Predicts answers for any prepared JSONL file |
| `qa/check_tokenization.py` | Checks that every training answer aligns to tokens |
| `data_prep/fetch_sources.py` | Downloads the raw sources into `data/raw/` |
| `data_prep/prepare_data.py` | Builds the splits in `data/` from `data/raw/` |
| `data_prep/package_data.py` | Rebuilds `milestones/m1/data.tar.gz` |

## 1. Setup

```sh
pip install -r requirements.txt
tar -xzf milestones/m1/data.tar.gz data/train/clean.jsonl
```

The second command extracts the training split, which is too large to
store in the repository. You can also rebuild it, together with the other
splits, from the raw downloads:

```sh
python3 code/data_prep/prepare_data.py
```

This reproduces the files in `data/` exactly. Run `fetch_sources.py` only
to re-download the sources. It overwrites `data/raw/` and
`references/source_manifest.json`.

## 2. Simple baseline

```sh
python3 code/simple-baseline.py --input data/dev/clean.jsonl \
  --output output/predictions/simple-dev.json
python3 code/simple-baseline.py --input data/test/clean.jsonl \
  --output output/predictions/simple-test.json
for condition in addsent addonesent; do
  python3 code/simple-baseline.py --input data/test/$condition.jsonl \
    --output output/predictions/simple-$condition.json
done
```

## 3. Strong baseline (DistilBERT)

```sh
python code/qa/prepare_features.py            # writes features/train and features/validation
python code/qa/train.py --output-dir runs/distilbert-baseline
python code/qa/predict.py --model runs/distilbert-baseline/best-model \
  --input data/test/clean.jsonl --output output/predictions/strong-test.json
```

`strong-baseline.md` gives the full configuration, how to resume training,
and the development and adversarial prediction commands.

## 4. Scoring

`output/README.md` lists the scoring command and expected score for every
prediction file. The metrics are defined in `scoring.md`.
