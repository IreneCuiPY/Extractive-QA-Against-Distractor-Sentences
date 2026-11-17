# Output

`predictions/` holds one JSON file per system and evaluation file, mapping
each question ID to the predicted answer string. Gold answers are in
`data/`. Run the commands below from the repository root.

## Clean data

```sh
python3 code/score.py --gold data/test/clean.jsonl \
  --predictions output/predictions/simple-test.json
```

| Prediction file | Gold file | EM | F1 |
| --- | --- | ---: | ---: |
| `simple-dev.json` | `data/dev/clean.jsonl` | 0.13 | 15.28 |
| `simple-test.json` | `data/test/clean.jsonl` | 0.22 | 18.70 |
| `strong-dev.json` (not yet committed) | `data/dev/clean.jsonl` | 67.05 | 79.63 |
| `strong-test.json` (not yet committed) | `data/test/clean.jsonl` | 76.26 | 84.70 |

## Adversarial data

Adversarial scoring also needs the system's clean test predictions:

```sh
python3 code/score-adversarial.py \
  --clean-predictions output/predictions/simple-test.json \
  --adv-gold data/test/addsent.jsonl \
  --adv-predictions output/predictions/simple-addsent.json
```

Scores on the 1,000-question adversarial cohort:

| Prediction file | Gold file | Adversarial EM | Adversarial F1 |
| --- | --- | ---: | ---: |
| `simple-addsent.json` | `data/test/addsent.jsonl` | 0.40 | 13.71 |
| `simple-addonesent.json` | `data/test/addonesent.jsonl` | 0.40 | 15.51 |

On the same 1,000 questions, the simple baseline's clean scores are
0.50 EM / 18.60 F1.
