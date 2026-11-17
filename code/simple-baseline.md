# Simple Baseline: Lexical Overlap Sentence Selection

## Method

For each question, we select the context sentence with the greatest
number of distinct normalized tokens shared with the question.
The selected sentence is returned as the predicted answer.

We split the context on whitespace following `.`, `!`, or `?`.
We normalize tokens using the same normalization as our scoring script:
lowercase text, remove punctuation and English articles, and collapse
whitespace.

Overlap is the size of the intersection between the question-token
set and each sentence-token set. Repeated words do not increase the
overlap score. Ties are resolved by selecting the earliest sentence.
If the context contains no nonempty sentences, the prediction is empty.

The method requires no training and uses only the question and context.
It never reads reference answers or answer offsets to make predictions.

## Example

Question: What is the capital of France?

Context: Paris is the capital of France. Berlin is in Germany.

Prediction: Paris is the capital of France.

The output JSON maps each question ID to its predicted answer:

```json
{
  "example-question": "Paris is the capital of France."
}
```

This example illustrates the method; it is not a dataset result.

## Running the baseline

Run from the project root.

Development predictions and evaluation:

```bash
python3 code/simple-baseline.py \
  --input data/dev/clean.jsonl \
  --output output/predictions/simple-dev.json

python3 code/score.py \
  --gold data/dev/clean.jsonl \
  --predictions output/predictions/simple-dev.json
```

Clean test predictions and evaluation:

```bash
python3 code/simple-baseline.py \
  --input data/test/clean.jsonl \
  --output output/predictions/simple-test.json

python3 code/score.py \
  --gold data/test/clean.jsonl \
  --predictions output/predictions/simple-test.json
```

The implementation uses only Python's standard library and imports
`normalize_answer` from `score.py`. Keep both scripts in the same directory.

## Results

Scores use SQuAD 1.1 Exact Match and token-level F1, expressed as
percentages from 0 to 100.

| Split | Questions | EM (%) | F1 (%) |
| --- | ---: | ---: | ---: |
| Development | 9,250 | 0.13 | 15.28 |
| Clean test | 10,570 | 0.22 | 18.70 |

The baseline rules were fixed before clean test evaluation.

Clean test scores exceed development scores partly because clean test
has about three reference answers per question and development has one
(see `strong-baseline.md`).

### Adversarial results

Adversarial predictions and evaluation:

```bash
for condition in addsent addonesent; do
  python3 code/simple-baseline.py \
    --input data/test/$condition.jsonl \
    --output output/predictions/simple-$condition.json

  python3 code/score-adversarial.py \
    --clean-predictions output/predictions/simple-test.json \
    --adv-gold data/test/$condition.jsonl \
    --adv-predictions output/predictions/simple-$condition.json
done
```

The 1,000-question adversarial cohort, scored as described in `scoring.md`:

| Condition | Questions | Clean EM | Clean F1 | Adversarial EM | Adversarial F1 | Distractor picked |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AddSent | 1,000 | 0.50 | 18.60 | 0.40 | 13.71 | — |
| AddOneSent | 1,000 | 0.50 | 18.60 | 0.40 | 15.51 | — |
| AddSent, attacked only | 787 | 0.13 | 17.73 | 0.00 | 11.51 | 45.2% |
| AddOneSent, attacked only | 787 | 0.13 | 17.73 | 0.00 | 13.80 | 37.9% |

"Distractor picked" is the share of attacked questions where the selected
attacked prediction appears only in the inserted sentence. The baseline
returns the distractor sentence for 45.2% of attacked questions under
AddSent and 37.9% under AddOneSent. This is the expected failure mode:
distractors are written to share many words with the question.

## Limitations

The baseline selects whole sentences rather than precise answer spans.
Even when it selects a sentence containing the answer, additional words
reduce token precision and usually prevent exact matches.

Lexical overlap does not establish semantic relevance. A distractor
sentence containing question words may receive a high score.
The punctuation-based sentence splitter can also split incorrectly
around abbreviations.

The proposal described a richer simple baseline that extracts a
location, date, or person entity from the selected sentence for
where/when/who questions. The implemented version returns the whole
sentence, which keeps it dependency-free but leaves EM near zero.