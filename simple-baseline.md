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
python3 simple-baseline.py \
  --input submission/data/dev/clean.jsonl \
  --output predictions/simple-dev.json

python3 score.py \
  --gold submission/data/dev/clean.jsonl \
  --predictions predictions/simple-dev.json
```

Clean test predictions and evaluation:

```bash
python3 simple-baseline.py \
  --input submission/data/test/clean.jsonl \
  --output predictions/simple-test.json

python3 score.py \
  --gold submission/data/test/clean.jsonl \
  --predictions predictions/simple-test.json
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

## Limitations

The baseline selects whole sentences rather than precise answer spans.
Even when it selects a sentence containing the answer, additional words
reduce token precision and usually prevent exact matches.

Lexical overlap does not establish semantic relevance. A distractor
sentence containing question words may receive a high score.
The punctuation-based sentence splitter can also split incorrectly
around abbreviations.

These results cover clean data only. Adversarial evaluation has not
yet been performed.