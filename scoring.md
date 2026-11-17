# Evaluation Metrics

## Task and input format

We evaluate extractive question answering using Exact Match (EM) and
token-level F1, following SQuAD 1.1 (Rajpurkar et al., 2016).

The evaluation script accepts:

- A gold JSONL file, with one question per line. Each record contains an
  `id` and an `answers` list containing reference answer strings under `text`.
- A predictions JSON object mapping each question ID to one predicted
  answer string.

Example prediction format:

```json
{
  "q1": "Paris",
  "q2": "New York"
}
```

Prediction IDs must exactly match the gold IDs. Missing or extra IDs
raise an error. Duplicate gold IDs and empty gold datasets also raise errors.

## Answer normalization

Before scoring, both predicted and reference answers are normalized by:

1. Converting text to lowercase.
2. Removing punctuation characters in Python's `string.punctuation`.
3. Removing the English articles `a`, `an`, and `the`.
4. Collapsing whitespace.

This follows the official SQuAD 1.1 normalization procedure.

## Exact Match

Let p_i be the prediction for question i, G_i its set of reference
answers, and N the normalization function.

EM_i = max over g in G_i of 1[N(p_i) = N(g)].

The dataset score is:

EM = (100 / n) × sum over i of EM_i,

where n is the number of questions.

## Token-level F1

Normalized answers are split into whitespace-separated tokens.
Token overlap is computed using multiset counts, so repeated tokens
are counted only up to their frequency in both answers.

For prediction p and reference g, let c be the number of overlapping
tokens, m the number of predicted tokens, and k the number of reference
tokens.

If c = 0, F1(p, g) = 0, matching the official SQuAD 1.1 evaluator.
Otherwise:

Precision = c / m

Recall = c / k

F1(p, g) = 2 × Precision × Recall / (Precision + Recall).

For each question:

F1_i = max over g in G_i of F1(p_i, g).

The dataset score is:

F1 = (100 / n) × sum over i of F1_i.

EM and F1 are maximized independently over the reference answers.
Both dataset scores range from 0 to 100.

## Running the evaluator

Run from the project root. Prediction files must first be generated
by the corresponding baseline.

Example: evaluate the strong baseline on clean test:

```bash
python3 score.py \
  --gold submission/data/test/clean.jsonl \
  --predictions predictions/strong-test.json
```

Example output:

```json
{
  "exact_match": 76.26300851466415,
  "f1": 84.69942180025888,
  "total": 10570
}
```

`exact_match` and `f1` are percentages from 0 to 100.
`total` is the number of evaluated questions.

To evaluate the simple baseline on development data:

```bash
python3 score.py \
  --gold submission/data/dev/clean.jsonl \
  --predictions predictions/simple-dev.json
```

Example output:

```json
{
  "exact_match": 0.12972972972972974,
  "f1": 15.279694386881282,
  "total": 9250
}
```

## Verification example

We verified the evaluator using three constructed questions:

| ID | Reference answers | Prediction | EM | F1 |
| --- | --- | --- | ---: | ---: |
| q1 | Paris | The Paris! | 1 | 1 |
| q2 | New York City | New York | 0 | 0.8 |
| q3 | USA; United States | United States | 1 | 1 |

The evaluator returned the following scores, rounded here for readability:

```json
{
  "exact_match": 66.6667,
  "f1": 93.3333,
  "total": 3
}
```

These are verification examples, not baseline performance results.
We also verified normalization, repeated-token overlap, zero overlap,
and rejection of missing predictions.

## Adversarial evaluation (AddSent / AddOneSent)

`score-adversarial.py` implements the grouped evaluation of Jia and Liang
(2017). AddSent has up to five attacked candidates per question, so the
candidates must not be averaged as independent questions.

Let C be the 1,000-question adversarial cohort (`adversarial_cohort_ids`
in `split_manifest.json`) and A_i the attacked candidates for question i,
in file order. A_i is empty for 213 questions with no attack. Let
a_i* be the candidate in A_i with the lowest F1, keeping the first
candidate on ties. The adversarial scores are:

Adv-F1 = (100 / |C|) × sum over i in C of F1_i(a_i*),
with F1_i(a_i*) replaced by the clean F1_i when A_i is empty.

Adv-EM uses the EM of the same selected candidate a_i*, not the
lowest EM. The clean prediction never competes with attacked candidates.
AddOneSent has one candidate per attacked question, so a_i* is that
candidate.

The script also reports:

- Clean EM/F1 on the same 1,000-question cohort, for a paired comparison.
- An `attacked_only` block restricted to the 787 attacked questions.
- `distractor_pick_rate`: among the 787 attacked questions, the share whose
  selected prediction appears only in the inserted distractor sentence.
  This is a text-match approximation. It does not use token offsets.

We checked this script against a Python 3 port of the authors' evaluator
(`references/upstream_eval_squad.py.txt`) run on the raw AddSent and
AddOneSent JSON files. Both gave identical scores for the simple baseline.

```bash
python3 score-adversarial.py \
  --clean-predictions predictions/simple-test.json \
  --adv-gold submission/data/test/addsent.jsonl \
  --adv-predictions predictions/simple-addsent.json
```

Example output (simple baseline, AddSent):

```json
{
  "condition": "addsent",
  "cohort_questions": 1000,
  "clean_exact_match": 0.5,
  "clean_f1": 18.599340325650168,
  "adversarial_exact_match": 0.4,
  "adversarial_f1": 13.706466374685503,
  "attacked_only": {
    "questions": 787,
    "clean_exact_match": 0.12706480304955528,
    "clean_f1": 17.72982412591085,
    "adversarial_exact_match": 0.0,
    "adversarial_f1": 11.51270347665459,
    "distractor_pick_rate": 45.23506988564168
  }
}
```

## References

Rajpurkar, P., Zhang, J., Lopyrev, K., and Liang, P. (2016).
SQuAD: 100,000+ Questions for Machine Comprehension of Text.
Proceedings of EMNLP.
https://aclanthology.org/D16-1264/

Official SQuAD 1.1 evaluation script:
https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v1.1.py

Jia, R. and Liang, P. (2017).
Adversarial Examples for Evaluating Reading Comprehension Systems.
Proceedings of EMNLP.
https://aclanthology.org/D17-1215/

Adversarial evaluator:
https://github.com/robinjia/adversarial-squad/blob/master/src/py/eval_squad.py