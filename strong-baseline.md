# Strong Baseline: Fine-tuned DistilBERT

## Model and task

We fine-tune `distilbert-base-uncased` for extractive question answering
using our prepared SQuAD 1.1 training split.

DistilBERT is a distilled Transformer encoder introduced by
Sanh et al. (2019). We initialize the encoder from its pretrained
checkpoint and attach a newly initialized question-answering head
that predicts answer start and end positions. The encoder and QA
head are fine-tuned jointly.

We use the base pretrained model rather than a checkpoint already
fine-tuned on SQuAD.

## Data and preprocessing

| Split | Questions | Tokenized windows |
| --- | ---: | ---: |
| Training | 78,349 | 79,137 |
| Development | 9,250 | 9,387 |
| Clean test | 10,570 | 10,784 |

Questions and contexts are tokenized together using a fast tokenizer.
Only the context is truncated. Long contexts are divided into
overlapping windows with:

- Maximum sequence length: 384 tokens.
- Context overlap between windows: 128 tokens.
- Padding to the maximum sequence length.

Training labels use the first reference answer and its character offset.
Character offsets are mapped to token start and end positions.
Windows that do not contain the complete answer are labeled at `[CLS]`.

We verified that all 78,349 training questions have an aligned answer
in at least one window.

Training uses the training split only. The development split is used
for checkpoint selection. Clean test data are used only for final
prediction and evaluation.

## Training configuration

| Setting | Value |
| --- | --- |
| Base model | distilbert-base-uncased |
| Optimizer | AdamW (`adamw_torch`) |
| Learning rate | 3e-5 |
| Learning-rate schedule | Linear decay after warmup |
| Warmup steps | 500 |
| Weight decay | 0.01 |
| Training batch size | 4 |
| Gradient accumulation steps | 4 |
| Effective batch size | 16 windows |
| Evaluation batch size | 4 |
| Epochs | 2 |
| Total optimizer steps | 9,894 |
| Evaluation interval | 500 optimizer steps |
| Checkpoint interval | 500 optimizer steps |
| Checkpoint selection | Lowest development loss |
| Random seed | 42 |
| Precision | Full precision; FP16/BF16 disabled |

Training began on an Apple Silicon Mac using MPS and resumed from
checkpoint-500 on an NVIDIA L4 GPU in Google Colab. The optimizer and
learning-rate scheduler states were restored.

CUDA random-number-generator state could not be restored from the
MPS checkpoint. Therefore, this cross-device run is not expected to
be bit-for-bit identical to uninterrupted training on one device.

The selected checkpoint was `checkpoint-8500`, with a logged
development loss of approximately 1.018. We used this checkpoint
for both development and clean test predictions.

## Prediction method

`predict_strong.py` uses only the question and context.
Reference answers and answer offsets are not used during prediction.

Prediction uses the same maximum sequence length and context overlap
as training. For each window, the model produces start and end logits.

For a candidate span starting at token s and ending at token e:

    span_score(s, e) = start_logit(s) + end_logit(e)

We consider context spans satisfying:

    s <= e
    e - s + 1 <= 30

Question tokens, special tokens, and padding are excluded.
The highest-scoring span across all windows is selected.
The answer is extracted directly from the original context using
the tokenizer's character offsets.

Ties retain the first encountered span. No unanswerable-question
threshold is applied because this experiment uses SQuAD 1.1.

The checkpoint and prediction rules were fixed before clean test
evaluation.

## Running the baseline

Run commands from the project root.

### Prepare features

The existing preprocessing script reads the prepared train/dev JSONL
files and creates `features/train` and `features/validation`:

```bash
python prepare-features.py
```

### Train

```bash
python train_strong.py \
  --train-features features/train \
  --dev-features features/validation \
  --output-dir runs/distilbert-baseline
```

To resume from a saved checkpoint:

```bash
python train_strong.py \
  --train-features features/train \
  --dev-features features/validation \
  --resume-from-checkpoint checkpoint-500 \
  --output-dir runs/distilbert-baseline-colab
```

### Generate development predictions

The commands below assume the selected checkpoint has been placed at
`runs/distilbert-baseline-colab/checkpoint-8500`.

```bash
python predict_strong.py \
  --model runs/distilbert-baseline-colab/checkpoint-8500 \
  --input submission/data/dev/clean.jsonl \
  --output predictions/strong-dev.json
```

### Evaluate development predictions

```bash
python score.py \
  --gold submission/data/dev/clean.jsonl \
  --predictions predictions/strong-dev.json
```

### Generate clean test predictions

```bash
python predict_strong.py \
  --model runs/distilbert-baseline-colab/checkpoint-8500 \
  --input submission/data/test/clean.jsonl \
  --output predictions/strong-test.json
```

### Evaluate clean test predictions

```bash
python score.py \
  --gold submission/data/test/clean.jsonl \
  --predictions predictions/strong-test.json
```

The prediction script automatically selects CUDA, MPS, or CPU.
A device can also be specified using `--device`.

## Results

Scores use SQuAD 1.1 Exact Match and token-level F1, expressed
as percentages from 0 to 100. See `scoring.md` for definitions.

| Method | Split | Questions | EM (%) | F1 (%) |
| --- | --- | ---: | ---: | ---: |
| Lexical overlap sentence selection | Development | 9,250 | 0.13 | 15.28 |
| Fine-tuned DistilBERT | Development | 9,250 | 67.05 | 79.63 |
| Lexical overlap sentence selection | Clean test | 10,570 | 0.22 | 18.70 |
| Fine-tuned DistilBERT | Clean test | 10,570 | 76.26 | 84.70 |

DistilBERT substantially outperforms the simple baseline on both splits.
Unlike whole-sentence selection, it predicts a specific answer span.

Clean test scores are higher than development scores for both methods.
These results describe the two evaluated splits; they do not establish
why their difficulty differs.

## Environment

The resumed Colab training environment used:

- Python 3.13.16
- PyTorch 2.14.1+cu130
- Transformers 5.23.0
- Datasets 5.2.0
- Accelerate 1.15.0
- PyArrow 26.0.1
- NVIDIA L4 GPU

The initial Mac training used Python 3.12 and PyTorch 2.15.0 with MPS.

## Limitations and scope

These are results from one training run, without multi-seed uncertainty
estimates. Checkpoint selection uses development loss rather than
development EM/F1.

Predictions are limited to contiguous context spans of at most
30 tokens. The model may still rely on lexical overlap and may be
sensitive to misleading distractor sentences.

This report covers clean-data evaluation only. Adversarial AddSent
and AddOneSent evaluation and conditional robustness retention
are not included in these results.

## References

Sanh, V., Debut, L., Chaumond, J., and Wolf, T. (2019).
DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.
https://arxiv.org/abs/1910.01108

Rajpurkar, P., Zhang, J., Lopyrev, K., and Liang, P. (2016).
SQuAD: 100,000+ Questions for Machine Comprehension of Text.
Proceedings of EMNLP.
https://aclanthology.org/D16-1264/