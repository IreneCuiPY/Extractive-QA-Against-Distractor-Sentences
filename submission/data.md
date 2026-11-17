# Data for Milestone 1

Project: **Robustness of Extractive QA Models Against Distractor Sentences**  
Authors: Irene Cui, Crystal Jin, Rita Wang  
Prepared: November 1, 2026

## Sources and collection

We collected the full public **SQuAD 1.1** training and development releases and the published **AddSent** and **AddOneSent** evaluation releases. We use SQuAD 1.1, which has answer spans, rather than SQuAD 2.0, which also has unanswerable questions.

| Source | Original download | Original contents |
| --- | --- | --- |
| SQuAD 1.1 training | [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json) | 87,599 questions across 442 articles |
| SQuAD 1.1 development | [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json) | 10,570 questions across 48 articles |
| AddSent | [Authors' CodaLab bundle](https://worksheets.codalab.org/rest/bundles/0xb765680b60c64d088f5daccac08b3905/contents/blob/) | 1,000 clean records plus 2,560 attacked variants |
| AddOneSent | [Authors' CodaLab bundle](https://worksheets.codalab.org/rest/bundles/0x3ac9349d16ba4e7bb9b5920e3b1af393/contents/blob/) | 1,000 clean records plus 787 attacked variants |

SQuAD was collected from Wikipedia passages with questions and answer spans written by crowdworkers. The [SQuAD site](https://rajpurkar.github.io/SQuAD-explorer/) distributes the data under CC BY-SA 4.0. Cite Rajpurkar et al. (2016), [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://aclanthology.org/D16-1264/).

The attack releases come directly from the [CodaLab worksheet](https://worksheets.codalab.org/worksheets/0xc86d3ebe69a3427d91f9aaa63f7d1e7d/) linked by [Jia and Liang's repository](https://github.com/robinjia/adversarial-squad). Cite Jia and Liang (2017), [Adversarial Examples for Evaluating Reading Comprehension Systems](https://aclanthology.org/D17-1215/). The existing distractor sentences were produced by the original authors, not by our team. We downloaded data directly; we did not execute a third-party dataset loading script.

All downloads were made November 1, 2026 in America/Los_Angeles (November 2 UTC). `data/source_manifest.json` records exact source URLs, retrieval timestamps, byte counts, and SHA-256 hashes. Prepared-file hashes are in `data/validation.json`.

## Submitted files and counts

**The archive contains the full prepared dataset, not a sample.** No external Drive download is needed.

Extract with:

```sh
tar -xzf data.tar.gz
```

The resulting directory is:

```text
data/
  train/clean.jsonl
  dev/clean.jsonl
  test/clean.jsonl
  test/addsent.jsonl
  test/addonesent.jsonl
  split_manifest.json
  source_manifest.json
  validation.json
  data.md
```

| Partition/file | Rows | Unique original questions | Articles |
| --- | ---: | ---: | ---: |
| `train/clean.jsonl` | 78,349 | 78,349 | 398 |
| `dev/clean.jsonl` | 9,250 | 9,250 | 44 |
| `test/clean.jsonl` | 10,570 | 10,570 | 48 |
| `test/addsent.jsonl` | 2,560 | 787 | 48 |
| `test/addonesent.jsonl` | 787 | 787 | 48 |

There are **98,169 unique clean/source questions** and **101,516 serialized rows** across the five files. The test partition contains 13,917 rows representing 10,570 source questions. Adversarial rows are repeated conditions for existing test questions; they are not independent new questions. AddOneSent's 787 attacked records are also present among the AddSent candidates, so IDs intentionally overlap between these two test condition files. IDs are unique within each file. Use `(file/condition, id)` if combining conditions into one table.

## Split construction and use

The official training file is divided **by article**, using the seed string `5300-m1-2026`. Compute SHA-256 of the UTF-8 string `seed + "\n" + article_title`, sort titles by this digest (then title to break ties), and assign the first 44 of 442 titles to development. The remaining 398 titles are training. All paragraphs and questions from an article retain the same split. `split_manifest.json` records the exact title lists.

The official SQuAD development file is reserved as our **project test set**, together with all of its AddSent/AddOneSent derivatives. This follows the proposal and produces approximately 79.8% training / 9.4% development / 10.8% test among public clean questions. These are project-specific partitions; our test set is not the hidden official SQuAD test set. We do not use an already SQuAD-fine-tuned checkpoint as our proposed starting model: we will fine-tune the proposal's base `distilbert-base-uncased` checkpoint ourselves.

- **Train:** fit model parameters; later derive augmentation answer/entity pools only from this partition.
- **Dev:** choose hyperparameters and compare approaches. Any generated development variants must come from development questions and stay there.
- **Test:** final evaluation after model and generator choices are fixed. Current access has only checked structure, IDs, counts, and answer offsets; no predictions or scores have been produced.

The public availability of test labels does not make this set externally sealed. Blindness here is a project protocol: do not use test performance or hand-inspected test errors to tune the method. Do not randomly repartition adversarial rows or move their clean originals into training/development.

## File format

All `.jsonl` files use UTF-8 **JSON Lines**: one complete JSON object per line, with no surrounding array. Contexts, questions, answer strings, and answer offsets are copied from the sources without trimming, case conversion, tokenization, or Unicode normalization. New fields record provenance and clean/adversarial pairing. Source nesting (`data -> articles -> paragraphs -> qas`) is flattened into one row per question/context condition.

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | Original release ID; attacked variants retain their suffix |
| `original_id` | string | Clean SQuAD question ID, used to join variants and group results |
| `title` | string | Wikipedia article title and split grouping key |
| `context` | string | Full passage, including an appended distractor for attacked records |
| `question` | string | Question to answer |
| `answers` | array of objects | All supplied references, each with `text` and `answer_start` |
| `variant` | string | `clean`, `addsent`, or `addonesent` |
| `source` | string | `squad_v1.1` for clean records; `jia_liang_2017` for attacks |

`answer_start` is a zero-based **character** offset into the decoded context, not a byte offset or token index. The end is exclusive: `answer_end = answer_start + len(text)`. Use Python string indexing (Unicode code points); JavaScript's UTF-16 indexing may differ for non-BMP characters. Keep all references when computing EM/F1. Multiple references are alternative valid answers, not extra training items.

## Actual training example

This record is present in `train/clean.jsonl`; the context is complete. It is pretty-printed here for readability, but occupies one line in the data file.

```json
{
  "id": "56df6bf35ca0a614008f99ff",
  "original_id": "56df6bf35ca0a614008f99ff",
  "title": "Christian",
  "context": "In the past, the Malays used to call the Portuguese Serani from the Arabic Nasrani, but the term now refers to the modern Kristang creoles of Malaysia.",
  "question": "What term did the Malays use for the Portuguese Serani?",
  "answers": [{"answer_start": 75, "text": "Nasrani"}],
  "variant": "clean",
  "source": "squad_v1.1"
}
```

For this example, `context[75:82] == "Nasrani"`.

Read the data using only Python's standard library:

```python
import json

with open("data/train/clean.jsonl", encoding="utf-8") as stream:
    for line in stream:
        record = json.loads(line)
        for answer in record["answers"]:
            start = answer["answer_start"]
            assert record["context"][start:start + len(answer["text"]) ] == answer["text"]
```

## Adversarial pairing and evaluation cohorts

The two original attack files contain **the same 1,000 clean questions**, all found in the official SQuAD development release. Of these, 787 have one or more generated distractors; 213 have none. We retain the 1,000 clean records within `test/clean.jsonl` and remove their redundant copies from the condition files. The split manifest contains:

- `adversarial_cohort_ids`: all 1,000 original question IDs.
- `adversarial_attack_available_ids`: the 787 with attacks.
- `adversarial_no_attack_ids`: the remaining 213.

**Planned main reporting:** clean EM/F1 on all 10,570 project test questions; clean-versus-attacked comparisons on the same 1,000-question cohort. For AddSent, group candidate predictions by `original_id` and select the attacked candidate with the lowest F1 against that question's references. Report EM on that same selected candidate; on tied F1, retain the first candidate in release/file order to match the authors' implementation. Use the clean prediction only when no attacked candidate exists. Do not take the minimum over clean and attacked predictions. AddOneSent already supplies one selected candidate for each of the 787 attacked questions, with clean fallback for the other 213.

This matches the grouping logic in the [authors' evaluator](https://github.com/robinjia/adversarial-squad/blob/master/src/py/eval_squad.py). All expected predictions must be present. As a secondary analysis, results may be restricted to the 787 attacked questions, comparing against exactly those 787 clean counterparts and explicitly labeling that denominator. Do not average all 2,560 AddSent candidates as if they were independent questions, or compare their scores directly with an unmatched clean average.

## Checks completed

The preparation script validated the complete files, rather than only a sample:

- All original source questions were retained; source train and development totals are 87,599 and 10,570.
- Zero overlap between train/dev/test in article title, original question ID, or whitespace-normalized full context. Repeated contexts within one split are expected because a passage can have several questions.
- Every answer span exactly matches its context substring, for every reference answer in every output row.
- Every attacked record's question, title, and answer list match its clean original; the original context is preserved as a prefix.
- AddOneSent's attacked records are a subset of AddSent's candidates.
- Output hashes and counts are recorded in `validation.json`.

These are structural checks, not a new human review of the published distractors' semantic validity. Model input windowing, token offsets, and truncation checks will be implemented with the baseline pipeline in M2.

## Custom augmentation status

No custom augmentation is included in the M1 row counts. In the approved plan, generating and validating this data is the M3 extension. The collected training split provides the source passages, questions, and answer pools. We will fix seen/held-out templates before evaluating, derive replacement entities/answers only from training, preserve each source question's partition, update offsets for prepended sentences, and check that distractors do not change the correct answer. This is an adaptation of Wang and Bansal (2018), [Robust Machine Comprehension Models via Adversarial Training](https://aclanthology.org/N18-2091/), not a claim to have downloaded or reproduced AddSentDiverse.
