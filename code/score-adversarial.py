"""Grouped AddSent / AddOneSent evaluation (Jia and Liang, 2017).

Each question is scored by its lowest-F1 attacked candidate (EM from the
same candidate), or by its clean prediction if it has no attack.
See scoring.md for details.

Example:
    python3 code/score-adversarial.py \
      --clean-predictions output/predictions/simple-test.json \
      --adv-gold data/test/addsent.jsonl \
      --adv-predictions output/predictions/simple-addsent.json
"""
import argparse
import json
from pathlib import Path

from score import exact_match, normalize_answer, token_f1


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_predictions(path):
    with open(path, encoding="utf-8") as stream:
        predictions = json.load(stream)
    if not isinstance(predictions, dict):
        raise ValueError(f"{path}: predictions must map IDs to strings.")
    if not all(isinstance(value, str) for value in predictions.values()):
        raise ValueError(f"{path}: every prediction must be a string.")
    return predictions


def score_pair(prediction, references):
    return (
        max(exact_match(prediction, ref) for ref in references),
        max(token_f1(prediction, ref) for ref in references),
    )


def picks_distractor(prediction, clean_context, attacked_context):
    """Text-based check: the prediction occurs only in the inserted sentence."""
    if not normalize_answer(prediction):
        return False
    if not attacked_context.startswith(clean_context):
        raise ValueError("Expected the distractor to be appended to the context.")
    distractor = attacked_context[len(clean_context):]
    return prediction in distractor and prediction not in clean_context


def evaluate(clean_gold, manifest, clean_predictions, adv_gold, adv_predictions):
    cohort = json.loads(Path(manifest).read_text(encoding="utf-8"))[
        "adversarial_cohort_ids"
    ]
    clean_rows = {row["id"]: row for row in read_jsonl(clean_gold)}
    adv_rows = read_jsonl(adv_gold)
    clean_preds = read_predictions(clean_predictions)
    adv_preds = read_predictions(adv_predictions)

    missing_clean = [qid for qid in cohort if qid not in clean_preds]
    if missing_clean:
        raise ValueError(
            f"Clean predictions are missing {len(missing_clean)} cohort IDs."
        )
    adv_ids = [row["id"] for row in adv_rows]
    if len(set(adv_ids)) != len(adv_ids):
        raise ValueError("The adversarial gold file contains duplicate IDs.")
    missing = set(adv_ids) - set(adv_preds)
    extra = set(adv_preds) - set(adv_ids)
    if missing or extra:
        raise ValueError(
            "Adversarial prediction IDs do not match gold IDs: "
            f"{len(missing)} missing, {len(extra)} extra."
        )

    cohort_set = set(cohort)
    selected = {}  # original_id -> (em, f1, row)
    for row in adv_rows:  # File order, so the first candidate wins ties.
        original_id = row["original_id"]
        if original_id not in cohort_set:
            raise ValueError(f"{row['id']} is not in the adversarial cohort.")
        references = [answer["text"] for answer in row["answers"]]
        em, f1 = score_pair(adv_preds[row["id"]], references)
        if original_id not in selected or f1 < selected[original_id][1]:
            selected[original_id] = (em, f1, row)

    totals = {key: 0.0 for key in ("clean_em", "clean_f1", "adv_em", "adv_f1")}
    attacked = {key: 0.0 for key in totals}
    distractor_picks = 0
    for original_id in cohort:
        clean = clean_rows[original_id]
        references = [answer["text"] for answer in clean["answers"]]
        clean_em, clean_f1 = score_pair(clean_preds[original_id], references)
        if original_id in selected:
            adv_em, adv_f1, row = selected[original_id]
            for key, value in zip(attacked, (clean_em, clean_f1, adv_em, adv_f1)):
                attacked[key] += value
            distractor_picks += picks_distractor(
                adv_preds[row["id"]], clean["context"], row["context"]
            )
        else:
            adv_em, adv_f1 = clean_em, clean_f1
        for key, value in zip(totals, (clean_em, clean_f1, adv_em, adv_f1)):
            totals[key] += value

    n, k = len(cohort), len(selected)
    return {
        "condition": Path(adv_gold).stem,
        "cohort_questions": n,
        "clean_exact_match": 100 * totals["clean_em"] / n,
        "clean_f1": 100 * totals["clean_f1"] / n,
        "adversarial_exact_match": 100 * totals["adv_em"] / n,
        "adversarial_f1": 100 * totals["adv_f1"] / n,
        "attacked_only": {
            "questions": k,
            "clean_exact_match": 100 * attacked["clean_em"] / k,
            "clean_f1": 100 * attacked["clean_f1"] / k,
            "adversarial_exact_match": 100 * attacked["adv_em"] / k,
            "adversarial_f1": 100 * attacked["adv_f1"] / k,
            "distractor_pick_rate": 100 * distractor_picks / k,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--clean-gold", default="data/test/clean.jsonl"
    )
    parser.add_argument(
        "--manifest", default="data/split_manifest.json"
    )
    parser.add_argument(
        "--clean-predictions", required=True,
        help="Predictions on clean test (must cover the 1,000 cohort IDs)",
    )
    parser.add_argument(
        "--adv-gold", required=True,
        help="data/test/addsent.jsonl or addonesent.jsonl",
    )
    parser.add_argument(
        "--adv-predictions", required=True,
        help="Predictions on the attacked rows of --adv-gold",
    )
    args = parser.parse_args()
    results = evaluate(
        args.clean_gold,
        args.manifest,
        args.clean_predictions,
        args.adv_gold,
        args.adv_predictions,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
