"""Evaluate QA predictions against the prepared SQuAD JSONL data."""
import argparse
import json
import re
import string
from collections import Counter


def normalize_answer(text):
    """Using the official SQuAD 1.1 answer normalization."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, reference):
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction, reference):
    predicted_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    overlap = Counter(predicted_tokens) & Counter(reference_tokens)
    num_shared = sum(overlap.values())
    if num_shared == 0:
        return 0.0
    precision = num_shared / len(predicted_tokens)
    recall = num_shared / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate(gold_path, predictions_path):
    with open(gold_path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError("The gold file is empty.")
    gold_ids = [row["id"] for row in rows]
    if len(set(gold_ids)) != len(gold_ids):
        raise ValueError("The gold file contains duplicate IDs.")
    with open(predictions_path, encoding="utf-8") as stream:
        predictions = json.load(stream)
    if not isinstance(predictions, dict):
        raise ValueError("Predictions must be a JSON object mapping IDs to strings.")
    if not all(isinstance(value, str) for value in predictions.values()):
        raise ValueError("Every prediction must be a string.")

    missing = set(gold_ids) - set(predictions)
    extra = set(predictions) - set(gold_ids)

    if missing or extra:
        raise ValueError(
            f"Prediction IDs do not match gold IDs: "
            f"{len(missing)} missing, {len(extra)} extra."
        )

    total_em = 0.0
    total_f1 = 0.0

    for row in rows:
        references = [answer["text"] for answer in row["answers"]]
        if not references:
            raise ValueError(f"No reference answers for {row['id']}.")
        prediction = predictions[row["id"]]
        total_em += max(exact_match(prediction, ref) for ref in references)
        total_f1 += max(token_f1(prediction, ref) for ref in references)

    return {
        "exact_match": 100 * total_em / len(rows),
        "f1": 100 * total_f1 / len(rows),
        "total": len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, help="Gold JSONL file")
    parser.add_argument("--predictions", required=True, help="Predictions JSON file")
    args = parser.parse_args()
    results = evaluate(args.gold, args.predictions)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()