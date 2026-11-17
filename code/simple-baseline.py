"""Predict the context sentence by finding the highest question-token overlap."""
import argparse
import json
import re
from pathlib import Path
from score import normalize_answer


def predict(question, context):
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", context)
        if sentence.strip()
    ]    # Split on whitespace after sentence-ending punctuation.
    if not sentences:
        return ""
    question_tokens = set(normalize_answer(question).split())
    
    def overlap(sentence):
        sentence_tokens = set(normalize_answer(sentence).split())
        return len(question_tokens & sentence_tokens)

    return max(sentences, key=overlap)  # Keeping the first sentence when scores tie.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output predictions JSON")
    args = parser.parse_args()
    predictions = {}

    with open(args.input, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = row["id"]
            if question_id in predictions:
                raise ValueError(f"Duplicate question ID: {question_id}")
            predictions[question_id] = predict(
                row["question"], row["context"]
            )

    if not predictions:
        raise ValueError("The input file is empty.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved {len(predictions)} predictions to {output}")


if __name__ == "__main__":
    main()