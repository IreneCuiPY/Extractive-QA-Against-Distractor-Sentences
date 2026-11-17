"""
Generate extractive QA predictions using a fine-tuned model.
Predictions use only question and context and never used the reference answers.
"""
import argparse
import json
from pathlib import Path
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Directory containing the fine-tuned model and tokenizer",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max-answer-length", type=int, default=30)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
    )
    args = parser.parse_args()
    if args.max_length <= 0:
        parser.error("--max-length must be positive.")
    if not 0 <= args.stride < args.max_length:
        parser.error("--stride must be nonnegative and below --max-length.")
    if args.max_answer_length <= 0:
        parser.error("--max-answer-length must be positive.")
    return args


def select_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_answer(
    question,
    context,
    tokenizer,
    model,
    device,
    max_length,
    stride,
    max_answer_length,
):
    encoded = tokenizer(
        question.lstrip(),
        context,
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        padding="max_length",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded["offset_mapping"].tolist()
    sequence_ids = [
        encoded.sequence_ids(index)
        for index in range(len(encoded["input_ids"]))
    ]
    inputs = {
        name: encoded[name].to(device)
        for name in tokenizer.model_input_names
        if name in encoded
    }
    with torch.inference_mode():
        outputs = model(**inputs)
    starts = outputs.start_logits.cpu()
    ends = outputs.end_logits.cpu()
    best_score = float("-inf")
    best_answer = ""

    for window_index, window_offsets in enumerate(offsets):
        context_positions = [
            index
            for index, sequence_id in enumerate(
                sequence_ids[window_index]
            )
            if sequence_id == 1
            and window_offsets[index][1] > window_offsets[index][0]
        ]
        if not context_positions:
            continue
        context_end = context_positions[-1]
        for start in context_positions:
            last_end = min(
                start + max_answer_length - 1,
                context_end,
            )
            end_scores = ends[window_index, start:last_end + 1]
            end = start + int(end_scores.argmax().item())
            score = float(
                starts[window_index, start]
                + ends[window_index, end]
            )
            if score > best_score:
                char_start = window_offsets[start][0]
                char_end = window_offsets[end][1]
                best_score = score
                best_answer = context[char_start:char_end]
    return best_answer, len(offsets)


def main():
    args = parse_args()
    device = select_device(args.device)
    with args.input.open(encoding="utf-8") as stream:
        rows = [
            json.loads(line)
            for line in stream
            if line.strip()
        ]
    if not rows:
        raise ValueError("The input dataset is empty.")
    question_ids = [row["id"] for row in rows]
    if not all(isinstance(question_id, str) for question_id in question_ids):
        raise ValueError("Every question ID must be a string.")
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("The input dataset contains duplicate IDs.")
    for row in rows:
        if not isinstance(row.get("question"), str):
            raise ValueError(f"Invalid question for ID {row['id']}.")
        if not isinstance(row.get("context"), str):
            raise ValueError(f"Invalid context for ID {row['id']}.")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise ValueError(
            "A fast tokenizer is required for character-offset mapping."
        )
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model
    ).to(device)
    model.eval()
    print("Prediction device:", device, flush=True)
    print("Questions:", len(rows), flush=True)
    predictions = {}
    window_count = 0
    for row in tqdm(rows, desc="Predicting"):
        answer, windows = predict_answer(
            question=row["question"],
            context=row["context"],
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=args.max_length,
            stride=args.stride,
            max_answer_length=args.max_answer_length,
        )
        predictions[row["id"]] = answer
        window_count += windows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            predictions,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print("Questions:", len(predictions))
    print("Windows:", window_count)
    print(
        "Empty predictions:",
        sum(answer == "" for answer in predictions.values()),
    )
    print("Saved predictions:", args.output)


if __name__ == "__main__":
    main()