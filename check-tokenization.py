"""Check answer-token alignment across all training examples."""
import json
from transformers import AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased",
        use_fast=True,
    )
    
    assert tokenizer.is_fast
    
    total = 0
    windows = 0
    failed = []

    with open(
        "submission/data/train/clean.jsonl", encoding="utf-8"
    ) as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            answer = row["answers"][0]
            char_start = answer["answer_start"]
            char_end = char_start + len(answer["text"])
            encoded = tokenizer(
                row["question"],
                row["context"],
                truncation="only_second",
                max_length=384,
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
            )
            found = False
            for index, offsets in enumerate(encoded["offset_mapping"]):
                sequence_ids = encoded.sequence_ids(index)
                context_tokens = [
                    i for i, sequence_id in enumerate(sequence_ids)
                    if sequence_id == 1
                ]
                if not context_tokens:
                    continue
                first = context_tokens[0]
                last = context_tokens[-1]
                if (
                    offsets[first][0] > char_start
                    or offsets[last][1] < char_end
                ):
                    continue
                start_token = first
                while (
                    start_token <= last
                    and offsets[start_token][0] <= char_start
                ):
                    start_token += 1
                start_token -= 1
                end_token = last
                while (
                    end_token >= first
                    and offsets[end_token][1] >= char_end
                ):
                    end_token -= 1
                end_token += 1
                if first <= start_token <= end_token <= last:
                    extracted = row["context"][
                        offsets[start_token][0]:offsets[end_token][1]
                    ]
                    if answer["text"] in extracted:
                        found = True
            total += 1
            windows += len(encoded["input_ids"])

            if not found:
                failed.append(row["id"])
            if total % 10000 == 0:
                print(f"Checked {total} questions...", flush=True)
                
    print(f"Questions: {total}")
    print(f"Windows: {windows}")
    print(f"Questions without an aligned answer: {len(failed)}")
    if failed:
        print("First failed IDs:", failed[:10])


if __name__ == "__main__":
    main()