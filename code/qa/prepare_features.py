"""Prepare sliding-window features for DistilBERT QA training."""
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased",
        use_fast=True,
    )
    
    assert tokenizer.is_fast

    data_files = {
        "train": "data/train/clean.jsonl",
        "validation": "data/dev/clean.jsonl",
    }

    datasets = load_dataset("json", data_files=data_files)

    def tokenize_batch(batch):
        encoded = tokenizer(
            batch["question"],
            batch["context"],
            truncation="only_second",
            max_length=384,
            stride=128,
            padding="max_length",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )

        sample_mapping = encoded.pop("overflow_to_sample_mapping")
        all_offsets = encoded.pop("offset_mapping")
        start_positions = []
        end_positions = []
        for window_index, offsets in enumerate(all_offsets):
            input_ids = encoded["input_ids"][window_index]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = encoded.sequence_ids(window_index)
            sample_index = sample_mapping[window_index]
            answer = batch["answers"][sample_index][0]
            char_start = answer["answer_start"]
            char_end = char_start + len(answer["text"])
            context_tokens = [
                index
                for index, sequence_id in enumerate(sequence_ids)
                if sequence_id == 1
            ]
            if not context_tokens:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue
            first = context_tokens[0]
            last = context_tokens[-1]
            # A window that does not contain the full answer would use CLS.
            if (
                offsets[first][0] > char_start
                or offsets[last][1] < char_end
            ):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
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
            
            if not first <= start_token <= end_token <= last:
                raise ValueError("Invalid answer-token alignment.")

            start_positions.append(start_token)
            end_positions.append(end_token)
            
        encoded["start_positions"] = start_positions
        encoded["end_positions"] = end_positions
        return encoded

    for split, dataset in datasets.items():
        output = Path("features") / split
        if output.exists():
            raise FileExistsError(
                f"{output} already exists. "
                "Keep it, or rename it before rebuilding."
            )
        features = dataset.map(
            tokenize_batch,
            batched=True,
            batch_size=512,
            remove_columns=dataset.column_names,
            desc=f"Preparing {split}",
        )
        features.save_to_disk(str(output))
        print(
            f"{split}: {len(dataset)} questions "
            f"-> {len(features)} windows saved to {output}"
        )


if __name__ == "__main__":
    main()