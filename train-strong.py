"""Fine-tune DistilBERT for extractive question answering.
Dependencies used:
    transformers = 5.23.0
    datasets = 5.2.0
    accelerate = 1.15.0
    torch (appropriate build for CUDA, MPS, or CPU)

Input:
    Prepared Hugging Face datasets saved with save_to_disk().
    Features use max_length = 384, stride = 128, and max-length padding.
    Required columns:
        input_ids, attention_mask, start_positions, end_positions

Train from scratch:
    python train_strong.py

Smoke test:
    python train_strong.py --smoke-test

Resume training:
    python train_strong.py \
        --resume-from-checkpoint checkpoint-500 \
        --output-dir runs/distilbert-baseline-colab

Custom feature paths:
    python train_strong.py \
        --train-features /path/to/features/train \
        --dev-features /path/to/features/validation \
        --output-dir /path/to/output

Checkpoint selection uses development loss.
Answer prediction and EM/F1 scoring are performed separately.
"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from time import time_ns


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-features",
        type=Path,
        default=Path("features/train"),
    )
    parser.add_argument(
        "--dev-features",
        type=Path,
        default=Path("features/validation"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument(
        "--previous-best-checkpoint",
        type=Path,
        help="Current location of the previous best checkpoint if it moved.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test and args.resume_from_checkpoint:
        parser.error("Smoke tests cannot resume a full training run.")
    if args.previous_best_checkpoint and not args.resume_from_checkpoint:
        parser.error(
            "--previous-best-checkpoint requires --resume-from-checkpoint."
        )
    return args


def has_weights(folder):
    filenames = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any((folder / name).is_file() for name in filenames)


def resolve_previous_best(checkpoint, override=None):
    """Locate a moved best checkpoint without modifying the source files."""
    state_path = checkpoint / "trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    previous_path = state.get("best_model_checkpoint")
    if not previous_path:
        return None
    if override is not None:
        candidates = [override]
    else:
        previous = Path(previous_path)
        candidates = [
            checkpoint if previous.name == checkpoint.name else None,
            previous,
            checkpoint.parent / previous.name,
        ]
    for candidate in candidates:
        if candidate is None:
            continue
        if (candidate / "config.json").is_file() and has_weights(candidate):
            return str(candidate.resolve())
    raise FileNotFoundError(
        f"Cannot locate previous best checkpoint: {previous_path}. "
        "Provide its current location with --previous-best-checkpoint."
    )


def main():
    args = parse_args()
    from datasets import load_from_disk
    from transformers import (
        AutoModelForQuestionAnswering,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        default_data_collator,
        set_seed,
    )
    set_seed(42)
    default_output = (
        "runs/distilbert-smoke"
        if args.smoke_test
        else "runs/distilbert-baseline"
    )
    output = (args.output_dir or Path(default_output)).resolve()
    checkpoint = args.resume_from_checkpoint
    previous_best = None
    if checkpoint is not None:
        checkpoint = checkpoint.resolve()
        for name in (
            "config.json",
            "trainer_state.json",
            "optimizer.pt",
            "scheduler.pt",
        ):
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(
                    f"Incomplete checkpoint: {checkpoint / name}"
                )
        if not has_weights(checkpoint):
            raise FileNotFoundError(
                f"No model weights found in {checkpoint}"
            )
        previous_best = resolve_previous_best(
            checkpoint,
            args.previous_best_checkpoint,
        )
    if output.exists() and any(output.iterdir()) and checkpoint is None:
        raise FileExistsError(
            f"{output} is not empty. Choose a new output directory "
            "or explicitly resume from a checkpoint."
        )
    train = load_from_disk(str(args.train_features))
    dev = load_from_disk(str(args.dev_features))
    required_columns = {
        "input_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
    }
    for split_name, dataset in [("train", train), ("development", dev)]:
        if not len(dataset):
            raise ValueError(f"The {split_name} dataset is empty.")
        missing = required_columns - set(dataset.column_names)
        if missing:
            raise ValueError(
                f"The {split_name} dataset is missing columns: "
                f"{sorted(missing)}"
            )
    if args.smoke_test:
        train = train.select(range(min(160, len(train))))
        dev = dev.select(range(min(64, len(dev))))
        
    model_source = (
        str(checkpoint)
        if checkpoint is not None
        else "distilbert-base-uncased"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(model_source)
    training_args = TrainingArguments(
        output_dir=str(output),
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        max_steps=10 if args.smoke_test else -1,
        weight_decay=0.01,
        warmup_steps=0 if args.smoke_test else 500,
        optim="adamw_torch",
        eval_strategy="steps",
        eval_steps=10 if args.smoke_test else 500,
        save_strategy="steps",
        save_steps=10 if args.smoke_test else 500,
        logging_steps=1 if args.smoke_test else 50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        fp16=False,
        bf16=False,
        report_to="none",
        seed=42,
        data_seed=42,
    )

    class RestoreBestPath(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            # Trainer loads checkpoint state before this callback.
            if previous_best is not None:
                state.best_model_checkpoint = previous_best

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=dev,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        callbacks=[RestoreBestPath()],
    )

    output.mkdir(parents=True, exist_ok=True)

    run_config = {
        "train_features": str(args.train_features.resolve()),
        "dev_features": str(args.dev_features.resolve()),
        "resume_from_checkpoint": (
            str(checkpoint) if checkpoint is not None else None
        ),
        "train_windows": len(train),
        "dev_windows": len(dev),
        "smoke_test": args.smoke_test,
        "versions": {
            package: version(package)
            for package in [
                "torch",
                "transformers",
                "datasets",
                "accelerate",
            ]
        },
        "training_arguments": training_args.to_dict(),
    }

    config_path = output / f"run_config_{time_ns()}.json"
    config_path.write_text(
        json.dumps(run_config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print("Training device:", training_args.device, flush=True)
    print("Training windows:", len(train), flush=True)
    print("Validation windows:", len(dev), flush=True)
    print("Resume checkpoint:", checkpoint, flush=True)

    result = trainer.train(
        resume_from_checkpoint=(
            str(checkpoint) if checkpoint is not None else None
        )
    )

    best_model_path = output / "best-model"
    trainer.save_model(str(best_model_path))
    tokenizer.save_pretrained(str(best_model_path))
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)

    print("Best checkpoint:", trainer.state.best_model_checkpoint)
    print("Best development loss:", trainer.state.best_metric)
    print("Saved model:", best_model_path)


if __name__ == "__main__":
    main()