"""Fine-tune extractive QA to locate distorted text in patient questions.

Run from any directory, for example:
    python distortion_mrc_extraction.py --check-data
    python distortion_mrc_extraction.py --epochs 2

The default CSV lives in data/ beside this script; use --data for another CSV.
Model output goes to artifacts/span-extractor/ beside this script.
"""

import argparse
from collections import Counter
from pathlib import Path
import re
import string


HERE = Path(__file__).resolve().parent
QUESTION = "What is the distorted part?"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=HERE / "data" / "distortion_processed.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=HERE / "artifacts" / "span-extractor"
    )
    parser.add_argument(
        "--checkpoint", default="distilbert/distilbert-base-cased-distilled-squad"
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--check-data", action="store_true", help="Validate spans without training")
    return parser.parse_args()


def build_examples(frame):
    """Recompute answer offsets because some CSV Answer Start values are wrong."""
    examples = []
    for row in frame.to_dict("records"):
        if row["Dominant Distortion"] == "No Distortion":
            continue
        context = row["Patient Question"]
        answer = row["processed_substrings"]
        if not isinstance(context, str) or not isinstance(answer, str):
            continue
        answer = answer.strip()
        start = context.find(answer)
        if answer and start >= 0:
            examples.append({
                "question": QUESTION, "context": context,
                "answer": answer, "answer_start": start,
            })
    return examples


def prepare_features(examples, tokenizer, max_length, stride):
    """Create sliding windows and align answer characters with token offsets."""
    import torch

    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for offset mappings")
    encoded = tokenizer(
        [example["question"] for example in examples],
        [example["context"] for example in examples],
        max_length=max_length, truncation="only_second", stride=stride,
        return_overflowing_tokens=True, return_offsets_mapping=True,
        padding="max_length",
    )
    sample_indices = encoded.pop("overflow_to_sample_mapping")
    offset_mappings = encoded.pop("offset_mapping")
    prepared = []
    covered_examples = set()
    for feature_index, example_index in enumerate(sample_indices):
        example = examples[example_index]
        offsets = offset_mappings[feature_index]
        context_tokens = [
            index for index, segment in enumerate(encoded.sequence_ids(feature_index))
            if segment == 1
        ]
        cls_index = encoded["input_ids"][feature_index].index(tokenizer.cls_token_id)
        start_token = end_token = cls_index
        start_char = example["answer_start"]
        end_char = start_char + len(example["answer"])
        if context_tokens:
            first, last = context_tokens[0], context_tokens[-1]
            if offsets[first][0] <= start_char and offsets[last][1] >= end_char:
                starts = [
                    index for index in context_tokens
                    if offsets[index][0] <= start_char < offsets[index][1]
                ]
                ends = [
                    index for index in context_tokens
                    if offsets[index][0] < end_char <= offsets[index][1]
                ]
                if starts and ends:
                    start_token, end_token = starts[0], ends[-1]
                    covered_examples.add(example_index)
        feature = {
            name: torch.tensor(values[feature_index])
            for name, values in encoded.items()
        }
        feature["start_positions"] = torch.tensor(start_token)
        feature["end_positions"] = torch.tensor(end_token)
        prepared.append((example_index, feature))
    # An answer longer than every token window cannot be supervised correctly.
    features = [feature for index, feature in prepared if index in covered_examples]
    return features, len(examples) - len(covered_examples)


def qa_scores(predictions, references):
    """Compute normalized exact match and token F1 on held-out examples."""
    if not predictions or len(predictions) != len(references):
        raise ValueError("Prediction and reference lists must have equal nonzero length")

    def tokens(text):
        return re.sub(f"[{re.escape(string.punctuation)}]", " ", text.lower()).split()

    exact_matches = []
    f1_scores = []
    for prediction, reference in zip(predictions, references):
        predicted_tokens, reference_tokens = tokens(prediction), tokens(reference)
        exact_matches.append(predicted_tokens == reference_tokens)
        if not predicted_tokens and not reference_tokens:
            f1_scores.append(1.0)
            continue
        common = sum((Counter(predicted_tokens) & Counter(reference_tokens)).values())
        f1_scores.append(
            2 * common / (len(predicted_tokens) + len(reference_tokens)) if common else 0.0
        )
    return {
        "exact_match": sum(exact_matches) / len(exact_matches),
        "token_f1": sum(f1_scores) / len(f1_scores),
    }


def main():
    args = arguments()
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 32:
        raise ValueError("epochs and batch-size must be positive; max-length must be at least 32")
    if not 0 <= args.stride < args.max_length // 2:
        raise ValueError("stride must be nonnegative and less than half max-length")

    import pandas as pd
    from sklearn.model_selection import train_test_split

    data = pd.read_csv(args.data)
    required = {"Patient Question", "Dominant Distortion", "processed_substrings"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    data = data.dropna(subset=["Patient Question", "Dominant Distortion"])
    train, test = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["Dominant Distortion"]
    )
    train_examples = build_examples(train)
    test_examples = build_examples(test)
    print(f"Usable QA examples: {len(train_examples)} train, {len(test_examples)} test")
    if args.check_data:
        return

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    train_features, skipped = prepare_features(
        train_examples, tokenizer, args.max_length, args.stride
    )
    if not train_features:
        raise ValueError("No annotated answers fit in the token windows")
    print(f"Training windows: {len(train_features)}; spans too long to align: {skipped}")
    train_loader = DataLoader(train_features, batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForQuestionAnswering.from_pretrained(args.checkpoint).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{args.epochs}: mean loss = {total_loss / len(train_loader):.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved model: {args.output_dir}")

    extractor = pipeline(
        "question-answering", model=model, tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )
    predictions = [
        extractor(
            question=example["question"], context=example["context"],
            max_seq_len=args.max_length, doc_stride=args.stride,
        )["answer"]
        for example in test_examples
    ]
    references = [example["answer"] for example in test_examples]
    print(qa_scores(predictions, references))


if __name__ == "__main__":
    main()
