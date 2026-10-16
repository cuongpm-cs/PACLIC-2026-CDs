"""Fine-tune a 10-class distortion classifier on annotated text spans.

Run from any directory, for example:
    python distortion_classification.py --check-data
    python distortion_classification.py --epochs 2

This evaluates gold annotated spans, so the score is an oracle-input result.
The default CSV lives in data/ beside this script; use --data for another CSV.
"""

import argparse
from pathlib import Path


HERE = Path(__file__).resolve().parent


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=HERE / "data" / "distortion_processed.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=HERE / "artifacts" / "type-transformer"
    )
    parser.add_argument("--checkpoint", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--check-data", action="store_true", help="Validate the CSV without training")
    return parser.parse_args()


def train_and_predict(train_texts, train_labels, test_texts, label_names, args):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import DataCollatorWithPadding

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)

    class TextDataset(Dataset):
        def __init__(self, texts, labels=None):
            self.encoded = tokenizer(list(texts), truncation=True, max_length=args.max_length)
            self.labels = labels

        def __len__(self):
            return len(self.encoded["input_ids"])

        def __getitem__(self, index):
            item = {key: values[index] for key, values in self.encoded.items()}
            if self.labels is not None:
                item["labels"] = int(self.labels[index])
            return item

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_loader = DataLoader(
        TextDataset(train_texts, train_labels), batch_size=args.batch_size,
        shuffle=True, collate_fn=collator,
    )
    test_loader = DataLoader(
        TextDataset(test_texts), batch_size=args.batch_size, collate_fn=collator,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, num_labels=len(label_names)
    ).to(device)
    model.config.id2label = {i: str(label) for i, label in enumerate(label_names)}
    model.config.label2id = {label: i for i, label in model.config.id2label.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{args.epochs}: mean loss = {total_loss / len(train_loader):.4f}")

    model.eval()
    predictions = []
    with torch.inference_mode():
        for batch in test_loader:
            inputs = {key: value.to(device) for key, value in batch.items()}
            predictions.extend(model(**inputs).logits.argmax(dim=-1).cpu().tolist())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved model: {args.output_dir}")
    return predictions


def main():
    args = arguments()
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 8:
        raise ValueError("epochs and batch-size must be positive; max-length must be at least 8")

    import pandas as pd
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    data = pd.read_csv(args.data)
    required = {"Patient Question", "Dominant Distortion", "processed_substrings"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    data = data.dropna(subset=["Patient Question", "Dominant Distortion"])
    train, test = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["Dominant Distortion"]
    )
    train = train[(train["Dominant Distortion"] != "No Distortion") &
                  train["processed_substrings"].notna()].copy()
    test = test[(test["Dominant Distortion"] != "No Distortion") &
                test["processed_substrings"].notna()].copy()
    encoder = LabelEncoder().fit(train["Dominant Distortion"])
    print(f"Train: {len(train)} positive spans; test: {len(test)} positive spans")
    print(dict(enumerate(encoder.classes_)))
    if args.check_data:
        return

    train_labels = encoder.transform(train["Dominant Distortion"]).tolist()
    test_labels = encoder.transform(test["Dominant Distortion"]).tolist()
    predictions = train_and_predict(
        train["processed_substrings"].tolist(), train_labels,
        test["processed_substrings"].tolist(), encoder.classes_, args,
    )
    print(classification_report(
        test_labels, predictions, labels=list(range(len(encoder.classes_))),
        target_names=encoder.classes_, zero_division=0,
    ))


if __name__ == "__main__":
    main()
