# train_detector.py

# train_detector.py

import os

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from configs import AuctionConfig, TrainConfig
from detector_model import DetectorClassifier, DetectorDataset
from labeling import add_behavioral_collusion_labels, make_detector_text
from utils import ensure_dir


def train_detector_for_cycle(
    data_path,
    output_dir,
    base_model="distilbert-base-uncased",
    epochs=None,
):
    auction_cfg = AuctionConfig()
    train_cfg = TrainConfig()

    epochs = epochs or train_cfg.detector_epochs_per_cycle

    ensure_dir(output_dir)

    df = pd.read_parquet(data_path)
    df = add_behavioral_collusion_labels(df, auction_cfg)

    df["label"] = df["behavioral_collusion_label"].astype(int)
    texts = df.apply(make_detector_text, axis=1).tolist()
    labels = df["label"].tolist()

    if len(set(labels)) < 2:
        return {
            "checkpoint_path": None,
            "accuracy": None,
            "auroc": None,
            "num_positive": int(sum(labels)),
            "num_total": len(labels),
        }

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=7,
        stratify=labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    train_ds = DetectorDataset(
        train_texts,
        train_labels,
        tokenizer,
        max_length=train_cfg.detector_max_length,
    )

    test_ds = DetectorDataset(
        test_texts,
        test_labels,
        tokenizer,
        max_length=train_cfg.detector_max_length,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.detector_batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg.detector_batch_size,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DetectorClassifier(base_model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.detector_learning_rate,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Detector epoch {epoch + 1}: loss={loss.item():.4f}")

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(batch["label"].numpy())

    preds = [1 if p > 0.5 else 0 for p in all_probs]

    accuracy = accuracy_score(all_labels, preds)
    auroc = roc_auc_score(all_labels, all_probs)

    checkpoint_path = os.path.join(output_dir, "detector.pt")
    torch.save(model.state_dict(), checkpoint_path)
    tokenizer.save_pretrained(output_dir)

    return {
        "checkpoint_path": checkpoint_path,
        "accuracy": accuracy,
        "auroc": auroc,
        "num_positive": int(sum(labels)),
        "num_total": len(labels),
    }


if __name__ == "__main__":
    train_cfg = TrainConfig()

    result = train_detector_for_cycle(
        data_path=os.path.join(train_cfg.run_dir, "data", "all_rollouts.parquet"),
        output_dir=os.path.join(train_cfg.run_dir, "checkpoints", "detector_manual"),
    )

    print(result)