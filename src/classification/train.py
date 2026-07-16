"""Train the OpenMIC instrument CNN with masked BCEWithLogitsLoss."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.classification.dataset import OpenMICDataset
from src.classification.model import Model
from src.config import CACHE_DIR, NUM_EPOCHS, OPENMIC_DIR, TRAIN_PARTITION

CHECKPOINT_PATH = Path("models/classifier.pt")


def main():
    dataloader, model, optimizer, loss_fn, device = setup()
    trained_model = train(dataloader, model, optimizer, loss_fn, NUM_EPOCHS, device)

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trained_model.state_dict(), CHECKPOINT_PATH)
    print(f"Saved checkpoint to {CHECKPOINT_PATH}")


def setup():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = OpenMICDataset(OPENMIC_DIR, TRAIN_PARTITION, CACHE_DIR)
    print(f"Train set size: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    return dataloader, model, optimizer, loss_fn, device


def train(dataloader, model, optimizer, loss_fn, num_epochs, device):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            specs, labels, label_mask = batch
            specs = specs.to(device)
            labels = labels.to(device)
            label_mask = label_mask.to(device)

            optimizer.zero_grad()

            logits = model(specs)

            per_element = loss_fn(logits, labels)
            loss = (per_element * label_mask).sum() / label_mask.sum().clamp(min=1.0)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}  loss={total_loss / len(dataloader):.4f}")

    return model


if __name__ == "__main__":
    main()
