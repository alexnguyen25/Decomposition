from torch.utils.data import DataLoader
from src.config import NUM_EPOCHS, OPENMIC_DIR, CACHE_DIR, TRAIN_PARTITION, CHECKPOINT_PATH
from src.classification.dataset import OpenMICDataset
from src.classification.model import Model

import torch
from torch import nn

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
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")


    return dataloader, model, optimizer, loss_fn, device

def train(dataloader, model, optimizer, loss_fn, num_epochs, device):
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            spec, labels, label_mask = [t.to(device) for t in batch]

            optimizer.zero_grad()

            logits = model(spec)

            per_element = loss_fn(logits, labels)
            loss = (per_element * label_mask).sum() / label_mask.sum()
            total_loss += loss.item()

            loss.backward()

            optimizer.step()
        print(total_loss / len(dataloader))


    return model
