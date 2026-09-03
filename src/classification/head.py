"""The trained classification head: 768-d BEATs embedding -> 20 instrument logits.

This is the only part of the BEATs path that was trained for this project. The
backbone stays frozen — fine-tuning it end-to-end was measured and tied the
frozen version at 300x the cost (docs/research, experiment E9).
"""

import torch
from torch import nn

from src.config import EMBED_DIM, NUM_CLASSES


class MLPHead(nn.Module):
    """768 -> 512 -> 20. Architecture must match the saved checkpoints exactly."""

    def __init__(self, d_in: int = EMBED_DIM, d_hidden: int = 512,
                 p_drop: float = 0.5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(d_hidden, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


def load_head(checkpoint_path, device) -> MLPHead:
    """Build the head, load weights, put it in eval mode."""
    head = MLPHead().to(device)
    head.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    head.eval()
    return head
