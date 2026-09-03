"""Frozen BEATs backbone: 16 kHz audio -> 768-d embeddings.

The protocol here is not a free choice — it must reproduce how the head was
trained, or the embeddings land in a different space than the head expects:

    10 s chunk -> two 5 s windows -> kaldi fbank -> BEATs encoder
               -> mean over tokens -> mean over the two windows -> (768,)

Kaldi fbank runs on CPU because the op is unsupported on Apple MPS; only the
transformer moves to the accelerator.
"""

import numpy as np
import torch

from src.classification.beats import BEATs, BEATsConfig
from src.config import CHUNK_SAMPLES, SR16, WINDOW_SAMPLES


def pick_device() -> torch.device:
    """CUDA if present, else Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_backbone(checkpoint_path, device) -> BEATs:
    """Load the frozen BEATs backbone. Never trained, only ever run in eval."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = BEATs(BEATsConfig(checkpoint["cfg"]))
    model.load_state_dict(checkpoint["model"])
    return model.eval().to(device)


def _encode(model: BEATs, fbank: torch.Tensor) -> torch.Tensor:
    """BEATs.extract_features minus the CPU-only fbank step.

    Reimplemented rather than called so the fbank can be computed on CPU while
    the transformer runs on MPS/CUDA — extract_features would force both onto
    one device.
    """
    x = model.patch_embedding(fbank.unsqueeze(1))
    x = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)
    x = model.layer_norm(x)
    if model.post_extract_proj is not None:
        x = model.post_extract_proj(x)
    x, _ = model.encoder(model.dropout_input(x), padding_mask=None)
    return x                                    # (batch, tokens, 768)


def embed_chunks(model: BEATs, chunks: np.ndarray, device) -> np.ndarray:
    """(n, CHUNK_SAMPLES) 16 kHz mono chunks -> (n, 768) embeddings.

    Both 5 s windows of every chunk go through the encoder in a single batch,
    then are averaged back together.
    """
    waves = torch.from_numpy(chunks.astype(np.float32))
    n = waves.shape[0]
    windows = torch.cat(
        [waves[:, :WINDOW_SAMPLES], waves[:, WINDOW_SAMPLES:WINDOW_SAMPLES * 2]],
        dim=0,
    )
    with torch.no_grad():
        fbank = model.preprocess(windows.cpu())          # kaldi fbank: CPU only
        tokens = _encode(model, fbank.to(device))
        pooled = tokens.mean(dim=1)                      # mean over tokens
        return ((pooled[:n] + pooled[n:]) / 2).cpu().numpy()


def to_chunks(waveform: np.ndarray) -> np.ndarray:
    """Split a 16 kHz mono waveform into (n, CHUNK_SAMPLES), zero-padding the tail.

    A trailing fragment shorter than 3 s is dropped rather than padded: mostly
    silence produces a meaningless embedding that still votes in aggregation.
    """
    minimum_tail = 3 * SR16                      # drop tails under 3 seconds
    chunks = []
    for start in range(0, max(1, len(waveform) - minimum_tail), CHUNK_SAMPLES):
        chunk = waveform[start:start + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        chunks.append(chunk)
    return np.stack(chunks)

