"""Model loading + instrument classification (BEATs embeddings -> MLP head).

WHY this shape: research showed frozen-pretrained embeddings + tiny head beats
everything else we tried (docs/research F1 doc §8). BEATs is frozen — we only
ever trained the 20-way head. Loading happens ONCE per process (module-level
singletons) because model init is expensive (~2 s) and requests are frequent.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from settings import CLASSIFIER_HEAD, CONFIDENCE_THRESHOLD, DEAD_WEIGHT, MODEL_DIR
from torch import nn

sys.path.insert(0, str(Path(__file__).parent / "beats"))
from BEATs import BEATs, BEATsConfig

# MPS (Apple GPU) if available; CPU otherwise (that's what HF Spaces gives).
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

SR16 = 16000
WIN5 = SR16 * 5          # BEATs protocol: 5-second windows
CHUNK10 = SR16 * 10      # classify per 10-second chunk, like training clips

with open(Path(__file__).parent / "assets" / "class-map.json") as f:
    CLASS_MAP = json.load(f)                       # name -> index
IDX_TO_NAME = {i: n for n, i in CLASS_MAP.items()}


class MLPHead(nn.Module):
    """Identical architecture to the trained checkpoint (768 -> 512 -> 20)."""

    def __init__(self, d_in=768, d_hidden=512, p_drop=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.BatchNorm1d(d_hidden), nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(d_hidden, 20),
        )

    def forward(self, x):
        return self.net(x)


def _load_beats() -> BEATs:
    ckpt = torch.load(MODEL_DIR / "beats" / "BEATs_iter3_plus_AS2M.pt",
                      map_location="cpu")
    model = BEATs(BEATsConfig(ckpt["cfg"]))
    model.load_state_dict(ckpt["model"])
    return model.eval().to(DEVICE)


def _load_head() -> MLPHead:
    name = {"stem_recalib": "ckpt_E10_stem_recalib.pt",
            "e10": "ckpt_E10_beats_head.pt"}[CLASSIFIER_HEAD]
    head = MLPHead()
    head.load_state_dict(torch.load(MODEL_DIR / name, map_location=DEVICE,
                                    weights_only=True))
    return head.eval().to(DEVICE)


BEATS = _load_beats()
HEAD = _load_head()


def _encode_fbank(fbank: torch.Tensor) -> torch.Tensor:
    """BEATs.extract_features body, minus the CPU-only kaldi fbank step —
    so the transformer can run on MPS while fbank runs on CPU."""
    m = BEATS
    x = fbank.unsqueeze(1)
    x = m.patch_embedding(x)
    x = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)
    x = m.layer_norm(x)
    if m.post_extract_proj is not None:
        x = m.post_extract_proj(x)
    x, _ = m.encoder(m.dropout_input(x), padding_mask=None)
    return x                                        # (B, tokens, 768)


def _embed_10s_chunks(chunks: np.ndarray) -> np.ndarray:
    """(B, 160000) 16 kHz chunks -> (B, 768) embeddings.

    Each 10 s chunk becomes two 5 s windows (training protocol), token-mean
    pooled, then averaged. Windows are batched together for one forward pass.
    """
    t = torch.from_numpy(chunks.astype(np.float32))
    b = t.shape[0]
    windows = torch.cat([t[:, :WIN5], t[:, WIN5:WIN5 * 2]], dim=0)
    with torch.no_grad():
        fbank = BEATS.preprocess(windows.cpu())     # kaldi fbank: CPU only
        tokens = _encode_fbank(fbank.to(DEVICE))
        emb = tokens.mean(dim=1)
        return ((emb[:b] + emb[b:]) / 2).cpu().numpy()


def classify_other_stem(other_16k: np.ndarray) -> tuple[list[dict], list[dict]]:
    """'other'-stem waveform (16 kHz mono) ->
    (song_level [{name, confidence}, ...],
     per_chunk  [{t: start_s, top: {name: prob, ...}}, ...]).

    Song-level aggregation is top-3-mean over chunk probabilities, NOT max:
    research showed max-pooling turns per-chunk noise into near-certain
    per-song false positives (F1 doc §6). Dead-weight classes are excluded
    because Demucs already answers vocals/drums/bass as stems.

    The per-chunk view exists for the chat agent's get_instruments(start,end)
    tool — chunk probs are noisier than the song-level verdict (that's WHY we
    aggregate), so it reports only the top few per chunk above 0.3.
    """
    chunks = []
    step = CHUNK10
    for s in range(0, max(1, len(other_16k) - SR16 * 3), step):
        c = other_16k[s:s + step]
        if len(c) < step:
            c = np.pad(c, (0, step - len(c)))
        chunks.append(c)

    probs = []
    for s in range(0, len(chunks), 16):             # bound peak memory
        batch = np.stack(chunks[s:s + 16])
        emb = _embed_10s_chunks(batch)
        with torch.no_grad():
            p = torch.sigmoid(HEAD(torch.from_numpy(emb).to(DEVICE)))
        probs.append(p.cpu().numpy())
    probs = np.concatenate(probs)                   # (n_chunks, 20)

    k = min(3, len(probs))
    song = np.sort(probs, axis=0)[-k:].mean(axis=0)  # top-3-mean

    out = []
    for c in np.argsort(-song):
        name = IDX_TO_NAME[int(c)]
        if name in DEAD_WEIGHT or song[c] < CONFIDENCE_THRESHOLD:
            continue
        out.append({"name": name, "confidence": round(float(song[c]), 3)})

    per_chunk = []
    for i, p in enumerate(probs):
        top = {IDX_TO_NAME[int(c)]: round(float(p[c]), 2)
               for c in np.argsort(-p)[:5]
               if IDX_TO_NAME[int(c)] not in DEAD_WEIGHT and p[c] >= 0.3}
        per_chunk.append({"t": i * 10, "top": top})
    return out, per_chunk
