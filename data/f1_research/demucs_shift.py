"""Measure the mix -> Demucs-"other"-stem distribution shift on OpenMIC test clips.

For N sampled test clips: classify (a) the full mix mel and (b) the mel of the
Demucs "other" stem. Compare masked F1 and false-positive counts on the 16
classes the pipeline actually reports (excluding bass/cymbals/drums/voice).

Ground truth = OpenMIC confirmed labels. Note: for (b), instruments removed by
Demucs (into vocals/drums/bass stems) shouldn't be found in "other" - that is
by design - so we evaluate only the 16 non-dead-weight classes.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from experiment import (  # noqa: E402
    CACHE_DIR, OPENMIC_DIR, PROJECT, BaselineModel, masked_f1,
)

import librosa  # noqa: E402
from demucs.apply import apply_model  # noqa: E402
from demucs.pretrained import get_model  # noqa: E402

SCRATCH = Path(__file__).parent
STEM_MEL_DIR = SCRATCH / "shift_other_mels"
N_CLIPS = 80
SEED = 42
SR = 22050
DEAD_WEIGHT = {2, 5, 6, 19}  # bass, cymbals, drums, voice
KEEP = [c for c in range(20) if c not in DEAD_WEIGHT]


def sample_clips():
    npz = np.load(OPENMIC_DIR / "openmic-2018.npz", allow_pickle=True)
    y_true, y_mask, sample_key = npz["Y_true"], npz["Y_mask"], npz["sample_key"]
    with open(OPENMIC_DIR / "partitions" / "split01_test.csv") as f:
        keys = {line.strip() for line in f if line.strip()}
    sel = np.array([k in keys for k in sample_key])
    y = (y_true[sel] >= 0.5).astype(np.float32)
    m = y_mask[sel].astype(np.float32)
    kept = sample_key[sel]

    # prefer clips with at least one confirmed positive among KEEP classes
    has_pos = (y[:, KEEP] * m[:, KEEP]).sum(axis=1) > 0
    idx = np.flatnonzero(has_pos)
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(idx, size=min(N_CLIPS, len(idx)), replace=False)
    return [str(k) for k in kept[chosen]], y[chosen], m[chosen]


def mel_from_wave(y):
    s = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=128, n_fft=2048,
                                       hop_length=512)
    mel = librosa.power_to_db(s, ref=np.max)
    out = np.full((128, 431), -80.0, dtype=np.float32)
    w = min(mel.shape[1], 431)
    out[:, :w] = mel[:, :w]
    return out


def separate_other(demucs_model, audio_path, device):
    wav, sr = librosa.load(audio_path, sr=44100, mono=False)
    if wav.ndim == 1:
        wav = np.stack([wav, wav])
    x = torch.from_numpy(wav).float().unsqueeze(0)
    with torch.no_grad():
        sources = apply_model(demucs_model, x, device=device, progress=False)[0]
    other = sources[demucs_model.sources.index("other")].mean(dim=0).numpy()
    return librosa.resample(other, orig_sr=44100, target_sr=SR)


def main():
    STEM_MEL_DIR.mkdir(exist_ok=True)
    keys, y, m = sample_clips()
    print(f"{len(keys)} clips sampled", flush=True)

    demucs_model = get_model("htdemucs")
    demucs_model.eval()
    dev = "cpu"  # MPS support in demucs 4.0.1 is flaky; CPU is fine for 80 clips

    mix_mels, other_mels = [], []
    for i, key in enumerate(keys):
        mix_mels.append(np.load(CACHE_DIR / f"{key}.npy").astype(np.float32))
        stem_mel_path = STEM_MEL_DIR / f"{key}.npy"
        if stem_mel_path.exists():
            other_mels.append(np.load(stem_mel_path))
        else:
            audio_path = OPENMIC_DIR / "audio" / key[:3] / f"{key}.ogg"
            other = separate_other(demucs_model, audio_path, dev)
            om = mel_from_wave(other)
            np.save(stem_mel_path, om)
            other_mels.append(om)
        if (i + 1) % 10 == 0:
            print(f"  separated {i + 1}/{len(keys)}", flush=True)

    def fix_width(mel):
        out = np.full((128, 431), -80.0, dtype=np.float32)
        w = min(mel.shape[1], 431)
        out[:, :w] = mel[:, :w]
        return out

    x_mix = np.stack([fix_width(mm) for mm in mix_mels])
    x_other = np.stack(other_mels)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = BaselineModel().to(device)
    state = torch.load(PROJECT / "models" / "classifier.pt",
                       map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    def probs_for(x):
        with torch.no_grad():
            xb = torch.from_numpy(x).unsqueeze(1).to(device)
            return torch.sigmoid(model(xb)).cpu().numpy()

    p_mix, p_other = probs_for(x_mix), probs_for(x_other)

    with open(OPENMIC_DIR / "class-map.json") as f:
        class_map = json.load(f)
    name_of = {i: n for n, i in class_map.items()}

    print("\nclass                    mixF1  stemF1   mixFP  stemFP  (confirmed neg)")
    rows = []
    for c in KEEP:
        conf = m[:, c] == 1
        yt = y[conf, c]
        pm = (p_mix[conf, c] >= 0.5).astype(int)
        po = (p_other[conf, c] >= 0.5).astype(int)

        def f1_of(yp):
            tp = int(((yp == 1) & (yt == 1)).sum())
            fp = int(((yp == 1) & (yt == 0)).sum())
            fn = int(((yp == 0) & (yt == 1)).sum())
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            return 2 * p * r / (p + r) if p + r else 0.0

        fp_mix = int(((pm == 1) & (yt == 0)).sum())
        fp_other = int(((po == 1) & (yt == 0)).sum())
        n_neg = int((yt == 0).sum())
        rows.append({"class": name_of[c], "mix_f1": f1_of(pm),
                     "stem_f1": f1_of(po), "mix_fp": fp_mix,
                     "stem_fp": fp_other, "n_neg": n_neg})
        print(f"{name_of[c]:<22} {f1_of(pm):6.3f} {f1_of(po):7.3f} "
              f"{fp_mix:7d} {fp_other:7d}   ({n_neg})")

    mix_macro = float(np.mean([r["mix_f1"] for r in rows]))
    stem_macro = float(np.mean([r["stem_f1"] for r in rows]))
    print(f"\nmacro-F1 over 16 kept classes: mix {mix_macro:.4f} -> "
          f"other-stem {stem_macro:.4f}")
    tot_fp_mix = sum(r["mix_fp"] for r in rows)
    tot_fp_other = sum(r["stem_fp"] for r in rows)
    print(f"total false positives on confirmed negatives: "
          f"mix {tot_fp_mix} -> other-stem {tot_fp_other}")

    with open(SCRATCH / "shift_results.json", "w") as f:
        json.dump({"rows": rows, "mix_macro": mix_macro,
                   "stem_macro": stem_macro, "keys": keys}, f, indent=2)
    print("SHIFT DONE", flush=True)


if __name__ == "__main__":
    main()
