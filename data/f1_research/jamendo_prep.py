"""Select, download, and Demucs-separate CC-licensed Jamendo tracks with
known instrument tags, for real-song evaluation of the classifier.

Outputs per track under scratchpad/jamendo/:
  <id>.mp3            original audio
  <id>_other_mel.npy  mel of the Demucs "other" stem (128 x T)
  <id>_mix_mel.npy    mel of the full mix (128 x T)
manifest.json         track id -> {tags, mapped_openmic_classes, duration}
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import librosa  # noqa: E402
from demucs.apply import apply_model  # noqa: E402
from demucs.pretrained import get_model  # noqa: E402

SCRATCH = Path(__file__).parent
OUT = SCRATCH / "jamendo"
TSV = SCRATCH / "jamendo_instrument.tsv"
SR = 22050

# MTG-Jamendo tag -> OpenMIC class (skip Demucs dead-weight classes)
TAG_MAP = {
    "accordion": "accordion", "cello": "cello", "clarinet": "clarinet",
    "flute": "flute", "guitar": "guitar", "electricguitar": "guitar",
    "acousticguitar": "guitar", "classicalguitar": "guitar",
    "organ": "organ", "pipeorgan": "organ", "piano": "piano",
    "saxophone": "saxophone", "synthesizer": "synthesizer",
    "trombone": "trombone", "trumpet": "trumpet", "ukulele": "ukulele",
    "violin": "violin", "banjo": "banjo", "mandolin": "mandolin",
}
# classes we most want represented (weak per-class F1 first)
PRIORITY = ["clarinet", "flute", "accordion", "organ", "trombone", "ukulele",
            "saxophone", "trumpet", "cello", "violin", "banjo", "mandolin",
            "piano", "guitar", "synthesizer"]
N_TRACKS = 12


def load_tracks():
    tracks = []
    with open(TSV) as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            tid = parts[0].split("_")[1].lstrip("0")
            dur = float(parts[4])
            raw = [t.replace("instrument---", "") for t in parts[5:] if t]
            mapped = sorted({TAG_MAP[t] for t in raw if t in TAG_MAP})
            tracks.append({"id": tid, "duration": dur, "raw_tags": raw,
                           "classes": mapped})
    return tracks


def select(tracks):
    # 2-5 mapped tags: enough ground truth to score, few enough to trust
    usable = [t for t in tracks
              if 90 <= t["duration"] <= 300 and 2 <= len(t["classes"]) <= 5]
    chosen_ids, chosen = set(), []
    per_class = {}
    for cls in PRIORITY:
        cands = [t for t in usable
                 if cls in t["classes"] and t["id"] not in chosen_ids]
        cands.sort(key=lambda t: -len(t["classes"]))
        for pick in cands[:2 - per_class.get(cls, 0)]:
            chosen.append(pick)
            chosen_ids.add(pick["id"])
            for c in pick["classes"]:
                per_class[c] = per_class.get(c, 0) + 1
            if len(chosen) >= N_TRACKS:
                return chosen
    return chosen


def download(tid: str, dest: Path) -> bool:
    url = f"https://mp3d.jamendo.com/?trackid={tid}&format=mp32"
    for attempt in range(3):
        r = subprocess.run(["curl", "-sL", "--max-time", "120", "-o",
                            str(dest), url], capture_output=True)
        if r.returncode == 0 and dest.exists() and dest.stat().st_size > 200_000:
            return True
        time.sleep(3)
    return False


def mel_of(y):
    s = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=128, n_fft=2048,
                                       hop_length=512)
    return librosa.power_to_db(s, ref=np.max).astype(np.float32)


def main():
    OUT.mkdir(exist_ok=True)
    chosen = select(load_tracks())
    print(f"selected {len(chosen)} tracks:", flush=True)
    for t in chosen:
        print(f"  {t['id']}: {t['classes']} ({t['duration']:.0f}s)", flush=True)

    demucs_model = get_model("htdemucs")
    demucs_model.eval()

    manifest = {}
    for t in chosen:
        tid = t["id"]
        mp3 = OUT / f"{tid}.mp3"
        other_mel_path = OUT / f"{tid}_other_mel.npy"
        if not mp3.exists() and not download(tid, mp3):
            print(f"  {tid}: DOWNLOAD FAILED, skipping", flush=True)
            continue
        if not other_mel_path.exists():
            wav, _ = librosa.load(mp3, sr=44100, mono=False)
            if wav.ndim == 1:
                wav = np.stack([wav, wav])
            x = torch.from_numpy(wav).float().unsqueeze(0)
            with torch.no_grad():
                sources = apply_model(demucs_model, x, device="cpu",
                                      progress=False)[0]
            other = sources[demucs_model.sources.index("other")].mean(0).numpy()
            other_32k = librosa.resample(other, orig_sr=44100, target_sr=32000)
            np.save(OUT / f"{tid}_other_32k.npy", other_32k.astype(np.float16))
            other = librosa.resample(other, orig_sr=44100, target_sr=SR)
            np.save(other_mel_path, mel_of(other))
            mix = wav.mean(axis=0)
            mix = librosa.resample(mix, orig_sr=44100, target_sr=SR)
            np.save(OUT / f"{tid}_mix_mel.npy", mel_of(mix))
        manifest[tid] = t
        print(f"  {tid}: done", flush=True)

    # keep any previously prepped tracks in the manifest
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            old = json.load(f)
        for tid, meta in old.items():
            manifest.setdefault(tid, meta)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"JAMENDO PREP DONE ({len(manifest)} tracks)", flush=True)


if __name__ == "__main__":
    main()
