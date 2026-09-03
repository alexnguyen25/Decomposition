"""Profile the real deployment pipeline on one full song to size hosting.

Stages timed (CPU only, to match cheap hosting): load -> demucs htdemucs ->
mel of "other" stem -> CNN14 embed + E7 head. Reports wall time per stage and
peak RSS. Uses data/test_audio/test.mp3 (the project's real test song).
"""

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
sys.path.insert(0, str(SCRATCH / "pylibs"))
PROJECT = Path("/Users/alexnguyen25/Documents/GitHub/Decomposition")

import torch  # noqa: E402
import librosa  # noqa: E402


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def main():
    song = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        PROJECT / "data" / "test_audio" / "test.mp3")
    timings, marks = {}, {"start_rss_gb": rss_gb()}

    t0 = time.time()
    wav, sr = librosa.load(song, sr=44100, mono=False)
    if wav.ndim == 1:
        wav = np.stack([wav, wav])
    dur = wav.shape[1] / 44100
    timings["load_audio_s"] = time.time() - t0
    marks["after_load_rss_gb"] = rss_gb()

    t0 = time.time()
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    dm = get_model("htdemucs")
    dm.eval()
    timings["demucs_model_load_s"] = time.time() - t0
    marks["after_demucs_load_rss_gb"] = rss_gb()

    t0 = time.time()
    with torch.no_grad():
        sources = apply_model(dm, torch.from_numpy(wav).float().unsqueeze(0),
                              device="cpu", progress=False)[0]
    timings["demucs_separate_s"] = time.time() - t0
    marks["after_separate_rss_gb"] = rss_gb()
    other = sources[dm.sources.index("other")].mean(0).numpy()

    t0 = time.time()
    other32 = librosa.resample(other, orig_sr=44100, target_sr=32000)
    timings["resample_s"] = time.time() - t0

    t0 = time.time()
    from panns_inference.models import Cnn14
    cnn14 = Cnn14(sample_rate=32000, window_size=1024, hop_size=320,
                  mel_bins=64, fmin=50, fmax=14000, classes_num=527)
    ckpt = torch.load(SCRATCH / "Cnn14_mAP=0.431.pth", map_location="cpu")
    cnn14.load_state_dict(ckpt["model"])
    cnn14.eval()
    timings["cnn14_load_s"] = time.time() - t0
    marks["after_cnn14_load_rss_gb"] = rss_gb()

    t0 = time.time()
    win = 32000 * 10
    chunks = [other32[s:s + win] for s in range(0, len(other32) - 32000 * 3, win)]
    chunks = [np.pad(c, (0, win - len(c))) if len(c) < win else c
              for c in chunks]
    from panns_head import MLPHead
    head = MLPHead()
    head.load_state_dict(torch.load(SCRATCH / "ckpt_E7_panns_head.pt",
                                    map_location="cpu", weights_only=True))
    head.eval()
    with torch.no_grad():
        x = torch.from_numpy(np.stack(chunks).astype(np.float32))
        emb = cnn14(x)["embedding"]
        probs = torch.sigmoid(head(emb)).numpy()
    timings["classify_cnn14_e7_s"] = time.time() - t0
    marks["peak_rss_gb"] = rss_gb()

    top3 = np.sort(probs, axis=0)[-3:].mean(0)
    report = {
        "song_duration_s": round(dur, 1),
        "timings_s": {k: round(v, 2) for k, v in timings.items()},
        "total_s": round(sum(timings.values()), 1),
        "memory_gb": {k: round(v, 2) for k, v in marks.items()},
        "n_chunks": len(chunks),
        "top_probs": {IDX: float(top3[i]) for i, IDX in enumerate(
            json.load(open(PROJECT / "data/openmic/openmic-2018/class-map.json")))
            if top3[i] >= 0.3},
    }
    print(json.dumps(report, indent=2), flush=True)
    with open(SCRATCH / "pipeline_profile.json", "w") as f:
        json.dump(report, f, indent=2)
    print("PROFILE DONE", flush=True)


if __name__ == "__main__":
    main()
