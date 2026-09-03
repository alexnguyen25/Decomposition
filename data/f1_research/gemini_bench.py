"""Gemini vs classifier benchmark on labeled audio.

Usage: gemini_bench.py dev|test|jamendo|all

- dev: iterate 3 prompt variants on 15 TRAIN-partition clips (never test) and
  pick the best by masked macro-F1.
- test: run winning prompt on 100 stratified TEST clips with gemini-flash-latest
  and gemini-pro-latest (50-clip subset); score E7/E4 on the same clips.
- jamendo: run on the 14 real tracks; compare tag recall/extras vs classifiers.

Key from repo .env; never printed. Results -> gemini_bench_*.json in scratchpad.
"""

import base64
import json
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
from experiment import OPENMIC_DIR, PROJECT  # noqa: E402

CTX = ssl._create_unverified_context()  # this python build lacks local certs
ENV = PROJECT / ".env"
FLASH = "gemini-flash-latest"
PRO = "gemini-pro-latest"
SEED = 42

# OpenMIC class -> human-readable prompt name
with open(OPENMIC_DIR / "class-map.json") as f:
    CLASS_MAP = json.load(f)  # name -> idx
IDX_TO_NAME = {i: n for n, i in CLASS_MAP.items()}
PROMPT_NAME = {
    "mallet_percussion": "mallet percussion (xylophone, marimba, vibraphone, glockenspiel)",
    "voice": "voice (human singing or speech)",
}
VOCAB_LINES = "\n".join(
    f"- {n}" + (f": {PROMPT_NAME[n].split(': ')[-1]}" if False else "")
    for n in sorted(CLASS_MAP)
)


def api_key() -> str:
    for line in open(ENV):
        if line.startswith("GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("no key in .env")


KEY = api_key()


def call_gemini(model: str, text: str, audio_path: Path | None = None,
                max_retries: int = 5):
    """Return (parsed_or_text, usage, seconds). JSON response mode."""
    parts = [{"text": text}]
    if audio_path is not None:
        mime = {"ogg": "audio/ogg", "mp3": "audio/mp3",
                "wav": "audio/wav"}[audio_path.suffix.lstrip(".")]
        data = base64.b64encode(audio_path.read_bytes()).decode()
        parts.append({"inline_data": {"mime_type": mime, "data": data}})
    body = json.dumps({
        "contents": [{"parts": parts}],
        "generationConfig": {"response_mime_type": "application/json"},
    }).encode()
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={KEY}")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=180, context=CTX) as r:
                d = json.load(r)
            dt = time.time() - t0
            txt = d["candidates"][0]["content"]["parts"][0]["text"]
            usage = d.get("usageMetadata", {})
            try:
                return json.loads(txt), usage, dt
            except json.JSONDecodeError:
                return txt, usage, dt
        except urllib.error.HTTPError as e:
            code = e.code
            msg = e.read().decode()[:200]
            if code in (429, 500, 503) and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    HTTP {code}, retry in {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP {code}: {msg}") from e
        except Exception as e:  # noqa: BLE001 - timeouts etc.
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise


PROMPTS = {
    "P1_basic": (
        "Listen to this audio clip. Which of the following instruments are "
        "audibly present?\n\nAllowed instrument names (use EXACTLY these "
        "strings):\n{vocab}\n\nRespond with JSON: "
        '{{"instruments": ["name1", "name2", ...]}}'
    ),
    "P2_conservative": (
        "Listen to this audio clip. Which of the following instruments are "
        "audibly present? Only include an instrument if you are confident it "
        "is clearly audible - do not guess or include instruments that merely "
        "might be there.\n\nAllowed instrument names (use EXACTLY these "
        "strings):\n{vocab}\n\nRespond with JSON: "
        '{{"instruments": ["name1", "name2", ...]}}'
    ),
    "P3_reasoned": (
        "You are an expert music analyst. Listen to this audio clip and "
        "identify instruments.\n\nAllowed instrument names (use EXACTLY these "
        "strings):\n{vocab}\n\nNotes: 'cymbals' and 'drums' refer to a drum "
        "kit; 'mallet_percussion' means xylophone/marimba/vibraphone/"
        "glockenspiel; 'voice' means human singing or speech; 'synthesizer' "
        "means electronic synth sounds. Include an instrument only if clearly "
        "audible.\n\nRespond with JSON: "
        '{{"instruments": [...], "uncertain": [...]}} where "instruments" are '
        "clearly audible and \"uncertain\" are possible but not clear."
    ),
}
VOCAB = "\n".join(f"- {n}" for n in sorted(CLASS_MAP))


def parse_instruments(resp) -> set[str]:
    if not isinstance(resp, dict):
        return set()
    names = resp.get("instruments", [])
    return {n for n in names if n in CLASS_MAP}


def load_labels():
    npz = np.load(OPENMIC_DIR / "openmic-2018.npz", allow_pickle=True)
    return (npz["Y_true"], npz["Y_mask"],
            np.array([str(k) for k in npz["sample_key"]]))


def partition_keys(partition):
    with open(OPENMIC_DIR / "partitions" / partition) as f:
        return [line.strip() for line in f if line.strip()]


def masked_scores(pred_sets, y, m, keys):
    """Per-class P/R/F1 over confirmed labels only; macro-F1."""
    per_class = {}
    for name, c in CLASS_MAP.items():
        tp = fp = fn = tn = 0
        for i, k in enumerate(keys):
            if m[i, c] != 1:
                continue
            true_pos = y[i, c] >= 0.5
            pred_pos = name in pred_sets[k]
            tp += true_pos and pred_pos
            fp += (not true_pos) and pred_pos
            fn += true_pos and not pred_pos
            tn += (not true_pos) and (not pred_pos)
        p = float(tp / (tp + fp)) if tp + fp else 0.0
        r = float(tp / (tp + fn)) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per_class[name] = {"p": p, "r": r, "f1": f1,
                           "support": int(tp + fp + fn + tn)}
    macro = float(np.mean([v["f1"] for v in per_class.values()]))
    return {"per_class": per_class, "macro_f1": macro}


def audio_path_of(key):
    return OPENMIC_DIR / "audio" / key[:3] / f"{key}.ogg"


def sample_clips(partition, n, min_confirmed_pos=1):
    y, m, all_keys = load_labels()
    keys = set(partition_keys(partition))
    sel = np.array([k in keys for k in all_keys])
    y, m, all_keys = y[sel], m[sel], all_keys[sel]
    pos_count = ((y >= 0.5) * m).sum(axis=1)
    idx = np.flatnonzero(pos_count >= min_confirmed_pos)
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(idx, size=min(n, len(idx)), replace=False)
    return [str(k) for k in all_keys[chosen]], y[chosen], m[chosen]


def run_set(model, prompt_text, keys, label="run"):
    cache_path = SCRATCH / f"gemini_cache_{model}.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    preds, usages, lats, raw = {}, [], [], {}
    for i, k in enumerate(keys):
        if k in cache:
            preds[k] = cache[k]
            continue
        resp, usage, dt = call_gemini(model, prompt_text, audio_path_of(k))
        preds[k] = sorted(parse_instruments(resp))
        raw[k] = resp if isinstance(resp, dict) else str(resp)[:300]
        usages.append(usage)
        lats.append(dt)
        cache[k] = preds[k]
        cache_path.write_text(json.dumps(cache))  # persist every call
        if (i + 1) % 10 == 0:
            print(f"  {label}: {i + 1}/{len(keys)}", flush=True)
        time.sleep(0.3)
    if not lats:  # fully cached rerun
        lats = [0.0]
    tok = {
        "prompt": sum(u.get("promptTokenCount", 0) for u in usages),
        "output": sum(u.get("candidatesTokenCount", 0) for u in usages),
        "thoughts": sum(u.get("thoughtsTokenCount", 0) for u in usages),
    }
    return preds, tok, {"mean_s": float(np.mean(lats)),
                        "p90_s": float(np.percentile(lats, 90))}, raw


def cmd_dev():
    keys, y, m = sample_clips("split01_train.csv", 15, min_confirmed_pos=3)
    print(f"dev set: {len(keys)} train clips", flush=True)
    out = {}
    for pname, ptmpl in PROMPTS.items():
        text = ptmpl.format(vocab=VOCAB)
        preds, tok, lat, raw = run_set(FLASH, text, keys, pname)
        pred_sets = {k: set(v) for k, v in preds.items()}
        sc = masked_scores(pred_sets, y, m, keys)
        out[pname] = {"macro_f1": sc["macro_f1"], "tokens": tok,
                      "latency": lat, "preds": preds}
        print(f"{pname}: dev masked macro-F1 = {sc['macro_f1']:.4f} "
              f"(mean {lat['mean_s']:.1f}s/clip)", flush=True)
    best = max(out, key=lambda p: out[p]["macro_f1"])
    out["best"] = best
    with open(SCRATCH / "gemini_bench_dev.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"BEST PROMPT: {best}", flush=True)


def classifier_preds_on(keys, y_all, m_all, key_order):
    """E7 (saved probs) and E4 (fresh inference) predictions for given keys."""
    import torch
    from experiment import CNN4, DEVICE

    results = {}
    # E7 from saved test probs
    e7 = np.load(SCRATCH / "E7_test_probs.npy")
    e7_keys = list(np.load(SCRATCH / "panns_split01_test.npz",
                           allow_pickle=True)["keys"])
    e7_row = {str(k): i for i, k in enumerate(e7_keys)}
    with open(SCRATCH / "results.jsonl") as f:
        recs = {json.loads(l)["name"]: json.loads(l) for l in f}
    th7 = np.array(recs["E7_panns_head"]["tuned_thresholds"])
    results["E7_panns@0.5"] = {
        k: sorted(IDX_TO_NAME[c] for c in range(20)
                  if e7[e7_row[k], c] >= 0.5) for k in keys}
    results["E7_panns@tuned"] = {
        k: sorted(IDX_TO_NAME[c] for c in range(20)
                  if e7[e7_row[k], c] >= th7[c]) for k in keys}

    # E4 fresh inference from mel cache
    model = CNN4().to(DEVICE)
    model.load_state_dict(torch.load(
        SCRATCH / "ckpt_E4_cnn4_specaug_posweight.pt",
        map_location=DEVICE, weights_only=True))
    model.eval()
    cache = PROJECT / "data" / "openmic" / "mel_cache"
    mels = []
    for k in keys:
        mel = np.load(cache / f"{k}.npy")
        out = np.full((128, 431), -80.0, dtype=np.float32)
        w = min(mel.shape[1], 431)
        out[:, :w] = mel[:, :w]
        mels.append(out)
    x = (torch.from_numpy(np.stack(mels)).unsqueeze(1) + 40.0) / 40.0
    with torch.no_grad():
        p4 = torch.sigmoid(model(x.to(DEVICE))).cpu().numpy()
    results["E4_cnn@0.5"] = {
        k: sorted(IDX_TO_NAME[c] for c in range(20) if p4[i, c] >= 0.5)
        for i, k in enumerate(keys)}
    return results


def cmd_test():
    with open(SCRATCH / "gemini_bench_dev.json") as f:
        best = json.load(f)["best"]
    print(f"using prompt: {best}", flush=True)
    text = PROMPTS[best].format(vocab=VOCAB)

    keys, y, m = sample_clips("split01_test.csv", 100)
    out = {"prompt": best, "n_clips": len(keys), "keys": keys}

    for model, subset in [(FLASH, keys), (PRO, keys[:50])]:
        preds, tok, lat, raw = run_set(model, text, subset, model)
        pred_sets = {k: set(v) for k, v in preds.items()}
        sub_idx = [keys.index(k) for k in subset]
        sc = masked_scores(pred_sets, y[sub_idx], m[sub_idx], subset)
        out[model] = {"macro_f1": sc["macro_f1"],
                      "per_class": sc["per_class"], "tokens": tok,
                      "latency": lat, "preds": preds}
        print(f"{model}: masked macro-F1 = {sc['macro_f1']:.4f} on "
              f"{len(subset)} clips", flush=True)

    clf = classifier_preds_on(keys, y, m, keys)
    for name, preds in clf.items():
        sc = masked_scores({k: set(v) for k, v in preds.items()}, y, m, keys)
        out[name] = {"macro_f1": sc["macro_f1"], "per_class": sc["per_class"]}
        print(f"{name}: masked macro-F1 = {sc['macro_f1']:.4f} (same 100 "
              f"clips)", flush=True)

    with open(SCRATCH / "gemini_bench_test.json", "w") as f:
        json.dump(out, f, indent=2)
    print("GEMINI TEST BENCH DONE", flush=True)


def cmd_jamendo():
    with open(SCRATCH / "gemini_bench_dev.json") as f:
        best = json.load(f)["best"]
    text = PROMPTS[best].format(vocab=VOCAB)
    with open(SCRATCH / "jamendo" / "manifest.json") as f:
        manifest = json.load(f)
    dead = {"bass", "cymbals", "drums", "voice"}

    out = {}
    tot_tags = tot_hits = tot_extras = 0
    lats = []
    for tid, meta in manifest.items():
        mp3 = SCRATCH / "jamendo" / f"{tid}.mp3"
        resp, usage, dt = call_gemini(FLASH, text, mp3)
        lats.append(dt)
        pred = parse_instruments(resp) - dead
        truth = set(meta["classes"])
        out[tid] = {"tags": sorted(truth), "pred": sorted(pred),
                    "hits": sorted(pred & truth),
                    "extras": sorted(pred - truth), "latency_s": dt}
        tot_tags += len(truth)
        tot_hits += len(pred & truth)
        tot_extras += len(pred - truth)
        print(f"  {tid}: hits {len(pred & truth)}/{len(truth)} "
              f"extras {len(pred - truth)} ({dt:.0f}s)", flush=True)
        time.sleep(0.5)

    summary = {"tag_recall": tot_hits / tot_tags,
               "extras_per_track": tot_extras / len(manifest),
               "mean_latency_s": float(np.mean(lats))}
    print(f"GEMINI jamendo: recall {summary['tag_recall']:.3f}, "
          f"extras/track {summary['extras_per_track']:.2f}", flush=True)
    with open(SCRATCH / "gemini_bench_jamendo.json", "w") as f:
        json.dump({"summary": summary, "per_track": out}, f, indent=2)
    print("GEMINI JAMENDO BENCH DONE", flush=True)


if __name__ == "__main__":
    cmds = sys.argv[1:] or ["all"]
    if "all" in cmds:
        cmds = ["dev", "test", "jamendo"]
    for c in cmds:
        {"dev": cmd_dev, "test": cmd_test, "jamendo": cmd_jamendo}[c]()
