"""Richer Gemini feature: full song DESCRIPTION (genre/mood/energy/tempo feel/
era/blurb), two grounding variants tested on real Jamendo tracks with real E7
pipeline output:

  A text_only  - Gemini sees only the pipeline JSON
  B multimodal - Gemini also hears a 60s mix excerpt

Contract: instrument names in output must be subset of pipeline instruments
+ generic stems (vocals/drums/bass). Tests flash-lite-latest and flash-latest.
"""

import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
sys.path.insert(0, str(SCRATCH / "pylibs"))
from gemini_bench import CLASS_MAP, KEY, CTX  # noqa: E402

import librosa  # noqa: E402
import torch  # noqa: E402
import urllib.request  # noqa: E402

FLASH = "gemini-flash-latest"
LITE = "gemini-flash-lite-latest"
TRACKS = ["867662", "783235", "6253"]
DEAD = {"bass", "cymbals", "drums", "voice"}
GENERIC_OK = {"vocals", "voice", "drums", "bass"}

SCHEMA_DOC = """{
  "blurb": "2-3 listener-facing sentences",
  "genre": "one short genre label",
  "moods": ["up to 3 adjectives"],
  "energy": "low | medium | high",
  "tempo_feel": "short phrase about the tempo/groove",
  "era_production": "short phrase about production style/era",
  "mentioned_instruments": ["exact names from the analysis JSON only"]
}"""

RULES = """Rules:
- In all text fields, name ONLY instruments from the analysis JSON's
  "instruments" list. You may also refer to vocals/drums/bass generically.
  NEVER name any other specific instrument, even if you think you hear one -
  describe such content by texture/role instead (e.g. "a bright lead line").
- Confidence bands: >=0.90 state plainly; 0.70-0.89 "clear"; 0.50-0.69 hedge.
- Use bpm and key naturally. No markdown, no numbers from the JSON in text.
- Respond with JSON exactly in this shape:
""" + SCHEMA_DOC

PROMPT_TEXT_ONLY = """You write track breakdowns for a music-analysis app.
Based ONLY on this analysis JSON (you cannot hear the song), write a
description. Be honest about uncertainty - genre/mood are educated guesses
from instrumentation and tempo.

Analysis JSON:
<PAYLOAD>

""" + RULES

PROMPT_MULTIMODAL = """You write track breakdowns for a music-analysis app.
You are given (1) the app's analysis JSON and (2) a 60-second excerpt of the
song. LISTEN to the excerpt for genre, mood, energy, tempo feel and
production style. For instruments, trust the analysis JSON.

Analysis JSON:
<PAYLOAD>

""" + RULES


def call(model, text, wav_bytes=None, retries=4):
    parts = [{"text": text}]
    if wav_bytes is not None:
        parts.append({"inline_data": {
            "mime_type": "audio/wav",
            "data": base64.b64encode(wav_bytes).decode()}})
    body = json.dumps({
        "contents": [{"parts": parts}],
        "generationConfig": {"response_mime_type": "application/json"},
    }).encode()
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={KEY}")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    for attempt in range(retries):
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=180, context=CTX) as r:
                d = json.load(r)
            txt = d["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(txt), d.get("usageMetadata", {}), time.time() - t0
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))


def build_pipeline_json(tid):
    """Real pipeline output: E7 on saved other stem + bpm/key from the mix."""
    from panns_inference.models import Cnn14
    from panns_head import MLPHead

    global _CNN14, _HEAD
    if "_CNN14" not in globals():
        _CNN14 = Cnn14(sample_rate=32000, window_size=1024, hop_size=320,
                       mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        _CNN14.load_state_dict(torch.load(SCRATCH / "Cnn14_mAP=0.431.pth",
                                          map_location="cpu")["model"])
        _CNN14.eval()
        _HEAD = MLPHead()
        _HEAD.load_state_dict(torch.load(SCRATCH / "ckpt_E7_panns_head.pt",
                                         map_location="cpu",
                                         weights_only=True))
        _HEAD.eval()

    wave = np.load(SCRATCH / "jamendo" / f"{tid}_other_32k.npy").astype(
        np.float32)
    win = 320000
    chunks = [np.pad(wave[s:s + win], (0, max(0, win - len(wave[s:s + win]))))
              for s in range(0, max(1, len(wave) - 96000), win)]
    with torch.no_grad():
        emb = _CNN14(torch.from_numpy(np.stack(chunks)))["embedding"]
        probs = torch.sigmoid(_HEAD(emb)).numpy()
    song = np.sort(probs, axis=0)[-3:].mean(0)
    name_of = {i: n for n, i in CLASS_MAP.items()}
    instruments = [{"name": name_of[c], "confidence": round(float(song[c]), 3)}
                   for c in np.argsort(-song)
                   if name_of[c] not in DEAD and song[c] >= 0.5]

    mix, _ = librosa.load(SCRATCH / "jamendo" / f"{tid}.mp3", sr=22050,
                          mono=True)
    tempo, _ = librosa.beat.beat_track(y=mix, sr=22050)
    bpm = round(float(np.atleast_1d(tempo)[0]), 1)
    try:
        import essentia.standard as es
        k, scale, _ = es.KeyExtractor()(mix.astype(np.float32))
        key_str = f"{k} {scale}"
    except Exception:  # noqa: BLE001
        key_str = "unknown"
    return {"bpm": bpm, "key": key_str, "instruments": instruments}, mix


def excerpt_wav_bytes(mix_22k):
    mid = len(mix_22k) // 2
    seg = mix_22k[max(0, mid - 30 * 22050): mid + 30 * 22050]
    buf = io.BytesIO()
    sf.write(buf, seg, 22050, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def validate(pipeline_json, resp):
    allowed = ({i["name"] for i in pipeline_json["instruments"]}
               | GENERIC_OK)
    if not isinstance(resp, dict) or "blurb" not in resp:
        return False, "bad schema"
    mentioned = set(resp.get("mentioned_instruments", []))
    if not mentioned <= allowed:
        return False, f"hallucinated: {sorted(mentioned - allowed)}"
    import re
    text = " ".join(str(resp.get(f, "")) for f in
                    ("blurb", "genre", "tempo_feel", "era_production")).lower()
    leaked = {n for n in CLASS_MAP
              if re.search(rf"\b{n.replace('_', ' ')}s?\b", text)
              } - allowed - GENERIC_OK
    if leaked:
        return False, f"leaked: {sorted(leaked)}"
    return True, "ok"


def main():
    results = {}
    for tid in TRACKS:
        pj, mix = build_pipeline_json(tid)
        payload = json.dumps(pj, indent=2)
        wav = excerpt_wav_bytes(mix)
        with open(SCRATCH / "jamendo" / "manifest.json") as f:
            tags = json.load(f)[tid]["classes"]
        results[tid] = {"pipeline_json": pj, "jamendo_tags": tags,
                        "runs": {}}
        print(f"\n== track {tid} | E7 instruments: "
              f"{[i['name'] for i in pj['instruments']]} | tags: {tags}",
              flush=True)
        for model in (LITE, FLASH):
            for variant, (prompt, audio) in {
                "text_only": (PROMPT_TEXT_ONLY, None),
                "multimodal": (PROMPT_MULTIMODAL, wav),
            }.items():
                resp, usage, dt = call(model, prompt.replace("<PAYLOAD>", payload),
                                       audio)
                ok, why = validate(pj, resp)
                results[tid]["runs"][f"{model}/{variant}"] = {
                    "ok": ok, "why": why, "latency_s": round(dt, 2),
                    "tokens": {k: usage.get(k, 0) for k in
                               ("promptTokenCount", "candidatesTokenCount",
                                "thoughtsTokenCount")},
                    "response": resp,
                }
                print(f"  {model}/{variant}: "
                      f"{'PASS' if ok else 'FAIL-' + why} ({dt:.1f}s) "
                      f"genre={resp.get('genre')!r} "
                      f"energy={resp.get('energy')!r}", flush=True)
                time.sleep(0.3)
    with open(SCRATCH / "gemini_description_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDESCRIPTION TEST DONE", flush=True)


if __name__ == "__main__":
    main()
