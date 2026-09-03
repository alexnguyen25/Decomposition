"""One-off: backfill `timeline` into examples/manifest.json.

The bundled examples were precomputed before the pipeline emitted
time-resolved data (per-chunk instruments + stem activity), which the chat
agent's tools need. This recomputes just that part from the cached stem MP3s
— no Demucs, no LLM — and patches the manifest in place.

Run:  cd reference-app/backend && ../../../.venv/bin/python make_timelines.py
(or the repo venv python; anything with the backend deps works)
"""

import json

import librosa

import settings
from models_util import classify_other_stem
from pipeline import SR_NATIVE, STEM_ORDER, _activity_envelope


def timeline_for(example_dir) -> dict:
    stems = {}
    for name in STEM_ORDER:
        y, _ = librosa.load(example_dir / f"{name}.mp3", sr=SR_NATIVE,
                            mono=True)
        stems[name] = y
    other_16k = librosa.resample(stems["other"], orig_sr=SR_NATIVE,
                                 target_sr=16000)
    _, chunk_instruments = classify_other_stem(other_16k)
    return {
        "chunk_s": 10,
        "instruments": chunk_instruments,
        "stem_activity": {"hop_s": 1.0,
                          **{n: _activity_envelope(stems[n])
                             for n in STEM_ORDER}},
    }


def main():
    manifest_path = settings.EXAMPLES_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for ex in manifest:
        if "timeline" in ex["result"]:
            print(f"{ex['id']}: already has timeline, skipping")
            continue
        print(f"{ex['id']}: computing timeline…")
        ex["result"]["timeline"] = timeline_for(settings.EXAMPLES_DIR / ex["id"])
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print("manifest updated")


if __name__ == "__main__":
    main()
