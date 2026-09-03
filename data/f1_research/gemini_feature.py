"""In-app Gemini feature test: pipeline JSON -> grounded natural-language
song analysis. Tests text-only grounding on flash-latest and
flash-lite-latest, and validates the no-hallucination contract.
"""

import json
import sys
import time
from pathlib import Path

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
from gemini_bench import CLASS_MAP, call_gemini, FLASH  # noqa: E402

LITE = "gemini-flash-lite-latest"

FEATURE_PROMPT = """You are the analysis writer for a music-tech app. The app
separated a song into stems and analyzed it. Write a short, engaging
"track breakdown" for the listener based ONLY on this analysis JSON:

{payload}

Rules:
- Mention ONLY instruments listed in the JSON. Never add instruments.
- Confidence phrasing: >=0.90 state plainly; 0.70-0.89 "clear"; 0.50-0.69
  hedge ("likely", "hints of"). Do not show numbers.
- Use bpm/key naturally (e.g. tempo feel). presence tells you whether
  vocals/drums/bass exist in the song - mention them accordingly.
- 2-4 sentences, friendly but not gushing. No markdown.

Respond with JSON:
{{"summary": "...", "mentioned_instruments": ["exact-names-from-json", ...]}}
"""

# three real pipeline outputs (test.mp3 measured today; two synthesized from
# real Jamendo E7 predictions)
CASES = {
    "rock_track": {
        "bpm": 132.5, "key": "C# minor",
        "presence": {"vocals": True, "drums": True, "bass": True},
        "instruments": [{"name": "guitar", "confidence": 0.984}],
    },
    "orchestral": {
        "bpm": 92.0, "key": "D major",
        "presence": {"vocals": False, "drums": False, "bass": False},
        "instruments": [{"name": "violin", "confidence": 0.93},
                        {"name": "cello", "confidence": 0.81},
                        {"name": "trumpet", "confidence": 0.55}],
    },
    "electronic": {
        "bpm": 124.0, "key": "F minor",
        "presence": {"vocals": True, "drums": True, "bass": True},
        "instruments": [{"name": "synthesizer", "confidence": 0.97},
                        {"name": "piano", "confidence": 0.62}],
    },
}


def validate(case, resp):
    """The grounding contract: mentioned instruments must be subset of input."""
    allowed = {i["name"] for i in case["instruments"]}
    # presence-derived stems are legitimate mentions too
    allowed |= {k for k, v in case["presence"].items() if v}
    allowed |= {"vocals", "voice"} if case["presence"].get("vocals") else set()
    if not isinstance(resp, dict) or "summary" not in resp:
        return False, "bad schema"
    mentioned = set(resp.get("mentioned_instruments", []))
    if not mentioned <= allowed:
        return False, f"hallucinated: {mentioned - allowed}"
    # crude text check: no other class name may appear in the summary
    text = resp["summary"].lower()
    leaked = {n for n in CLASS_MAP if n.replace("_", " ") in text} - allowed
    # 'voice'/'drums'/'bass' may legitimately appear via presence
    leaked -= {"voice", "drums", "bass", "cymbals"}
    if leaked:
        return False, f"leaked into text: {leaked}"
    return True, "ok"


def main():
    results = {}
    for model in (FLASH, LITE):
        results[model] = {}
        for name, case in CASES.items():
            payload = json.dumps(case, indent=2)
            t0 = time.time()
            resp, usage, dt = call_gemini(model,
                                          FEATURE_PROMPT.format(payload=payload))
            ok, why = validate(case, resp)
            results[model][name] = {
                "ok": ok, "why": why, "latency_s": round(dt, 2),
                "tokens": {k: usage.get(k, 0) for k in
                           ("promptTokenCount", "candidatesTokenCount",
                            "thoughtsTokenCount")},
                "summary": resp.get("summary") if isinstance(resp, dict)
                           else str(resp)[:200],
            }
            print(f"{model} / {name}: {'PASS' if ok else 'FAIL - ' + why} "
                  f"({dt:.1f}s)", flush=True)
            print(f"   {results[model][name]['summary']}", flush=True)
            time.sleep(0.3)
    with open(SCRATCH / "gemini_feature_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("FEATURE TEST DONE", flush=True)


if __name__ == "__main__":
    main()
