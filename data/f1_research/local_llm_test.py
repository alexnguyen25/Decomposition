"""Concrete quality/latency test: local open LLM (Ollama, OpenAI-compatible
endpoint) on the SAME grounded song-description task + cases as the Gemini test.
Validates the grounding contract and compares to the saved Gemini outputs.
"""

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
# reuse the three real pipeline-output cases + contract from the Gemini test
from gemini_feature import CASES  # noqa: E402
from gemini_bench import CLASS_MAP  # noqa: E402

OLLAMA = "http://localhost:11434/v1/chat/completions"
MODEL = "llama3.2:3b"
GENERIC_OK = {"vocals", "voice", "drums", "bass"}

SCHEMA = """{"blurb": "...", "genre": "...", "moods": ["..."],
"energy": "low|medium|high", "tempo_feel": "...", "era_production": "...",
"mentioned_instruments": ["exact names from JSON only"]}"""

PROMPT = """You write track breakdowns for a music app. Based ONLY on this
analysis JSON, write a description. Name ONLY instruments in the JSON's
"instruments" list (you may refer to vocals/drums/bass generically); never name
any other instrument. 2-3 sentence blurb. Respond ONLY with JSON of this shape:
""" + SCHEMA + "\n\nAnalysis JSON:\n<PAYLOAD>"


def call_ollama(text):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "temperature": 0.4,
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
    dt = time.time() - t0
    txt = d["choices"][0]["message"]["content"]
    try:
        return json.loads(txt), dt
    except json.JSONDecodeError:
        return {"_raw": txt[:300]}, dt


def validate(case, resp):
    allowed = {i["name"] for i in case["instruments"]} | GENERIC_OK
    allowed |= {k for k, v in case["presence"].items() if v}
    if not isinstance(resp, dict) or "blurb" not in resp:
        return False, "bad schema"
    mentioned = set(resp.get("mentioned_instruments", []))
    if not mentioned <= allowed:
        return False, f"hallucinated {sorted(mentioned - allowed)}"
    text = " ".join(str(resp.get(f, "")) for f in
                    ("blurb", "genre", "tempo_feel", "era_production")).lower()
    leaked = {n for n in CLASS_MAP
              if re.search(rf"\b{n.replace('_', ' ')}s?\b", text)
              } - allowed - GENERIC_OK
    return (not leaked), (f"leaked {sorted(leaked)}" if leaked else "ok")


def main():
    results = {}
    lats = []
    for name, case in CASES.items():
        payload = json.dumps(case, indent=2)
        resp, dt = call_ollama(PROMPT.replace("<PAYLOAD>", payload))
        lats.append(dt)
        ok, why = validate(case, resp)
        results[name] = {"ok": ok, "why": why, "latency_s": round(dt, 1),
                         "blurb": resp.get("blurb"), "genre": resp.get("genre")}
        print(f"{name}: {'PASS' if ok else 'FAIL-' + why} ({dt:.1f}s) "
              f"genre={resp.get('genre')!r}", flush=True)
        print(f"   {resp.get('blurb')}", flush=True)
    results["_summary"] = {"model": MODEL, "mean_latency_s": sum(lats) /
                           len(lats), "passes": sum(r["ok"] for k, r in
                           results.items() if k != "_summary")}
    with open(SCRATCH / "local_llm_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nLOCAL LLM: {results['_summary']['passes']}/{len(CASES)} pass, "
          f"mean {results['_summary']['mean_latency_s']:.1f}s/call", flush=True)
    print("LOCAL LLM TEST DONE", flush=True)


if __name__ == "__main__":
    main()
