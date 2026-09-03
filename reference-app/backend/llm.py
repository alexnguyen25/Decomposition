"""Swappable LLM layer: pipeline JSON -> grounded natural-language description.

Design (from the research, MVP doc §7b/§11c):
- ONE client speaking the OpenAI-compatible chat API. Ollama, Groq, Cerebras,
  Gemini, OpenRouter all expose it, so "which LLM" is just env config:
  local dev -> Ollama (localhost:11434), public prod -> a free hosted open
  model (no card on the key -> no bill possible).
- The GROUNDING CONTRACT is enforced in code, not hoped for: the model may
  only name instruments the classifier found (plus stems from `presence`).
  Violations -> one retry with feedback -> deterministic template fallback.
  A missing/broken LLM never breaks the app; it just degrades gracefully.
"""

import json
import re
import urllib.error
import urllib.request

# word-boundary matching matters: a naive substring check flags the word
# "organic" as the instrument "organ" (found the hard way — see research doc).
from models_util import CLASS_MAP
from settings import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT_S

GENERIC_OK = {"vocals", "voice", "drums", "bass"}

_PROMPT = """You write track breakdowns for a music-analysis app. Based ONLY
on this analysis JSON, write a short description for the listener.

Analysis JSON:
<PAYLOAD>

Rules:
- Name ONLY instruments in the JSON's "instruments" list. You may refer to
  vocals/drums/bass generically when "presence" says they exist. NEVER name
  any other instrument.
- Confidence: >=0.90 state plainly; 0.70-0.89 "clear"; 0.50-0.69 hedge
  ("hints of"). Never show numbers.
- Mention the tempo feel and key naturally if given. No markdown.
- Respond ONLY with JSON: {"blurb": "2-3 sentences", "genre": "short label",
  "moods": ["up to 3"], "energy": "low|medium|high",
  "mentioned_instruments": ["exact names from the JSON"]}
"""


def _chat(prompt: str) -> dict:
    """Minimal OpenAI-compatible chat call. No SDK: one fewer dependency,
    and the request shape is worth understanding anyway."""
    body = json.dumps({
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "response_format": {"type": "json_object"},
    }).encode()
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    req = urllib.request.Request(f"{LLM_BASE_URL}/chat/completions",
                                 data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_S) as r:
        data = json.load(r)
    return json.loads(data["choices"][0]["message"]["content"])


def _validate(result: dict, resp: dict) -> tuple[bool, str]:
    """The grounding contract, as executable code."""
    allowed = {i["name"] for i in result["instruments"]} | GENERIC_OK
    allowed |= {k for k, v in result.get("presence", {}).items() if v}
    if not isinstance(resp, dict) or not resp.get("blurb"):
        return False, "bad schema"
    mentioned = set(resp.get("mentioned_instruments", []))
    if not mentioned <= allowed:
        return False, f"named instruments not in the analysis: {sorted(mentioned - allowed)}"
    text = " ".join(str(resp.get(f, "")) for f in ("blurb", "genre")).lower()
    leaked = {n for n in CLASS_MAP
              if re.search(rf"\b{n.replace('_', ' ')}s?\b", text)
              } - allowed - GENERIC_OK
    if leaked:
        return False, f"mentioned {sorted(leaked)} in prose"
    return True, "ok"


def _template_fallback(result: dict) -> dict:
    """Deterministic, always-grounded description when no LLM is reachable."""
    names = [i["name"].replace("_", " ") for i in result["instruments"]]
    lead = (f"featuring {', '.join(names[:-1])} and {names[-1]}" if len(names) > 1
            else f"featuring {names[0]}" if names else "with a minimal arrangement")
    bpm = result.get("bpm")
    tempo = ("an up-tempo" if bpm and bpm >= 120 else
             "a mid-tempo" if bpm and bpm >= 90 else "a laid-back")
    key = f" in {result['key']}" if result.get("key") else ""
    return {"blurb": f"{tempo.capitalize()} track{key}, {lead}.",
            "genre": None, "moods": [], "energy": None,
            "mentioned_instruments": [i["name"] for i in result["instruments"]],
            "source": "template"}


def describe(result: dict) -> dict:
    """Public entry point: result dict -> description dict (never raises)."""
    payload = json.dumps(
        {k: result[k] for k in ("bpm", "key", "presence", "instruments")},
        indent=2)
    prompt = _PROMPT.replace("<PAYLOAD>", payload)
    feedback = ""
    for _ in range(2):                               # initial try + one retry
        try:
            resp = _chat(prompt + feedback)
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError, KeyError):
            return _template_fallback(result)        # LLM down -> degrade
        ok, why = _validate(result, resp)
        if ok:
            resp["source"] = LLM_MODEL
            return resp
        feedback = (f"\n\nYour previous answer violated the rules ({why}). "
                    "Fix it and respond again with ONLY the JSON.")
    return _template_fallback(result)                # contract never met
