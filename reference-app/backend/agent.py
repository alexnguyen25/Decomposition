"""Grounded chat agent: tool-calling loop over ONE track's analysis JSON.

Design (extends llm.py's philosophy from one-shot to conversational):
- The model NEVER sees raw audio and is never asked to "know" music. Every
  fact flows through three tools that read the pipeline's result dict, so
  answers are grounded by construction — the model's job is phrasing, not
  recall.
- Same swappable OpenAI-compatible layer as llm.py (Ollama dev, hosted-open
  prod). Tool calling is part of that API surface, so no new dependency.
- The grounding contract is enforced in code AFTER generation too (belt and
  suspenders): instrument names outside the analysis, wrong BPM/key claims
  -> one retry with feedback -> honest refusal. A chat agent that sometimes
  says "I can't verify that" beats one that sometimes lies.
"""

import json
import re
import urllib.error
import urllib.request

from models_util import CLASS_MAP
from settings import (CHAT_MAX_ROUNDS, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
                      LLM_TIMEOUT_S)

# Generic stem words -> the presence flag that makes them true for THIS track.
# Previously an unconditional allowlist, which let the agent assert "there is a
# bass line" about a track whose bass stem is silent (found by the eval harness).
GENERIC_STEM_ALIASES = {"vocals": "vocals", "voice": "vocals",
                        "drums": "drums", "percussion": "drums",
                        "bass": "bass"}

# ── tools: each one reads the result dict, returns JSON-safe data ───────────

def _fmt_t(s: float) -> str:
    return f"{int(s // 60)}:{int(s % 60):02d}"


def _tool_get_bpm_key(result: dict) -> dict:
    dur = result.get("duration_s")
    return {"bpm": result.get("bpm"), "key": result.get("key"),
            # pre-formatted because models botch seconds->m:ss arithmetic
            "duration": _fmt_t(dur) if dur else None,
            "note": "key may be null when the detector was unavailable"}


def _num_or_none(v):
    """Small models send junk arguments ('null', 'None', '1:30', '') —
    coerce charitably, treat garbage as absent rather than crashing."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    if s in ("", "null", "none", "nan"):
        return None
    if re.fullmatch(r"(\d+):([0-5]?\d)", s):         # "1:30" -> 90.0
        m, sec = s.split(":")
        return float(m) * 60 + float(sec)
    try:
        return float(s)
    except ValueError:
        return None


def _tool_get_instruments(result: dict, start_s=None, end_s=None) -> dict:
    """Song-level instrument list, or per-chunk detections within a window."""
    start_s, end_s = _num_or_none(start_s), _num_or_none(end_s)
    if (start_s is not None and end_s is not None and end_s <= start_s):
        start_s = end_s = None          # degenerate window -> whole song
    if start_s is None and end_s is None:
        return {"scope": "whole song",
                "instruments": result.get("instruments", []),
                "note": "detected in the non-vocal/drum/bass ('other') stem; "
                        "confidence is the classifier's probability"}
    tl = (result.get("timeline") or {}).get("instruments")
    if not tl:
        return {"error": "no time-resolved data for this track; "
                         "only whole-song instruments are available",
                "instruments": result.get("instruments", [])}
    start_s = float(start_s or 0)
    end_s = float(end_s if end_s is not None else result.get("duration_s", 1e9))
    chunk_s = result["timeline"].get("chunk_s", 10)
    merged: dict[str, float] = {}
    for entry in tl:
        t0 = entry["t"]
        if t0 + chunk_s <= start_s or t0 >= end_s:
            continue
        for name, p in entry["top"].items():
            merged[name] = max(merged.get(name, 0.0), p)
    return {"scope": f"{_fmt_t(start_s)}-{_fmt_t(min(end_s, result.get('duration_s', end_s)))}",
            "instruments": [{"name": n, "confidence": round(p, 3)}
                            for n, p in sorted(merged.items(),
                                               key=lambda kv: -kv[1])],
            "note": "confidences are per-10s-chunk maxima within the window"}


def _tool_get_stem_activity(result: dict, stem: str) -> dict:
    """Where a stem is audible: active spans + overall fraction."""
    stem = str(stem).lower().strip()
    tl = (result.get("timeline") or {}).get("stem_activity") or {}
    if stem not in ("vocals", "drums", "bass", "other"):
        return {"error": f"unknown stem '{stem}' — use vocals/drums/bass/other"}
    # presence gate first: the envelope is normalized to the stem's OWN peak,
    # so a near-silent stem would show residual Demucs bleed as "activity"
    # (found by the eval harness — the model faithfully relayed a lying tool).
    if result.get("presence", {}).get(stem) is False:
        return {"stem": stem, "present": False,
                "note": f"this track has no meaningful {stem} — the stem is "
                        "essentially silent"}
    env = tl.get(stem)
    if env is None:
        presence = result.get("presence", {})
        return {"stem": stem, "error": "no time-resolved data for this track",
                "present_overall": presence.get(stem)}
    hop = tl.get("hop_s", 1.0)
    thresh = 0.15                       # of the stem's own peak loudness
    spans, run_start = [], None
    for i, v in enumerate(env + [0.0]):
        if v >= thresh and run_start is None:
            run_start = i * hop
        elif v < thresh and run_start is not None:
            if i * hop - run_start >= 2:            # ignore <2s blips
                spans.append([run_start, i * hop])
            run_start = None
    active_frac = sum(1 for v in env if v >= thresh) / max(1, len(env))
    return {"stem": stem,
            "active_fraction": round(active_frac, 2),
            "active_spans": [[_fmt_t(a), _fmt_t(b)] for a, b in spans[:25]],
            "note": "spans where the stem is above 15% of its peak loudness"}


TOOL_IMPLS = {
    "get_bpm_key": _tool_get_bpm_key,
    "get_instruments": _tool_get_instruments,
    "get_stem_activity": _tool_get_stem_activity,
}

TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": "get_bpm_key",
        "description": "Tempo (BPM), musical key and duration of the track.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_instruments",
        "description": "Instruments detected by the classifier. Without "
                       "arguments: the whole song. With start_s/end_s "
                       "(seconds): what was detected inside that window.",
        # ["number", "null"], not "number": models signal "whole song" by
        # sending null for both bounds, and some providers (Groq) validate
        # tool arguments against this schema server-side and reject the call
        # before _num_or_none ever gets to coerce it.
        "parameters": {"type": "object", "properties": {
            "start_s": {"type": ["number", "null"],
                        "description": "window start in seconds; null for whole song"},
            "end_s": {"type": ["number", "null"],
                      "description": "window end in seconds; null for whole song"}}}}},
    {"type": "function", "function": {
        "name": "get_stem_activity",
        "description": "When a stem (vocals, drums, bass or other) is "
                       "audible: active time spans and overall fraction.",
        "parameters": {"type": "object", "properties": {
            "stem": {"type": "string",
                     "enum": ["vocals", "drums", "bass", "other"]}},
            "required": ["stem"]}}},
]

_SYSTEM = """You are the analysis console of a music-decomposition app,
answering questions about ONE analyzed track.

Hard rules:
- Every musical fact you state MUST come from a tool result in this
  conversation. Call tools first, answer after.
- The analysis covers: stems (vocals/drums/bass/other), instruments in the
  'other' stem, BPM, key, duration, and when each stem is audible. It does
  NOT cover lyrics, song title, artist, album, year, genre history or
  influences — if asked about those, your ENTIRE answer must be that the
  analysis doesn't include that. Never invent lyrical content or meaning;
  never guess an artist or title.
- Never name a specific instrument unless a tool returned it. If a user asks
  about an instrument the tools don't show, say it wasn't detected (it may
  still be there — the classifier isn't perfect — but you can only speak to
  what was detected).
- Confidence wording: >=0.9 state plainly; 0.7-0.89 "clearly"; 0.5-0.69
  "probably"; below 0.5 "faint hints, not confirmed". Don't show raw numbers
  unless asked.
- Write times as m:ss. Keep answers to 1-4 sentences, plain text, no markdown.
"""


# ── OpenAI-compatible chat with tools (kept SDK-free like llm.py) ────────────

def _chat_raw(messages: list[dict], use_tools: bool = True) -> dict:
    body: dict = {"model": LLM_MODEL, "messages": messages, "temperature": 0.2}
    if use_tools:
        body["tools"] = TOOLS_SPEC
    req = urllib.request.Request(
        f"{LLM_BASE_URL}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {LLM_API_KEY}"}
                    if LLM_API_KEY else {})})
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_S) as r:
        return json.load(r)["choices"][0]["message"]


# ── the post-hoc grounding check (code, not hope) ────────────────────────────

_NEGATORS = ("no ", "not ", "n't ", "n't.", "without", "lack", "absent",
             "wasn't", "isn't", "aren't", "don't", "doesn't", "didn't",
             "couldn't", "can't", "cannot", "never", "none", "neither",
             "instead of", "rather than")
# "detect" was once in this list for "didn't detect X", but it also made
# "there is a detected bass line" read as a denial. The explicit negators
# above already cover the intended phrasings.


def _is_negated(text: str, match_start: int) -> bool:
    """True if a negation word appears shortly before the instrument name —
    denying an instrument is grounded speech, asserting it is not."""
    window = text[max(0, match_start - 60):match_start].lower()
    return any(n in window for n in _NEGATORS)


def check_grounding(result: dict, reply: str,
                    tool_outputs: list[dict]) -> tuple[bool, list[str]]:
    """Verify every checkable claim in `reply` against the analysis.

    Checkable claims: instrument names (vs. the 20-class map), BPM numbers,
    key statements. Returns (ok, list of violations). Used by the API (to
    retry/refuse) and by the eval harness (to measure hallucination rate).
    """
    violations = []
    text = reply.lower()

    # 1. instruments: allowed = analysis + presence stems + anything a tool
    #    actually returned this conversation (per-chunk names may sit below
    #    the song-level threshold; if the tool said it, it's grounded).
    presence = result.get("presence", {})
    allowed = {i["name"] for i in result.get("instruments", [])}
    allowed |= {word for word, stem in GENERIC_STEM_ALIASES.items()
                if presence.get(stem) is not False}
    allowed |= {k for k, v in presence.items() if v}
    for out in tool_outputs:
        for inst in out.get("instruments", []) or []:
            allowed.add(inst["name"])
    for name in set(CLASS_MAP) | set(GENERIC_STEM_ALIASES):
        pretty = name.replace("_", " ")
        if name in allowed:
            continue
        for m in re.finditer(rf"\b{re.escape(pretty)}s?\b", text):
            if not _is_negated(text, m.start()):
                violations.append(f"asserted undetected instrument: {pretty}")
                break

    # 2. BPM: any number the reply attaches to "bpm" must match the analysis.
    bpm = result.get("bpm")
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*bpm", text):
        if bpm is None or abs(float(m.group(1)) - float(bpm)) > 1.5:
            violations.append(f"claimed {m.group(1)} BPM (analysis: {bpm})")

    # 3. duration: m:ss values the reply presents AS the length must match
    #    (plain timestamps like "vocals enter at 1:12" are not length claims).
    dur = result.get("duration_s")
    if dur:
        for m in re.finditer(
                r"(?:duration|length|long|lasts|runs)\D{0,20}?(\d+):([0-5]\d)"
                r"|(\d+):([0-5]\d)\s*(?:long|in length|total)", text):
            mm, ss = (m.group(1), m.group(2)) if m.group(1) else \
                     (m.group(3), m.group(4))
            claimed = int(mm) * 60 + int(ss)
            if abs(claimed - dur) > 3:
                violations.append(f"claimed duration {mm}:{ss} "
                                  f"(analysis: {_fmt_t(dur)})")

    # 4. key: "in X major/minor" style claims must match.
    key = (result.get("key") or "").lower()
    for m in re.finditer(r"\b([a-g](?:\s?(?:sharp|flat)|[#b])?)\s+(major|minor)\b",
                         text):
        claimed = f"{m.group(1).strip()} {m.group(2)}".replace("#", " sharp")
        if claimed != key.replace("#", " sharp"):
            if not _is_negated(text, m.start()):
                violations.append(f"claimed key '{m.group(0)}' "
                                  f"(analysis: {key or 'unknown'})")

    return (not violations, violations)


# ── the agent loop ───────────────────────────────────────────────────────────

REFUSAL = ("I couldn't give a reliable answer to that from the analysis — "
           "try asking about the instruments, stems, tempo or key.")


def chat(result: dict, messages: list[dict]) -> dict:
    """messages: [{'role': 'user'|'assistant', 'content': str}, ...]
    Returns {'reply', 'grounded', 'trace': [{'tool', 'args'}...]}.
    Never raises: LLM failures degrade to an honest refusal."""
    msgs = [{"role": "system", "content": _SYSTEM}] + [
        {"role": m["role"], "content": str(m.get("content", ""))[:2000]}
        for m in messages if m.get("role") in ("user", "assistant")]
    trace: list[dict] = []
    tool_outputs: list[dict] = []
    retried = False

    for _ in range(CHAT_MAX_ROUNDS):
        try:
            msg = _chat_raw(msgs)
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError, KeyError):
            return {"reply": "The chat model isn't reachable right now — "
                             "the analysis above is still all yours.",
                    "grounded": True, "trace": trace}

        calls = msg.get("tool_calls") or []
        if calls:
            msgs.append({"role": "assistant", "content": msg.get("content"),
                         "tool_calls": calls})
            for call in calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                impl = TOOL_IMPLS.get(name)
                try:
                    out = (impl(result, **args) if impl
                           else {"error": f"unknown tool {name}"})
                except (TypeError, ValueError) as e:  # bad/extra args
                    out = {"error": f"bad arguments for {name}: {e}"}
                if isinstance(out, dict) and not out.get("error"):
                    tool_outputs.append(out)
                trace.append({"tool": name, "args": args})
                msgs.append({"role": "tool",
                             "tool_call_id": call.get("id", name),
                             "content": json.dumps(out)})
            continue

        reply = (msg.get("content") or "").strip()
        # reasoning models (qwen3 etc.) may inline their thinking; drop it
        reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S).strip()
        if not reply:
            continue
        ok, why = check_grounding(result, reply, tool_outputs)
        if ok:
            return {"reply": reply, "grounded": True, "trace": trace}
        if not retried:                              # one shot at self-repair
            retried = True
            msgs.append({"role": "assistant", "content": reply})
            msgs.append({"role": "user", "content":
                         "Your answer contained claims not backed by the "
                         f"analysis ({'; '.join(why)}). Answer again using "
                         "only tool results; if the analysis can't answer, "
                         "say so."})
            continue
        return {"reply": REFUSAL, "grounded": False, "trace": trace,
                "violations": why}

    return {"reply": REFUSAL, "grounded": False, "trace": trace}
