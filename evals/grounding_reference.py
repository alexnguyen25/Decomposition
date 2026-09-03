"""Reference implementation of the grounding contract, in Python.

This is deliberately a SECOND implementation. The shipped agent enforces the
contract in TypeScript (reference-app/frontend/lib/agent/grounding.ts); this
one re-checks the replies that agent produces. Grading a validator with itself
proves nothing — if the port drifts, this is what notices.

Kept in sync by hand with the TS version; evals/run_evals_http.py is the test
that says whether they still agree.
"""

import json
import re
from pathlib import Path

_CLASS_MAP_PATH = (Path(__file__).resolve().parents[1]
                   / "src" / "classification" / "assets" / "class-map.json")
CLASS_MAP = json.loads(_CLASS_MAP_PATH.read_text())

# Each generic word maps to the presence flag that decides whether it is
# actually true for THIS track. An unconditional allowlist let the agent assert
# "there is a bass line" about a track whose bass stem is silent.
GENERIC_STEM_ALIASES = {"vocals": "vocals", "voice": "vocals",
                        "drums": "drums", "percussion": "drums",
                        "bass": "bass"}


def _fmt_t(s: float) -> str:
    return f"{int(s // 60)}:{int(s % 60):02d}"


def _tool_get_bpm_key(result: dict) -> dict:
    dur = result.get("duration_s")
    return {"bpm": result.get("bpm"), "key": result.get("key"),
            # pre-formatted because models botch seconds->m:ss arithmetic
            "duration": _fmt_t(dur) if dur else None,
            "note": "key may be null when the detector was unavailable"}



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
    allowed |= {k for k, v in result.get("presence", {}).items() if v}
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


