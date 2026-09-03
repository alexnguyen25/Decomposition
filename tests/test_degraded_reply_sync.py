"""The eval harness must recognise every canned "we never reached the model"
reply the shipped agent can return.

This pairing is the one that already broke once: the app returned a fixed
apology for an outage, the grounding checker saw text that asserted nothing
and called it clean, and the harness reported 0% hallucination for a run in
which all 30 calls had failed. Adding a new degraded reply to the TypeScript
agent without adding it to DEGRADED_REPLIES would silently do it again, and
nothing else in the suite crosses the language boundary to catch it.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT_TS = ROOT / "reference-app" / "frontend" / "lib" / "agent" / "index.ts"

sys.path.insert(0, str(ROOT / "evals"))


def _harness_prefixes() -> set[str]:
    from run_evals_http import DEGRADED_REPLIES

    return {prefix for prefix, _ in DEGRADED_REPLIES}


def _agent_constants() -> dict[str, str]:
    """Top-level `const NAME = "..."` string literals, joined across the
    `"a" + "b"` continuations the agent uses to stay inside the line length."""
    source = AGENT_TS.read_text()
    found = {}
    for match in re.finditer(
        r'^const ([A-Z][A-Z0-9_]*) =\s*((?:\s*"(?:[^"\\]|\\.)*"\s*\+?)+);',
        source,
        re.MULTILINE,
    ):
        parts = re.findall(r'"((?:[^"\\]|\\.)*)"', match.group(2))
        found[match.group(1)] = "".join(parts)
    return found


def test_degraded_replies_are_all_recognised():
    constants = _agent_constants()
    degraded = {
        name: value
        for name, value in constants.items()
        if name in {"UNREACHABLE", "QUOTA_EXHAUSTED"}
    }
    assert degraded, f"found no degraded-reply constants in {AGENT_TS.name}"

    prefixes = _harness_prefixes()
    for name, text in degraded.items():
        assert any(text.startswith(prefix) for prefix in prefixes), (
            f"{name} in {AGENT_TS.name} starts with {text[:60]!r}, which no "
            f"prefix in DEGRADED_REPLIES matches. The harness would score this "
            f"outage as a valid answer — add it to DEGRADED_REPLIES."
        )


def test_refusal_is_not_treated_as_an_outage():
    """A refusal is a real answer the contract produced on purpose. Counting it
    as an outage would hide the agent's most important behaviour."""
    refusal = _agent_constants()["REFUSAL"]
    assert not any(refusal.startswith(p) for p in _harness_prefixes())
