"""Eval harness for the chat agent as deployed: measured grounding, not vibes.

Every question in questions.jsonl is sent to a running site's POST
/api/chat/{track}, and the FINAL user-visible reply is graded. That matters:
it measures the TypeScript agent that visitors actually talk to, over the
provider actually configured, rather than a Python function that no longer
ships.

Metrics
    hallucination rate   % of replies containing >=1 claim that fails the
                         grounding contract. The README headline number.
    trap resistance      % of leading questions about absent instruments
                         ("that guitar solo!") the agent does NOT play along with.
    factual accuracy     % of answerable questions whose reply has the expected fact.
    refusal correctness  % of out-of-scope questions (artist, lyrics, album)
                         answered with an explicit "not covered" response.
    validator agreement  how often the shipped TS validator and the independent
                         Python one reach the same verdict. Disagreement means
                         the port has drifted and one of them is wrong.

Usage
    python evals/run_evals_http.py --base-url http://localhost:3002
"""

import argparse
import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from grounding_reference import check_grounding

EVALS_DIR = Path(__file__).parent
MANIFEST = (Path(__file__).resolve().parents[1] / "reference-app" / "frontend"
            / "public" / "examples" / "manifest.json")

REFUSAL_MARKERS = ("not", "n't", "no ", "unable", "only", "beyond",
                   "unavailable", "unknown")

# Canned replies the app returns when it never reached the model at all. These
# assert nothing, so the grounding checker calls them clean — which once
# produced a "0% hallucination, 6/6 traps" summary for a run in which all 30
# calls had failed. Any new degraded reply in lib/agent/index.ts must be added
# here, or the harness will silently score an outage as a pass.
DEGRADED_REPLIES = (
    ("The chat model isn't reachable", "provider unreachable"),
    ("This demo's daily AI quota is used up", "provider quota exhausted"),
)


def _ssl_context() -> ssl.SSLContext | None:
    """CA bundle for HTTPS targets.

    A framework Python on macOS ships without a usable CA store, so verifying
    an https:// deployment fails with CERTIFICATE_VERIFY_FAILED even though
    curl works. certifi is already a dependency; use its bundle rather than
    disabling verification, which would make this harness lie about reachability.
    """
    try:
        import certifi
    except ImportError:
        return None
    return ssl.create_default_context(cafile=certifi.where())


_CONTEXT = _ssl_context()


def ask(base_url: str, track: str, question: str, timeout: int) -> dict:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat/{track}",
        data=json.dumps({"messages": [{"role": "user", "content": question}]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    kwargs = {"timeout": timeout}
    if base_url.startswith("https://") and _CONTEXT is not None:
        kwargs["context"] = _CONTEXT
    with urllib.request.urlopen(request, **kwargs) as response:
        return json.load(response)


def grade(question: dict, reply: str, trace: list, grounded: bool) -> tuple[bool, str]:
    """Category-specific pass/fail for one reply (lowercased matching)."""
    text = reply.lower()
    category = question["category"]
    if category == "trap":
        return (grounded,
                "ok" if grounded else "asserted the absent instrument")
    if category == "out_of_scope":
        refused = any(marker in text for marker in REFUSAL_MARKERS)
        return ((grounded and refused),
                "ok" if refused else "answered instead of refusing")
    if category == "temporal":
        used = any(t["tool"] == question["expect_tool"] for t in trace)
        return ((grounded and used),
                "ok" if used else f"never called {question['expect_tool']}")
    hit = any(fact in text for fact in question["expect_any"])
    return ((grounded and hit),
            "ok" if hit else f"missing expected fact {question['expect_any']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:3002")
    parser.add_argument("--label", default="local",
                        help="tag for the output file, e.g. the model name")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--delay", type=float, default=45.0,
                        help="seconds between questions. Lower it for a local "
                             "Ollama endpoint, where there is no quota to spend.")
    parser.add_argument("--checkpoint", default=None,
                        help="partial-results file. Questions already answered "
                             "there are skipped, so a run can span several days "
                             "of a daily quota. Defaults to "
                             ".checkpoint_{label}.json next to this script.")
    parser.add_argument("--give-up-after", type=int, default=3,
                        help="stop after this many consecutive provider "
                             "failures. The daily quota runs out mid-run on "
                             "free tiers, and continuing just burns wall-clock "
                             "time on calls that cannot succeed.")
    args = parser.parse_args()

    manifest = {ex["id"]: ex["result"] for ex in json.loads(MANIFEST.read_text())}
    questions = [json.loads(line) for line
                 in (EVALS_DIR / "questions.jsonl").read_text().splitlines()
                 if line.strip()]

    safe_label = re.sub(r"[^\w.-]", "_", args.label)
    checkpoint_path = (Path(args.checkpoint) if args.checkpoint
                       else EVALS_DIR / f".checkpoint_{safe_label}.json")

    # Answers already collected on a previous run. Only graded rows are stored,
    # never failures, so a question that hit the quota wall is retried rather
    # than frozen as a permanent failure.
    done: dict[str, dict] = {}
    if checkpoint_path.exists():
        done = {r["id"]: r for r in json.loads(checkpoint_path.read_text())}
        print(f"resuming: {len(done)}/{len(questions)} answers already in "
              f"{checkpoint_path.name}\n")

    rows = []
    disagreements = 0
    started = time.time()
    consecutive_failures = 0
    asked_this_run = 0

    for question in questions:
        if question["id"] in done:
            rows.append(done[question["id"]])
            continue
        if consecutive_failures >= args.give_up_after:
            rows.append({**question, "error": "not attempted (gave up early)",
                         "passed": False})
            continue
        if asked_this_run:
            time.sleep(args.delay)
        asked_this_run += 1
        result = manifest[question["track"]]
        t0 = time.time()
        try:
            response = ask(args.base_url, question["track"], question["q"], args.timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            print(f"✗ [{question['category']:12s}] {question['id']:22s} "
                  f"TRANSPORT ERROR {error}")
            rows.append({**question, "error": str(error), "passed": False})
            consecutive_failures += 1
            continue
        latency = time.time() - t0

        reply = response.get("reply", "")
        trace = response.get("trace", [])
        shipped_verdict = bool(response.get("grounded"))

        # The app degrades to a fixed line when the provider is unreachable.
        # That text asserts nothing, so the grounding checker calls it clean —
        # which once produced a "0% hallucination, 6/6 traps" summary for a run
        # where all 30 calls had failed. A metric that scores an outage as a
        # pass is worse than no metric: count these as errors and refuse to
        # summarise a run that contains them.
        degraded = next(
            (label for prefix, label in DEGRADED_REPLIES if reply.startswith(prefix)),
            None,
        )
        if degraded:
            print(f"✗ [{question['category']:12s}] {question['id']:22s} "
                  f"{degraded.upper()}")
            rows.append({**question, "reply": reply,
                         "error": degraded, "passed": False})
            consecutive_failures += 1
            continue
        consecutive_failures = 0

        # Independent re-check. Tool outputs are not returned by the API, so
        # replay is impossible here — instead allow anything the analysis or a
        # timeline entry legitimately contains, which is what the tools expose.
        allowed_from_timeline = [{
            "instruments": [
                {"name": name, "confidence": probability}
                for entry in (result.get("timeline") or {}).get("instruments", [])
                for name, probability in entry["top"].items()
            ]
        }]
        independent_ok, violations = check_grounding(result, reply, allowed_from_timeline)
        # A refusal legitimately reads as grounded=False from the agent (the
        # model failed the contract) but clean to this checker (the visible text
        # asserts nothing). That is agreement, not drift, so skip refusals.
        is_refusal = reply.startswith("I couldn't give a reliable answer")
        if not is_refusal and independent_ok != shipped_verdict:
            disagreements += 1
            print(f"    ! validator drift on {question['id']}: "
                  f"shipped={shipped_verdict} independent={independent_ok}")

        passed, why = grade(question, reply, trace, independent_ok)
        row = {**question, "reply": reply, "latency_s": round(latency, 1),
               "tools_used": [t["tool"] for t in trace],
               "grounded_shipped": shipped_verdict,
               "grounded_independent": independent_ok,
               "violations": violations, "passed": passed, "why": why}
        rows.append(row)
        done[question["id"]] = row
        checkpoint_path.write_text(json.dumps(list(done.values()), indent=1))
        print(f"{'✓' if passed else '✗'} [{question['category']:12s}] "
              f"{question['id']:22s} {latency:5.1f}s  {why}")

    graded = [r for r in rows if "reply" in r and not r.get("error")]
    failed = [r for r in rows if r.get("error")]
    if failed:
        print(f"\n{len(failed)}/{len(rows)} questions failed to reach the model "
              f"({failed[0]['error']}).")
    if not graded:
        sys.exit("no usable answers — nothing to score. Check the provider "
                 "configuration and the server logs before trusting any number.")
    if failed:
        sys.exit(
            f"refusing to summarise: {len(failed)} of {len(rows)} questions "
            "never got an answer, so every rate below would be computed over a "
            f"biased subset.\n{len(done)} answers are saved in "
            f"{checkpoint_path.name}; re-run the same command once the provider "
            "is available and only the missing questions will be asked."
        )

    ungrounded = [r for r in graded if not r["grounded_independent"]]
    by_category: dict[str, list[bool]] = {}
    for row in graded:
        by_category.setdefault(row["category"], []).append(row["passed"])

    summary = {
        "label": args.label,
        "base_url": args.base_url,
        "n_questions": len(graded),
        "hallucination_rate": round(len(ungrounded) / len(graded), 3),
        "per_category_pass": {c: f"{sum(v)}/{len(v)}"
                              for c, v in sorted(by_category.items())},
        "validator_disagreements": disagreements,
        "mean_latency_s": round(sum(r["latency_s"] for r in graded) / len(graded), 1),
        "mean_tool_calls": round(sum(len(r["tools_used"]) for r in graded) / len(graded), 1),
        "total_runtime_s": round(time.time() - started, 1),
    }
    print("\n" + json.dumps(summary, indent=2))
    if ungrounded:
        print("\nungrounded replies:")
        for row in ungrounded:
            print(f"  {row['id']}: {row['violations']} :: {row['reply'][:120]}")

    out_path = EVALS_DIR / f"results_{safe_label}.json"
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
