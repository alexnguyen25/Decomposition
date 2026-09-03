"""Unit tests for agent.py's deterministic parts: tools + grounding checker.

The LLM itself is NOT under test here (that's the eval harness's job —
evals/run_evals.py measures the end-to-end hallucination rate). These tests
pin down the code the whole grounding story rests on.

Run:  ../../.venv/bin/python -m pytest test_agent.py -q
"""

import agent

RESULT = {
    "duration_s": 200.0,
    "bpm": 120.0,
    "key": "A minor",
    "presence": {"vocals": True, "drums": True, "bass": False},
    "instruments": [{"name": "guitar", "confidence": 0.9},
                    {"name": "piano", "confidence": 0.6}],
    "timeline": {
        "chunk_s": 10,
        "instruments": [
            {"t": 0, "top": {"guitar": 0.9}},
            {"t": 10, "top": {"guitar": 0.8, "violin": 0.4}},
            {"t": 20, "top": {"piano": 0.7}},
        ],
        "stem_activity": {
            "hop_s": 1.0,
            # vocals: silent 0-9s, active 10-19s, silent again
            "vocals": [0.0] * 10 + [0.8] * 10 + [0.0] * 10,
            "drums": [0.5] * 30,
            "bass": [0.0] * 30,
            "other": [0.9] * 30,
        },
    },
}


# ── tools ────────────────────────────────────────────────────────────────────

def test_instruments_whole_song():
    out = agent._tool_get_instruments(RESULT)
    assert [i["name"] for i in out["instruments"]] == ["guitar", "piano"]


def test_instruments_window_merges_chunk_maxima():
    out = agent._tool_get_instruments(RESULT, start_s=5, end_s=15)
    # chunks [0,10) and [10,20) overlap the window; violin appears only there
    names = {i["name"]: i["confidence"] for i in out["instruments"]}
    assert names["guitar"] == 0.9          # max over the two chunks
    assert "violin" in names
    assert "piano" not in names            # only in chunk [20,30)


def test_instruments_window_without_timeline_degrades():
    bare = {k: v for k, v in RESULT.items() if k != "timeline"}
    out = agent._tool_get_instruments(bare, start_s=0, end_s=10)
    assert "error" in out and out["instruments"]   # still gives song level


def test_stem_activity_spans():
    out = agent._tool_get_stem_activity(RESULT, "vocals")
    assert out["active_spans"] == [["0:10", "0:20"]]
    assert 0.3 < out["active_fraction"] < 0.4


def test_stem_activity_unknown_stem():
    assert "error" in agent._tool_get_stem_activity(RESULT, "kazoo")


def test_stem_activity_absent_stem_reports_absent():
    # bass presence is False; peak-normalized residue must NOT become spans
    out = agent._tool_get_stem_activity(RESULT, "bass")
    assert out.get("present") is False
    assert "active_spans" not in out


def test_bpm_key():
    out = agent._tool_get_bpm_key(RESULT)
    assert out["bpm"] == 120.0 and out["key"] == "A minor"


def test_junk_tool_args_coerced():
    # small models send 'null' as a string, timestamps as "1:30", etc.
    out = agent._tool_get_instruments(RESULT, start_s="null", end_s="None")
    assert out["scope"] == "whole song"
    out = agent._tool_get_instruments(RESULT, start_s="0:05", end_s="0:15")
    assert any(i["name"] == "violin" for i in out["instruments"])


# ── grounding checker ────────────────────────────────────────────────────────

def _check(reply, tool_outputs=()):
    return agent.check_grounding(RESULT, reply, list(tool_outputs))


def test_ok_when_only_detected_instruments_named():
    ok, v = _check("Clear guitar throughout, with piano appearing later.")
    assert ok, v


def test_flags_undetected_instrument():
    ok, v = _check("The banjo carries the melody.")
    assert not ok and "banjo" in v[0]


def test_negated_mention_is_grounded():
    ok, v = _check("No banjo was detected in this track.")
    assert ok, v


def test_tool_returned_instrument_is_allowed():
    # violin is below the song-level threshold but a tool surfaced it
    tool_out = agent._tool_get_instruments(RESULT, start_s=10, end_s=20)
    ok, v = _check("There are hints of violin around 0:10.", [tool_out])
    assert ok, v


def test_wrong_bpm_flagged_right_bpm_ok():
    ok, v = _check("It sits at 90 BPM.")
    assert not ok and "90" in v[0]
    ok, _ = _check("It sits at a steady 120 bpm.")
    assert ok


def test_wrong_key_flagged():
    ok, _ = _check("The song is in C major.")
    assert not ok
    ok, _ = _check("The song is in A minor.")
    assert ok


def test_wrong_duration_flagged_timestamps_ok():
    ok, v = _check("The track's duration is about 5:17.")     # truth: 3:20
    assert not ok and "duration" in v[0]
    ok, _ = _check("The duration is 3:20.")
    assert ok
    ok, _ = _check("Vocals enter at 5:17 in some remix.")     # timestamp, not length
    assert ok


def test_multiword_class_names_match():
    # class map has e.g. mallet_percussion; prose says "mallet percussion"
    ok, v = _check("Bright mallet percussion opens the track.")
    assert not ok and "mallet percussion" in v[0]
