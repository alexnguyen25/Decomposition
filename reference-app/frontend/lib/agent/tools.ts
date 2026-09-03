/** The three tools the agent may call. Each one reads the analysis object and
 *  returns JSON-safe data — the model never sees audio and is never asked to
 *  recall anything, so answers are grounded by construction. */

import type { Analysis, ToolOutput } from "./types";

export function formatTime(seconds: number): string {
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}

/** Small models send junk arguments ("null", "None", "1:30", ""). Coerce
 *  charitably and treat garbage as absent rather than throwing. */
export function numberOrNull(value: unknown): number | null {
  if (value === null || value === undefined || typeof value === "boolean") return null;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  const text = String(value).trim().toLowerCase();
  if (["", "null", "none", "nan"].includes(text)) return null;
  const clock = /^(\d+):([0-5]?\d)$/.exec(text);
  if (clock) return Number(clock[1]) * 60 + Number(clock[2]);
  const parsed = Number(text);
  return Number.isFinite(parsed) ? parsed : null;
}

function getBpmKey(result: Analysis): ToolOutput {
  return {
    bpm: result.bpm ?? null,
    key: result.key ?? null,
    // pre-formatted because models botch seconds -> m:ss arithmetic
    duration: result.duration_s ? formatTime(result.duration_s) : null,
    note: "key may be null when the detector was unavailable",
  };
}

function getInstruments(result: Analysis, args: Record<string, unknown>): ToolOutput {
  let start = numberOrNull(args.start_s);
  let end = numberOrNull(args.end_s);
  if (start !== null && end !== null && end <= start) {
    start = null;
    end = null;                                  // degenerate window -> whole song
  }
  if (start === null && end === null) {
    return {
      scope: "whole song",
      instruments: result.instruments ?? [],
      note:
        "detected in the non-vocal/drum/bass ('other') stem; confidence is " +
        "the classifier's probability",
    };
  }

  const entries = result.timeline?.instruments;
  if (!entries || entries.length === 0) {
    return {
      error:
        "no time-resolved data for this track; only whole-song instruments are available",
      instruments: result.instruments ?? [],
    };
  }

  const from = start ?? 0;
  const to = end ?? result.duration_s ?? 1e9;
  const chunk = result.timeline?.chunk_s ?? 10;
  const merged = new Map<string, number>();
  for (const entry of entries) {
    if (entry.t + chunk <= from || entry.t >= to) continue;
    for (const [name, probability] of Object.entries(entry.top)) {
      merged.set(name, Math.max(merged.get(name) ?? 0, probability));
    }
  }
  return {
    scope: `${formatTime(from)}-${formatTime(Math.min(to, result.duration_s ?? to))}`,
    instruments: [...merged.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([name, confidence]) => ({ name, confidence: Number(confidence.toFixed(3)) })),
    note: "confidences are per-10s-chunk maxima within the window",
  };
}

function getStemActivity(result: Analysis, args: Record<string, unknown>): ToolOutput {
  const stem = String(args.stem ?? "").toLowerCase().trim();
  if (!["vocals", "drums", "bass", "other"].includes(stem)) {
    return { error: `unknown stem '${stem}' — use vocals/drums/bass/other` };
  }
  // Presence gate first: the envelope is normalised to the stem's OWN peak, so
  // a near-silent stem would show residual Demucs bleed as "activity". The eval
  // harness caught the model faithfully relaying that lying tool.
  if (result.presence?.[stem] === false) {
    return {
      stem,
      present: false,
      note: `this track has no meaningful ${stem} — the stem is essentially silent`,
    };
  }
  const activity = result.timeline?.stem_activity;
  const envelope = activity?.[stem] as number[] | undefined;
  if (!envelope) {
    return {
      stem,
      error: "no time-resolved data for this track",
      present_overall: result.presence?.[stem],
    };
  }
  const hop = (activity?.hop_s as number | undefined) ?? 1.0;
  const threshold = 0.15;                        // of the stem's own peak
  const spans: [number, number][] = [];
  let runStart: number | null = null;
  [...envelope, 0].forEach((value, index) => {
    if (value >= threshold && runStart === null) {
      runStart = index * hop;
    } else if (value < threshold && runStart !== null) {
      if (index * hop - runStart >= 2) spans.push([runStart, index * hop]);
      runStart = null;
    }
  });
  const active = envelope.filter((v) => v >= threshold).length;
  return {
    stem,
    active_fraction: Number((active / Math.max(1, envelope.length)).toFixed(2)),
    active_spans: spans.slice(0, 25).map(([a, b]) => [formatTime(a), formatTime(b)]),
    note: "spans where the stem is above 15% of its peak loudness",
  };
}

export function runTool(
  name: string,
  result: Analysis,
  args: Record<string, unknown>,
): ToolOutput {
  switch (name) {
    case "get_bpm_key":
      return getBpmKey(result);
    case "get_instruments":
      return getInstruments(result, args);
    case "get_stem_activity":
      return getStemActivity(result, args);
    default:
      return { error: `unknown tool ${name}` };
  }
}

export const TOOLS_SPEC = [
  {
    type: "function",
    function: {
      name: "get_bpm_key",
      description: "Tempo (BPM), musical key and duration of the track.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "get_instruments",
      description:
        "Instruments detected by the classifier. Without arguments: the whole " +
        "song. With start_s/end_s (seconds): what was detected inside that window.",
      parameters: {
        type: "object",
        properties: {
          // ["number", "null"], not "number": models signal "whole song" by
          // sending null for both bounds, and Groq validates tool arguments
          // against this schema SERVER-SIDE and rejects the call with a 400
          // before our own charitable coercion in numberOrNull ever runs.
          start_s: {
            type: ["number", "null"],
            description: "window start in seconds; null or omitted for the whole song",
          },
          end_s: {
            type: ["number", "null"],
            description: "window end in seconds; null or omitted for the whole song",
          },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_stem_activity",
      description:
        "When a stem (vocals, drums, bass or other) is audible: active time " +
        "spans and overall fraction.",
      parameters: {
        type: "object",
        properties: {
          stem: { type: "string", enum: ["vocals", "drums", "bass", "other"] },
        },
        required: ["stem"],
      },
    },
  },
];
