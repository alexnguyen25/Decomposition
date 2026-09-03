/** Post-hoc verification of the model's answer against the analysis.
 *
 *  The system prompt asks the model to stay grounded; this checks that it did.
 *  Prompting is a request, not a guarantee — so every checkable claim
 *  (instrument names, BPM, duration, key) is re-verified in code after
 *  generation. A chat agent that sometimes says "I can't verify that" beats
 *  one that sometimes lies.
 */

import { CLASS_NAMES } from "./classes";
import { formatTime } from "./tools";
import type { Analysis, ToolOutput } from "./types";

/** Words a track may be described with even though the classifier never
 *  reports them, because Demucs answers them as stems instead. Each maps to
 *  the presence flag that decides whether it is actually true HERE — an
 *  unconditional allowlist let the agent assert "there is a bass line" about a
 *  track whose bass stem is silent (caught by the eval harness). */
const GENERIC_STEM_ALIASES: Record<string, string> = {
  vocals: "vocals",
  voice: "vocals",
  drums: "drums",
  percussion: "drums",
  bass: "bass",
};

const NEGATORS = [
  "no ", "not ", "n't ", "n't.", "without", "lack", "absent", "wasn't",
  "isn't", "aren't", "don't", "doesn't", "didn't", "couldn't", "can't",
  "cannot", "never", "none", "neither", "instead of", "rather than",
];
// "detect" used to be in this list for "didn't detect X", but it also made
// "there is a detected bass line" read as a denial and slip past the check.
// The explicit negators above already cover the intended phrasings.

/** True when a negation appears shortly before the mention. Denying an
 *  instrument is grounded speech; asserting one is what needs backing. */
function isNegated(text: string, matchStart: number): boolean {
  const window = text.slice(Math.max(0, matchStart - 60), matchStart).toLowerCase();
  return NEGATORS.some((negator) => window.includes(negator));
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function checkGrounding(
  result: Analysis,
  reply: string,
  toolOutputs: ToolOutput[],
): { ok: boolean; violations: string[] } {
  const violations: string[] = [];
  const text = reply.toLowerCase();

  // 1. Instruments. Allowed = the song-level analysis, the present stems, and
  //    anything a tool actually returned this conversation (per-chunk names can
  //    sit below the song-level threshold — if the tool said it, it's grounded).
  const allowed = new Set<string>();
  for (const [word, stem] of Object.entries(GENERIC_STEM_ALIASES)) {
    // absent only when presence says so explicitly; unknown stays permissive
    if (result.presence?.[stem] !== false) allowed.add(word);
  }
  for (const instrument of result.instruments ?? []) allowed.add(instrument.name);
  for (const [stem, present] of Object.entries(result.presence ?? {})) {
    if (present) allowed.add(stem);
  }
  for (const output of toolOutputs) {
    for (const instrument of output.instruments ?? []) allowed.add(instrument.name);
  }

  const checkable = new Set<string>([...CLASS_NAMES, ...Object.keys(GENERIC_STEM_ALIASES)]);
  for (const name of checkable) {
    if (allowed.has(name)) continue;
    const pretty = name.replace("_", " ");
    const pattern = new RegExp(`\\b${escapeRegExp(pretty)}s?\\b`, "g");
    for (const match of text.matchAll(pattern)) {
      if (!isNegated(text, match.index ?? 0)) {
        violations.push(`asserted undetected instrument: ${pretty}`);
        break;
      }
    }
  }

  // 2. BPM: any number the reply attaches to "bpm" must match the analysis.
  const bpm = result.bpm;
  for (const match of text.matchAll(/(\d+(?:\.\d+)?)\s*bpm/g)) {
    if (bpm === null || bpm === undefined || Math.abs(Number(match[1]) - bpm) > 1.5) {
      violations.push(`claimed ${match[1]} BPM (analysis: ${bpm ?? "unknown"})`);
    }
  }

  // 3. Duration: m:ss values presented AS the length must match. A bare
  //    timestamp ("vocals enter at 1:12") is not a length claim.
  const duration = result.duration_s;
  if (duration) {
    const pattern =
      /(?:duration|length|long|lasts|runs)\D{0,20}?(\d+):([0-5]\d)|(\d+):([0-5]\d)\s*(?:long|in length|total)/g;
    for (const match of text.matchAll(pattern)) {
      const minutes = match[1] ?? match[3];
      const seconds = match[2] ?? match[4];
      const claimed = Number(minutes) * 60 + Number(seconds);
      if (Math.abs(claimed - duration) > 3) {
        violations.push(
          `claimed duration ${minutes}:${seconds} (analysis: ${formatTime(duration)})`,
        );
      }
    }
  }

  // 4. Key: "in X major/minor" claims must match.
  const key = (result.key ?? "").toLowerCase().replace("#", " sharp");
  for (const match of text.matchAll(
    /\b([a-g](?:\s?(?:sharp|flat)|[#b])?)\s+(major|minor)\b/g,
  )) {
    const claimed = `${match[1].trim()} ${match[2]}`.replace("#", " sharp");
    if (claimed !== key && !isNegated(text, match.index ?? 0)) {
      violations.push(`claimed key '${match[0]}' (analysis: ${key || "unknown"})`);
    }
  }

  return { ok: violations.length === 0, violations };
}
