// PROTOTYPE — throwaway. Shared mock data + deterministic fake waveforms so
// all three art directions render at realistic density without depending on
// the backend. Numbers mirror the real ex_867662 example.

export const TRACK = {
  title: "Folk ensemble — accordion, brass & strings",
  bpm: 147.7,
  key: "A minor",
  duration_s: 257.3,
  genre: "Indie / Alternative",
  blurb:
    "A haunting A minor track with a steady, driving tempo at 147.7 BPM. Vocals, drums and bass lay a solid foundation, while accordion, trumpet, trombone, violin, cello, guitar and saxophone add depth and complexity.",
  instruments: [
    { name: "accordion", confidence: 0.993 },
    { name: "trumpet", confidence: 0.958 },
    { name: "trombone", confidence: 0.87 },
    { name: "violin", confidence: 0.855 },
    { name: "cello", confidence: 0.851 },
    { name: "guitar", confidence: 0.843 },
    { name: "saxophone", confidence: 0.707 },
  ],
};

export const STEMS = ["vocals", "drums", "bass", "other"] as const;
export type Stem = (typeof STEMS)[number];

// deterministic PRNG so the "waveform" is stable across renders/reloads
function rng(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
}

/** Per-stem envelope with a distinct character:
 *  vocals enter late and phrase in bursts, drums are dense and steady,
 *  bass is a smooth floor, other breathes. */
export function peaks(stem: Stem, n = 96): number[] {
  const r = rng(stem.length * 9871 + stem.charCodeAt(0) * 131);
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = i / n;
    let v: number;
    switch (stem) {
      case "vocals": {
        const phrase = Math.sin(t * Math.PI * 9) * 0.5 + 0.5;
        const gate = t < 0.08 || (t > 0.42 && t < 0.5) ? 0.05 : 1;
        v = phrase * gate * (0.55 + r() * 0.45);
        break;
      }
      case "drums": {
        const build = Math.min(1, t * 5);
        v = build * (0.62 + r() * 0.38);
        break;
      }
      case "bass": {
        v = 0.45 + Math.sin(t * Math.PI * 3) * 0.12 + r() * 0.16;
        break;
      }
      default: {
        v = 0.4 + Math.sin(t * Math.PI * 6) * 0.22 + r() * 0.3;
      }
    }
    out.push(Math.max(0.04, Math.min(1, v)));
  }
  return out;
}

export const CHAT = [
  { role: "user", text: "when do the vocals come in?" },
  {
    role: "assistant",
    text: "The vocals start around 0:13 and stay audible until about 4:10.",
    tools: ["stem activity: vocals"],
  },
];
