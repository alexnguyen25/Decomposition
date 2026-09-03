// PROTOTYPE — throwaway. One distinctive type pairing per art direction.
// next/font requires module-scope calls, so they all live here.
import {
  Anton,
  DM_Mono,
  Fraunces,
  JetBrains_Mono,
  Newsreader,
} from "next/font/google";

// A — Tape Lab: wonky optical serif, warm and characterful
export const fraunces = Fraunces({
  subsets: ["latin"],
  axes: ["SOFT", "WONK", "opsz"],
  variable: "--f-tape-display",
});
export const dmMono = DM_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--f-tape-mono",
});

// B — Spectrum: heavy condensed display against a technical mono
export const anton = Anton({
  subsets: ["latin"],
  weight: "400",
  variable: "--f-spec-display",
});
export const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--f-spec-mono",
});

// C — Field Notes: literary serif with optical sizing
export const newsreader = Newsreader({
  subsets: ["latin"],
  axes: ["opsz"],
  style: ["normal", "italic"],
  variable: "--f-field-display",
});

export const allFontVars = [
  fraunces.variable,
  dmMono.variable,
  anton.variable,
  jetbrains.variable,
  newsreader.variable,
].join(" ");
