// PROTOTYPE — throwaway route. Three art directions for the Decomposition
// overhaul, switchable via ?variant=A|B|C (arrow keys work too).
//   A — Tape Lab    : warm analog maximalism, light, tactile
//   B — Spectrum    : dark cinematic visualizer, waveform-as-hero
//   C — Field Notes : editorial/scientific specimen sheet
// Delete this whole folder once a direction wins.

"use client";

import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Switcher from "./Switcher";
import VariantA, { name as nameA } from "./VariantA";
import VariantB, { name as nameB } from "./VariantB";
import VariantC, { name as nameC } from "./VariantC";
import { allFontVars } from "./fonts";

const VARIANTS = [
  { key: "A", name: nameA },
  { key: "B", name: nameB },
  { key: "C", name: nameC },
];

function Stage() {
  const key = useSearchParams().get("variant") ?? "A";
  return (
    <>
      {key === "A" && <VariantA />}
      {key === "B" && <VariantB />}
      {key === "C" && <VariantC />}
      <Switcher variants={VARIANTS} />
    </>
  );
}

export default function ArtDirectionPrototype() {
  return (
    <div className={allFontVars}>
      {/* the app's global studio-console body styling (dark bg, amber wash,
          grain overlay) would bleed through the light variants — mute it
          for this throwaway route only. */}
      <style>{`
        body{background:#fff !important;background-image:none !important}
        body::after{display:none !important}
      `}</style>
      <Suspense fallback={<div style={{ padding: 40 }}>loading variants…</div>}>
        <Stage />
      </Suspense>
    </div>
  );
}
