"use client";

// PROTOTYPE — throwaway. Variant C: "FIELD NOTES"
// Editorial/scientific. The song is a specimen on a herbarium sheet: numbered
// samples, hairline rules, marginalia, a literary serif. Delight lives in the
// typographic craft, not spectacle. Motion: crisp, short, no bounce.

import { useState } from "react";
import { CHAT, peaks, STEMS, TRACK, type Stem } from "./data";

const SPEC: Record<Stem, { c: string; latin: string }> = {
  vocals: { c: "#8C2F27", latin: "vox humana" },
  drums: { c: "#A8752C", latin: "percussio" },
  bass: { c: "#3F5165", latin: "fundamentum" },
  other: { c: "#4A6B4F", latin: "cetera" },
};

export const name = "Field Notes";

export default function VariantC() {
  const [open, setOpen] = useState<Stem | null>("vocals");

  return (
    <div className="v-field">
      <style>{fieldCSS}</style>

      {/* ── masthead ─────────────────────────────────────────── */}
      <header className="fn-mast">
        <div className="fn-mast-rule">
          <span>Decomposition</span>
          <span>Specimen No. 867662</span>
          <span>OpenMIC-2018 · macro-F1 0.80</span>
        </div>
        <h1 className="fn-title">
          A song, <em>dissected</em>
        </h1>
        <div className="fn-lede">
          <p>
            <span className="fn-drop">U</span>pload a recording and it is taken
            apart: four isolated stems, every instrument identified inside the
            mix, tempo and key measured, and a written account of what was
            found — each claim traceable to the analysis that produced it.
          </p>
          <aside className="fn-margin">
            <b>Method.</b> Separation by Demucs; instrument identification by a
            frozen BEATs encoder with a trained classification head. Nothing in
            the write-up is asserted without evidence.
          </aside>
        </div>
        <button className="fn-cta">
          Submit a specimen <span aria-hidden>→</span>
        </button>
      </header>

      {/* ── plate: the four stems ────────────────────────────── */}
      <section className="fn-plate">
        <div className="fn-sec-head">
          <h2>Plate I — Separation</h2>
          <span>four constituent parts, isolated</span>
        </div>

        <div className="fn-specimens">
          {STEMS.map((s, i) => {
            const isOpen = open === s;
            return (
              <article
                key={s}
                className={`fn-spec${isOpen ? " fn-spec-open" : ""}`}
                style={{ ["--ink" as string]: SPEC[s].c, animationDelay: `${i * 70}ms` }}
              >
                <button className="fn-spec-hd" onClick={() => setOpen(isOpen ? null : s)}>
                  <span className="fn-num">{String(i + 1).padStart(2, "0")}</span>
                  <span className="fn-spec-t">
                    {s}
                    <em>{SPEC[s].latin}</em>
                  </span>
                  <span className="fn-spec-x" aria-hidden>
                    {isOpen ? "−" : "+"}
                  </span>
                </button>
                {/* single grid child — the 0fr→1fr collapse only constrains
                    the first row, so everything must live inside one wrapper */}
                <div className="fn-spec-body">
                  <div className="fn-spec-inner">
                    <div className="fn-trace">
                      {peaks(s, 80).map((p, j) => (
                        <i key={j} style={{ height: `${p * 100}%` }} />
                      ))}
                    </div>
                    <div className="fn-spec-meta">
                      <span>continuous throughout</span>
                      <span>·</span>
                      <span>isolate</span>
                      <span>·</span>
                      <span>download</span>
                    </div>
                  </div>
                </div>
              </article>
            );
          })}
        </div>
      </section>

      {/* ── plate II: findings ───────────────────────────────── */}
      <section className="fn-findings">
        <div className="fn-sec-head">
          <h2>Plate II — Identification</h2>
          <span>seven instruments above threshold</span>
        </div>

        <div className="fn-cols">
          <ol className="fn-list">
            {TRACK.instruments.map((it, i) => (
              <li key={it.name} style={{ animationDelay: `${i * 50}ms` }}>
                <span className="fn-li-n">{String(i + 1).padStart(2, "0")}</span>
                <span className="fn-li-name">{it.name.replace("_", " ")}</span>
                <span className="fn-rule" aria-hidden />
                <span className="fn-li-c">{it.confidence.toFixed(3)}</span>
              </li>
            ))}
          </ol>

          <div className="fn-obs">
            <div className="fn-measures">
              <div>
                <b>{TRACK.bpm}</b>
                <span>beats / min</span>
              </div>
              <div>
                <b>{TRACK.key}</b>
                <span>estimated key</span>
              </div>
              <div>
                <b>4:17</b>
                <span>duration</span>
              </div>
            </div>
            <h3 className="fn-h3">Observations</h3>
            <p className="fn-body">{TRACK.blurb}</p>
            <p className="fn-attrib">
              Written by the model, constrained to the analysis above.
            </p>
          </div>
        </div>
      </section>

      {/* ── correspondence / chat ────────────────────────────── */}
      <section className="fn-corr">
        <div className="fn-sec-head">
          <h2>Plate III — Enquiry</h2>
          <span>ask; answers are cited</span>
        </div>
        <div className="fn-corr-body">
          {CHAT.map((m, i) =>
            m.role === "user" ? (
              <p key={i} className="fn-q">
                {m.text}
              </p>
            ) : (
              <blockquote key={i} className="fn-a">
                {m.text}
                <cite>consulted {m.tools?.join(", ")}</cite>
              </blockquote>
            ),
          )}
          <div className="fn-input">
            <input placeholder="put a question to the analysis…" readOnly />
            <button>Enquire</button>
          </div>
        </div>
      </section>
    </div>
  );
}

const fieldCSS = `
.v-field{
  --paper:#FBFAF6; --ink:#14140F; --soft:#6E6A5E; --hair:#D8D3C6; --red:#8C2F27;
  background:var(--paper); color:var(--ink); min-height:100vh;
  font-family:var(--font-mono),ui-monospace,monospace;
  padding:0 clamp(20px,6vw,88px) 140px;
  background-image:linear-gradient(var(--hair) 1px,transparent 1px);
  background-size:100% 34px; background-position:0 -1px;
}
.v-field *{box-sizing:border-box}
.v-field ::selection{background:#EDE6D2}

.fn-mast{max-width:1120px;margin:0 auto;padding:clamp(34px,7vh,84px) 0 0}
.fn-mast-rule{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;
  border-bottom:1.5px solid var(--ink);padding-bottom:9px;font-size:10px;letter-spacing:.2em;
  text-transform:uppercase;color:var(--soft)}
.fn-mast-rule span:first-child{color:var(--ink)}
.fn-title{font-family:var(--f-field-display),Georgia,serif;font-weight:400;
  font-size:clamp(46px,8.6vw,116px);line-height:.98;letter-spacing:-.022em;margin:clamp(26px,5vh,52px) 0 0;
  font-variation-settings:'opsz' 60}
.fn-title em{font-style:italic;color:var(--red)}
.fn-lede{display:grid;grid-template-columns:1.6fr .9fr;gap:clamp(22px,4vw,60px);
  margin-top:clamp(26px,4vh,44px);align-items:start}
@media(max-width:860px){.fn-lede{grid-template-columns:1fr}}
.fn-lede p{font-family:var(--f-field-display),Georgia,serif;font-size:clamp(17px,2vw,21px);
  line-height:1.62;margin:0;font-variation-settings:'opsz' 20}
.fn-drop{float:left;font-size:3.05em;line-height:.78;padding:5px 9px 0 0;color:var(--red);
  font-variation-settings:'opsz' 72}
.fn-margin{border-left:2px solid var(--red);padding-left:15px;font-size:11.5px;line-height:1.72;
  color:var(--soft)}
.fn-margin b{color:var(--ink);font-weight:500}
.fn-cta{margin-top:clamp(26px,4vh,44px);font-family:inherit;font-size:11.5px;letter-spacing:.15em;
  text-transform:uppercase;background:var(--ink);color:var(--paper);border:0;padding:14px 26px;
  cursor:pointer;display:inline-flex;gap:12px;align-items:center;
  transition:transform 150ms cubic-bezier(.23,1,.32,1),background 180ms ease}
.fn-cta span{transition:transform 200ms cubic-bezier(.23,1,.32,1)}
.fn-cta:hover{background:var(--red)}
.fn-cta:hover span{transform:translateX(4px)}
.fn-cta:active{transform:scale(.98)}

.fn-sec-head{display:flex;align-items:baseline;gap:16px;border-bottom:1px solid var(--ink);
  padding-bottom:8px;margin-bottom:clamp(20px,3vh,34px)}
.fn-sec-head h2{font-family:var(--f-field-display),Georgia,serif;font-size:22px;font-weight:400;
  margin:0;letter-spacing:-.01em}
.fn-sec-head span{font-size:10px;letter-spacing:.18em;text-transform:uppercase;color:var(--soft);
  margin-left:auto}

.fn-plate,.fn-findings,.fn-corr{max-width:1120px;margin:clamp(48px,9vh,110px) auto 0}

/* specimens */
.fn-spec{border-bottom:1px solid var(--hair);opacity:0;
  animation:fn-in 420ms cubic-bezier(.23,1,.32,1) forwards}
@keyframes fn-in{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:none}}
.fn-spec-hd{display:flex;align-items:center;gap:18px;width:100%;background:none;border:0;
  padding:17px 4px;cursor:pointer;font:inherit;color:inherit;text-align:left;
  transition:padding-left 200ms cubic-bezier(.23,1,.32,1)}
.fn-spec-hd:hover{padding-left:12px}
.fn-num{font-size:10.5px;color:var(--soft);width:22px;flex:none}
.fn-spec-t{font-family:var(--f-field-display),Georgia,serif;font-size:23px;letter-spacing:-.01em;
  display:flex;align-items:baseline;gap:12px}
.fn-spec-t em{font-size:11.5px;font-style:italic;color:var(--ink-2,#8A8474);letter-spacing:.02em}
.fn-spec-x{margin-left:auto;font-size:17px;color:var(--soft)}
.fn-spec-body{display:grid;grid-template-rows:0fr;
  transition:grid-template-rows 280ms cubic-bezier(.23,1,.32,1)}
.fn-spec-open .fn-spec-body{grid-template-rows:1fr}
.fn-spec-inner{overflow:hidden;min-height:0}
.fn-trace{display:flex;align-items:flex-end;gap:1.5px;height:74px;padding:0 4px 14px;min-width:0}
.fn-trace i{flex:1;background:var(--ink);opacity:.16;transition:opacity 200ms ease}
.fn-spec-open .fn-trace i{opacity:.42;background:var(--ink)}
.fn-spec-meta{display:flex;gap:11px;padding:0 4px 18px;font-size:10.5px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--soft)}

/* findings */
.fn-cols{display:grid;grid-template-columns:1fr 1.12fr;gap:clamp(26px,5vw,72px)}
@media(max-width:860px){.fn-cols{grid-template-columns:1fr}}
.fn-list{list-style:none;margin:0;padding:0}
.fn-list li{display:flex;align-items:baseline;gap:11px;padding:11px 0;
  border-bottom:1px dotted var(--hair);opacity:0;
  animation:fn-in 400ms cubic-bezier(.23,1,.32,1) forwards}
.fn-li-n{font-size:10px;color:var(--soft);width:20px;flex:none}
.fn-li-name{font-family:var(--f-field-display),Georgia,serif;font-size:19px;
  font-variation-settings:'opsz' 18}
.fn-rule{flex:1;border-bottom:1px dotted var(--hair);transform:translateY(-3px)}
.fn-li-c{font-size:12px;font-variant-numeric:tabular-nums;color:var(--soft)}
.fn-measures{display:flex;gap:clamp(18px,4vw,44px);border-bottom:1.5px solid var(--ink);
  padding-bottom:16px;margin-bottom:22px;flex-wrap:wrap}
.fn-measures b{display:block;font-family:var(--f-field-display),Georgia,serif;font-size:30px;
  font-weight:400;line-height:1.05}
.fn-measures span{font-size:9.5px;letter-spacing:.17em;text-transform:uppercase;color:var(--soft)}
.fn-h3{font-size:10px;letter-spacing:.22em;text-transform:uppercase;color:var(--soft);
  margin:0 0 11px;font-weight:400}
.fn-body{font-family:var(--f-field-display),Georgia,serif;font-size:17.5px;line-height:1.68;
  margin:0;font-variation-settings:'opsz' 18}
.fn-attrib{font-size:10.5px;color:var(--soft);margin-top:14px;font-style:italic}

/* correspondence */
.fn-corr-body{max-width:730px}
.fn-q{font-family:var(--f-field-display),Georgia,serif;font-style:italic;font-size:19px;
  margin:0 0 14px;color:var(--red)}
.fn-a{margin:0 0 26px;padding-left:20px;border-left:2px solid var(--hair);
  font-family:var(--f-field-display),Georgia,serif;font-size:17px;line-height:1.66}
.fn-a cite{display:block;font-family:var(--font-mono),monospace;font-style:normal;font-size:9.5px;
  letter-spacing:.14em;text-transform:uppercase;color:var(--soft);margin-top:11px}
.fn-input{display:flex;gap:0;border:1.5px solid var(--ink)}
.fn-input input{flex:1;border:0;background:none;font:inherit;font-size:12.5px;padding:13px 15px;
  outline:none;color:var(--ink)}
.fn-input button{font:inherit;font-size:10.5px;letter-spacing:.15em;text-transform:uppercase;
  background:var(--ink);color:var(--paper);border:0;padding:0 22px;cursor:pointer;
  transition:background 180ms ease}
.fn-input button:hover{background:var(--red)}
@media(prefers-reduced-motion:reduce){
  .v-field *{animation-duration:.01ms!important;transition-duration:.01ms!important}
}
`;
