"use client";

// PROTOTYPE — throwaway. Floating variant switcher. Deliberately styled to
// look like devtooling, not part of any design being judged.

import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect } from "react";

export default function Switcher({
  variants,
}: {
  variants: { key: string; name: string }[];
}) {
  const router = useRouter();
  const params = useSearchParams();
  const current = params.get("variant") ?? variants[0].key;
  const idx = Math.max(
    0,
    variants.findIndex((v) => v.key === current),
  );

  const go = useCallback(
    (delta: number) => {
      const next = variants[(idx + delta + variants.length) % variants.length];
      router.replace(`?variant=${next.key}`, { scroll: false });
    },
    [idx, router, variants],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (
        t &&
        (t.tagName === "INPUT" ||
          t.tagName === "TEXTAREA" ||
          t.isContentEditable)
      )
        return;
      if (e.key === "ArrowLeft") go(-1);
      if (e.key === "ArrowRight") go(1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go]);

  if (process.env.NODE_ENV === "production") return null;

  return (
    <div
      style={{
        position: "fixed",
        bottom: 20,
        left: "50%",
        transform: "translateX(-50%)",
        zIndex: 9999,
        display: "flex",
        alignItems: "center",
        gap: 4,
        padding: 4,
        borderRadius: 999,
        background: "#0B0B0F",
        border: "1px solid #33333F",
        boxShadow: "0 8px 30px rgba(0,0,0,.45)",
        fontFamily: "ui-monospace, monospace",
        fontSize: 12,
        color: "#fff",
      }}
    >
      <button onClick={() => go(-1)} aria-label="Previous variant" style={btn}>
        ←
      </button>
      <span style={{ padding: "0 12px", letterSpacing: ".04em" }}>
        <b style={{ color: "#7CE0B0" }}>{variants[idx].key}</b>
        {"  "}
        {variants[idx].name}
        <span style={{ opacity: 0.45, marginLeft: 10 }}>
          {idx + 1}/{variants.length}
        </span>
      </span>
      <button onClick={() => go(1)} aria-label="Next variant" style={btn}>
        →
      </button>
    </div>
  );
}

const btn: React.CSSProperties = {
  width: 30,
  height: 30,
  borderRadius: 999,
  border: "1px solid #33333F",
  background: "#17171F",
  color: "#fff",
  cursor: "pointer",
  fontSize: 13,
  lineHeight: 1,
};
