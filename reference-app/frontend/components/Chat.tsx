"use client";

import { useEffect, useRef, useState } from "react";

type Msg = { role: "user" | "assistant"; content: string };
type ChatResponse = {
  reply: string;
  grounded: boolean;
  trace: { tool: string; args: Record<string, unknown> }[];
};

const STARTERS = [
  "when do the vocals come in?",
  "what's playing in the first 30 seconds?",
  "what tempo and key is this?",
];

// human-readable trace line: get_stem_activity {stem: vocals} -> "stem activity: vocals"
function traceLabel(t: ChatResponse["trace"][number]): string {
  const name = t.tool.replace("get_", "").replace(/_/g, " ");
  const args = Object.values(t.args ?? {})
    .map(String)
    .filter((v) => v && v !== "null")
    .join("–");
  return args ? `${name}: ${args}` : name;
}

/** Grounded Q&A about the analyzed track. The backend agent answers ONLY
 *  from the pipeline's analysis via tools (see backend/agent.py) — the
 *  trace of which tools it consulted is shown under each answer, because
 *  "how do you know?" deserves a visible answer in a console aesthetic. */
export default function Chat({ jobId }: { jobId: string }) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [traces, setTraces] = useState<Record<number, string[]>>({});
  const [input, setInput] = useState("");
  const [pending, setPending] = useState(false);
  const scroller = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight });
  }, [messages, pending]);

  async function send(text: string) {
    const q = text.trim();
    if (!q || pending) return;
    const history = [...messages, { role: "user" as const, content: q }];
    setMessages(history);
    setInput("");
    setPending(true);
    try {
      const res = await fetch(`/api/chat/${jobId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: history }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: ChatResponse = await res.json();
      // the assistant reply lands at index history.length
      setTraces((t) => ({
        ...t,
        [history.length]: (data.trace ?? []).map(traceLabel),
      }));
      setMessages([...history, { role: "assistant", content: data.reply }]);
    } catch {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content:
            "console link dropped — the backend isn't reachable. The analysis above still stands.",
        },
      ]);
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="rise mt-10" style={{ animationDelay: "0.35s" }}>
      <h2 className="border-b border-[var(--line)] pb-2 text-sm uppercase tracking-[0.25em] text-[var(--muted)]">
        ask the console
      </h2>
      <p className="mt-2 text-[10px] text-[var(--muted)]">
        answers come only from the analysis — the agent checks its instruments
        before speaking
      </p>

      <div
        ref={scroller}
        role="log"
        aria-live="polite"
        aria-label="chat with the analysis"
        className="mt-4 max-h-80 space-y-4 overflow-y-auto pr-2"
      >
        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="text-sm">
              <span className="text-[var(--amber)]">&gt; </span>
              {m.content}
            </div>
          ) : (
            <div key={i} className="border-l-2 border-[var(--line)] pl-3">
              <div className="text-sm leading-relaxed">{m.content}</div>
              {(traces[i] ?? []).length > 0 && (
                <div className="mt-1 text-[10px] text-[var(--muted)]">
                  · consulted {traces[i].join(" · ")}
                </div>
              )}
            </div>
          ),
        )}
        {pending && (
          <div className="border-l-2 border-[var(--line)] pl-3 text-sm text-[var(--muted)]">
            <span className="blink">▮</span> checking the analysis…
          </div>
        )}
      </div>

      {messages.length === 0 && (
        <div className="mt-4 flex flex-wrap gap-2">
          {STARTERS.map((s) => (
            <button
              key={s}
              onClick={() => send(s)}
              className="border border-[var(--line)] px-3 py-1.5 text-xs text-[var(--muted)] transition-colors hover:border-[var(--amber)] hover:text-[var(--amber)]"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      <form
        className="mt-4 flex items-center gap-3 border border-[var(--line)] bg-[var(--panel)] px-3"
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
      >
        <span aria-hidden className="text-[var(--amber)]">
          &gt;
        </span>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="ask about instruments, stems, tempo…"
          aria-label="ask a question about this track"
          disabled={pending}
          className="min-w-0 flex-1 bg-transparent py-3 text-sm outline-none placeholder:text-[var(--muted)] disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={pending || !input.trim()}
          className="text-xs uppercase tracking-wider text-[var(--muted)] transition-colors hover:text-[var(--amber)] disabled:opacity-40"
        >
          send
        </button>
      </form>
    </div>
  );
}
