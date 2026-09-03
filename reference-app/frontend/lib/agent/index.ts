/** Grounded chat agent: a tool-calling loop over ONE track's analysis.
 *
 *  Ported from reference-app/backend/agent.py, which stays in the repo as the
 *  readable original. This version runs in a Next.js route handler so the whole
 *  deployed app is one Vercel project — the tools read precomputed JSON, so no
 *  Python, no Demucs and no model weights are needed at request time.
 *
 *  Provider-agnostic: any OpenAI-compatible endpoint with tool calling works.
 *  Groq in production, Ollama locally, swapped with two env vars.
 */

import { checkGrounding } from "./grounding";
import { TOOLS_SPEC, runTool } from "./tools";
import type { Analysis, ChatMessage, ChatResult, ToolOutput } from "./types";

const BASE_URL = process.env.LLM_BASE_URL ?? "https://api.groq.com/openai/v1";
// Groq shut down llama-3.1-8b-instant on 2026-08-16; gpt-oss-20b is their
// named replacement and supports tool calling. Model IDs churn — check
// https://console.groq.com/docs/deprecations before trusting this default.
const MODEL = process.env.LLM_MODEL ?? "openai/gpt-oss-20b";
const API_KEY = process.env.LLM_API_KEY ?? "";
const TIMEOUT_MS = Number(process.env.LLM_TIMEOUT_S ?? "60") * 1000;
/** Small models occasionally loop on tools; this is the circuit breaker. */
const MAX_ROUNDS = Number(process.env.CHAT_MAX_ROUNDS ?? "6");

const SYSTEM = `You are the analysis console of a music-decomposition app,
answering questions about ONE analyzed track.

Hard rules:
- Every musical fact you state MUST come from a tool result in this
  conversation. Call tools first, answer after.
- The analysis covers: stems (vocals/drums/bass/other), instruments in the
  'other' stem, BPM, key, duration, and when each stem is audible. It does
  NOT cover lyrics, song title, artist, album, year, genre history or
  influences — if asked about those, your ENTIRE answer must be that the
  analysis doesn't include that. Never invent lyrical content or meaning;
  never guess an artist or title.
- Never name a specific instrument unless a tool returned it. If a user asks
  about an instrument the tools don't show, say it wasn't detected (it may
  still be there — the classifier isn't perfect — but you can only speak to
  what was detected).
- Confidence wording: >=0.9 state plainly; 0.7-0.89 "clearly"; 0.5-0.69
  "probably"; below 0.5 "faint hints, not confirmed". Don't show raw numbers
  unless asked.
- Write times as m:ss. Keep answers to 1-4 sentences, plain text, no markdown.`;

const REFUSAL =
  "I couldn't give a reliable answer to that from the analysis — try asking " +
  "about the instruments, stems, tempo or key.";

const UNREACHABLE =
  "The chat model isn't reachable right now — the analysis above is still all yours.";

// Groq's free tier allows 60 requests per DAY across the whole project, which a
// public demo can exhaust in an afternoon. Saying "unreachable" for that is
// misleading: nothing is broken, the day's budget is simply spent. Visitors
// deserve to know it comes back rather than assuming the demo is dead.
const QUOTA_EXHAUSTED =
  "This demo's daily AI quota is used up — it resets within 24 hours. " +
  "Everything above is precomputed and still works.";

/** True for a provider 429. The error carries the provider's own JSON body,
 *  which is the only place the distinction survives. */
function isRateLimited(error: unknown): boolean {
  return error instanceof Error && /provider returned 429\b/.test(error.message);
}

type ToolCall = {
  id?: string;
  function?: { name?: string; arguments?: string };
};

type ProviderMessage = {
  role: string;
  content?: string | null;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
};

async function callModel(messages: ProviderMessage[]): Promise<ProviderMessage> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const response = await fetch(`${BASE_URL}/chat/completions`, {
      method: "POST",
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(API_KEY ? { Authorization: `Bearer ${API_KEY}` } : {}),
      },
      body: JSON.stringify({
        model: MODEL,
        messages,
        temperature: 0.2,
        tools: TOOLS_SPEC,
      }),
    });
    if (!response.ok) {
      // Include the provider's own message: a bare status code cannot tell a
      // dead model ID from a bad key from a rate limit, and this failure is
      // invisible in the browser by design.
      const detail = await response.text().catch(() => "");
      throw new Error(`provider returned ${response.status}: ${detail.slice(0, 500)}`);
    }
    const data = await response.json();
    return data.choices[0].message as ProviderMessage;
  } finally {
    clearTimeout(timer);
  }
}

export async function chat(
  result: Analysis,
  messages: ChatMessage[],
): Promise<ChatResult> {
  const conversation: ProviderMessage[] = [
    { role: "system", content: SYSTEM },
    ...messages
      .filter((m) => m.role === "user" || m.role === "assistant")
      .map((m) => ({ role: m.role, content: String(m.content ?? "").slice(0, 2000) })),
  ];

  const trace: ChatResult["trace"] = [];
  const toolOutputs: ToolOutput[] = [];
  let retried = false;

  for (let round = 0; round < MAX_ROUNDS; round += 1) {
    let message: ProviderMessage;
    try {
      message = await callModel(conversation);
    } catch (error) {
      // Visitors get an honest one-liner, never a stack trace — but the cause
      // has to reach the server logs or a misconfigured model is undebuggable.
      console.error("[chat] provider call failed:", error);
      return {
        reply: isRateLimited(error) ? QUOTA_EXHAUSTED : UNREACHABLE,
        grounded: true,
        trace,
      };
    }

    const calls = message.tool_calls ?? [];
    if (calls.length > 0) {
      conversation.push({
        role: "assistant",
        content: message.content ?? "",
        tool_calls: calls,
      });
      for (const call of calls) {
        const name = call.function?.name ?? "";
        let args: Record<string, unknown> = {};
        try {
          args = JSON.parse(call.function?.arguments || "{}");
        } catch {
          args = {};                              // malformed args -> no args
        }
        const output = runTool(name, result, args);
        if (!output.error) toolOutputs.push(output);
        trace.push({ tool: name, args });
        conversation.push({
          role: "tool",
          tool_call_id: call.id ?? name,
          content: JSON.stringify(output),
        });
      }
      continue;
    }

    // Reasoning models may inline their thinking; drop it before checking.
    const reply = (message.content ?? "")
      .replace(/<think>[\s\S]*?<\/think>/g, "")
      .trim();
    if (!reply) continue;

    const { ok, violations } = checkGrounding(result, reply, toolOutputs);
    if (ok) return { reply, grounded: true, trace };

    if (!retried) {                               // one shot at self-repair
      retried = true;
      conversation.push({ role: "assistant", content: reply });
      conversation.push({
        role: "user",
        content:
          "Your answer contained claims not backed by the analysis (" +
          `${violations.join("; ")}). Answer again using only tool results; ` +
          "if the analysis can't answer, say so.",
      });
      continue;
    }
    return { reply: REFUSAL, grounded: false, trace, violations };
  }

  return { reply: REFUSAL, grounded: false, trace };
}
