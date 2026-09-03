/** POST /api/chat/{jobId} — grounded Q&A about one analyzed track.
 *
 *  Stateless on purpose: the client sends the whole (capped) history each time,
 *  so there is no server-side session store to keep, expire or scale.
 *
 *  Only example tracks are answerable in production. A real upload's result
 *  lives in the local Python backend's memory, and in that setup next.config.ts
 *  proxies /api/jobs there — but chat is always served here, so an unknown id
 *  gets an honest 404 rather than a fabricated answer.
 */

import { readFile } from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";

import { chat } from "@/lib/agent";
import type { Analysis, ChatMessage } from "@/lib/agent/types";

const MAX_MESSAGES = 24;
const MAX_CHARS = 24_000;

async function findExample(jobId: string): Promise<Analysis | null> {
  if (!jobId.startsWith("ex_")) return null;
  const manifestPath = path.join(process.cwd(), "public", "examples", "manifest.json");
  try {
    const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
    const example = manifest.find((e: { id: string }) => e.id === jobId);
    return example ? (example.result as Analysis) : null;
  } catch {
    return null;
  }
}

export async function POST(
  request: Request,
  { params }: { params: Promise<{ jobId: string }> },
) {
  const { jobId } = await params;

  const result = await findExample(jobId);
  if (!result) {
    return NextResponse.json(
      {
        detail:
          "Chat is available for the example tracks. Uploaded tracks are " +
          "analysed locally — see the README.",
      },
      { status: 404 },
    );
  }

  let body: { messages?: ChatMessage[] };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ detail: "Invalid JSON body." }, { status: 400 });
  }

  let messages = body.messages;
  if (!Array.isArray(messages) || messages.length === 0) {
    return NextResponse.json(
      { detail: "messages must be a non-empty list." },
      { status: 400 },
    );
  }
  if (messages.length > MAX_MESSAGES) messages = messages.slice(-MAX_MESSAGES);
  const size = messages.reduce((sum, m) => sum + String(m.content ?? "").length, 0);
  if (size > MAX_CHARS) {
    return NextResponse.json(
      { detail: "Conversation too long — refresh to reset." },
      { status: 400 },
    );
  }

  return NextResponse.json(await chat(result, messages));
}
