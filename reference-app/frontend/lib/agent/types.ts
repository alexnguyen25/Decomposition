/** The analysis result the agent answers questions about. Mirrors the dict
 *  produced by src/main.py's analyze(). */
export type Instrument = { name: string; confidence: number };

export type Analysis = {
  duration_s?: number;
  bpm?: number | null;
  key?: string | null;
  presence?: Record<string, boolean>;
  instruments?: Instrument[];
  stems?: Record<string, string>;
  timeline?: {
    chunk_s?: number;
    instruments?: { t: number; top: Record<string, number> }[];
    stem_activity?: Record<string, number[]> & { hop_s?: number };
  };
};

export type ToolOutput = Record<string, unknown> & {
  error?: string;
  instruments?: Instrument[];
};

export type ChatMessage = { role: string; content?: string };

export type ChatResult = {
  reply: string;
  grounded: boolean;
  trace: { tool: string; args: Record<string, unknown> }[];
  violations?: string[];
};
