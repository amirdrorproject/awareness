export type MessageRole = "user" | "assistant";
export type MessageSource = "chat" | "langgraph";

export type KnowledgeMode = "none" | "search" | "full";

export interface ToolCall {
  name: string;
  input: Record<string, unknown>;
  // score is only present for search hits; the full bank has none.
  output?: { query?: string; hits?: { title: string; score?: number }[] };
  error?: string;
}

// How a reply came about, from the backend (api/engine.py): shown in the
// run-details panel under each reply. A failed turn carries only `error`.
export interface RunTrace {
  model?: string;
  effort?: string;
  stop_reason?: string;
  prompt?: { source: "admin" | "file"; created_at: string | null; note: string | null };
  system_suffix?: string | null;
  memory_loaded?: boolean;
  knowledge_mode?: string;
  history_messages?: number;
  api_calls?: number;
  thinking?: string[];
  usage?: {
    input_tokens: number;
    output_tokens: number;
    cache_read_input_tokens: number;
    cache_creation_input_tokens: number;
  };
  cost_usd?: number | null;
  latency_s?: number;
  error?: string;
}

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  createdAt: number;
  source?: MessageSource;
  internalAuditLog?: string;
  toolCalls?: ToolCall[];
  trace?: RunTrace;
}
