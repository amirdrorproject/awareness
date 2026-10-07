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

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  createdAt: number;
  source?: MessageSource;
  internalAuditLog?: string;
  toolCalls?: ToolCall[];
}
