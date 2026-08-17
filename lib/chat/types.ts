export type MessageRole = "user" | "assistant";
export type MessageSource = "chat" | "langgraph";

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  createdAt: number;
  source?: MessageSource;
  internalAuditLog?: string;
}
