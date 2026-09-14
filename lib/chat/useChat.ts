"use client";

import { useCallback, useRef, useState } from "react";
import type { Message, MessageSource } from "./types";

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isAssistantTyping, setIsAssistantTyping] = useState(false);
  const [useLangGraph, setUseLangGraph] = useState(true);

  const threadIdRef = useRef<string | null>(null);
  if (threadIdRef.current === null) {
    threadIdRef.current = crypto.randomUUID();
  }

  const previousAuditLogRef = useRef("");

  const sendMessage = useCallback(
    (content: string) => {
      const trimmed = content.trim();
      if (!trimmed) return;

      const message: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content: trimmed,
        createdAt: Date.now(),
      };

      setMessages((prev) => {
        const next = [...prev, message];

        setIsAssistantTyping(true);

        (async () => {
          const source: MessageSource = useLangGraph ? "langgraph" : "chat";
          // A turn can produce zero, one, or (e.g. classify_professional_content's
          // disclaimer followed by the normal reply) more than one message, in order.
          let replyContents: string[] = [];
          let internalAuditLog: string | undefined;

          try {
            if (useLangGraph) {
              const res = await fetch("/api/chat-langgraph", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  message: trimmed,
                  thread_id: threadIdRef.current,
                }),
              });
              if (!res.ok) throw new Error(`Request failed: ${res.status}`);
              const data = await res.json();
              if (data.error) throw new Error(data.error);

              const fullAuditLog: string | undefined = data.internal_audit_log;
              if (typeof fullAuditLog === "string") {
                internalAuditLog = fullAuditLog
                  .slice(previousAuditLogRef.current.length)
                  .trimStart();
                previousAuditLogRef.current = fullAuditLog;
              }

              // A turn can legitimately produce no reply (e.g. classify_direction_choice,
              // or reaching END directly on emotional_clear/practical_clear) - it only
              // classified and logged internally. Nothing to show, but not an error either.
              // responses (plural) carries every message the turn produced, in order;
              // fall back to the single response field if it's ever absent.
              replyContents = Array.isArray(data.responses)
                ? data.responses
                : data.response
                  ? [data.response]
                  : [];
            } else {
              const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  messages: next.map(({ role, content }) => ({ role, content })),
                }),
              });
              if (!res.ok) throw new Error(`Request failed: ${res.status}`);
              const data = await res.json();
              replyContents = data.content ? [data.content] : [];
            }
          } catch (err) {
            replyContents = [
              err instanceof Error
                ? `שגיאה בפנייה לשרת: ${err.message}`
                : "שגיאה בפנייה לשרת.",
            ];
          }

          if (replyContents.length > 0) {
            const replies: Message[] = replyContents.map((content, index) => ({
              id: crypto.randomUUID(),
              role: "assistant",
              content,
              createdAt: Date.now(),
              source,
              // Audit log diff belongs with the turn's last message only, to
              // avoid showing the same debug info under multiple bubbles.
              internalAuditLog: index === replyContents.length - 1 ? internalAuditLog : undefined,
            }));
            setMessages((prev) => [...prev, ...replies]);
          }
          setIsAssistantTyping(false);
        })();

        return next;
      });
    },
    [useLangGraph]
  );

  return {
    messages,
    sendMessage,
    isAssistantTyping,
    useLangGraph,
    setUseLangGraph,
  };
}
