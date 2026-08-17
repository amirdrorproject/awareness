"use client";

import { useCallback, useState } from "react";
import type { Message, MessageSource } from "./types";

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isAssistantTyping, setIsAssistantTyping] = useState(false);
  const [useLangGraph, setUseLangGraph] = useState(true);

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
          let replyContent: string;
          let internalAuditLog: string | undefined;

          try {
            if (useLangGraph) {
              const res = await fetch("/api/chat-langgraph", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                credentials: "include",
                body: JSON.stringify({ message: trimmed }),
              });
              if (!res.ok) throw new Error(`Request failed: ${res.status}`);
              const data = await res.json();
              if (data.error) throw new Error(data.error);
              replyContent = data.response;
              internalAuditLog = data.internal_audit_log;
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
              replyContent = data.content;
            }
          } catch (err) {
            replyContent =
              err instanceof Error
                ? `שגיאה בפנייה לשרת: ${err.message}`
                : "שגיאה בפנייה לשרת.";
          }

          const reply: Message = {
            id: crypto.randomUUID(),
            role: "assistant",
            content: replyContent,
            createdAt: Date.now(),
            source,
            internalAuditLog,
          };
          setMessages((prev) => [...prev, reply]);
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
