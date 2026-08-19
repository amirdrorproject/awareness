"use client";

import { useCallback, useRef, useState } from "react";
import type { Message } from "./types";

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isAssistantTyping, setIsAssistantTyping] = useState(false);

  const threadIdRef = useRef<string | null>(null);
  if (threadIdRef.current === null) {
    threadIdRef.current = crypto.randomUUID();
  }

  const sendMessage = useCallback((content: string) => {
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
        let replyContent: string;
        let internalAuditLog: string | undefined;

        try {
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
          replyContent = data.response;
          internalAuditLog = data.internal_audit_log;
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
          source: "langgraph",
          internalAuditLog,
        };
        setMessages((prev) => [...prev, reply]);
        setIsAssistantTyping(false);
      })();

      return next;
    });
  }, []);

  return { messages, sendMessage, isAssistantTyping };
}
