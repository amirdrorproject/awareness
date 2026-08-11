"use client";

import { useCallback, useState } from "react";
import type { Message } from "./types";

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isAssistantTyping, setIsAssistantTyping] = useState(false);

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
        try {
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
        };
        setMessages((prev) => [...prev, reply]);
        setIsAssistantTyping(false);
      })();

      return next;
    });
  }, []);

  return { messages, sendMessage, isAssistantTyping };
}
