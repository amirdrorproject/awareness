"use client";

import { useCallback, useState } from "react";
import type { Message } from "./types";

const MOCK_ASSISTANT_REPLY =
  "תודה על ההודעה. זו תגובת דמה - החיבור האמיתי יתווסף בהמשך.";

function randomDelay() {
  return 700 + Math.random() * 300;
}

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

    setMessages((prev) => [...prev, message]);
    setIsAssistantTyping(true);

    setTimeout(() => {
      const reply: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: MOCK_ASSISTANT_REPLY,
        createdAt: Date.now(),
      };
      setMessages((prev) => [...prev, reply]);
      setIsAssistantTyping(false);
    }, randomDelay());
  }, []);

  return { messages, sendMessage, isAssistantTyping };
}
