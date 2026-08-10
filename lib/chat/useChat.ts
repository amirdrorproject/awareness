"use client";

import { useCallback, useState } from "react";
import type { Message } from "./types";

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);

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
  }, []);

  return { messages, sendMessage };
}
