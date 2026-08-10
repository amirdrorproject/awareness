"use client";

import { useEffect, useRef } from "react";
import type { Message } from "@/lib/chat/types";
import MessageBubble from "./MessageBubble";

export default function MessageList({ messages }: { messages: Message[] }) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
      {messages.length === 0 ? (
        <p className="text-center text-sm text-gray-400 mt-8">
          עדיין אין הודעות. כתבו משהו כדי להתחיל.
        </p>
      ) : (
        messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))
      )}
      <div ref={bottomRef} />
    </div>
  );
}
