"use client";

import { useEffect, useRef } from "react";
import type { Message } from "@/lib/chat/types";
import MessageBubble from "./MessageBubble";
import TypingIndicator from "./TypingIndicator";

export default function MessageList({
  messages,
  isAssistantTyping = false,
}: {
  messages: Message[];
  isAssistantTyping?: boolean;
}) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isAssistantTyping]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
      {messages.length === 0 && !isAssistantTyping ? (
        <p className="text-center text-sm text-gray-400 mt-8">
          עדיין אין הודעות. כתבו משהו כדי להתחיל.
        </p>
      ) : (
        messages.map((message, index) => (
          <MessageBubble
            key={message.id}
            message={message}
            // The client's answer to this reply, to judge whether it moved things on.
            nextClientMessage={messages.slice(index + 1).find((m) => m.role === "user")?.content}
          />
        ))
      )}
      {isAssistantTyping && <TypingIndicator />}
      <div ref={bottomRef} />
    </div>
  );
}
