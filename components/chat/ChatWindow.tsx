"use client";

import { useChat } from "@/lib/chat/useChat";
import MessageList from "./MessageList";
import MessageInput from "./MessageInput";

export default function ChatWindow() {
  const { messages, sendMessage } = useChat();

  return (
    <div className="flex h-[80vh] w-full max-w-lg flex-col overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-lg">
      <div className="border-b border-gray-200 px-4 py-3">
        <h1 className="text-lg font-semibold text-gray-900">
          Awareness Helper
        </h1>
      </div>
      <MessageList messages={messages} />
      <MessageInput onSend={sendMessage} />
    </div>
  );
}
