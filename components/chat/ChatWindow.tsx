"use client";

import { useChat } from "@/lib/chat/useChat";
import type { KnowledgeMode } from "@/lib/chat/types";
import MessageList from "./MessageList";
import MessageInput from "./MessageInput";

export default function ChatWindow() {
  const {
    messages,
    sendMessage,
    isAssistantTyping,
    useLangGraph,
    setUseLangGraph,
    knowledgeMode,
    setKnowledgeMode,
  } = useChat();

  return (
    <div className="flex h-[80vh] w-full max-w-2xl flex-col overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-lg">
      <div className="flex items-center justify-between border-b border-gray-200 px-4 py-3">
        <h1 className="text-lg font-semibold text-gray-900">
          Awareness Helper
        </h1>
        <div className="flex flex-col items-end gap-1">
          <label className="flex items-center gap-1.5 text-xs text-gray-500">
            Expression bank
            <select
              value={knowledgeMode}
              disabled={useLangGraph}
              onChange={(e) => setKnowledgeMode(e.target.value as KnowledgeMode)}
              className="rounded border border-gray-300 bg-white px-1 py-0.5 text-xs"
            >
              {/* "search" (retrieval) stays available to the API and the
                  comparison script, but full bank was chosen over it. */}
              <option value="full">Full bank</option>
              <option value="none">None</option>
            </select>
          </label>
          <label className="flex items-center gap-1.5 text-xs text-gray-500">
            <input
              type="checkbox"
              checked={useLangGraph}
              onChange={(e) => setUseLangGraph(e.target.checked)}
            />
            Use LangGraph (experimental)
          </label>
        </div>
      </div>
      <MessageList messages={messages} isAssistantTyping={isAssistantTyping} />
      <MessageInput onSend={sendMessage} />
    </div>
  );
}
