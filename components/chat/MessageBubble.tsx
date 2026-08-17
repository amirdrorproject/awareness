"use client";

import { useState } from "react";
import type { Message } from "@/lib/chat/types";

export default function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  const [showAuditLog, setShowAuditLog] = useState(false);

  const hasAuditLog = message.source === "langgraph" && !!message.internalAuditLog;

  return (
    <div className={`flex flex-col ${isUser ? "items-end" : "items-start"}`}>
      <div
        className={`max-w-[75%] rounded-2xl px-4 py-2 text-sm shadow-sm ${
          isUser
            ? "bg-blue-600 text-white rounded-br-sm"
            : "bg-gray-100 text-gray-900 rounded-bl-sm"
        }`}
      >
        <p className="whitespace-pre-wrap break-words">{message.content}</p>
      </div>

      {hasAuditLog && (
        <div className="mt-1 max-w-[75%]">
          <button
            onClick={() => setShowAuditLog((prev) => !prev)}
            className="text-xs text-gray-500 underline"
          >
            {showAuditLog ? "הסתר יומן פנימי" : "הצג יומן פנימי (LangGraph)"}
          </button>
          {showAuditLog && (
            <pre className="mt-1 whitespace-pre-wrap break-words rounded-lg border border-gray-200 bg-gray-50 p-2 text-xs text-gray-600">
              {message.internalAuditLog}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
