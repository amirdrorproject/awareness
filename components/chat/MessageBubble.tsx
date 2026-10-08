import ReactMarkdown from "react-markdown";
import type { Message } from "@/lib/chat/types";
import RunDetails from "./RunDetails";

export default function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        // dir="auto" lays each bubble out by its own text, so Hebrew reads
        // right-to-left with punctuation in place even though the page is LTR.
        dir="auto"
        className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-[15px] leading-relaxed shadow-sm ${
          isUser
            ? "bg-blue-600 text-white rounded-br-sm"
            : "bg-gray-100 text-gray-900 rounded-bl-sm"
        }`}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap break-words">{message.content}</p>
        ) : (
          <div className="space-y-2 break-words">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}
        {/* Shown to whoever runs the simulation, not part of Claude's reply. */}
        {message.trace && <RunDetails trace={message.trace} toolCalls={message.toolCalls} />}
      </div>
    </div>
  );
}
