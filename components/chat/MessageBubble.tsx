import ReactMarkdown from "react-markdown";
import type { Message, ToolCall } from "@/lib/chat/types";

// Debug note under a reply: what Claude searched for and which bank entries
// came back. Shown to whoever runs the simulation, not part of Claude's text.
function ToolCallNote({ call }: { call: ToolCall }) {
  const query = call.input.query;
  const hits = call.output?.hits ?? [];
  return (
    <div className="mt-2 border-t border-gray-300 pt-1.5 text-xs text-gray-500">
      <span>
        🔎 {call.name}
        {typeof query === "string" && <>: &quot;{query}&quot;</>}
      </span>
      {call.error ? (
        <div className="text-red-600">{call.error}</div>
      ) : hits.some((hit) => hit.score !== undefined) ? (
        <ul className="mt-0.5">
          {hits.map((hit) => (
            <li key={hit.title}>
              {hit.title} ({hit.score})
            </li>
          ))}
        </ul>
      ) : (
        <div className="mt-0.5">Full bank: {hits.length} entries</div>
      )}
    </div>
  );
}

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
        {message.toolCalls?.map((call, index) => (
          <ToolCallNote key={index} call={call} />
        ))}
      </div>
    </div>
  );
}
