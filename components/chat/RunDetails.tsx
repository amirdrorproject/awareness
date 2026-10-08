import type { RunTrace, ToolCall } from "@/lib/chat/types";

// "Run details" under a reply: how it came about, layer by layer - the input
// Claude got, knowledge and tools, the model, and performance - plus the
// summary of Claude's thinking. Collapsed by default so the conversation
// still reads as a conversation.

function formatDate(iso: string) {
  return new Date(iso).toLocaleString("he-IL", { dateStyle: "short", timeStyle: "short" });
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-2">
      <span className="w-20 shrink-0 font-medium text-gray-600">{label}</span>
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  );
}

function ToolCallLine({ call }: { call: ToolCall }) {
  const query = call.input.query;
  const hits = call.output?.hits ?? [];
  const scored = hits.some((hit) => hit.score !== undefined);
  return (
    <div>
      <span dir="ltr">{call.name}</span>
      {typeof query === "string" && <>: &quot;{query}&quot;</>}
      {call.error ? (
        <div className="text-red-600">{call.error}</div>
      ) : scored ? (
        <ul>
          {hits.map((hit) => (
            <li key={hit.title}>
              {hit.title} ({hit.score})
            </li>
          ))}
        </ul>
      ) : (
        <span> · הבנק המלא, {hits.length} רשומות</span>
      )}
    </div>
  );
}

export default function RunDetails({ trace, toolCalls }: { trace: RunTrace; toolCalls?: ToolCall[] }) {
  const calls = toolCalls ?? [];
  const usage = trace.usage;
  const headline = trace.error
    ? "הריצה נכשלה"
    : [
        trace.latency_s !== undefined && `${trace.latency_s} שנ'`,
        calls.length > 0 ? `${calls.length} כלים` : "בלי כלים",
        trace.memory_loaded && "עם זיכרון",
      ]
        .filter(Boolean)
        .join(" · ");

  return (
    <details dir="rtl" className="mt-2 border-t border-gray-300 pt-1.5 text-xs text-gray-700">
      <summary className="cursor-pointer select-none text-gray-500">פרטי ריצה · {headline}</summary>
      <div className="mt-2 flex flex-col gap-2">
        {trace.error ? (
          <Row label="שגיאה">
            <span className="text-red-600">{trace.error}</span>
          </Row>
        ) : (
          <>
            <Row label="חשיבה">
              {trace.thinking && trace.thinking.length > 0 ? (
                trace.thinking.map((text, i) => (
                  <p key={i} className="whitespace-pre-wrap">
                    {text}
                  </p>
                ))
              ) : (
                <span className="text-gray-500">Claude לא החזיר סיכום חשיבה לתשובה הזו.</span>
              )}
            </Row>

            <Row label="קלט">
              <div>
                פרומפט המנוע:{" "}
                {trace.prompt?.source === "admin" && trace.prompt.created_at
                  ? `גרסה מ-${formatDate(trace.prompt.created_at)}${trace.prompt.note ? ` (${trace.prompt.note})` : ""}`
                  : "הגרסה שבקוד"}
              </div>
              <div>היסטוריה: {trace.history_messages} הודעות</div>
              {trace.system_suffix ? (
                <details>
                  <summary className="cursor-pointer">זיכרון משיחות קודמות: כן (פתיחה)</summary>
                  <p className="mt-1 whitespace-pre-wrap text-gray-600">{trace.system_suffix}</p>
                </details>
              ) : (
                <div>זיכרון משיחות קודמות: לא</div>
              )}
            </Row>

            <Row label="ידע וכלים">
              <div>
                בנק הביטויים: {trace.knowledge_mode === "none" ? "כבוי" : trace.knowledge_mode === "full" ? "בנק מלא" : "חיפוש"}
              </div>
              {calls.length > 0 ? calls.map((call, i) => <ToolCallLine key={i} call={call} />) : <div>לא הופעל כלי</div>}
            </Row>

            <Row label="מודל">
              <span dir="ltr">
                {trace.model} · effort {trace.effort}
              </span>
              {trace.api_calls && trace.api_calls > 1 && <span> · {trace.api_calls} קריאות</span>}
            </Row>

            <Row label="ביצועים">
              <div>{trace.latency_s} שניות</div>
              {usage && (
                <div>
                  טוקנים: {(usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens).toLocaleString()} בקלט
                  {usage.cache_read_input_tokens > 0 && ` (${usage.cache_read_input_tokens.toLocaleString()} מהמטמון)`},{" "}
                  {usage.output_tokens.toLocaleString()} בפלט
                </div>
              )}
              {trace.cost_usd != null && <div>עלות משוערת: ${trace.cost_usd.toFixed(3)}</div>}
            </Row>
          </>
        )}
      </div>
    </details>
  );
}
