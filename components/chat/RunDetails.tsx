import type { AnalysisState, ReplyChecks, RunTrace, ToolCall, TurnAnalysis, TurnLog } from "@/lib/chat/types";

// "Run details" under a reply. On top, the turn in the four parts Amir reviews:
// understanding, focus and rationale (an observer's reading of the
// conversation - not Claude's own reasoning) and the reply itself. Below, how
// it came about technically: Claude's thinking summary, the input, knowledge
// and tools, the model, and performance.
// Collapsed by default so the conversation still reads as a conversation.

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

const NOT_INFERABLE = "לא ניתן להסיק";

function AnalysisField({ label, text }: { label: string; text: string }) {
  const missing = text.trim() === NOT_INFERABLE;
  return (
    <div>
      <div className="font-semibold text-gray-800">{label}</div>
      <p className={`whitespace-pre-wrap ${missing ? "text-gray-400" : "text-gray-800"}`}>{text}</p>
    </div>
  );
}

function TurnAnalysisBlock({
  analysis,
  state,
  reply,
}: {
  analysis?: TurnAnalysis;
  state?: AnalysisState;
  reply: string;
}) {
  return (
    <div className="flex flex-col gap-2 rounded-lg bg-white/70 p-2.5">
      <p className="text-gray-500">
        ניתוח של צופה: פירוש של מה שעולה מהשיחה ומהתשובה, לא ההסבר של הסוכן עצמו. ההסבר של Claude נמצא ב&quot;סיכום החשיבה&quot; למטה.
      </p>
      {state === "done" && analysis ? (
        <>
          <AnalysisField label="הבנה" text={analysis.understanding} />
          <AnalysisField label="מוקד" text={analysis.focus} />
          <AnalysisField label="היגיון הבחירה" text={analysis.rationale} />
        </>
      ) : (
        <p className="text-gray-500">
          {state === "pending" ? "מנתח את התור..." : "הניתוח נכשל. סיכום החשיבה של Claude מופיע למטה."}
        </p>
      )}
      <AnalysisField label="תגובה" text={reply} />
    </div>
  );
}

// The progress sign of each state, from Amir's strategy (prompts/module_states.md),
// shown next to the client's next message so the two can be compared.
const PROGRESS_SIGNS: Record<string, string> = {
  א: "נוצר מוקד ראשוני או שהלקוח מוסיף מידע בעל משמעות מרצונו.",
  ב: "הסיפור הגיע לנקודה שבה אפשר להבין את עניינו המרכזי או לברר עם הלקוח מה חשוב לו.",
  ג: "הלקוח מצליח לבחור עניין שחשוב לו להתחיל ממנו, גם אם יתר המורכבות נותרת פתוחה.",
  ד: "מתגבש מוקד שהלקוח מאשר, מתקן או מנסח בעצמו.",
  ה: "הלקוח מפתח הבנה, מזהה אפשרות, מחדד החלטה או מגדיר פעולה בעלת משמעות עבורו.",
};

function TurnLogBlock({
  log,
  checks,
  nextClientMessage,
}: {
  log?: TurnLog | null;
  checks?: ReplyChecks;
  nextClientMessage?: string;
}) {
  return (
    <div className="flex flex-col gap-2 rounded-lg bg-white/70 p-2.5">
      <p className="text-gray-500">לוג הסוכן: מה הסוכן עצמו כתב לפני התגובה, ומה התגובה עשתה בפועל.</p>
      {log ? (
        <>
          <AnalysisField label="מצב השיחה" text={log.state ?? "—"} />
          <AnalysisField label="מאפיינים נצפים" text={log.observed ?? "—"} />
          <AnalysisField label="משימה תקשורתית" text={log.task ?? "—"} />
          <AnalysisField label="תנועה שנבחרה" text={log.move ?? "—"} />
          <AnalysisField label="ממה להימנע" text={log.avoid ?? "—"} />
          <AnalysisField label="השערה שהוחזקה ולא הוצגה" text={log.held_hypothesis ?? "—"} />
        </>
      ) : (
        <p className="text-gray-500">לא נכתב לוג לתור הזה.</p>
      )}
      {checks && (
        <div>
          <div className="font-semibold text-gray-800">בפועל</div>
          <p>
            {checks.words} מילים · {checks.questions} סימני שאלה
          </p>
          {checks.flags.map((flag) => (
            <p key={flag} className="text-amber-700">
              ⚠ {flag}
            </p>
          ))}
        </div>
      )}
      {log && log.states.length > 0 && (
        <div>
          <div className="font-semibold text-gray-800">התקדמות</div>
          {log.states.map((letter) => (
            <p key={letter} className="text-gray-600">
              סימן להתקדמות במצב {letter}׳: {PROGRESS_SIGNS[letter]}
            </p>
          ))}
          <p className="whitespace-pre-wrap text-gray-800">
            {nextClientMessage ? `תגובת הלקוח: ${nextClientMessage}` : "הלקוח עוד לא הגיב."}
          </p>
        </div>
      )}
    </div>
  );
}

export default function RunDetails({
  trace,
  toolCalls,
  reply,
  analysis,
  analysisState,
  nextClientMessage,
}: {
  trace: RunTrace;
  toolCalls?: ToolCall[];
  reply: string;
  analysis?: TurnAnalysis;
  analysisState?: AnalysisState;
  nextClientMessage?: string;
}) {
  const calls = toolCalls ?? [];
  const usage = trace.usage;
  const headline = trace.error
    ? "הריצה נכשלה"
    : [
        trace.latency_s !== undefined && `${trace.latency_s} שנ'`,
        calls.length > 0 ? `${calls.length} כלים` : "בלי כלים",
        trace.success_module && "עם מודול הצלחה",
        trace.turn_log?.states.length && `מצב ${trace.turn_log.states.map((l) => `${l}׳`).join("+")}`,
        trace.checks && trace.checks.flags.length > 0 && `⚠ ${trace.checks.flags.length}`,
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
            {trace.states_module && (
              <TurnLogBlock log={trace.turn_log} checks={trace.checks} nextClientMessage={nextClientMessage} />
            )}

            <TurnAnalysisBlock analysis={analysis} state={analysisState} reply={reply} />

            <Row label="סיכום החשיבה של Claude">
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
              <div>
                מודול ניתוח הצלחה:{" "}
                {!trace.success_module
                  ? "כבוי"
                  : trace.success_module !== true && trace.success_module.source === "admin" && trace.success_module.created_at
                    ? `גרסה מ-${formatDate(trace.success_module.created_at)}${trace.success_module.note ? ` (${trace.success_module.note})` : ""}`
                    : "הגרסה שבקוד"}
              </div>
              <div>
                מודול מצבי שיחה ולוג:{" "}
                {!trace.states_module
                  ? "כבוי"
                  : trace.states_module !== true && trace.states_module.source === "admin" && trace.states_module.created_at
                    ? `גרסה מ-${formatDate(trace.states_module.created_at)}${trace.states_module.note ? ` (${trace.states_module.note})` : ""}`
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
