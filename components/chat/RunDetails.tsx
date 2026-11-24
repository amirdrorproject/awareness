import { useId, useState } from "react";
import type {
  HumanComparison,
  ObserverMeta,
  ObserverRun,
  PromptVersion,
  ReplyChecks,
  RunTrace,
  ToolCall,
  TurnLog,
  TurnOutcome,
} from "@/lib/chat/types";

// "Run details" under a reply. On top, the agent's own log, then the research
// observer (an independent reading of the story and of the agent's choice -
// never Claude's own reasoning), hidden until Amir writes what he sees or
// skips. Below, how the reply came about technically: Claude's thinking
// summary, the input, knowledge and tools, the model, and performance.
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

// Answers that mean "nothing found", shown dimmed.
const EMPTY_ANSWERS = ["לא ניתן לקבוע", "לא זוהה", "אין"];

function AnalysisField({ label, text }: { label: string; text: string }) {
  const missing = EMPTY_ANSWERS.includes(text.trim());
  return (
    <div>
      <div className="font-semibold text-gray-800">{label}</div>
      <p className={`whitespace-pre-wrap ${missing ? "text-gray-400" : "text-gray-800"}`}>{text}</p>
    </div>
  );
}

function formatVersion(version?: PromptVersion | null) {
  return version?.source === "admin" && version.created_at
    ? `גרסה מ-${formatDate(version.created_at)}${version.note ? ` (${version.note})` : ""}`
    : "הגרסה שבקוד";
}

function TurnLogBlock({ log, checks }: { log?: TurnLog | null; checks?: ReplyChecks }) {
  return (
    <div className="flex flex-col gap-2 rounded-lg bg-white/70 p-2.5">
      <p className="text-gray-500">לוג הסוכן: מה הסוכן עצמו כתב לפני התגובה.</p>
      {log ? (
        <>
          <AnalysisField label="מצב השיחה" text={log.state ?? "—"} />
          <AnalysisField label="מאפיינים נצפים" text={log.observed ?? "—"} />
          <AnalysisField label="משימה תקשורתית" text={log.task ?? "—"} />
          <AnalysisField label="תנועה שנבחרה" text={log.move ?? "—"} />
          <AnalysisField label="ממה להימנע" text={log.avoid ?? "—"} />
          {log.held_hypothesis !== undefined && log.hypothesis === undefined ? (
            <AnalysisField label="השערה שהוחזקה ולא הוצגה (שדה קודם)" text={log.held_hypothesis} />
          ) : (
            <AnalysisField label="השערה על המצב" text={log.hypothesis ?? "—"} />
          )}
        </>
      ) : (
        <p className="text-gray-500">לא נכתב לוג לתור הזה.</p>
      )}
      {checks && (
        <div>
          <div className="font-semibold text-gray-800">בפועל</div>
          <p>{checks.words} מילים</p>
          {checks.flags.map((flag) => (
            <p key={flag} className="text-amber-700">
              ⚠ {flag}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

// Amir's own reading of the turn, written before the observer's is shown.
function HumanNoteForm({ onSave }: { onSave: (text: string | null) => void }) {
  const [text, setText] = useState("");
  const id = useId();
  return (
    <div className="flex flex-col gap-1.5">
      <label className="font-semibold text-gray-800" htmlFor={id}>
        מה אני מזהה בסיפור?
      </label>
      <textarea
        id={id}
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={3}
        className="rounded border border-gray-300 bg-white p-1.5 text-xs text-gray-900"
      />
      <div className="flex gap-2">
        <button
          onClick={() => onSave(text)}
          disabled={!text.trim()}
          className="rounded-full border border-gray-300 bg-white px-3 py-0.5 hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-50"
        >
          שמירה
        </button>
        <button onClick={() => onSave(null)} className="rounded-full px-3 py-0.5 text-gray-500 hover:bg-gray-50">
          דילוג
        </button>
      </div>
      <p className="text-gray-500">ניתוח הצופה יוצג אחרי השמירה או הדילוג.</p>
    </div>
  );
}

function HumanComparisonBlock({
  note,
  comparison,
  state,
}: {
  note: string;
  comparison?: HumanComparison;
  state?: "pending" | "done" | "error";
}) {
  return (
    <div className="flex flex-col gap-2 rounded-md border border-blue-100 bg-blue-50/60 p-2">
      <AnalysisField label="מה זיהיתי" text={note} />
      {comparison ? (
        <>
          <AnalysisField label={`הסוכן: ${comparison.agent}`} text={comparison.agent_evidence} />
          <AnalysisField label={`הצופה: ${comparison.observer}`} text={comparison.observer_evidence} />
        </>
      ) : (
        <p className="text-gray-500">
          {state === "error"
            ? "ההשוואה נכשלה."
            : state === "pending"
              ? "משווה למה שזיהיתי..."
              : "ההשוואה תרוץ כשהצופה יסיים..."}
        </p>
      )}
    </div>
  );
}

function OutcomeBlock({ outcome, nextClientMessage }: { outcome?: TurnOutcome; nextClientMessage?: string }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="font-semibold text-gray-800">תוצאות התור (מתוך תגובת הלקוח הבאה)</div>
      {outcome?.no_client_reply ? (
        <p className="text-gray-500">אין תגובת לקוח</p>
      ) : !nextClientMessage ? (
        <p className="text-gray-500">הלקוח עוד לא הגיב.</p>
      ) : (
        <>
          <p className="whitespace-pre-wrap text-gray-600">תגובת הלקוח: {nextClientMessage}</p>
          {outcome ? (
            <>
              <AnalysisField label="האם הובהר המוקד" text={outcome.focus} />
              <AnalysisField label="האם החשיבה התקדמה" text={outcome.progress} />
              <AnalysisField label="מה הלקוח עשה עם מה שהוצע" text={outcome.offer} />
            </>
          ) : (
            <p className="text-gray-500">התוצאות ייכתבו בקריאה 1 של התור הבא.</p>
          )}
        </>
      )}
    </div>
  );
}

const STAGE_LABELS: Record<string, string> = {
  story: "קריאה 1: מנתח את הסיפור...",
  choice: "קריאה 2: מנתח את בחירת הסוכן...",
  comparison: "קריאה 3: משווה ללוג הסוכן...",
};

function ObserverBlock({
  observer,
  humanNote,
  onSaveNote,
  nextClientMessage,
}: {
  observer: ObserverRun;
  humanNote?: { text: string | null };
  onSaveNote: (text: string | null) => void;
  nextClientMessage?: string;
}) {
  const { story, choice, comparison } = observer;
  // Every observer call made for this turn (outcome's runs with the next turn).
  const metas = [story?.meta, choice?.meta, comparison?.meta, observer.humanComparison?.meta].filter(
    (meta): meta is ObserverMeta => Boolean(meta)
  );
  const cost = metas.reduce((sum, meta) => sum + (meta.cost_usd ?? 0), 0);
  // On a turn the observer didn't run on there is nothing to hide.
  const hidden = Boolean(observer.stage) && !humanNote;

  return (
    <div className="flex flex-col gap-2 rounded-lg bg-white/70 p-2.5">
      <p className="text-gray-500">
        הצופה המחקרי: ניתוח עצמאי של הסיפור ושל בחירת הסוכן. הוא לא רואה את סיכום החשיבה של Claude.
      </p>
      {hidden ? (
        <HumanNoteForm onSave={onSaveNote} />
      ) : (
        <>
          {humanNote?.text && (
            <HumanComparisonBlock note={humanNote.text} comparison={observer.humanComparison} state={observer.humanState} />
          )}

          {story && (
            <>
              <div className="font-semibold text-gray-500">קריאה 1: הסיפור (לפני תגובת הסוכן)</div>
              <AnalysisField label="מה ניתן לזהות אצל המספר" text={story.narrator} />
              <AnalysisField label="מתח בדברי הלקוח" text={story.tension} />
              <div>
                <div className="font-semibold text-gray-800">הזדמנויות לחשיבה</div>
                {story.opportunities.length > 0 ? (
                  <ul className="list-disc ps-4">
                    {story.opportunities.map((o) => (
                      <li key={o}>{o}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-gray-400">לא זוהו</p>
                )}
              </div>
            </>
          )}

          {choice && (
            <>
              <div className="font-semibold text-gray-500">קריאה 2: בחירת הסוכן</div>
              <AnalysisField label="מה נראה שהוביל לבחירת הכיוון" text={choice.direction} />
              <AnalysisField label={`יותר משאלה עצמאית אחת: ${choice.multiple_questions}`} text={choice.questions_note} />
            </>
          )}

          {comparison && (
            <>
              <div className="font-semibold text-gray-500">קריאה 3: השוואה ללוג הסוכן</div>
              {comparison.skipped ? (
                <p className="text-gray-400">{comparison.skipped}</p>
              ) : (
                <>
                  {comparison.items.map((item) => (
                    <AnalysisField key={item.opportunity} label={`${item.classification}: ${item.opportunity}`} text={item.evidence} />
                  ))}
                  <AnalysisField label="מסקנה מחקרית" text={comparison.conclusion} />
                </>
              )}
            </>
          )}

          {observer.stage && STAGE_LABELS[observer.stage] && <p className="text-gray-500">{STAGE_LABELS[observer.stage]}</p>}
          {observer.stage === "error" && <p className="text-red-600">הצופה נכשל: {observer.error}</p>}

          <OutcomeBlock outcome={observer.outcome} nextClientMessage={nextClientMessage} />

          {metas.length > 0 && (
            <p className="text-gray-500" dir="rtl">
              <span dir="ltr">
                {metas[0].model} · effort {metas[0].effort}
              </span>{" "}
              · פרומפט הצופה: {formatVersion(metas[0].prompt)} · {metas.length} קריאות · עלות משוערת: ${cost.toFixed(3)}
            </p>
          )}
        </>
      )}
    </div>
  );
}

export default function RunDetails({
  trace,
  toolCalls,
  observer,
  humanNote,
  onSaveNote,
  nextClientMessage,
}: {
  trace: RunTrace;
  toolCalls?: ToolCall[];
  observer?: ObserverRun;
  humanNote?: { text: string | null };
  onSaveNote: (text: string | null) => void;
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
            {trace.states_module && <TurnLogBlock log={trace.turn_log} checks={trace.checks} />}

            {observer && (
              <ObserverBlock
                observer={observer}
                humanNote={humanNote}
                onSaveNote={onSaveNote}
                nextClientMessage={nextClientMessage}
              />
            )}

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
                סביבה:{" "}
                {trace.runtime ? (
                  <span dir="ltr">
                    {[trace.runtime.env, trace.runtime.branch, trace.runtime.commit].filter(Boolean).join(" · ")}
                  </span>
                ) : (
                  "לא תועדה"
                )}
              </div>
              <div>פרומפט המנוע: {formatVersion(trace.prompt)}</div>
              <div>
                מודול ניתוח הצלחה:{" "}
                {!trace.success_module
                  ? "לא נטען"
                  : `נטען, ${formatVersion(trace.success_module === true ? null : trace.success_module)}. אין לו מנגנון הפעלה נפרד: כשהוא נטען, הוא חלק מהפרומפט בכל תור.`}
              </div>
              <div>
                מודול מצבי שיחה ולוג:{" "}
                {!trace.states_module ? "כבוי" : formatVersion(trace.states_module === true ? null : trace.states_module)}
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
