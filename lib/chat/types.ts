export type MessageRole = "user" | "assistant";
export type MessageSource = "chat" | "langgraph";

export type KnowledgeMode = "none" | "search" | "full";

export interface ToolCall {
  name: string;
  input: Record<string, unknown>;
  // score is only present for search hits; the full bank has none.
  output?: { query?: string; hits?: { title: string; score?: number }[] };
  error?: string;
}

// How a reply came about, from the backend (api/engine.py): shown in the
// run-details panel under each reply. A failed turn carries only `error`.
export interface PromptVersion {
  source: "admin" | "file";
  created_at: string | null;
  note: string | null;
}

export interface Usage {
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens: number;
  cache_creation_input_tokens: number;
}

export interface RunTrace {
  model?: string;
  effort?: string;
  stop_reason?: string;
  // Where the turn ran: Vercel environment, branch and commit ("local" when run locally).
  runtime?: { env: string; branch: string | null; commit: string | null; dirty?: boolean | null };
  prompt?: PromptVersion;
  // The success-analysis module's version, when it was part of the prompt.
  // A failed turn only knows that it was on.
  success_module?: { source: "admin" | "file"; created_at: string | null; note: string | null } | true | null;
  // Same for the conversation-states module, under which the agent logs each turn.
  states_module?: { source: "admin" | "file"; created_at: string | null; note: string | null } | true | null;
  turn_log?: TurnLog | null;
  checks?: ReplyChecks;
  system_suffix?: string | null;
  memory_loaded?: boolean;
  knowledge_mode?: string;
  history_messages?: number;
  api_calls?: number;
  thinking?: string[];
  usage?: Usage;
  cost_usd?: number | null;
  latency_s?: number;
  error?: string;
}

// The agent's own log, written before its reply (api/turn_log.py). Any field
// can be missing if the agent left it out.
export interface TurnLog {
  state?: string;
  observed?: string;
  task?: string;
  move?: string;
  avoid?: string;
  // The hypothesis on the situation, or "אין".
  hypothesis?: string;
  // The earlier field (a hypothesis held back), in logs written before it changed.
  held_hypothesis?: string;
  // State letters found in `state` (א-ה); more than one means mixed.
  states: string[];
  // The move from the fixed list that `move` starts with, if any.
  move_type: string | null;
  raw: string;
}

// What the reply actually did, checked in code.
export interface ReplyChecks {
  words: number;
  flags: string[];
}

// How one research-observer call ran (api/observer.py).
export interface ObserverMeta {
  model: string;
  effort: string;
  prompt: PromptVersion;
  usage: Usage;
  cost_usd: number | null;
  latency_s: number;
}

// Call 1: the story, read before the agent's reply.
export interface ObserverStory {
  narrator: string;
  tension: string;
  opportunities: string[];
  meta: ObserverMeta;
}

// The turn's outcome, judged from the client's next message (in call 1 of the
// next turn), or marked when the conversation ended on this reply.
export interface TurnOutcome {
  focus: string;
  progress: string;
  offer: string;
  no_client_reply?: boolean;
  meta?: ObserverMeta;
}

// Call 2: the agent's choice.
export interface ObserverChoice {
  direction: string;
  multiple_questions: "כן" | "לא";
  questions_note: string;
  meta: ObserverMeta;
}

export type Classification = "לא זיהה" | "זיהה ולא נתן משקל" | "זיהה ובחר אחרת" | "לא ניתן לקבוע";

// Call 3: the opportunities from call 1 against the agent's log. `skipped`
// says why no call was made (no log, or no opportunities).
export interface ObserverComparison {
  items: { opportunity: string; classification: Classification; evidence: string }[];
  conclusion: string;
  skipped?: string;
  meta?: ObserverMeta;
}

export type Recognition = "זיהה" | "זיהה חלקית" | "לא זיהה" | "לא ניתן לקבוע";

// Whether the agent or the observer saw what Amir wrote about the story.
export interface HumanComparison {
  agent: Recognition;
  agent_evidence: string;
  observer: Recognition;
  observer_evidence: string;
  meta: ObserverMeta;
}

export type ObserverStage = "story" | "choice" | "comparison" | "done" | "error";

export interface ObserverRun {
  // The step running now; absent on a turn the observer didn't run on (it
  // can still get an outcome from the next turn).
  stage?: ObserverStage;
  error?: string;
  story?: ObserverStory;
  choice?: ObserverChoice;
  comparison?: ObserverComparison;
  outcome?: TurnOutcome;
  humanComparison?: HumanComparison;
  humanState?: "pending" | "done" | "error";
}

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  createdAt: number;
  source?: MessageSource;
  internalAuditLog?: string;
  toolCalls?: ToolCall[];
  trace?: RunTrace;
  // The saved reply's row id, so later steps can add to its trace.
  dbId?: string | null;
  observer?: ObserverRun;
  // What Amir sees in the story; text null means he skipped. Until it is set
  // the observer's results stay hidden.
  humanNote?: { text: string | null };
}
