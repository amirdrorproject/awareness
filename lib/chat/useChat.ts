"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  HumanComparison,
  KnowledgeMode,
  Message,
  MessageSource,
  ObserverChoice,
  ObserverComparison,
  ObserverRun,
  ObserverStory,
  RunTrace,
  ToolCall,
  TurnOutcome,
} from "./types";

const CLIENT_NAME_KEY = "awareness.clientName";

async function postJson(url: string, body: unknown) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isAssistantTyping, setIsAssistantTyping] = useState(false);
  const [useLangGraph, setUseLangGraph] = useState(false);
  const [knowledgeMode, setKnowledgeMode] = useState<KnowledgeMode>("full");
  // Whether the success-analysis module is added to the engine prompt.
  // On by default on this branch, which exists to try the module out.
  const [successModule, setSuccessModule] = useState(true);
  // Whether the conversation-states module is added, with its per-turn log.
  // On by default on this branch, which exists to try the log out.
  const [statesModule, setStatesModule] = useState(true);
  // The research switch: whether the observer reads each turn (three more
  // model calls per turn). On by default on this branch, which exists to try it.
  const [researchObserver, setResearchObserver] = useState(true);
  // Who the client is: conversations under the same name share memory.
  // Remembered in this browser so a simulation can be resumed after a refresh.
  const [clientName, setClientNameState] = useState("");
  // Whether the last reply was written with memory from earlier conversations.
  const [memoryLoaded, setMemoryLoaded] = useState<boolean | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [isEnding, setIsEnding] = useState(false);

  useEffect(() => {
    try {
      setClientNameState(localStorage.getItem(CLIENT_NAME_KEY) ?? "");
    } catch {
      // Storage can be unavailable (private mode); the field just starts empty.
    }
  }, []);

  const setClientName = useCallback((name: string) => {
    setClientNameState(name);
    try {
      localStorage.setItem(CLIENT_NAME_KEY, name);
    } catch {
      // Not remembered across refreshes, but still used for this session.
    }
  }, []);

  // Mirrors `messages` so sendMessage can read the current transcript without
  // doing its fetch inside a setState updater - React may run updaters twice
  // (Strict Mode in dev), which sent every message to the backend twice.
  const messagesRef = useRef<Message[]>([]);

  const threadIdRef = useRef<string | null>(null);
  if (threadIdRef.current === null) {
    threadIdRef.current = crypto.randomUUID();
  }

  const previousAuditLogRef = useRef("");

  const updateMessage = useCallback((id: string, changes: Partial<Message>) => {
    messagesRef.current = messagesRef.current.map((m) => (m.id === id ? { ...m, ...changes } : m));
    setMessages(messagesRef.current);
  }, []);

  const updateObserver = useCallback((id: string, changes: Partial<ObserverRun>) => {
    messagesRef.current = messagesRef.current.map((m) =>
      m.id === id ? { ...m, observer: { ...m.observer, ...changes } } : m
    );
    setMessages(messagesRef.current);
  }, []);

  // Once Amir has written what he sees (not skipped) and the observer is
  // done, a further call checks whether the agent or the observer saw it.
  const compareWithNote = useCallback(
    async (id: string, history: Message[]) => {
      const reply = messagesRef.current.find((m) => m.id === id);
      const observer = reply?.observer;
      const note = reply?.humanNote?.text;
      if (!reply || !note || observer?.stage !== "done" || observer.humanState) return;
      if (!observer.story || !observer.choice || !observer.comparison) return;
      updateObserver(id, { humanState: "pending" });
      try {
        const data = await postJson("/api/observer/human-comparison", {
          note,
          messages: history.map(({ role, content }) => ({ role, content })),
          reply: reply.content,
          turn_log: reply.trace?.turn_log ?? null,
          story: observer.story,
          choice: observer.choice,
          comparison: observer.comparison,
          message_id: reply.dbId ?? null,
        });
        updateObserver(id, { humanComparison: data as HumanComparison, humanState: "done" });
      } catch {
        updateObserver(id, { humanState: "error" });
      }
    },
    [updateObserver]
  );

  // The research observer, once a reply is on screen: three calls in turn,
  // each given the earlier results (api/observer.py). Runs in the background,
  // so the reply never waits on it. `history` ends with the client's message
  // and carries no logs.
  const runObserver = useCallback(
    async (reply: Message, history: Message[]) => {
      const previous = history.length >= 2 && history[history.length - 2].role === "assistant" ? history[history.length - 2] : null;
      const messages = history.map(({ role, content }) => ({ role, content }));
      const messageId = reply.dbId ?? null;
      try {
        updateObserver(reply.id, { stage: "story" });
        const first = await postJson("/api/observer/story", {
          messages,
          message_id: messageId,
          previous_message_id: previous?.dbId ?? null,
        });
        const story = first.story as ObserverStory;
        // The previous turn's outcome goes on the previous reply.
        if (previous && first.outcome) updateObserver(previous.id, { outcome: first.outcome as TurnOutcome });
        updateObserver(reply.id, { story, stage: "choice" });

        const choice = (await postJson("/api/observer/choice", {
          messages,
          reply: reply.content,
          story,
          message_id: messageId,
        })) as ObserverChoice;
        updateObserver(reply.id, { choice, stage: "comparison" });

        const comparison = (await postJson("/api/observer/comparison", {
          story,
          choice,
          turn_log: reply.trace?.turn_log ?? null,
          message_id: messageId,
        })) as ObserverComparison;
        updateObserver(reply.id, { comparison, stage: "done" });
      } catch (err) {
        updateObserver(reply.id, { stage: "error", error: err instanceof Error ? err.message : String(err) });
        return;
      }
      compareWithNote(reply.id, history);
    },
    [updateObserver, compareWithNote]
  );

  // Amir's own reading of a turn, or his choice to skip it (text null). Only
  // then are the observer's results shown.
  const saveHumanNote = useCallback(
    (id: string, text: string | null) => {
      const index = messagesRef.current.findIndex((m) => m.id === id);
      if (index === -1) return;
      const note = text?.trim() || null;
      updateMessage(id, { humanNote: { text: note } });
      const reply = messagesRef.current[index];
      postJson("/api/observer/note", { message_id: reply.dbId ?? null, note }).catch(() => {
        // Still shown here; only the saved copy is missing.
      });
      compareWithNote(id, messagesRef.current.slice(0, index));
    },
    [updateMessage, compareWithNote]
  );

  const appendMessages = useCallback((added: Message[]) => {
    messagesRef.current = [...messagesRef.current, ...added];
    setMessages(messagesRef.current);
  }, []);

  const sendMessage = useCallback(
    (content: string) => {
      const trimmed = content.trim();
      if (!trimmed) return;

      const message: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content: trimmed,
        createdAt: Date.now(),
      };

      appendMessages([message]);
      const next = messagesRef.current;
      setIsAssistantTyping(true);

      (async () => {
        const source: MessageSource = useLangGraph ? "langgraph" : "chat";
        // A turn can produce zero, one, or (e.g. classify_professional_content's
        // disclaimer followed by the normal reply) more than one message, in order.
        let replyContents: string[] = [];
        let internalAuditLog: string | undefined;
        let toolCalls: ToolCall[] | undefined;
        let trace: RunTrace | undefined;
        let messageDbId: string | null = null;

        try {
          if (useLangGraph) {
            const res = await fetch("/api/chat-langgraph", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                message: trimmed,
                thread_id: threadIdRef.current,
              }),
            });
            if (!res.ok) throw new Error(`Request failed: ${res.status}`);
            const data = await res.json();
            if (data.error) throw new Error(data.error);

            const fullAuditLog: string | undefined = data.internal_audit_log;
            if (typeof fullAuditLog === "string") {
              internalAuditLog = fullAuditLog
                .slice(previousAuditLogRef.current.length)
                .trimStart();
              previousAuditLogRef.current = fullAuditLog;
            }

            // A turn can legitimately produce no reply (e.g. classify_direction_choice,
            // or reaching END directly on emotional_clear/practical_clear) - it only
            // classified and logged internally. Nothing to show, but not an error either.
            // responses (plural) carries every message the turn produced, in order;
            // fall back to the single response field if it's ever absent.
            replyContents = Array.isArray(data.responses)
              ? data.responses
              : data.response
                ? [data.response]
                : [];
          } else {
            const res = await fetch("/api/chat", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                // A reply's log goes back with it, so the agent sees its own
                // earlier logs (and the hypotheses it held back).
                messages: next.map(({ role, content, trace }) => ({
                  role,
                  content: trace?.turn_log ? `<log>\n${trace.turn_log.raw}\n</log>\n\n${content}` : content,
                })),
                conversation_id: threadIdRef.current,
                knowledge_mode: knowledgeMode,
                client_name: clientName.trim() || null,
                success_module: successModule,
                states_module: statesModule,
              }),
            });
            if (!res.ok) throw new Error(`Request failed: ${res.status}`);
            const data = await res.json();
            replyContents = data.content ? [data.content] : [];
            toolCalls = data.tool_calls;
            trace = data.trace;
            messageDbId = data.message_id ?? null;
            setMemoryLoaded(Boolean(data.memory_loaded));
          }
        } catch (err) {
          replyContents = [
            err instanceof Error
              ? `שגיאה בפנייה לשרת: ${err.message}`
              : "שגיאה בפנייה לשרת.",
          ];
        }

        if (replyContents.length > 0) {
          const replies: Message[] = replyContents.map((content, index) => ({
            id: crypto.randomUUID(),
            role: "assistant",
            content,
            createdAt: Date.now(),
            source,
            // Audit log diff belongs with the turn's last message only, to
            // avoid showing the same debug info under multiple bubbles.
            internalAuditLog: index === replyContents.length - 1 ? internalAuditLog : undefined,
            toolCalls: index === replyContents.length - 1 ? toolCalls : undefined,
            trace: index === replyContents.length - 1 ? trace : undefined,
            dbId: index === replyContents.length - 1 ? messageDbId : undefined,
          }));
          appendMessages(replies);
          const last = replies[replies.length - 1];
          if (researchObserver && last.trace && !last.trace.error) runObserver(last, next);
        }
        setIsAssistantTyping(false);
      })();
    },
    [useLangGraph, knowledgeMode, successModule, statesModule, researchObserver, clientName, appendMessages, runObserver]
  );

  // Ends the conversation: Claude summarises what was established and it is
  // saved under the client's name, then a fresh conversation starts. The
  // client's next conversation is answered with that memory.
  const endConversation = useCallback(async () => {
    const name = clientName.trim();
    if (!name) {
      setStatus("כדי לשמור את השיחה צריך למלא שם לקוח.");
      return;
    }
    if (messagesRef.current.length === 0) {
      setStatus("אין עדיין מה לשמור.");
      return;
    }
    setIsEnding(true);
    setStatus(null);
    // A conversation that ends on the agent's reply: that turn's outcome is
    // "no client reply", with no inference about progress.
    const last = messagesRef.current[messagesRef.current.length - 1];
    if (last.role === "assistant" && last.observer?.stage && last.dbId) {
      postJson("/api/observer/no-client-reply", { message_id: last.dbId }).catch(() => {
        // Only the saved mark is missing; the conversation is saved either way.
      });
    }
    try {
      const res = await fetch("/api/conversation/end", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conversation_id: threadIdRef.current,
          client_name: name,
          messages: messagesRef.current.map(({ role, content }) => ({ role, content })),
        }),
      });
      if (!res.ok) throw new Error(`Request failed: ${res.status}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      messagesRef.current = [];
      setMessages([]);
      threadIdRef.current = crypto.randomUUID();
      previousAuditLogRef.current = "";
      setMemoryLoaded(null);
      setStatus(`השיחה נשמרה. השיחה הבאה עם ${name} תיפתח ממה שהתברר בה.`);
    } catch (err) {
      setStatus(`שמירת השיחה נכשלה: ${err instanceof Error ? err.message : err}`);
    } finally {
      setIsEnding(false);
    }
  }, [clientName]);

  return {
    messages,
    sendMessage,
    isAssistantTyping,
    useLangGraph,
    setUseLangGraph,
    knowledgeMode,
    setKnowledgeMode,
    successModule,
    setSuccessModule,
    statesModule,
    setStatesModule,
    researchObserver,
    setResearchObserver,
    saveHumanNote,
    clientName,
    setClientName,
    memoryLoaded,
    status,
    isEnding,
    endConversation,
  };
}
