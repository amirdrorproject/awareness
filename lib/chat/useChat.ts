"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { KnowledgeMode, Message, MessageSource, ToolCall } from "./types";

const CLIENT_NAME_KEY = "awareness.clientName";

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isAssistantTyping, setIsAssistantTyping] = useState(false);
  const [useLangGraph, setUseLangGraph] = useState(false);
  const [knowledgeMode, setKnowledgeMode] = useState<KnowledgeMode>("full");
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
                messages: next.map(({ role, content }) => ({ role, content })),
                conversation_id: threadIdRef.current,
                knowledge_mode: knowledgeMode,
                client_name: clientName.trim() || null,
              }),
            });
            if (!res.ok) throw new Error(`Request failed: ${res.status}`);
            const data = await res.json();
            replyContents = data.content ? [data.content] : [];
            toolCalls = data.tool_calls;
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
          }));
          appendMessages(replies);
        }
        setIsAssistantTyping(false);
      })();
    },
    [useLangGraph, knowledgeMode, clientName, appendMessages]
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
    clientName,
    setClientName,
    memoryLoaded,
    status,
    isEnding,
    endConversation,
  };
}
