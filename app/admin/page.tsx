"use client";

import { useCallback, useEffect, useState } from "react";

// Prompt editor for Amir: the engine prompt and the two memory prompts.
// Every save is a new version (prompt_versions in Supabase); the chat uses the
// newest one from its very next message. Guarded by ADMIN_PASSWORD.

interface PromptRecord {
  name: string;
  label: string;
  content: string;
  note: string | null;
  created_at: string | null;
  source: "admin" | "file";
}

interface PromptVersion {
  id: string;
  content: string;
  note: string | null;
  created_at: string;
}

const PASSWORD_KEY = "awareness.adminPassword";

function formatDate(iso: string) {
  return new Date(iso).toLocaleString("he-IL", { dateStyle: "short", timeStyle: "short" });
}

async function adminFetch(path: string, password: string, init?: RequestInit) {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", "X-Admin-Password": password, ...init?.headers },
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail ?? `Request failed: ${res.status}`);
  return data;
}

export default function AdminPromptsPage() {
  const [password, setPassword] = useState("");
  const [authed, setAuthed] = useState(false);
  const [prompts, setPrompts] = useState<PromptRecord[]>([]);
  const [active, setActive] = useState("engine");
  const [draft, setDraft] = useState("");
  const [note, setNote] = useState("");
  const [versions, setVersions] = useState<PromptVersion[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ kind: "ok" | "error"; text: string } | null>(null);

  const current = prompts.find((p) => p.name === active);
  const dirty = current !== undefined && draft !== current.content;

  const loadPrompts = useCallback(async (pw: string) => {
    const data = await adminFetch("/api/admin/prompts", pw);
    setPrompts(data.prompts);
    return data.prompts as PromptRecord[];
  }, []);

  const login = useCallback(
    async (pw: string) => {
      setBusy(true);
      setMessage(null);
      try {
        const loaded = await loadPrompts(pw);
        setAuthed(true);
        setDraft(loaded.find((p) => p.name === "engine")?.content ?? "");
        try {
          sessionStorage.setItem(PASSWORD_KEY, pw);
        } catch {
          // Not remembered for this tab; the page still works.
        }
      } catch (err) {
        setMessage({ kind: "error", text: err instanceof Error ? err.message : String(err) });
      } finally {
        setBusy(false);
      }
    },
    [loadPrompts]
  );

  // Stay signed in across refreshes within the same tab.
  useEffect(() => {
    let saved: string | null = null;
    try {
      saved = sessionStorage.getItem(PASSWORD_KEY);
    } catch {
      // Storage unavailable: start at the password screen.
    }
    if (saved) {
      setPassword(saved);
      login(saved);
    }
  }, [login]);

  // Warn before leaving with unsaved edits.
  useEffect(() => {
    if (!dirty) return;
    const warn = (e: BeforeUnloadEvent) => e.preventDefault();
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  const switchTo = (name: string) => {
    if (dirty && !confirm("יש שינויים שלא נשמרו. לעבור בכל זאת?")) return;
    setActive(name);
    setDraft(prompts.find((p) => p.name === name)?.content ?? "");
    setNote("");
    setVersions(null);
    setMessage(null);
  };

  const save = async () => {
    setBusy(true);
    setMessage(null);
    try {
      await adminFetch(`/api/admin/prompts/${active}`, password, {
        method: "POST",
        body: JSON.stringify({ content: draft, note }),
      });
      await loadPrompts(password);
      setNote("");
      setVersions(null);
      setMessage({ kind: "ok", text: "נשמר. הצ'אט משתמש בגרסה הזאת מההודעה הבאה." });
    } catch (err) {
      setMessage({ kind: "error", text: err instanceof Error ? err.message : String(err) });
    } finally {
      setBusy(false);
    }
  };

  const showVersions = async () => {
    try {
      const data = await adminFetch(`/api/admin/prompts/${active}/versions`, password);
      setVersions(data.versions);
    } catch (err) {
      setMessage({ kind: "error", text: err instanceof Error ? err.message : String(err) });
    }
  };

  if (!authed) {
    return (
      <main dir="rtl" className="mx-auto flex max-w-sm flex-col gap-3 p-6">
        <h1 className="text-xl font-semibold text-gray-900">עריכת פרומפטים</h1>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && login(password)}
          placeholder="סיסמה"
          className="rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none focus:border-blue-500"
        />
        <button
          onClick={() => login(password)}
          disabled={busy || !password}
          className="rounded-full bg-gray-800 px-5 py-2 text-sm font-medium text-white hover:bg-gray-900 disabled:opacity-50"
        >
          כניסה
        </button>
        {message && <p className="text-sm text-red-600">{message.text}</p>}
      </main>
    );
  }

  return (
    <main dir="rtl" className="mx-auto flex max-w-4xl flex-col gap-4 p-6">
      <h1 className="text-xl font-semibold text-gray-900">עריכת פרומפטים</h1>

      <div className="flex flex-wrap gap-2 border-b border-gray-200 pb-2">
        {prompts.map((p) => (
          <button
            key={p.name}
            onClick={() => switchTo(p.name)}
            className={`rounded-full px-4 py-1.5 text-sm ${
              p.name === active ? "bg-gray-800 text-white" : "text-gray-600 hover:bg-gray-100"
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>

      {current && (
        <p className="text-xs text-gray-500">
          {current.source === "admin"
            ? `הגרסה הפעילה נשמרה כאן ב-${formatDate(current.created_at!)}${current.note ? ` · ${current.note}` : ""}`
            : "הגרסה הפעילה היא עדיין זו שבקוד. השמירה הראשונה כאן תחליף אותה."}
        </p>
      )}

      <textarea
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        dir="rtl"
        className="h-[60vh] w-full rounded-lg border border-gray-300 p-4 font-mono text-sm leading-relaxed outline-none focus:border-blue-500"
      />

      <div className="flex flex-wrap items-center gap-2">
        <input
          type="text"
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="מה שינית? (לא חובה)"
          className="min-w-64 flex-1 rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none focus:border-blue-500"
        />
        <button
          onClick={save}
          disabled={busy || !dirty}
          className="rounded-full bg-blue-600 px-5 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {busy ? "שומר..." : "שמירה"}
        </button>
        {dirty && (
          <button
            onClick={() => setDraft(current!.content)}
            className="rounded-full px-4 py-2 text-sm text-gray-600 hover:bg-gray-100"
          >
            ביטול השינויים
          </button>
        )}
      </div>

      {message && (
        <p className={`text-sm ${message.kind === "ok" ? "text-green-700" : "text-red-600"}`}>{message.text}</p>
      )}

      <section className="flex flex-col gap-2 border-t border-gray-200 pt-4">
        {versions === null ? (
          <button onClick={showVersions} className="w-fit text-sm text-blue-700 hover:underline">
            הצגת גרסאות קודמות
          </button>
        ) : versions.length === 0 ? (
          <p className="text-sm text-gray-500">עוד אין גרסאות שמורות לפרומפט הזה.</p>
        ) : (
          <>
            <h2 className="text-sm font-semibold text-gray-700">גרסאות קודמות</h2>
            <p className="text-xs text-gray-500">
              &quot;טעינה לעריכה&quot; מעתיקה גרסה לעורך. כדי שהיא תחזור לפעול, שומרים אותה.
            </p>
            <ul className="flex flex-col divide-y divide-gray-100 rounded-lg border border-gray-200">
              {versions.map((v, i) => (
                <li key={v.id} className="flex items-center gap-3 px-3 py-2 text-sm">
                  <span className="text-gray-500">{formatDate(v.created_at)}</span>
                  <span className="flex-1 text-gray-800">
                    {v.note || "—"}
                    {i === 0 && <span className="ms-2 text-xs text-green-700">(פעילה)</span>}
                  </span>
                  <button
                    onClick={() => setDraft(v.content)}
                    className="text-xs text-blue-700 hover:underline"
                  >
                    טעינה לעריכה
                  </button>
                </li>
              ))}
            </ul>
          </>
        )}
      </section>
    </main>
  );
}
