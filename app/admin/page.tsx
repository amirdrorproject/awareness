"use client";

import { useEffect, useState } from "react";

export default function AdminSystemPromptPage() {
  const [content, setContent] = useState("");
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch("/api/admin/system-prompt");
        if (!res.ok) throw new Error(`Request failed: ${res.status}`);
        const data = await res.json();
        setContent(data.content ?? "");
        setUpdatedAt(data.updated_at ?? null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load system prompt");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const res = await fetch("/api/admin/system-prompt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      });
      if (!res.ok) throw new Error(`Request failed: ${res.status}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setContent(data.content ?? content);
      setUpdatedAt(data.updated_at ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save system prompt");
    } finally {
      setSaving(false);
    }
  };

  return (
    <main className="mx-auto flex max-w-2xl flex-col gap-4 p-6">
      <h1 className="text-xl font-semibold text-gray-900">System Prompt</h1>

      {loading ? (
        <p className="text-sm text-gray-500">Loading...</p>
      ) : (
        <>
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            rows={10}
            className="w-full rounded-lg border border-gray-300 p-3 text-sm outline-none focus:border-blue-500"
          />

          <button
            onClick={handleSave}
            disabled={saving}
            className="w-fit rounded-full bg-gray-800 px-5 py-2 text-sm font-medium text-white transition hover:bg-gray-900 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {saving ? "Saving..." : "Save"}
          </button>

          <p className="text-xs text-gray-500">
            Last updated: {updatedAt ?? "unknown"}
          </p>

          {error && <p className="text-sm text-red-600">{error}</p>}
        </>
      )}
    </main>
  );
}
