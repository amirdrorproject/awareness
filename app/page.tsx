"use client";

import { useState } from "react";
import ChatWindow from "@/components/chat/ChatWindow";

export default function Home() {
  const [backendTime, setBackendTime] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const testBackend = async () => {
    setError(null);
    setBackendTime(null);
    try {
      const res = await fetch("/api/time");
      if (!res.ok) throw new Error(`Request failed: ${res.status}`);
      const data = await res.json();
      setBackendTime(data.time);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to reach backend");
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-4 bg-gray-50 p-4">
      <ChatWindow />

      {/* Temporary: verifies frontend-backend connectivity */}
      <div className="flex flex-col items-center gap-2">
        <button
          onClick={testBackend}
          className="rounded-full bg-gray-800 px-5 py-2 text-sm font-medium text-white transition hover:bg-gray-900"
        >
          Test Backend
        </button>
        {backendTime && (
          <p className="text-sm text-gray-700">Backend time: {backendTime}</p>
        )}
        {error && <p className="text-sm text-red-600">{error}</p>}
      </div>
    </main>
  );
}
