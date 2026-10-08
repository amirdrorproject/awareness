-- Run this manually in the Supabase SQL editor, after tool_calls.sql.
-- The run trace of every assistant turn: prompt version, memory given,
-- thinking summaries, tokens, latency, cost, and the error if it failed.
-- Safe to re-run.

alter table conversation_messages
  add column if not exists trace jsonb;
