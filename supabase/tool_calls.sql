-- Run this manually in the Supabase SQL editor, after conversations.sql.
-- Records, per assistant message, how the agent could reach the expression
-- banks (knowledge_mode: none / search / full) and which tool calls Claude
-- made (name, input, and what came back). Safe to re-run.

alter table conversation_messages
  add column if not exists tools_enabled boolean,
  add column if not exists tool_calls jsonb not null default '[]'::jsonb,
  add column if not exists knowledge_mode text;
