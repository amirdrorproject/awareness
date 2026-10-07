-- Run this manually in the Supabase SQL editor, after conversations.sql.
-- Records, per assistant message, whether tools were enabled and which tool
-- calls Claude made (name, input, and what came back).

alter table conversation_messages
  add column if not exists tools_enabled boolean,
  add column if not exists tool_calls jsonb not null default '[]'::jsonb;
