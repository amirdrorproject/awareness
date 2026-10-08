-- Run this manually in the Supabase SQL editor, after conversations.sql.
-- Memory between conversations: when a conversation is ended, what was
-- established in it is summarised into client_memories; the client's next
-- conversation starts from their latest row. Safe to re-run.

alter table conversations
  add column if not exists client_name text;

create table if not exists client_memories (
  id uuid primary key default gen_random_uuid(),
  client_name text not null,
  conversation_id uuid references conversations (id) on delete set null,
  -- Cumulative: each row already folds in the client's earlier memory.
  memory jsonb not null,
  created_at timestamptz not null default now()
);

create index if not exists client_memories_client_name_created_at_idx
  on client_memories (client_name, created_at desc);
