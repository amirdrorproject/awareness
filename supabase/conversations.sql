-- Run this manually in the Supabase SQL editor.
-- Creates the tables that store V0 chat transcripts (/api/chat).

create extension if not exists pgcrypto;

create table if not exists conversations (
  id uuid primary key,
  created_at timestamptz not null default now()
);

create table if not exists conversation_messages (
  id uuid primary key default gen_random_uuid(),
  conversation_id uuid not null references conversations (id) on delete cascade,
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  model text,
  stop_reason text,
  created_at timestamptz not null default clock_timestamp()
);

create index if not exists conversation_messages_conversation_id_created_at_idx
  on conversation_messages (conversation_id, created_at);
