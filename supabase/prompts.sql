-- Run this manually in the Supabase SQL editor.
-- Versions of the prompts edited from /admin. Every save is a new row; the
-- newest row per name is the one the agent uses. Until a prompt has a row,
-- the agent falls back to its file in prompts/. Safe to re-run.

create extension if not exists pgcrypto;

create table if not exists prompt_versions (
  id uuid primary key default gen_random_uuid(),
  -- engine | memory_summary | memory_context
  name text not null,
  content text not null,
  note text,
  created_at timestamptz not null default now()
);

create index if not exists prompt_versions_name_created_at_idx
  on prompt_versions (name, created_at desc);
