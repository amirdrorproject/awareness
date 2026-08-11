-- Run this manually in the Supabase SQL editor.
-- Creates the system_prompt table and seeds it with the initial prompt.

create extension if not exists pgcrypto;

create table if not exists system_prompt (
  id uuid primary key default gen_random_uuid(),
  content text not null,
  updated_at timestamptz not null default now()
);

insert into system_prompt (content)
values ('You are Awareness Helper, a supportive assistant.');
