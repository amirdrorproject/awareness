-- Run this manually in the Supabase SQL editor, after traces.sql.
-- The research observer adds its results to a reply's trace in several steps
-- (its three calls, the outcome from the next turn, Amir's note), sometimes at
-- the same moment. This merges new top-level keys into the trace in one
-- update, so two steps never overwrite each other. Safe to re-run.

create or replace function merge_message_trace(message_id uuid, patch jsonb)
returns void
language sql
as $$
  update conversation_messages
  set trace = coalesce(trace, '{}'::jsonb) || patch
  where id = message_id;
$$;
