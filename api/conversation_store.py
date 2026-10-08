import logging
from typing import Optional

from .system_prompt import get_supabase_client

logger = logging.getLogger("api.conversation_store")

CONVERSATIONS_TABLE = "conversations"
MESSAGES_TABLE = "conversation_messages"


def save_turn(
    conversation_id: str,
    user_message: str,
    assistant_message: str,
    model: str,
    stop_reason: Optional[str] = None,
    knowledge_mode: str = "none",
    tool_calls: Optional[list] = None,
    client_name: Optional[str] = None,
) -> None:
    # Persistence is best-effort: a missing Supabase config or a failed write
    # must never break the chat itself, so this only logs.
    client = get_supabase_client()
    if client is None:
        logger.info("Supabase not configured - conversation %s not saved.", conversation_id)
        return

    user_row = {"conversation_id": conversation_id, "role": "user", "content": user_message}
    assistant_row = {
        "conversation_id": conversation_id,
        "role": "assistant",
        "content": assistant_message,
        "model": model,
        "stop_reason": stop_reason,
    }
    tool_fields = {
        "tools_enabled": knowledge_mode != "none",
        "tool_calls": tool_calls or [],
        "knowledge_mode": knowledge_mode,
    }

    try:
        client.table(CONVERSATIONS_TABLE).upsert(
            {"id": conversation_id}, on_conflict="id", ignore_duplicates=True
        ).execute()
        if client_name:
            # Its own write: the column comes from supabase/client_memory.sql,
            # and a missing column must not stop the messages being saved.
            try:
                client.table(CONVERSATIONS_TABLE).update({"client_name": client_name}).eq("id", conversation_id).execute()
            except Exception:
                logger.exception("Failed to set client_name on conversation %s", conversation_id)
        try:
            client.table(MESSAGES_TABLE).insert([user_row, {**assistant_row, **tool_fields}]).execute()
        except Exception:
            # The tool columns come from supabase/tool_calls.sql. Until it has been
            # run, still save the messages themselves rather than losing the turn.
            logger.exception("Saving with tool columns failed - retrying without them")
            client.table(MESSAGES_TABLE).insert([user_row, assistant_row]).execute()
    except Exception:
        logger.exception("Failed to save turn for conversation %s", conversation_id)
