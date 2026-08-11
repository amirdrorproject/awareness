import os
from datetime import datetime, timezone
from typing import Optional

from supabase import Client, create_client

DEFAULT_SYSTEM_PROMPT = "You are Awareness Helper, a supportive assistant."

SYSTEM_PROMPT_TABLE = "system_prompt"


def get_supabase_client() -> Optional[Client]:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SECRET_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def get_system_prompt_record() -> dict:
    client = get_supabase_client()
    if client is None:
        return {"content": DEFAULT_SYSTEM_PROMPT, "updated_at": None}

    try:
        result = (
            client.table(SYSTEM_PROMPT_TABLE)
            .select("id, content, updated_at")
            .order("updated_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = result.data or []
        if not rows:
            return {"content": DEFAULT_SYSTEM_PROMPT, "updated_at": None}
        return rows[0]
    except Exception:
        return {"content": DEFAULT_SYSTEM_PROMPT, "updated_at": None}


def get_system_prompt() -> str:
    return get_system_prompt_record().get("content") or DEFAULT_SYSTEM_PROMPT


def update_system_prompt(content: str) -> dict:
    client = get_supabase_client()
    if client is None:
        raise RuntimeError("Supabase is not configured (missing SUPABASE_URL/SUPABASE_SECRET_KEY).")

    now = datetime.now(timezone.utc).isoformat()
    current = get_system_prompt_record()
    row_id = current.get("id")

    if row_id:
        result = (
            client.table(SYSTEM_PROMPT_TABLE)
            .update({"content": content, "updated_at": now})
            .eq("id", row_id)
            .execute()
        )
    else:
        result = (
            client.table(SYSTEM_PROMPT_TABLE)
            .insert({"content": content, "updated_at": now})
            .execute()
        )

    rows = result.data or []
    if not rows:
        raise RuntimeError("Failed to update system prompt.")
    return rows[0]
