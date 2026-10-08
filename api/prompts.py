"""The prompts Amir edits from /admin.

Each save is a new row in prompt_versions, so every earlier version stays
available. The agent always reads the newest row; a prompt that was never
saved there falls back to its file in prompts/ - which is also the fallback if
Supabase is unreachable, so the chat never runs without a prompt.
"""

import logging
from pathlib import Path
from typing import Optional

from .system_prompt import get_supabase_client

logger = logging.getLogger("api.prompts")

VERSIONS_TABLE = "prompt_versions"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# name -> (file fallback, label shown in /admin)
PROMPTS = {
    "engine": ("engine.md", "פרומפט המנוע"),
    "memory_summary": ("memory_summary.md", "מה נשמר בסוף שיחה"),
    "memory_context": ("memory_context.md", "איך משתמשים בזיכרון בשיחה הבאה"),
}


def _file_content(name: str) -> str:
    return (PROMPTS_DIR / PROMPTS[name][0]).read_text(encoding="utf-8")


def get_prompt_record(name: str) -> dict:
    """The prompt in effect: {content, source: "admin"|"file", created_at, note}."""
    db = get_supabase_client()
    if db is not None:
        try:
            rows = (
                db.table(VERSIONS_TABLE)
                .select("content, note, created_at")
                .eq("name", name)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
                .data
            )
            if rows:
                return {**rows[0], "source": "admin"}
        except Exception:
            logger.exception("Failed to load prompt %r from Supabase - using its file", name)
    return {"content": _file_content(name), "note": None, "created_at": None, "source": "file"}


def get_prompt(name: str) -> str:
    return get_prompt_record(name)["content"]


def list_versions(name: str, limit: int = 50) -> list[dict]:
    db = get_supabase_client()
    if db is None:
        return []
    return (
        db.table(VERSIONS_TABLE)
        .select("id, content, note, created_at")
        .eq("name", name)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
        .data
        or []
    )


def save_version(name: str, content: str, note: Optional[str]) -> dict:
    db = get_supabase_client()
    if db is None:
        raise RuntimeError("Supabase is not configured (missing SUPABASE_URL/SUPABASE_SECRET_KEY).")
    rows = (
        db.table(VERSIONS_TABLE)
        .insert({"name": name, "content": content, "note": note or None})
        .execute()
        .data
    )
    if not rows:
        raise RuntimeError("Saving the prompt returned no row.")
    return {**rows[0], "source": "admin"}
