"""Memory between conversations.

When a conversation is ended, Claude summarises what was established into a
structured memory (folding in the client's earlier memory, so the latest row
is always the whole picture). The client's next conversation gets that memory
appended to the engine prompt. Both texts are edited from /admin like the
engine prompt (see prompts.py).
"""

import json
import logging
from typing import Literal, Optional

import anthropic
from pydantic import BaseModel

from .engine import CLAUDE_MODEL
from .prompts import get_prompt
from .system_prompt import get_supabase_client

logger = logging.getLogger("api.memory")

MEMORIES_TABLE = "client_memories"
CONVERSATIONS_TABLE = "conversations"



class Capability(BaseModel):
    description: str
    # Exactly as the client named it; empty when they gave it no name.
    client_name_for_it: str


class ClientMemory(BaseModel):
    situation: str
    goal: str
    friction: str
    emotional_aspect: str
    confirmed_hypotheses: list[str]
    rejected_hypotheses: list[str]
    capabilities: list[Capability]
    practical_step: str
    open_threads: list[str]
    grammatical_gender: Literal["male", "female", "unknown"]


def _transcript_text(messages: list[dict]) -> str:
    return "\n\n".join(
        f"{'לקוח' if m['role'] == 'user' else 'סוכן'}: {m['content']}" for m in messages
    )


def summarize_conversation(
    client: anthropic.Anthropic, messages: list[dict], previous: Optional[dict]
) -> ClientMemory:
    previous_text = (
        json.dumps(previous, ensure_ascii=False, indent=2) if previous else "אין - זו השיחה הראשונה עם הלקוח."
    )
    response = client.messages.parse(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        output_config={"effort": "medium"},
        system=get_prompt("memory_summary"),
        messages=[{
            "role": "user",
            "content": f"זיכרון משיחות קודמות:\n{previous_text}\n\n---\n\nהשיחה שהסתיימה:\n\n{_transcript_text(messages)}",
        }],
        output_format=ClientMemory,
    )
    return response.parsed_output


def format_memory_for_prompt(memory: dict) -> str:
    gender = {"male": "זכר", "female": "נקבה"}.get(memory.get("grammatical_gender"), "לא ידוע")
    lines = [
        f"- המצב: {memory['situation']}",
        f"- המטרה: {memory['goal']}",
        f"- החיכוך: {memory['friction']}",
        f"- ההיבט הרגשי: {memory['emotional_aspect']}",
        *(f"- השערה שאושרה: {h}" for h in memory["confirmed_hypotheses"]),
        *(f"- השערה שנדחתה: {h}" for h in memory["rejected_hypotheses"]),
        *(
            f"- יכולת: {c['description']}"
            + (f" (השם שהלקוח נתן לה: \"{c['client_name_for_it']}\")" if c["client_name_for_it"] else "")
            for c in memory["capabilities"]
        ),
        f"- הצעד המעשי שהלקוח בנה: {memory['practical_step']}",
        *(f"- חוט פתוח: {t}" for t in memory["open_threads"]),
        f"- לשון פנייה: {gender}",
    ]
    template = get_prompt("memory_context")
    return template.replace("{memory}", "\n".join(lines))


def load_latest_memory(client_name: str) -> Optional[dict]:
    db = get_supabase_client()
    if db is None or not client_name:
        return None
    try:
        rows = (
            db.table(MEMORIES_TABLE)
            .select("memory")
            .eq("client_name", client_name)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
            .data
        )
        return rows[0]["memory"] if rows else None
    except Exception:
        logger.exception("Failed to load memory for client %r", client_name)
        return None


def save_memory(client_name: str, conversation_id: str, memory: ClientMemory) -> None:
    db = get_supabase_client()
    if db is None:
        raise RuntimeError("Supabase is not configured (missing SUPABASE_URL/SUPABASE_SECRET_KEY).")
    db.table(CONVERSATIONS_TABLE).upsert(
        {"id": conversation_id, "client_name": client_name}, on_conflict="id"
    ).execute()
    db.table(MEMORIES_TABLE).insert(
        {"client_name": client_name, "conversation_id": conversation_id, "memory": memory.model_dump()}
    ).execute()
