"""An observer's reading of a turn, in the parts Amir reviews: understanding,
focus, and the rationale for the reply (the reply itself is the fourth).

The observer is a separate call that sees only the conversation and the
reply - never Claude's thinking (passing the thinking summary to another call
is blocked as reasoning extraction under Anthropic's terms, and is not worked
around). So it is an interpretation of what the agent did, labelled as such
in the chat, next to Claude's own thinking summary. It runs after the reply
is on screen, so it adds no wait and never changes how the agent answers.
"""

import logging
from pathlib import Path

import anthropic
from pydantic import BaseModel

from .engine import CLAUDE_MODEL
from .system_prompt import get_supabase_client

logger = logging.getLogger("api.turn_analysis")

MESSAGES_TABLE = "conversation_messages"
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "turn_analysis.md"


class TurnAnalysis(BaseModel):
    understanding: str
    focus: str
    rationale: str


def analyze_turn(client: anthropic.Anthropic, messages: list[dict], reply: str) -> TurnAnalysis:
    transcript = "\n\n".join(
        f"{'לקוח' if m['role'] == 'user' else 'סוכן'}: {m['content']}" for m in messages
    )
    response = client.messages.parse(
        model=CLAUDE_MODEL,
        max_tokens=8000,
        output_config={"effort": "low"},
        system=PROMPT_PATH.read_text(encoding="utf-8"),
        messages=[{
            "role": "user",
            "content": f"השיחה עד עכשיו:\n\n{transcript}\n\n---\n\nהתשובה האחרונה של הסוכן:\n{reply}",
        }],
        output_format=TurnAnalysis,
    )
    if response.parsed_output is None:
        raise RuntimeError(f"No analysis returned (stop_reason: {response.stop_reason}).")
    return response.parsed_output


def save_analysis(message_id: str, analysis: TurnAnalysis) -> None:
    # Best-effort: the analysis is shown in the chat either way; a failed save
    # only means it won't be there when the conversation is reopened later.
    db = get_supabase_client()
    if db is None:
        return
    try:
        rows = db.table(MESSAGES_TABLE).select("trace").eq("id", message_id).limit(1).execute().data
        if not rows:
            return
        trace = {**(rows[0].get("trace") or {}), "analysis": analysis.model_dump()}
        db.table(MESSAGES_TABLE).update({"trace": trace}).eq("id", message_id).execute()
    except Exception:
        logger.exception("Failed to save the analysis of message %s", message_id)
