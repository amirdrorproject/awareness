"""The research observer: an independent reading of each turn, in three
separate calls, to find out why the agent missed an opportunity for thinking -
it didn't see it, saw it and gave it no weight, or saw it and chose otherwise.

1. The story, from the conversation up to the client's message - before the
   agent's reply, so what could be seen doesn't depend on the agent. The same
   call also judges the previous turn's outcome from the client's answer.
2. The agent's choice, from the reply and call 1.
3. The comparison of calls 1-2 with the agent's own <log>. Never the thinking
   summary: passing Claude's thinking to another call counts as reasoning
   extraction under Anthropic's terms, so the log is used instead.

Each call gets the earlier results and never rewrites them; each is saved
separately on the reply's trace. Amir can also write what he sees in the story
before the observer's results are shown, and a fourth call checks whether the
agent or the observer saw it. All of it runs after the reply is on screen,
only when the research switch is on, and never changes the agent's reply.
"""

import logging
import time
from typing import Literal

import anthropic
from pydantic import BaseModel

from .engine import CLAUDE_MODEL, _estimate_cost
from .prompts import get_prompt_record
from .system_prompt import get_supabase_client
from .turn_log import FIELDS

logger = logging.getLogger("api.observer")

MESSAGES_TABLE = "conversation_messages"
# Quality is kept up and the cost saved by the switch instead (Amir, Nov 2026).
OBSERVER_EFFORT = "high"
NO_CLIENT_REPLY = "אין תגובת לקוח"


class StoryAnalysis(BaseModel):
    # 1. What can be seen in the narrator (not only in the people described).
    narrator: str
    # 2. Two things the client said that pull apart, or "לא זוהה".
    tension: str
    # 3. Directions that could let the client see something new; may be empty.
    opportunities: list[str]


class TurnOutcome(BaseModel):
    focus: str
    progress: str
    # What the client did with what the agent offered.
    offer: str


class StoryWithOutcome(StoryAnalysis):
    previous_turn: TurnOutcome


class ChoiceAnalysis(BaseModel):
    # 4. The direction the agent chose and what seems to have led to it.
    direction: str
    # 5. More than one question that differs in substance (not "?" marks).
    multiple_questions: Literal["כן", "לא"]
    questions_note: str


class OpportunityCheck(BaseModel):
    opportunity: str
    classification: Literal["לא זיהה", "זיהה ולא נתן משקל", "זיהה ובחר אחרת", "לא ניתן לקבוע"]
    evidence: str


class LogComparison(BaseModel):
    items: list[OpportunityCheck]
    conclusion: str


Recognition = Literal["זיהה", "זיהה חלקית", "לא זיהה", "לא ניתן לקבוע"]


class HumanComparison(BaseModel):
    agent: Recognition
    agent_evidence: str
    observer: Recognition
    observer_evidence: str


def _transcript(messages: list[dict]) -> str:
    return "\n\n".join(f"{'לקוח' if m['role'] == 'user' else 'סוכן'}: {m['content']}" for m in messages)


def _run(client: anthropic.Anthropic, prompt_name: str, content: str, schema: type[BaseModel]) -> dict:
    """One observer call; returns its fields plus how it ran (model, prompt
    version, tokens, cost), so each saved step documents itself."""
    started = time.monotonic()
    prompt = get_prompt_record(prompt_name)
    response = client.messages.parse(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        output_config={"effort": OBSERVER_EFFORT},
        system=prompt["content"],
        messages=[{"role": "user", "content": content}],
        output_format=schema,
    )
    if response.parsed_output is None:
        raise RuntimeError(f"No observer output (stop_reason: {response.stop_reason}).")
    usage = {
        key: getattr(response.usage, key, None) or 0
        for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    }
    return {
        **response.parsed_output.model_dump(),
        "meta": {
            "model": response.model,
            "effort": OBSERVER_EFFORT,
            "prompt": {"source": prompt["source"], "created_at": prompt["created_at"], "note": prompt["note"]},
            "usage": usage,
            "cost_usd": _estimate_cost(response.model, usage),
            "latency_s": round(time.monotonic() - started, 2),
        },
    }


def _format_story(story: dict) -> str:
    opportunities = story.get("opportunities") or []
    listed = "\n".join(f"- {o}" for o in opportunities) if opportunities else "לא זוהו"
    return (
        f"מה ניתן לזהות אצל המספר: {story.get('narrator')}\n"
        f"מתח בדברי הלקוח: {story.get('tension')}\n"
        f"הזדמנויות לחשיבה:\n{listed}"
    )


def _format_choice(choice: dict) -> str:
    return (
        f"מה נראה שהוביל לבחירת הכיוון: {choice.get('direction')}\n"
        f"יותר משאלה עצמאית אחת: {choice.get('multiple_questions')}. {choice.get('questions_note')}"
    )


def _format_log(turn_log: dict) -> str:
    # The labels the agent itself wrote, in its order; older logs carry the
    # earlier hypothesis label, which FIELDS still maps.
    return "\n".join(f"{label}: {turn_log[key]}" for label, key in FIELDS.items() if turn_log.get(key))


def analyze_story(client: anthropic.Anthropic, messages: list[dict]) -> dict:
    """Call 1. `messages` ends with the client's current message; the agent's
    reply to it is not there, and earlier replies come without their logs.
    When the agent spoke before it, the previous turn's outcome is judged too."""
    content = f"השיחה עד הודעת הלקוח הנוכחית:\n\n{_transcript(messages)}\n\n---\n\nבצע את קריאה 1: ניתוח עצמאי של הסיפור."
    has_previous = len(messages) >= 2 and messages[-2]["role"] == "assistant"
    if not has_previous:
        return {"story": _run(client, "observer", content, StoryAnalysis), "outcome": None}
    content += (
        "\n\nבנוסף, בצע את ניתוח תוצאות התור הקודם, בנפרד מניתוח הסיפור (בשדה previous_turn):\n\n"
        f"תגובת הסוכן בתור הקודם:\n{messages[-2]['content']}\n\n"
        f"הודעת הלקוח שבאה אחריה:\n{messages[-1]['content']}"
    )
    result = _run(client, "observer", content, StoryWithOutcome)
    outcome = {**result.pop("previous_turn"), "meta": result["meta"]}
    return {"story": result, "outcome": outcome}


def analyze_choice(client: anthropic.Anthropic, messages: list[dict], reply: str, story: dict) -> dict:
    """Call 2: the agent's reply, read against call 1."""
    content = (
        f"השיחה עד הודעת הלקוח הנוכחית:\n\n{_transcript(messages)}\n\n---\n\n"
        f"תגובת הסוכן הנוכחית:\n{reply}\n\n---\n\n"
        f"תוצאות קריאה 1:\n{_format_story(story)}\n\n---\n\n"
        "בצע את קריאה 2: ניתוח בחירת הסוכן."
    )
    return _run(client, "observer", content, ChoiceAnalysis)


def compare_with_log(client: anthropic.Anthropic, story: dict, choice: dict, turn_log: dict | None) -> dict:
    """Call 3: calls 1-2 against the agent's log. Without a log (the states
    module was off) or without opportunities there is nothing to compare, so
    no call is made and the reason is saved instead."""
    if not turn_log:
        return {"items": [], "conclusion": "", "skipped": "אין לוג סוכן בתור הזה (מודול מצבי השיחה כבוי או שהסוכן לא כתב לוג), ולכן אין למה להשוות."}
    if not story.get("opportunities"):
        return {"items": [], "conclusion": "", "skipped": "בקריאה 1 לא זוהו הזדמנויות לחשיבה, ולכן אין מה להשוות."}
    content = (
        f"תוצאות קריאה 1:\n{_format_story(story)}\n\n---\n\n"
        f"תוצאות קריאה 2:\n{_format_choice(choice)}\n\n---\n\n"
        f"הלוג המובנה של הסוכן:\n{_format_log(turn_log)}\n\n---\n\n"
        "בצע את קריאה 3: השוואה ללוג הסוכן."
    )
    return _run(client, "observer", content, LogComparison)


def compare_with_human(
    client: anthropic.Anthropic,
    note: str,
    messages: list[dict],
    reply: str,
    turn_log: dict | None,
    story: dict,
    choice: dict,
    comparison: dict,
) -> dict:
    """Whether the agent (log and reply) or the observer saw what Amir wrote."""
    observer_text = f"{_format_story(story)}\n{_format_choice(choice)}"
    if comparison.get("items"):
        observer_text += "\n" + "\n".join(
            f"- {item['opportunity']}: {item['classification']}" for item in comparison["items"]
        )
    if comparison.get("conclusion"):
        observer_text += f"\nמסקנה: {comparison['conclusion']}"
    content = (
        f"השיחה עד הודעת הלקוח הנוכחית:\n\n{_transcript(messages)}\n\n---\n\n"
        f"מה אמיר מזהה בסיפור:\n{note}\n\n---\n\n"
        f"לוג הסוכן:\n{_format_log(turn_log) if turn_log else 'לא נכתב לוג'}\n\n"
        f"תגובת הסוכן:\n{reply}\n\n---\n\n"
        f"ניתוח הצופה המחקרי:\n{observer_text}"
    )
    return _run(client, "observer_human", content, HumanComparison)


def save_to_trace(message_id: str | None, patch: dict) -> None:
    """Adds top-level keys to a saved reply's trace. Best-effort: the results
    are shown in the chat either way; a failed save only means they won't be
    in the database. The merge runs in one update (supabase/observer.sql), so
    steps saved at the same moment don't overwrite each other; without that
    function it falls back to read-then-write."""
    db = get_supabase_client()
    if db is None or not message_id:
        return
    try:
        db.rpc("merge_message_trace", {"message_id": message_id, "patch": patch}).execute()
        return
    except Exception:
        logger.warning("merge_message_trace failed - run supabase/observer.sql; falling back to read-then-write")
    try:
        rows = db.table(MESSAGES_TABLE).select("trace").eq("id", message_id).limit(1).execute().data
        if not rows:
            return
        trace = {**(rows[0].get("trace") or {}), **patch}
        db.table(MESSAGES_TABLE).update({"trace": trace}).eq("id", message_id).execute()
    except Exception:
        logger.exception("Failed to save observer results on message %s", message_id)
