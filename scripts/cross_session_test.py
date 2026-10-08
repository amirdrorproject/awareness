"""Tests memory between conversations, end to end but without Supabase.

Session 1 is a full simulated conversation (the "michal" scenario from
long_conversation_test). It is summarised exactly as the "end conversation"
button does, and that memory is handed to session 2 - a second conversation
with the same client, run twice: with the memory and without it, so the
difference memory makes is visible. Session 2 comes in two kinds: the client
comes back to the same story, or arrives with something new (where the agent
should follow her, not drag last time's topic back in).

Usage (from the repo root):
    python -m scripts.cross_session_test --out results.json
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import anthropic
from dotenv import dotenv_values
from pydantic import BaseModel

from api.engine import CLAUDE_MODEL, generate_reply, get_engine_prompt
from api.memory import format_memory_for_prompt, summarize_conversation
from scripts.long_conversation_test import SCENARIOS, simulate_client, transcript_text

KNOWLEDGE_MODE = "full"
FIRST_SESSION = SCENARIOS["michal"]

# The persona carries over; what happened between the sessions is new.
SECOND_SESSIONS = {
    "follow_up": {
        "opening": "היי, חזרתי. הייתה לי השבוע שיחת הרבעון עם רוני",
        "persona": FIRST_SESSION["persona"] + """

Since the last conversation (a week ago): the quarterly review with Roni happened. You managed to
speak up for part of it - you brought the technical points you had prepared - but when Roni moved
to your targets you went quiet again.""",
        "plan": {
            1: "Answer the agent's question about how the review went: you spoke up on the technical part "
               "you had prepared, then went quiet when Roni moved to targets.",
            2: "Answer naturally.",
            3: "TRAP: say maybe in the end it's all about the money after all - the bonus.",
            4: "Say it happened again this week in a meeting with the VP: you went quiet.",
            5: "Answer naturally.",
        },
        "checks": """- Builds on the first session without re-asking what is known (who Roni is, that she goes quiet with seniors, her preparation capability).
- Asks how the review went rather than assuming an outcome.
- The money/bonus reading was rejected in session 1: at the trap, the agent notices it contradicts what she said, rather than following it.
- When the VP meeting comes up, connects it to the known pattern (quiet with seniors) as a hypothesis, using her own name for her capability if she gave one.
- Addresses her in the feminine from the first reply.""",
    },
    "new_topic": {
        "opening": "היי, הפעם משהו אחר לגמרי. יש לי לקוח שדורש הנחה של 30% ואני לא יודעת מה לעשות",
        "persona": FIRST_SESSION["persona"] + """

Since the last conversation (a week ago): something new came up. A client demands a 30% discount to
renew. His CEO will join the next meeting.""",
        "plan": {
            1: "Answer about the discount demand naturally.",
            2: "Answer naturally.",
            3: "Mention that the client's CEO will join the next meeting.",
            4: "Answer naturally.",
            5: "Answer naturally.",
        },
        "checks": """- Follows the new topic she brought; does not steer back to Roni or the lost deal on its own.
- Gives no tactics on the discount (per its prompt).
- When the client's CEO comes up, it MAY offer the known pattern (quiet with seniors) as a hypothesis - gently, as a question - but must not assume it.
- Addresses her in the feminine from the first reply.""",
    },
}


class Issue(BaseModel):
    agent_turn: int
    type: Literal[
        "re_asked_known",
        "reused_rejected_hypothesis",
        "dragged_old_topic",
        "assumed_outcome",
        "invented_detail",
        "wrong_gender_address",
        "missed_relevant_memory",
        "other",
    ]
    quote: str
    explanation: str


class SessionJudgement(BaseModel):
    checks_passed: list[str]
    checks_failed: list[str]
    issues: list[Issue]
    summary: str


def run_session(client: anthropic.Anthropic, scenario: dict, system_suffix: str | None) -> list[dict]:
    turns = [{"role": "user", "content": scenario["opening"]}]
    for client_turn in [*scenario["plan"], None]:
        reply = generate_reply(client, turns, KNOWLEDGE_MODE, system_suffix=system_suffix)
        turns.append({"role": "assistant", "content": reply["content"]})
        if client_turn is not None:
            turns.append({"role": "user", "content": simulate_client(client, scenario, turns, client_turn)})
    return turns


def judge(client: anthropic.Anthropic, memory: dict, scenario: dict, turns: list[dict], with_memory: bool) -> SessionJudgement:
    note = (
        "The agent WAS given the memory below." if with_memory
        else "The agent was NOT given the memory below (control run) - judge it against the same checks anyway."
    )
    response = client.messages.parse(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        output_config={"effort": "high"},
        system="You evaluate the second conversation between a coaching agent and a returning client. "
               "Be strict but fair: report only what the transcript shows. Quote the agent's exact words "
               "(Hebrew). Write the checks, explanations and summary in Hebrew.",
        messages=[{
            "role": "user",
            "content": f"The agent's instructions (for context):\n\n{get_engine_prompt()}\n\n---\n\n"
                       f"What was established in the first conversation (the memory):\n"
                       f"{json.dumps(memory, ensure_ascii=False, indent=2)}\n\n{note}\n\n"
                       f"Checks for this second conversation:\n{scenario['checks']}\n\n---\n\n"
                       f"Transcript of the second conversation:\n\n{transcript_text(turns)}\n\n---\n\n"
                       "List which checks passed and which failed (restate each check briefly in Hebrew), "
                       "and every issue.",
        }],
        output_format=SessionJudgement,
    )
    return response.parsed_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="Path of the JSON results file to write.")
    args = parser.parse_args()

    os.environ.update({k: v for k, v in dotenv_values(".env.local").items() if v and k not in os.environ})
    client = anthropic.Anthropic()

    print("Session 1 (michal)...")
    first = run_session(client, FIRST_SESSION, None)
    memory = summarize_conversation(client, first, None).model_dump()
    print("Memory:\n" + json.dumps(memory, ensure_ascii=False, indent=2))
    memory_prompt = format_memory_for_prompt(memory)

    jobs = [(kind, with_memory) for kind in SECOND_SESSIONS for with_memory in (True, False)]
    print(f"Session 2: {len(jobs)} conversations...")

    def run_job(job):
        kind, with_memory = job
        scenario = SECOND_SESSIONS[kind]
        turns = run_session(client, scenario, memory_prompt if with_memory else None)
        verdict = judge(client, memory, scenario, turns, with_memory)
        return {"kind": kind, "with_memory": with_memory, "turns": turns, "judgement": verdict.model_dump()}

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        second = list(pool.map(run_job, jobs))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {"model": CLAUDE_MODEL, "first_session": first, "memory": memory, "second_sessions": second},
            f, ensure_ascii=False, indent=2,
        )
    print(f"Done. Wrote {args.out}")


if __name__ == "__main__":
    main()
