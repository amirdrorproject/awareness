"""Runs the same conversation openings through the agent with each knowledge
mode (no bank / retrieval / full bank) and writes the replies side by side.

The question it serves: does access to the expression bank improve how the
agent identifies what the client brings, without making the conversation
mechanical - and does Claude reach for the bank only when it's relevant?

Usage (from the repo root):
    python -m scripts.compare_knowledge --out results.json [--runs 2]
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import anthropic
from dotenv import dotenv_values

from api.engine import CLAUDE_EFFORT, CLAUDE_MODEL, generate_reply
from api.knowledge import KNOWLEDGE_MODES

# Each scenario is a conversation so far, ending with the client's message the
# agent has to answer. `expect_bank` is the hypothesis being tested: whether
# reaching for the bank is warranted here at all.
SCENARIOS = [
    {
        "id": "explicit_burnout",
        "label": "Explicit emotion: burnout",
        "expect_bank": True,
        "messages": ["אני כבר גמור מהצוות הזה, נשחקתי. כל בוקר אני קם בלי חשק"],
    },
    {
        "id": "trust_hint",
        "label": "Emotional hint: losing the team's trust",
        "expect_bank": True,
        "messages": ["היה לי רבעון גרוע והצוות מתחיל לאבד בי אמון"],
    },
    {
        "id": "hollow_win",
        "label": "Subtle: a win that doesn't feel like one",
        "expect_bank": True,
        "messages": ["סגרתי את העסקה הכי גדולה של השנה, אבל משום מה זה לא מרגיש כמו שחשבתי"],
    },
    {
        "id": "identity_doubt",
        "label": "Doubt: not built for management",
        "expect_bank": True,
        "messages": ["אני לא בטוח שאני בנוי לניהול. אולי הייתי צריך להישאר איש מכירות"],
    },
    {
        "id": "threat_new_vp",
        "label": "Threat: a new VP above me",
        "expect_bank": True,
        "messages": ['שמעתי שמביאים סמנכ"ל מכירות חדש מעליי'],
    },
    {
        "id": "factual_renewal",
        "label": "Factual: contract renewal meeting",
        "expect_bank": False,
        "messages": ["יש לי מחר פגישה עם לקוח על חידוש חוזה שנתי, רוצה לחשוב איתך על זה"],
    },
    {
        "id": "tactic_request",
        "label": "Tactic request: underperforming rep",
        "expect_bank": False,
        "messages": ["איך אני גורם לנציג שלא עומד ביעדים להתחיל לעמוד בהם?"],
    },
    {
        "id": "mid_conversation_shift",
        "label": "Mid-conversation: factual start, emotion surfaces",
        "expect_bank": True,
        "messages": [
            "יש לי לקוח גדול שהיינו כבר כמעט בחתימה, ופתאום הוא נעלם. שבועיים לא עונה לי.",
            'שבועיים שקט אחרי שהייתם כמעט בחתימה. מה היה הדבר האחרון שקרה ביניכם לפני שהוא נעלם?',
            "הייתה שיחה על מחיר. ובכנות, מאז אני לא ישן טוב. אם העסקה הזאת נופלת, אני לא יודע מה אגיד למנכ\"ל",
        ],
    },
]


def as_api_messages(texts: list[str]) -> list[dict]:
    # Turns alternate user / assistant, starting and ending with the client.
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": text} for i, text in enumerate(texts)]


def run_one(client: anthropic.Anthropic, scenario: dict, mode: str, run: int) -> dict:
    started = time.time()
    try:
        reply = generate_reply(client, as_api_messages(scenario["messages"]), mode)
        error = None
    except Exception as exc:
        reply, error = {"content": "", "tool_calls": [], "stop_reason": None}, str(exc)
    return {
        "scenario": scenario["id"],
        "mode": mode,
        "run": run,
        "seconds": round(time.time() - started, 1),
        "reply": reply["content"],
        "stop_reason": reply["stop_reason"],
        "error": error,
        "tool_calls": [
            {
                "name": call["name"],
                "query": call["input"].get("query"),
                "hits": [hit["title"] for hit in (call.get("output") or {}).get("hits", [])],
                "error": call.get("error"),
            }
            for call in reply["tool_calls"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="Path of the JSON results file to write.")
    parser.add_argument("--runs", type=int, default=1, help="Runs per scenario and mode (replies vary between runs).")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    os.environ.update({k: v for k, v in dotenv_values(".env.local").items() if v and k not in os.environ})
    client = anthropic.Anthropic()

    jobs = [(s, mode, run) for s in SCENARIOS for mode in KNOWLEDGE_MODES for run in range(args.runs)]
    print(f"Running {len(jobs)} turns ({len(SCENARIOS)} scenarios x {len(KNOWLEDGE_MODES)} modes x {args.runs} runs)...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda job: run_one(client, *job), jobs))

    output = {
        "model": CLAUDE_MODEL,
        "effort": CLAUDE_EFFORT,
        "modes": list(KNOWLEDGE_MODES),
        "scenarios": SCENARIOS,
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    errors = [r for r in results if r["error"]]
    print(f"Done. Wrote {args.out}" + (f" ({len(errors)} errors)" if errors else ""))


if __name__ == "__main__":
    main()
