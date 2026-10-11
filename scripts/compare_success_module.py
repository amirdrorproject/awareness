"""Runs the same conversation openings through the agent with and without the
success-analysis module and writes the replies side by side.

The question it serves: at each of the module's entry gates, does the agent
offer the direction briefly and ask for consent (rather than launching a
track), without advice, compliments or a promised discovery - and does it stay
out of the way when no gate is open?

Usage (from the repo root):
    python -m scripts.compare_success_module --out results.json [--runs 2] [--knowledge full]
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
from api.prompts import get_prompt_record
from scripts.compare_knowledge import as_api_messages

VARIANTS = ("off", "on")

# One scenario per entry gate of the module, plus a control where no gate is
# open and the module should change nothing.
SCENARIOS = [
    {
        "id": "gate_success",
        "label": "Gate 1: a success the client wants to understand",
        "messages": ["סגרתי השבוע עסקה שכולם אמרו שהיא אבודה. אני רוצה להבין מה עשיתי שם, כדי שאוכל לעשות את זה שוב"],
    },
    {
        "id": "gate_dilemma",
        "label": "Gate 2: a dilemma / decision",
        "messages": ["הציעו לי לנהל את האזור הצפוני. אני לא יודעת אם לקחת את זה או להישאר עם הצוות שלי"],
    },
    {
        "id": "gate_stuck",
        "label": "Gate 3: stuck, foggy, unsatisfied",
        "messages": ["אני עומד ביעדים כבר שנה, ובכל זאת אני מרגיש תקוע. משהו שם לא מספק אותי ואני לא יודע מה"],
    },
    {
        "id": "gate_cannot_explain",
        "label": "Gate 4: performs well but can't explain how",
        "messages": [
            "ביקשו ממני להעביר לנציגים החדשים הדרכה על שיחות חידוש, כי שלי תמיד נסגרות.",
            "מה היית רוצה שייצא לך מהשיחה שלנו לקראת ההדרכה?",
            "האמת שאני לא יודעת מה להגיד להם. אני פשוט עושה את זה, אין לי מושג איך להסביר את זה",
        ],
    },
    {
        "id": "gate_pattern",
        "label": "Gate 5: a pattern across stories",
        "messages": [
            "אני רוצה לחשוב איתך על הלקוח שכמעט עזב בחודש שעבר ובסוף נשאר.",
            "מה קרה שם שגרם לו להישאר?",
            "ישבתי איתו שעתיים בלי להציע כלום, רק שאלתי. מצחיק, זה בדיוק מה שעשיתי עם הנציג שרצה להתפטר לפני חצי שנה, וגם אז זה עבד",
        ],
    },
    {
        "id": "control_factual",
        "label": "Control: plain preparation for a meeting",
        "messages": ["יש לי מחר פגישה עם לקוח על חידוש חוזה שנתי, רוצה לחשוב איתך על זה"],
    },
]


def run_one(client: anthropic.Anthropic, scenario: dict, variant: str, run: int, knowledge_mode: str, module: dict) -> dict:
    started = time.time()
    try:
        reply = generate_reply(
            client,
            as_api_messages(scenario["messages"]),
            knowledge_mode,
            module=module if variant == "on" else None,
        )
        error = None
    except Exception as exc:
        reply, error = {"content": "", "tool_calls": [], "stop_reason": None}, str(exc)
    return {
        "scenario": scenario["id"],
        "module": variant,
        "run": run,
        "seconds": round(time.time() - started, 1),
        "reply": reply["content"],
        "stop_reason": reply["stop_reason"],
        "error": error,
        "tool_calls": [call["name"] for call in reply["tool_calls"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="Path of the JSON results file to write.")
    parser.add_argument("--runs", type=int, default=1, help="Runs per scenario and variant (replies vary between runs).")
    parser.add_argument("--knowledge", default="none", choices=KNOWLEDGE_MODES, help="Expression bank mode for every turn.")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    os.environ.update({k: v for k, v in dotenv_values(".env.local").items() if v and k not in os.environ})
    client = anthropic.Anthropic()
    # The version in effect (the newest saved in /admin, else the file), read
    # once so every turn of the comparison uses the same text.
    module = get_prompt_record("module_success")

    jobs = [(s, variant, run) for s in SCENARIOS for variant in VARIANTS for run in range(args.runs)]
    print(f"Running {len(jobs)} turns ({len(SCENARIOS)} scenarios x {len(VARIANTS)} variants x {args.runs} runs)...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda job: run_one(client, *job, args.knowledge, module), jobs))

    output = {
        "model": CLAUDE_MODEL,
        "effort": CLAUDE_EFFORT,
        "knowledge_mode": args.knowledge,
        "module_source": module["source"],
        "scenarios": SCENARIOS,
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    errors = [r for r in results if r["error"]]
    print(f"Done. Wrote {args.out}" + (f" ({len(errors)} errors)" if errors else ""))


if __name__ == "__main__":
    main()
