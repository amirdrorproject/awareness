"""Runs the same conversation openings through the agent in three variants and
writes the replies side by side:

- off:   the engine prompt alone
- states: plus the conversation-states module, without its log instruction
- log:   plus the module with the log - what the chat runs

The question it serves: does writing the log change the replies, beyond what
the states module itself changes? If "states" and "log" read alike, the log
only records; if they differ, the difference is the log's own effect.

Usage (from the repo root):
    python -m scripts.compare_turn_log --out results.json [--runs 2] [--knowledge none]
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
from api.turn_log import check_reply
from scripts.compare_knowledge import as_api_messages

VARIANTS = ("off", "states", "log")
LOG_HEADING = "לוג תור"

# One scenario per state of the strategy.
SCENARIOS = [
    {
        "id": "a_short",
        "label": "State A: a short client",
        "messages": ["אני לא מרוצה בעבודה"],
    },
    {
        "id": "b_story",
        "label": "State B: in the middle of a story",
        "messages": [
            "אז ככה. בשבוע שעבר היה לנו כנס לקוחות, ואני הייתי אחראית על כל הלוגיסטיקה. "
            "בהתחלה הכול הלך כמו שתכננתי, ואז בבוקר של הכנס הספק של ההסעות התקשר ואמר שחסרים לו שני אוטובוסים. "
            "התחלתי להתקשר לכל מי שאני מכירה, ובינתיים המנהל שלי כבר היה בדרך לשם, ו"
        ],
    },
    {
        "id": "c_whirl",
        "label": "State C: a whirlpool",
        "messages": [
            "יש לי בעיה עם המנהל שלי",
            "מה קורה ביניכם?",
            "הוא כל הזמן מבקש ממני דברים ברגע האחרון, ואז אתמול הוא גם העיר לי מול כל הצוות שהמצגת לא מוכנה, "
            "למרות שהוא שלח לי את הנתונים רק בבוקר. ואני כבר לא יודעת אם זה אני או הוא, ואם בכלל כדאי לי להישאר שם, "
            "וגם בבית אומרים לי שאני לוקחת את זה קשה מדי",
        ],
    },
    {
        "id": "d_no_request",
        "label": "State D: a clear story without a request",
        "messages": [
            "לפני חודשיים העבירו אליי את הצוות של רונית אחרי שהיא עזבה. יש שם שני עובדים ותיקים שממשיכים לפנות אליה "
            "בוואטסאפ לכל שאלה, למרות שהיא כבר בחברה אחרת. היא עונה להם, ואני מגלה על החלטות רק אחרי שהן התקבלו."
        ],
    },
    {
        "id": "e_goal",
        "label": "State E: a clear goal",
        "messages": [
            "אני רוצה להחליט עד סוף השבוע אם לבקש מהמנהלת שלי לעבור לעבודה היברידית.",
            "מה מקשה עלייך להחליט?",
            "אני חוששת שזה ייראה כאילו אני פחות מחויבת, בדיוק כשמדברים על קידום בצוות. מצד שני הנסיעות גומרות אותי",
        ],
    },
]


def run_one(client: anthropic.Anthropic, scenario: dict, variant: str, run: int, knowledge_mode: str, modules: dict) -> dict:
    started = time.time()
    try:
        reply = generate_reply(
            client,
            as_api_messages(scenario["messages"]),
            knowledge_mode,
            states_module=modules.get(variant),
        )
        error = None
    except Exception as exc:
        reply, error = {"content": "", "stop_reason": None, "trace": {}}, str(exc)
    log = reply["trace"].get("turn_log")
    return {
        "scenario": scenario["id"],
        "variant": variant,
        "run": run,
        "seconds": round(time.time() - started, 1),
        "reply": reply["content"],
        "stop_reason": reply["stop_reason"],
        "error": error,
        "log": log and {k: v for k, v in log.items() if k != "raw"},
        "checks": check_reply(reply["content"], log, log_expected=variant == "log"),
        "output_tokens": reply["trace"].get("usage", {}).get("output_tokens"),
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
    # The version in effect, read once so every turn uses the same text. The
    # "states" variant is the same module cut before its log section.
    module = get_prompt_record("module_states")
    if LOG_HEADING not in module["content"]:
        raise SystemExit(f"The module has no '{LOG_HEADING}' section to cut.")
    without_log = {**module, "content": module["content"].split(LOG_HEADING)[0].rstrip()}
    modules = {"states": without_log, "log": module}

    jobs = [(s, variant, run) for s in SCENARIOS for variant in VARIANTS for run in range(args.runs)]
    print(f"Running {len(jobs)} turns ({len(SCENARIOS)} scenarios x {len(VARIANTS)} variants x {args.runs} runs)...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda job: run_one(client, *job, args.knowledge, modules), jobs))

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
