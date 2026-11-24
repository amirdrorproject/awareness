"""The agent's own log of a turn, and simple checks on the reply it wrote.

With the conversation-states module on, the agent opens each reply with a
short <log> block: the state it identified, what it rests on, the
communicative task, the move it chose, what to avoid, and its hypothesis on
the situation, if it has one. The log is cut out of the reply before the
client sees it and kept in the run trace, so identification, intent and
execution can be compared turn by turn. The checks are plain code - no model
call; whether a reply asks more than one question is judged by the research
observer (api/observer.py), since counting "?" marks miscounted it.
"""

import re

LOG_PATTERN = re.compile(r"<log>(.*?)</log>\s*", re.DOTALL)

# Line label in the log -> key in the trace.
FIELDS = {
    "מצב": "state",
    "מאפיינים": "observed",
    "משימה": "task",
    "תנועה": "move",
    "הימנעות": "avoid",
    "השערה על המצב": "hypothesis",
    # The label before Nov 2026 ("a hypothesis you hold and don't show"),
    # kept so older logs still read; a new log has only one of the two.
    "השערה מוחזקת": "held_hypothesis",
}
REQUIRED = ["state", "observed", "task", "move", "avoid"]

MOVES = [
    "הקשבה",
    "הכרה במורכבות",
    "בירור",
    "הצעת השערה",
    "העמקה",
    "התרוממות לתמונה רחבה",
    "חיבור בין פרטים",
    "התקדמות לפעולה",
]

STATE_LETTERS = "אבגדה"


def extract_log(text: str) -> tuple[dict | None, str]:
    """Splits the reply into (log, text the client sees). The log is None
    when the agent didn't write one."""
    match = LOG_PATTERN.search(text)
    if match is None:
        return None, text
    raw = match.group(1).strip()
    fields = {}
    for line in raw.splitlines():
        label, sep, value = line.partition(":")
        key = FIELDS.get(label.strip())
        if sep and key:
            fields[key] = value.strip()
    states = sorted(set(re.findall(f"([{STATE_LETTERS}])[׳']", fields.get("state", ""))))
    log = {
        **fields,
        "states": states,
        "move_type": next((m for m in MOVES if fields.get("move", "").startswith(m)), None),
        # Sent back with the reply in the history, so the agent keeps its own
        # earlier logs (and the hypotheses it held) from turn to turn.
        "raw": raw,
    }
    return log, (text[: match.start()] + text[match.end() :]).strip()


def check_reply(reply: str, log: dict | None, log_expected: bool) -> dict:
    """What the reply actually did, and the simple deviations in it."""
    words = len(reply.split())
    flags = []
    if log_expected:
        if log is None:
            flags.append("לא נכתב לוג")
        else:
            labels = {key: label for label, key in FIELDS.items()}
            missing = [labels[key] for key in REQUIRED if not log.get(key)]
            # "אין" is a valid hypothesis; only a missing line counts.
            if not (log.get("hypothesis") or log.get("held_hypothesis")):
                missing.append(labels["hypothesis"])
            if missing:
                flags.append(f"חסרים בלוג: {', '.join(missing)}")
            if log.get("move") and not log["move_type"]:
                flags.append("התנועה אינה מהרשימה")
    return {"words": words, "flags": flags}
