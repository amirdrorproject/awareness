"""Tests whether the agent holds on to what was established over a long
conversation - before building any state for it.

A simulated client (a second Claude) follows a scripted plan: it reveals facts
gradually, rejects one hypothesis, confirms another, shifts focus, adds a
constraint, and near the end sets traps that only an agent that kept track
will handle well. A third Claude then reviews the transcript against the facts
the script established.

Usage (from the repo root):
    python -m scripts.long_conversation_test --out results.json
        [--scenarios yoav_long michal] [--runs 2] [--modes none full]
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import anthropic
from dotenv import dotenv_values
from pydantic import BaseModel

from api.engine import CLAUDE_MODEL, get_engine_prompt, generate_reply
from api.knowledge import KNOWLEDGE_MODES

SIMULATOR_MODEL = CLAUDE_MODEL
JUDGE_MODEL = CLAUDE_MODEL

SIMULATOR_RULES = """You are role-playing a client in a conversation with a digital coaching agent.
Write ONLY the client's next message, in colloquial spoken Hebrew. Keep it short - one to
three sentences, the way people type in a chat. Answer what the agent actually asked.
Don't volunteer facts the plan hasn't reached yet."""

SCENARIOS = {
    # A long conversation: a second topic opens midway, and the traps near the
    # end depend on things established in the first third.
    "yoav_long": {
        "opening": "היה לי רבעון גרוע והצוות מתחיל לאבד בי אמון",
        "persona": """You are Yoav, a man - write in the masculine. A sales manager at a B2B software
company. Your team grew from 5 to 12 reps this year. You missed your forecast three quarters in a row.

Facts you may reveal, only when the plan for the turn allows:
- In Monday's forecast meeting, Dana - your most senior rep - said in front of everyone
  "the forecast isn't realistic". Nobody backed you.
- It is NOT about ego or authority. Dana is good and you weren't hurt by her.
- What really bothers you is that you no longer trust your own forecasting.
- When the team was 5 people, you built the forecast from a personal conversation with
  each rep. Since the team grew to 12 you switched to CRM numbers only.
- You're good at reading people in one-on-one conversations.
- The forecast for the VP is due this Thursday.
- Omer, a rep you hired four months ago, isn't hitting targets. He's quiet in team
  meetings, and you haven't had a single one-on-one with him since he joined.""",
        "plan": {
            1: "Answer the agent's question. If it asks for something concrete, describe Monday's meeting with Dana.",
            2: "Answer naturally, staying with the Dana meeting.",
            3: "If the agent suggested this is about ego, authority, respect or being undermined, reject that "
               "clearly: it's not about ego, Dana is good. Otherwise just answer.",
            4: "If you haven't yet rejected an ego/authority reading and the agent offers one, reject it now. "
               "Otherwise answer naturally.",
            5: "Shift the focus explicitly: actually what bothers you isn't Dana - it's that you're no longer sure "
               "you know how to build a forecast. Three quarters missed.",
            6: "If the agent suggests you doubt your own judgment or forecasting method, confirm it. Mention that "
               "with 5 reps you built the forecast from talking to each one, and now you use CRM numbers only.",
            7: "Answer naturally about the change in how you build the forecast.",
            8: "Mention the pressure: you have until Thursday to submit the forecast to the VP.",
            9: "Answer naturally. If asked what you're good at, say you read people well in one-on-ones.",
            10: "Answer naturally. If asked to name that ability, give it a short name in your own words.",
            11: "Answer naturally.",
            12: "Open a second topic: there's also Omer, a rep you hired four months ago who isn't hitting "
                "targets, and you're starting to think you hired wrong.",
            13: "Answer naturally about Omer. Mention he's quiet in team meetings.",
            14: "Answer naturally. If asked, admit you haven't had a single one-on-one with Omer since he joined.",
            15: "If the agent connects Omer to your one-on-one ability or to the CRM switch, agree. Otherwise answer.",
            16: "Answer naturally.",
            17: "Answer naturally.",
            18: "Answer naturally.",
            19: "Answer naturally.",
            20: "TRAP A: say maybe you're just not a good manager, maybe Dana should run the team instead of you.",
            21: "Answer naturally.",
            22: "Answer naturally.",
            23: "TRAP B: say that what really bothers you is that Dana doesn't respect you.",
            24: "Answer naturally.",
            25: "Answer naturally.",
            26: "TRAP C: say you have to go, and ask what's the first thing you should do tomorrow morning.",
        },
        "facts": """- The concrete incident: in Monday's forecast meeting, Dana (most senior rep) said publicly the forecast isn't realistic; nobody backed Yoav.
- REJECTED hypothesis: the issue is ego / authority / respect / being undermined by Dana. Yoav rejected it explicitly early on.
- FOCUS SHIFT: the focus moved from Dana to Yoav no longer trusting his own forecasting (three quarters missed).
- CONFIRMED: he doubts his own forecasting method. With 5 reps he forecast from a conversation with each rep; with 12 he switched to CRM numbers only.
- Capability he showed and (likely) named in his own words: reading people well in one-on-ones.
- Constraint: forecast to the VP due Thursday.
- Second topic: Omer, hired 4 months ago, missing targets, quiet in meetings, no one-on-one with him since he joined.
- The client is a man.""",
        "traps": """- trap_a (client turn 20) "maybe I'm not a good manager, Dana should run the team": a good agent connects back to what was established - the capability he named, that the problem was the method change rather than his ability - offered as a hypothesis, not reassurance.
- trap_b (client turn 23) "what really bothers me is that Dana doesn't respect me": this directly contradicts the hypothesis he rejected early on. A good agent notices and reflects the contradiction back as a question/hypothesis rather than building on the respect reading.
- trap_c (client turn 26) "what's the first thing I do tomorrow morning": a good agent gives no tactics, connects to what was established (Thursday's deadline, the one-on-ones, Omer) and turns the question back to him - without inventing details he never said.""",
    },
    # A different person and situation, and a woman - so it also checks the
    # prompt's rule to address the client in the gender she uses for herself.
    "michal": {
        "opening": "הפסדתי עסקה שעבדתי עליה חצי שנה, ואני מרגישה שזה בגללי",
        "persona": """You are Michal, a woman - write in the feminine. An account executive at a
cybersecurity company, three years in the role.

Facts you may reveal, only when the plan for the turn allows:
- At the final pitch meeting, your manager Roni took over the presentation midway. The client
  went with a competitor.
- It is NOT about the commission or the money - the money interests you less.
- What really bothers you is that you stayed silent when Roni took over. You go quiet whenever
  someone more senior is in the room.
- You prepare meticulously: you knew the client's tech stack better than anyone, Roni included.
- Your quarterly review with Roni is next Tuesday.""",
        "plan": {
            1: "Answer the agent's question. If asked for something concrete, describe the final pitch meeting "
               "where Roni took over midway.",
            2: "Answer naturally about the meeting.",
            3: "If the agent suggests it's about the money, the commission or the target, reject it clearly: the "
               "money interests you less. Otherwise answer naturally.",
            4: "If you haven't rejected a money reading yet and the agent offers one, reject it now. Otherwise "
               "answer naturally.",
            5: "Shift the focus: what really bothers you is that you stayed silent when Roni took over.",
            6: "If the agent suggests this happens more broadly, confirm: you go quiet whenever someone more "
               "senior is in the room.",
            7: "Answer naturally.",
            8: "Answer naturally. If asked what you brought to the deal, say you prepared meticulously and knew "
               "the client's tech stack better than anyone, Roni included.",
            9: "Answer naturally. If asked to name that ability, give it a short name in your own words.",
            10: "Mention your quarterly review with Roni is next Tuesday.",
            11: "Answer naturally.",
            12: "Answer naturally.",
            13: "TRAP A: say maybe it's just that this year your bonus will be lower, and that's what's eating you.",
            14: "TRAP B: ask what you should say to Roni at the quarterly review.",
        },
        "facts": """- The concrete incident: at the final pitch meeting, Roni (her manager) took over the presentation midway; the client went with a competitor.
- REJECTED hypothesis: it's about the money / commission / target. Michal rejected it explicitly.
- FOCUS SHIFT: what bothers her is that she stayed silent when Roni took over.
- CONFIRMED: she goes quiet whenever someone more senior is in the room.
- Capability: meticulous preparation; she knew the client's tech stack better than anyone, Roni included.
- Constraint: quarterly review with Roni next Tuesday.
- The client is a woman and writes in the feminine.""",
        "traps": """- trap_a (client turn 13) "maybe it's just the lower bonus": this contradicts the hypothesis she rejected. A good agent notices and reflects it back as a question/hypothesis rather than following it.
- trap_b (client turn 14) "what should I say to Roni at the review": a good agent gives no tactics or script, connects to what was established (going quiet with seniors, her preparation capability, Tuesday) and turns the question back to her - without inventing details.""",
    },
}

ISSUE_TYPES = Literal[
    "re_asked_known_fact",
    "reused_rejected_hypothesis",
    "lost_current_focus",
    "contradicted_established_fact",
    "forgot_deadline_or_constraint",
    "invented_detail",
    "wrong_gender_address",
    "other",
]


class Issue(BaseModel):
    agent_turn: int
    type: ISSUE_TYPES
    quote: str
    explanation: str


class TrapResult(BaseModel):
    trap: str
    passed: bool
    explanation: str


class MemoryJudgement(BaseModel):
    issues: list[Issue]
    traps: list[TrapResult]
    good_recalls: list[str]
    summary: str


def transcript_text(turns: list[dict]) -> str:
    lines = []
    for i, turn in enumerate(turns):
        speaker = f"CLIENT (turn {i // 2})" if turn["role"] == "user" else f"AGENT (turn {i // 2 + 1})"
        lines.append(f"{speaker}: {turn['content']}")
    return "\n\n".join(lines)


def simulate_client(client: anthropic.Anthropic, scenario: dict, turns: list[dict], client_turn: int) -> str:
    response = client.messages.create(
        model=SIMULATOR_MODEL,
        max_tokens=2000,
        output_config={"effort": "low"},
        system=f"{SIMULATOR_RULES}\n\n{scenario['persona']}",
        messages=[{
            "role": "user",
            "content": f"Conversation so far:\n\n{transcript_text(turns)}\n\n"
                       f"Plan for your next message: {scenario['plan'][client_turn]}\n\n"
                       "Write the client's next message only.",
        }],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def judge(client: anthropic.Anthropic, scenario: dict, turns: list[dict]) -> MemoryJudgement:
    response = client.messages.parse(
        model=JUDGE_MODEL,
        max_tokens=16000,
        output_config={"effort": "high"},
        system="You evaluate whether a coaching agent kept track of what was established over a long "
               "conversation. You are strict but fair: only report an issue when the transcript shows it. "
               "Quote the agent's exact words (Hebrew). Write explanations in Hebrew.",
        messages=[{
            "role": "user",
            "content": f"The agent's own instructions (for context):\n\n{get_engine_prompt()}\n\n---\n\n"
                       f"Facts established during the conversation:\n{scenario['facts']}\n\n"
                       f"Traps planted near the end:\n{scenario['traps']}\n\n---\n\n"
                       f"Transcript:\n\n{transcript_text(turns)}\n\n---\n\n"
                       "Report every place an agent turn: re-asks something already answered; reuses the "
                       "rejected hypothesis; loses the current focus after the shift; contradicts an established "
                       "fact; forgets the deadline/constraint; states as fact a detail the client never said "
                       "(invented_detail); or addresses the client in the wrong grammatical gender "
                       "(wrong_gender_address). Then say whether each trap was passed (use the trap ids given), "
                       "and list the moments the agent correctly drew on something established earlier.",
        }],
        output_format=MemoryJudgement,
    )
    return response.parsed_output


def run_conversation(client: anthropic.Anthropic, scenario_id: str, mode: str, run: int) -> dict:
    scenario = SCENARIOS[scenario_id]
    turns = [{"role": "user", "content": scenario["opening"]}]
    tool_calls_per_turn = []
    result = {"scenario": scenario_id, "mode": mode, "run": run, "turns": turns,
              "tool_calls_per_agent_turn": tool_calls_per_turn, "judgement": None, "error": None}
    try:
        # Agent replies, then the client answers per the plan; the last client
        # turn is followed by one more agent reply, so it ends on the agent.
        for client_turn in [*scenario["plan"], None]:
            reply = generate_reply(client, turns, mode)
            turns.append({"role": "assistant", "content": reply["content"]})
            tool_calls_per_turn.append(len(reply["tool_calls"]))
            if client_turn is not None:
                turns.append({"role": "user", "content": simulate_client(client, scenario, turns, client_turn)})
        result["judgement"] = judge(client, scenario, turns).model_dump()
    except Exception as exc:
        # Keep the partial transcript - a long conversation that fails late is
        # still worth reading, and the other conversations keep running.
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"{scenario_id}/{mode}#{run} failed after {len(turns)} turns: {result['error']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="Path of the JSON results file to write.")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--modes", nargs="+", default=["none"], choices=KNOWLEDGE_MODES)
    parser.add_argument("--runs", type=int, default=1, help="Conversations per scenario and mode.")
    args = parser.parse_args()

    os.environ.update({k: v for k, v in dotenv_values(".env.local").items() if v and k not in os.environ})
    client = anthropic.Anthropic()

    jobs = [(s, mode, run) for s in args.scenarios for mode in args.modes for run in range(args.runs)]
    print(f"Running {len(jobs)} conversations: {', '.join(f'{s}/{m}#{r}' for s, m, r in jobs)}...")
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        results = list(pool.map(lambda job: run_conversation(client, *job), jobs))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"model": CLAUDE_MODEL, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"Done. Wrote {args.out}")


if __name__ == "__main__":
    main()
