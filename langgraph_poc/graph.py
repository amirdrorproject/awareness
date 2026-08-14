import os
import re
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

CLASSIFICATION_MODEL = "claude-sonnet-4-6"

CLASSIFY_OPENING_SYSTEM_PROMPT = """Classify the user's message into one of these opening modes based on length and content:
1 = minimal message (up to 5 words)
2 = short situation description (up to 2 sentences, no emotional depth)
3 = detailed situation (3+ sentences, clear context)
4 = short dilemma (up to 2 sentences)
5 = detailed dilemma (3+ sentences, includes background/considerations)
Respond with ===INTERNAL=== followed by your reasoning and the chosen mode number, then ===RESPONSE=== followed by nothing (leave empty for now - this is a classification-only test)."""

MARKER_PATTERN = re.compile(r"===INTERNAL===(.*?)===RESPONSE===", re.DOTALL)
MODE_NUMBER_PATTERN = re.compile(r"(?<!\d)[1-5](?!\d)")


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    internal_audit_log: str
    opening_status: int


def classify_opening(state: GraphState) -> dict:
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not user_messages:
        return {"internal_audit_log": "WARNING: no user message found to classify."}

    first_message = user_messages[0].content

    llm = ChatAnthropic(
        model=CLASSIFICATION_MODEL,
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    response = llm.invoke(
        [
            {"role": "system", "content": CLASSIFY_OPENING_SYSTEM_PROMPT},
            {"role": "user", "content": first_message},
        ]
    )
    raw = response.content if isinstance(response.content, str) else str(response.content)

    match = MARKER_PATTERN.search(raw)
    if not match:
        return {
            "internal_audit_log": f"WARNING: could not parse ===INTERNAL===/===RESPONSE=== markers from model output: {raw!r}",
        }

    internal_section = match.group(1).strip()
    numbers = MODE_NUMBER_PATTERN.findall(internal_section)
    if not numbers:
        return {
            "internal_audit_log": f"WARNING: no mode number (1-5) found in internal reasoning: {internal_section!r}",
        }

    return {
        "internal_audit_log": internal_section,
        "opening_status": int(numbers[-1]),
    }


def respond_direct(state: GraphState) -> dict:
    opening_status = state.get("opening_status")

    if opening_status == 1:
        response_text = "למען הסדר הטוב, אני העוזר הדיגיטלי של אמיר דרור. אני מקשיב לך."
        note = "[respond_direct] Triggered by opening_status=1 (minimal message) - proceeding without asking permission."
    else:
        response_text = "למען הסדר הטוב, אני העוזר הדיגיטלי של אמיר דרור."
        note = f"[respond_direct] Triggered by opening_status={opening_status} (detailed situation/dilemma) - proceeding without asking permission."

    return {
        "messages": [AIMessage(content=response_text)],
        "internal_audit_log": state.get("internal_audit_log", "") + "\n" + note,
    }


def respond_with_check(state: GraphState) -> dict:
    opening_status = state.get("opening_status")

    if opening_status == 2:
        acknowledgment = "אני מבין שיש כאן משהו שתרצה לברר."
        note = "[respond_with_check] Triggered by opening_status=2 (short situation description)."
    elif opening_status == 4:
        acknowledgment = "אני מבין שיש כאן התלבטות."
        note = "[respond_with_check] Triggered by opening_status=4 (short dilemma)."
    else:
        acknowledgment = "אני מבין שיש כאן משהו שתרצה לברר."
        note = f"WARNING: opening_status missing/invalid ({opening_status!r}) - falling back to respond_with_check as a safe default."

    response_text = f"למען הסדר הטוב, אני העוזר הדיגיטלי של אמיר דרור. {acknowledgment} רוצה להמשיך?"

    return {
        "messages": [AIMessage(content=response_text)],
        "internal_audit_log": state.get("internal_audit_log", "") + "\n" + note,
    }


def route_after_classification(state: GraphState) -> str:
    if state.get("opening_status") in (1, 3, 5):
        return "respond_direct"
    return "respond_with_check"


graph_builder = StateGraph(GraphState)
graph_builder.add_node("classify_opening", classify_opening)
graph_builder.add_node("respond_direct", respond_direct)
graph_builder.add_node("respond_with_check", respond_with_check)

graph_builder.add_edge(START, "classify_opening")
graph_builder.add_conditional_edges(
    "classify_opening",
    route_after_classification,
    {
        "respond_direct": "respond_direct",
        "respond_with_check": "respond_with_check",
    },
)
graph_builder.add_edge("respond_direct", END)
graph_builder.add_edge("respond_with_check", END)

graph = graph_builder.compile()
