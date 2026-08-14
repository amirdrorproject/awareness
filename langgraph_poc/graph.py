import os
import re
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
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


graph_builder = StateGraph(GraphState)
graph_builder.add_node("classify_opening", classify_opening)
graph_builder.add_edge(START, "classify_opening")
graph_builder.add_edge("classify_opening", END)

graph = graph_builder.compile()
