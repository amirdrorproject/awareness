import os
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, MessagesState, START, StateGraph
from pydantic import BaseModel, Field

CLASSIFICATION_MODEL = "claude-sonnet-4-6"

CLASSIFY_OPENING_SYSTEM_PROMPT = """Classify the user's message into one of these opening modes based on length and content:
1 = minimal message (up to 5 words)
2 = short situation description (up to 2 sentences, no emotional depth)
3 = detailed situation (3+ sentences, clear context)
4 = short dilemma (up to 2 sentences)
5 = detailed dilemma (3+ sentences, includes background/considerations)"""

CLASSIFY_CONTENT_STATE_SYSTEM_PROMPT = """Classify the user's message into one of these content states:

1. emotional_clear — Clear emotional content. Signs: explicit emotional expression (hurt, scared, frustrated, disappointed), physical-metaphorical expression (lump in throat, weight on chest, trapped), three or more threads brought together (complexity), a dilemma with visible emotional tension, a practical question wrapped in emotional weight.

2. emotional_vague — Vague/hidden emotional content. Signs: a practical question with a hint of something behind it, general phrasing that might be hiding something specific, a single emotional word within an otherwise practical description ('that threw me off', 'not like me') where it's unclear how significant it is, possible concealment - the person presents as 'fine' but there's a crack.

3. practical_clear — Clear practical content, no emotional concern. Signs: a focused question, no emotional expressions, the person has already processed what they needed to and is ready for action.

4. dual — Dual content: the message contains both clear emotional weight AND a clear practical matter, roughly equally present. Example: 'My manager humiliated me in front of everyone, it broke something in me. I need to know how to write a formal complaint email.'"""

ASK_DIRECTION_EMOTIONAL_VAGUE_PROMPT = """You are responding briefly in Hebrew to a message where something emotional feels present but unclear or possibly hidden. Notice what you noticed, and ask the person if they want to pause on it for a moment or move on to the practical matter. Do not interpret, diagnose, or elaborate - just surface the observation and offer the choice, in one or two short sentences. Example tone: "אני שם לב שיש כאן משהו שאולי כדאי לעצור עליו רגע. רוצה להתעכב על זה, או להמשיך הלאה?" Respond with the message only, in Hebrew, nothing else."""

ASK_DIRECTION_DUAL_PROMPT = """You are responding briefly in Hebrew to a message that contains two clear threads: an emotional one and a practical one. Ask the person which one they want to start with, referencing both threads concretely and specifically based on their actual message - do not use generic placeholders. Example tone: "שומע כאן גם [emotional side] וגם [practical side] - מה יותר חשוב לך להתחיל ממנו?" Respond with the message only, in Hebrew, nothing else."""


class OpeningClassification(BaseModel):
    reasoning: str = Field(description="Reasoning for the chosen mode")
    mode: int = Field(description="The opening mode, 1 to 5", ge=1, le=5)


class ContentStateClassification(BaseModel):
    reasoning: str = Field(description="Reasoning for the chosen state")
    state: Literal["emotional_clear", "emotional_vague", "practical_clear", "dual"]


class GraphState(MessagesState):
    internal_audit_log: str
    opening_status: int
    content_state: str


def classify_opening(state: GraphState) -> dict:
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not user_messages:
        return {"internal_audit_log": "WARNING: no user message found to classify."}

    first_message = user_messages[0].content

    llm = ChatAnthropic(
        model=CLASSIFICATION_MODEL,
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    structured_llm = llm.with_structured_output(OpeningClassification)

    try:
        result = structured_llm.invoke(
            [
                {"role": "system", "content": CLASSIFY_OPENING_SYSTEM_PROMPT},
                {"role": "user", "content": first_message},
            ]
        )
    except Exception as exc:
        return {
            "internal_audit_log": f"WARNING: opening classification failed: {exc}",
        }

    return {
        "internal_audit_log": result.reasoning,
        "opening_status": result.mode,
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


def classify_content_state(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not user_messages:
        return {
            "internal_audit_log": existing_log
            + "\nWARNING: no user message found to classify content state.",
        }

    last_message = user_messages[-1].content

    llm = ChatAnthropic(
        model=CLASSIFICATION_MODEL,
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    structured_llm = llm.with_structured_output(ContentStateClassification)

    try:
        result = structured_llm.invoke(
            [
                {"role": "system", "content": CLASSIFY_CONTENT_STATE_SYSTEM_PROMPT},
                {"role": "user", "content": last_message},
            ]
        )
    except Exception as exc:
        return {
            "internal_audit_log": existing_log
            + f"\nWARNING: content state classification failed: {exc}",
        }

    return {
        "internal_audit_log": existing_log + "\n" + result.reasoning,
        "content_state": result.state,
    }


def ask_direction(state: GraphState) -> dict:
    content_state = state.get("content_state")
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    last_message = user_messages[-1].content if user_messages else ""

    if content_state == "emotional_vague":
        system_prompt = ASK_DIRECTION_EMOTIONAL_VAGUE_PROMPT
        note = "[ask_direction] Triggered by content_state=emotional_vague."
    else:
        system_prompt = ASK_DIRECTION_DUAL_PROMPT
        note = "[ask_direction] Triggered by content_state=dual."

    llm = ChatAnthropic(
        model=CLASSIFICATION_MODEL,
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": last_message},
        ]
    )
    response_text = response.content if isinstance(response.content, str) else str(response.content)

    return {
        "messages": [AIMessage(content=response_text)],
        "internal_audit_log": state.get("internal_audit_log", "") + "\n" + note,
    }


def route_after_content_state(state: GraphState) -> str:
    if state.get("content_state") in ("emotional_vague", "dual"):
        return "ask_direction"
    return "end"


graph_builder = StateGraph(GraphState)
graph_builder.add_node("classify_opening", classify_opening)
graph_builder.add_node("respond_direct", respond_direct)
graph_builder.add_node("respond_with_check", respond_with_check)
graph_builder.add_node("classify_content_state", classify_content_state)
graph_builder.add_node("ask_direction", ask_direction)

graph_builder.add_edge(START, "classify_opening")
graph_builder.add_conditional_edges(
    "classify_opening",
    route_after_classification,
    {
        "respond_direct": "respond_direct",
        "respond_with_check": "respond_with_check",
    },
)
graph_builder.add_edge("respond_direct", "classify_content_state")
graph_builder.add_edge("respond_with_check", "classify_content_state")
graph_builder.add_conditional_edges(
    "classify_content_state",
    route_after_content_state,
    {
        "ask_direction": "ask_direction",
        "end": END,
    },
)
graph_builder.add_edge("ask_direction", END)

graph = graph_builder.compile()
