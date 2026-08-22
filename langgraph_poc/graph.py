import os
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, MessagesState, START, StateGraph
from pinecone import Pinecone
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

CLASSIFY_DIRECTION_CHOICE_SYSTEM_PROMPT = """The user was just asked whether they want to pause on an emotional thread that was noticed, or continue toward the practical matter. Classify their reply as 'pause' (they want to stay with/explore the emotional thread) or 'continue' (they want to move to the practical matter)."""

BUILD_EXPRESSIONS_TABLE_SYSTEM_PROMPT = """Scan the client's words across the conversation for emotionally meaningful expressions - explicit, implied, or physical/somatic. For each expression found, match it against the retrieved bank content provided below and produce one table row per expression, using these exact field definitions:

- row_number: sequential row number, starting from 1
- expression: the exact quote from the client's own words
- expression_type: one of "גלוי" (explicit), "מרומז" (implied), or "פיזי" (physical/somatic)
- bank_name: the name of the bank (module) the matched expression came from
- matched_expression: the specific matching expression found in the bank content
- match_level: one of "זהה" (identical), "דומה" (similar), "קרוב" (close), or "מנוגד" (opposite)

Only use the bank content provided below as the source for bank_name and matched_expression - do not invent matches that aren't grounded in it."""

BUILD_BLOCKS_SYSTEM_PROMPT = """Review the table rows below and group them into blocks (topic groups). Each block should have a short, factual-neutral topic phrase - not emotional or interpretive language. Every row must be assigned to exactly one block_id - no orphaned rows. Decide the number of blocks naturally based on the content; do not force a specific count."""


class OpeningClassification(BaseModel):
    reasoning: str = Field(description="Reasoning for the chosen mode")
    mode: int = Field(description="The opening mode, 1 to 5", ge=1, le=5)


class ContentStateClassification(BaseModel):
    reasoning: str = Field(description="Reasoning for the chosen state")
    state: Literal["emotional_clear", "emotional_vague", "practical_clear", "dual"]


class DirectionChoiceClassification(BaseModel):
    reasoning: str = Field(description="Reasoning for the chosen direction")
    choice: Literal["pause", "continue"]


class TableRow(BaseModel):
    row_number: int
    expression: str
    expression_type: Literal["גלוי", "מרומז", "פיזי"]
    bank_name: str
    matched_expression: str
    match_level: Literal["זהה", "דומה", "קרוב", "מנוגד"]
    block_id: int


class ExpressionsTableResult(BaseModel):
    reasoning: str
    rows: list[TableRow]


class Block(BaseModel):
    block_id: int
    topic: str


class BuildBlocksResult(BaseModel):
    reasoning: str
    blocks: list[Block]
    rows: list[TableRow]


class GraphState(MessagesState):
    internal_audit_log: str
    opening_status: int
    content_state: str
    direction_choice: str
    expressions_content: list[dict]
    expressions_table: list[dict]
    blocks: list[dict]


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


def route_after_respond_direct(state: GraphState) -> str:
    if state.get("opening_status") == 1:
        return "end"
    return "continue"


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
    if state.get("content_state") == "emotional_clear":
        return "retrieve_expressions_content"
    return "end"


def _extract_hit_value(hit, *keys):
    # hit may be a plain dict or a typed SDK object - try dict-style access
    # (both with and without a leading underscore) before falling back to
    # attribute-style access, since we can't be certain which this SDK version uses.
    for key in keys:
        if isinstance(hit, dict) and key in hit:
            return hit[key]
        try:
            value = hit.get(key)
        except AttributeError:
            value = None
        if value is not None:
            return value
        value = getattr(hit, key, None)
        if value is not None:
            return value
    return None


def retrieve_expressions_content(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    query_text = " ".join(
        m.content if isinstance(m.content, str) else str(m.content)
        for m in human_messages
    ).strip()

    if not query_text:
        return {
            "internal_audit_log": existing_log
            + "\nWARNING: no user messages found to build expressions_content query.",
        }

    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    if not api_key or not index_name:
        return {
            "internal_audit_log": existing_log
            + "\nWARNING: PINECONE_API_KEY/PINECONE_INDEX_NAME not set - skipping expressions content retrieval.",
        }

    try:
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)

        # Auto-detect which namespace actually has data, same as /api/pinecone-query.
        # Not applying a module/doc_type filter for "bank" content yet - we haven't
        # confirmed the actual metadata values in this index, and a wrong filter
        # would silently return zero results rather than erroring.
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces") or {}
        namespace = ""
        if not (namespaces.get("") or {}).get("vector_count"):
            populated = [
                name for name, info in namespaces.items() if (info or {}).get("vector_count")
            ]
            if populated:
                namespace = populated[0]

        results = index.search(
            namespace=namespace,
            query={"inputs": {"text": query_text}, "top_k": 5},
        )
        hits = (results.get("result") or {}).get("hits") or []

        expressions_content = []
        for hit in hits:
            fields = _extract_hit_value(hit, "fields") or {}
            expressions_content.append(
                {
                    "id": _extract_hit_value(hit, "_id", "id"),
                    "score": _extract_hit_value(hit, "_score", "score"),
                    "text": fields.get("text"),
                    "module": fields.get("module"),
                    "chunk_title": fields.get("chunk_title"),
                    "doc_type": fields.get("doc_type"),
                }
            )
    except Exception as exc:
        return {
            "internal_audit_log": existing_log
            + f"\nWARNING: expressions content retrieval failed: {exc}",
        }

    note = f"[retrieve_expressions_content] Retrieved {len(expressions_content)} bank chunks for expression matching."
    return {
        "expressions_content": expressions_content,
        "internal_audit_log": existing_log + "\n" + note,
    }


def build_expressions_table(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    conversation_text = "\n".join(
        m.content if isinstance(m.content, str) else str(m.content)
        for m in human_messages
    )

    expressions_content = state.get("expressions_content") or []
    bank_content_text = "\n\n".join(
        f"[{chunk.get('module')} - {chunk.get('chunk_title')}]\n{chunk.get('text')}"
        for chunk in expressions_content
    )

    llm = ChatAnthropic(
        model=CLASSIFICATION_MODEL,
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    structured_llm = llm.with_structured_output(ExpressionsTableResult)

    user_content = (
        f"Client conversation:\n{conversation_text}\n\n"
        f"Retrieved bank content:\n{bank_content_text}"
    )

    try:
        result = structured_llm.invoke(
            [
                {"role": "system", "content": BUILD_EXPRESSIONS_TABLE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
        )
    except Exception as exc:
        return {
            "internal_audit_log": existing_log
            + f"\nWARNING: build_expressions_table failed: {exc}",
        }

    rows = [row.model_dump() for row in result.rows]
    bank_count = len({row["bank_name"] for row in rows})
    note = f"[build_expressions_table] Identified {len(rows)} expressions across {bank_count} banks."

    return {
        "expressions_table": rows,
        "internal_audit_log": existing_log + "\n" + note,
    }


def build_blocks(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    expressions_table = state.get("expressions_table") or []

    if not expressions_table:
        return {
            "internal_audit_log": existing_log
            + "\nWARNING: no expressions_table rows found to build blocks.",
        }

    rows_text = "\n".join(
        f"{row.get('row_number')}. expression={row.get('expression')!r}, "
        f"expression_type={row.get('expression_type')}, bank_name={row.get('bank_name')}, "
        f"matched_expression={row.get('matched_expression')!r}, match_level={row.get('match_level')}"
        for row in expressions_table
    )

    llm = ChatAnthropic(
        model=CLASSIFICATION_MODEL,
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    structured_llm = llm.with_structured_output(BuildBlocksResult)

    try:
        result = structured_llm.invoke(
            [
                {"role": "system", "content": BUILD_BLOCKS_SYSTEM_PROMPT},
                {"role": "user", "content": rows_text},
            ]
        )
    except Exception as exc:
        return {
            "internal_audit_log": existing_log
            + f"\nWARNING: build_blocks failed: {exc}",
        }

    blocks = [block.model_dump() for block in result.blocks]
    rows = [row.model_dump() for row in result.rows]
    note = f"[build_blocks] Organized {len(rows)} rows into {len(blocks)} blocks."

    return {
        "blocks": blocks,
        "expressions_table": rows,
        "internal_audit_log": existing_log + "\n" + note,
    }


def classify_direction_choice(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not user_messages:
        return {
            "internal_audit_log": existing_log
            + "\nWARNING: no user message found to classify direction choice.",
        }

    last_message = user_messages[-1].content

    llm = ChatAnthropic(
        model=CLASSIFICATION_MODEL,
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    structured_llm = llm.with_structured_output(DirectionChoiceClassification)

    try:
        result = structured_llm.invoke(
            [
                {"role": "system", "content": CLASSIFY_DIRECTION_CHOICE_SYSTEM_PROMPT},
                {"role": "user", "content": last_message},
            ]
        )
    except Exception as exc:
        return {
            "internal_audit_log": existing_log
            + f"\nWARNING: direction choice classification failed: {exc}",
        }

    return {
        "internal_audit_log": existing_log + "\n" + result.reasoning,
        "direction_choice": result.choice,
    }


def route_from_start(state: GraphState) -> str:
    if (
        state.get("content_state") in ("emotional_vague", "dual")
        and state.get("direction_choice") is None
    ):
        return "classify_direction_choice"  # ask_direction just asked a question - classify the reply
    if state.get("opening_status") is not None:
        return "classify_content_state"  # opening already handled in a previous turn - skip
    return "classify_opening"  # first turn - no opening_status yet


def build_graph_builder() -> StateGraph:
    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("classify_opening", classify_opening)
    graph_builder.add_node("respond_direct", respond_direct)
    graph_builder.add_node("respond_with_check", respond_with_check)
    graph_builder.add_node("classify_content_state", classify_content_state)
    graph_builder.add_node("ask_direction", ask_direction)
    graph_builder.add_node("classify_direction_choice", classify_direction_choice)
    graph_builder.add_node("retrieve_expressions_content", retrieve_expressions_content)
    graph_builder.add_node("build_expressions_table", build_expressions_table)
    graph_builder.add_node("build_blocks", build_blocks)

    graph_builder.add_conditional_edges(
        START,
        route_from_start,
        {
            "classify_opening": "classify_opening",
            "classify_content_state": "classify_content_state",
            "classify_direction_choice": "classify_direction_choice",
        },
    )
    graph_builder.add_conditional_edges(
        "classify_opening",
        route_after_classification,
        {
            "respond_direct": "respond_direct",
            "respond_with_check": "respond_with_check",
        },
    )
    graph_builder.add_conditional_edges(
        "respond_direct",
        route_after_respond_direct,
        {
            "continue": "classify_content_state",
            "end": END,
        },
    )
    graph_builder.add_edge("respond_with_check", END)
    graph_builder.add_conditional_edges(
        "classify_content_state",
        route_after_content_state,
        {
            "ask_direction": "ask_direction",
            "retrieve_expressions_content": "retrieve_expressions_content",
            "end": END,
        },
    )
    graph_builder.add_edge("ask_direction", END)
    graph_builder.add_edge("classify_direction_choice", END)
    graph_builder.add_edge("retrieve_expressions_content", "build_expressions_table")
    graph_builder.add_edge("build_expressions_table", "build_blocks")
    graph_builder.add_edge("build_blocks", END)

    return graph_builder


graph = build_graph_builder().compile()
