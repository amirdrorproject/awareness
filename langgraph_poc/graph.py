import json
import os
import re
from pathlib import Path
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, MessagesState, START, StateGraph
from pinecone import Pinecone
from pydantic import BaseModel, Field

CLASSIFICATION_MODEL = "claude-sonnet-4-6"

SUCCESS_ANALYSIS_RAG_TOP_K = 3

# Fixed response text sent to the client verbatim, with no LLM call involved in
# producing it, lives in messages.json. System prompts that guide LLM-generated
# text (never sent to the client as-is) live in prompts.json. Both are loaded
# once here at import time - nodes read from these dicts at call time, not from
# disk per call.
with open(Path(__file__).parent / "messages.json", "r", encoding="utf-8") as _messages_file:
    MESSAGES: dict = json.load(_messages_file)

with open(Path(__file__).parent / "prompts.json", "r", encoding="utf-8") as _prompts_file:
    PROMPTS: dict = json.load(_prompts_file)

# Developer-owned prompts (classifier/extraction system prompts and fixed prompt
# scaffolding). Unlike prompts.json/messages.json, not exposed in Keystatic.
with open(Path(__file__).parent / "const_prompts.json", "r", encoding="utf-8") as _const_prompts_file:
    CONST_PROMPTS: dict = json.load(_const_prompts_file)


def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=CLASSIFICATION_MODEL,
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )


def _content_text(message) -> str:
    return message.content if isinstance(message.content, str) else str(message.content)


def _human_messages(messages) -> list:
    return [m for m in messages if isinstance(m, HumanMessage)]


def _format_blocks(blocks: list[dict], include_color: bool = False) -> str:
    if include_color:
        return "\n".join(
            f"{block.get('block_id')}. topic={block.get('topic')!r}, color={block.get('color')!r}"
            for block in blocks
        )
    return "\n".join(
        f"{block.get('block_id')}. topic={block.get('topic')!r}" for block in blocks
    )


def _detect_namespace(index) -> str:
    # Auto-detect which namespace actually has data (same as /api/pinecone-query):
    # the default namespace if populated, otherwise the first populated one.
    stats = index.describe_index_stats()
    namespaces = stats.get("namespaces") or {}
    namespace = ""
    if not (namespaces.get("") or {}).get("vector_count"):
        populated = [
            name for name, info in namespaces.items() if (info or {}).get("vector_count")
        ]
        if populated:
            namespace = populated[0]
    return namespace


PRACTICAL_TRACK_PAUSE_PHRASE = "יצאתי לחשוב"

PRACTICAL_TRACK_PAUSE_STRIP_CHARS = " \t\n\r.,!?;:\"'״׳"


class OpeningClassification(BaseModel):
    reasoning: str = Field(description="Reasoning for the chosen mode")
    mode: int = Field(description="The opening mode, 1 to 5", ge=1, le=5)


class ContentStateClassification(BaseModel):
    reasoning: str = Field(description="Reasoning for the chosen state")
    state: Literal["emotional_clear", "emotional_vague", "practical_clear", "dual"]


class DirectionChoiceClassification(BaseModel):
    reasoning: str = Field(description="Reasoning for the chosen direction")
    choice: Literal["pause", "continue"]


class YesNoOtherResult(BaseModel):
    reasoning: str
    answer: Literal["yes", "no", "other"]


class TableRow(BaseModel):
    row_number: int
    expression: str
    expression_type: Literal["גלוי", "מרומז", "פיזי"]
    bank_name: str
    matched_expression: str
    match_level: Literal["זהה", "דומה", "קרוב", "מנוגד"]
    block_id: int | None = None


class ExpressionsTableResult(BaseModel):
    reasoning: str
    rows: list[TableRow]


class Block(BaseModel):
    block_id: int
    topic: str
    color: str | None = None


class BuildBlocksResult(BaseModel):
    reasoning: str
    blocks: list[Block]
    rows: list[TableRow]


class BlockColorAssignment(BaseModel):
    block_id: int
    color: str


class ColorBlocksResult(BaseModel):
    reasoning: str
    colors: list[BlockColorAssignment]


class PresentChoiceResult(BaseModel):
    reasoning: str
    intent: Literal["practical", "deepen"]
    current_block_id: int | None = None


class BlockTargetResult(BaseModel):
    reasoning: str
    current_block_id: int


class DeepenRoundResult(BaseModel):
    reasoning: str
    new_rows: list[TableRow]
    block_updates: list[Block]


class FocusChoiceResult(BaseModel):
    reasoning: str
    current_block_id: int


class ReadinessResult(BaseModel):
    reasoning: str
    readiness: Literal["ready", "half_ready", "not_ready"]


class ScopeCreepResult(BaseModel):
    reasoning: str
    status: Literal["in_scope", "drifting"]


class FromPracticalTrackResult(BaseModel):
    reasoning: str
    status: Literal["pausing", "capability_doubt", "continuing"]


class ReflectionContext(BaseModel):
    reflection_stage_1: str = Field(description=PROMPTS["reflection_stage_1"])
    reflection_stage_2: str = Field(description=PROMPTS["reflection_stage_2"])
    reflection_stage_3: str = Field(description=PROMPTS["reflection_stage_3"])
    reflection_stage_4: str = Field(description=PROMPTS["reflection_stage_4"])


class ProfessionalContentClassification(BaseModel):
    reasoning: str = Field(description="Reasoning for the classification")
    is_professional: bool


class GraphState(MessagesState):
    internal_audit_log: str
    opening_status: int
    content_state: str
    direction_choice: str
    expressions_content: list[dict]
    expressions_table: list[dict]
    blocks: list[dict]
    intent: str
    current_block_id: int | None
    block_chosen_unprompted: bool
    deepen_round_count: int
    deepen_round_new_rows: list[dict]
    deepen_round_block_updates: list[dict]
    readiness: str
    respond_check_answer: Literal["yes", "no", "other"] | None
    practical_check_answer: Literal["yes", "no", "other"] | None
    success_analysis_start_index: int | None
    success_analysis_rag_context: list[dict]
    success_analysis_rag_query: str | None
    success_consent_answer: Literal["yes", "no", "other"] | None
    scope_creep_status: Literal["in_scope", "drifting"] | None
    pivot_consent_answer: Literal["yes", "no", "other"] | None
    practical_track_consent_answer: Literal["yes", "no", "other"] | None
    practical_track_pause_status: Literal["pausing", "capability_doubt", "continuing"] | None
    practical_to_success_consent_answer: Literal["yes", "no", "other"] | None
    last_visited_node: str | None


def classify_opening(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    user_messages = _human_messages(state["messages"])
    if not user_messages:
        return {
            "internal_audit_log": existing_log
            + "\nWARNING: no user message found to classify.",
            "last_visited_node": "classify_opening",
        }

    first_message = user_messages[0].content

    llm = _get_llm()
    structured_llm = llm.with_structured_output(OpeningClassification)

    try:
        result = structured_llm.invoke(
            [
                {"role": "system", "content": CONST_PROMPTS["classify_opening"]},
                {"role": "user", "content": first_message},
            ]
        )
    except Exception as exc:
        return {
            "internal_audit_log": existing_log
            + f"\nWARNING: opening classification failed: {exc}",
            "last_visited_node": "classify_opening",
        }

    return {
        "internal_audit_log": existing_log + "\n" + result.reasoning,
        "opening_status": result.mode,
        "last_visited_node": "classify_opening",
    }


def respond_direct(state: GraphState) -> dict:
    opening_status = state.get("opening_status")

    if opening_status == 1:
        disclosure = MESSAGES["_shared"]["AGENT_DISCLOSURE_TEXT"]
        response_text = f"{disclosure} {MESSAGES['respond_direct_opening_status_1']}"
        note = "[respond_direct] Triggered by opening_status=1 (minimal message) - proceeding without asking permission."
    else:
        response_text = MESSAGES["_shared"]["AGENT_DISCLOSURE_TEXT"]
        note = f"[respond_direct] Triggered by opening_status={opening_status} (detailed situation/dilemma) - proceeding without asking permission."

    return {
        "messages": [AIMessage(content=response_text)],
        "internal_audit_log": state.get("internal_audit_log", "") + "\n" + note,
        "last_visited_node": "respond_direct",
    }


def route_after_respond_direct(state: GraphState) -> str:
    if state.get("opening_status") == 1:
        return "end"
    return "continue"


def _build_template_fill_system_prompt(template: str, content_principles: str) -> str:
    # Generic, reusable for any node that needs the model to fill a single
    # placeholder inside a fixed template and write out the complete reply
    # itself, rather than Python assembling it via string concatenation.
    # template_fill_instruction (format/mechanics, not content-specific) is
    # combined with content_principles (what the generated content should
    # actually say - varies per caller) and an example rendering of the
    # template, so the model both understands the technique and sees the
    # placeholder's exact position.
    example = template.format(reflection="[שיקוף שלך כאן]")
    return f"{PROMPTS['template_fill_instruction']}\n\n{content_principles}\n\nTemplate:\n{example}"


def reflect_on_situation(last_message: str) -> ReflectionContext:
    # Reusable helper, not a graph node - respond_with_check calls this
    # directly. No try/except here - respond_with_check is a free-text node
    # (like ask_direction/invite_to_share/present_and_ask), not a
    # classification node, and deliberately has no try/except of its own
    # either; an error here propagates up uncaught, consistent with that.
    llm = _get_llm()
    structured_llm = llm.with_structured_output(ReflectionContext)
    return structured_llm.invoke(
        [
            {"role": "system", "content": PROMPTS["reflection_context_instruction"]},
            {"role": "user", "content": last_message},
        ]
    )


def _build_reflection_final_system_prompt(template: str, context: ReflectionContext) -> str:
    return _build_template_fill_system_prompt(template, context.model_dump_json())


def respond_with_check(state: GraphState) -> dict:
    opening_status = state.get("opening_status")
    check_prompts = MESSAGES["respond_with_check"]
    user_messages = _human_messages(state["messages"])
    last_message = user_messages[-1].content if user_messages else ""

    if opening_status == 2:
        template = check_prompts["template_situation"]
        note = "[respond_with_check] Triggered by opening_status=2 (short situation description)."
    elif opening_status == 4:
        template = check_prompts["template_dilemma"]
        note = "[respond_with_check] Triggered by opening_status=4 (short dilemma)."
    else:
        template = check_prompts["template_situation"]
        note = f"WARNING: opening_status missing/invalid ({opening_status!r}) - falling back to respond_with_check as a safe default."

    context = reflect_on_situation(last_message)
    note += f" Reflection context: {context.model_dump_json()}"

    llm = _get_llm()
    response = llm.invoke(
        [
            {"role": "system", "content": _build_reflection_final_system_prompt(template, context)},
            {"role": "user", "content": last_message},
        ]
    )
    acknowledgment = _content_text(response)

    disclosure = MESSAGES["_shared"]["AGENT_DISCLOSURE_TEXT"]
    response_text = f"{disclosure} {acknowledgment}"

    return {
        "messages": [AIMessage(content=response_text)],
        "internal_audit_log": state.get("internal_audit_log", "") + "\n" + note,
        "last_visited_node": "respond_with_check",
    }


def classify_yes_no_other(message_text: str) -> Literal["yes", "no", "other"]:
    # Reusable helper, not a graph node - any node can call this directly.
    # No try/except here - error handling stays with whichever node calls it,
    # matching that node's own audit-log/fallback conventions.
    llm = _get_llm()
    structured_llm = llm.with_structured_output(YesNoOtherResult)
    result = structured_llm.invoke(
        [
            {"role": "system", "content": CONST_PROMPTS["classify_yes_no_other"]},
            {"role": "user", "content": message_text},
        ]
    )
    return result.answer


def classify_respond_check_choice(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    user_messages = _human_messages(state["messages"])
    if not user_messages:
        return {
            "internal_audit_log": existing_log
            + "\nWARNING: no user message found to classify respond-check choice.",
            "last_visited_node": "classify_respond_check_choice",
        }

    last_message = user_messages[-1].content

    try:
        answer = classify_yes_no_other(last_message)
    except Exception as exc:
        return {
            "internal_audit_log": existing_log
            + f"\nWARNING: respond-check choice classification failed: {exc}",
            "last_visited_node": "classify_respond_check_choice",
        }

    note = f"[classify_respond_check_choice] Client's reply classified as: {answer}."

    return {
        "respond_check_answer": answer,
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "classify_respond_check_choice",
    }


def route_after_respond_check_choice(state: GraphState) -> str:
    answer = state.get("respond_check_answer")
    if answer == "yes":
        return "invite_to_share"
    if answer == "no":
        return "end"
    return "classify_content_state"


def invite_to_share(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    user_messages = _human_messages(state["messages"])
    last_message = user_messages[-1].content if user_messages else ""

    llm = _get_llm()
    response = llm.invoke(
        [
            {"role": "system", "content": PROMPTS["invite_to_share"]},
            {"role": "user", "content": last_message},
        ]
    )
    response_text = _content_text(response)

    note = "[invite_to_share] Invited client to share content after confirming they want to continue."

    return {
        "messages": [AIMessage(content=response_text)],
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "invite_to_share",
    }


def route_after_classification(state: GraphState) -> str:
    if state.get("opening_status") in (1, 3, 5):
        return "respond_direct"
    return "respond_with_check"


def _find_emotional_vague_word(text: str) -> str | None:
    # An empty or missing bank means the feature is off, not an error. Entries may
    # be separated by commas or newlines (or a mix); blank entries are dropped, since
    # an empty string would otherwise match every message. Matches whole words only -
    # an entry must not be embedded in a longer word - so prefixed or inflected forms
    # (e.g. "מפחד" for "פחד") have to be listed in the bank as their own entries.
    bank = PROMPTS.get("emotional_vague_words_bank", "")
    haystack = " ".join(text.casefold().split())
    for entry in bank.replace("\n", ",").split(","):
        word = entry.strip()
        if not word:
            continue
        pattern = r"(?<!\w)" + re.escape(" ".join(word.casefold().split())) + r"(?!\w)"
        if re.search(pattern, haystack):
            return word
    return None


def classify_content_state(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    user_messages = _human_messages(state["messages"])
    if not user_messages:
        return {
            "internal_audit_log": existing_log
            + "\nWARNING: no user message found to classify content state.",
            "last_visited_node": "classify_content_state",
        }

    last_message = user_messages[-1].content

    matched_word = _find_emotional_vague_word(
        last_message if isinstance(last_message, str) else str(last_message)
    )
    if matched_word is not None:
        note = (
            f"[classify_content_state] Message contains emotional_vague_words_bank entry "
            f"{matched_word!r} - classified as emotional_vague without an LLM call."
        )
        return {
            "internal_audit_log": existing_log + "\n" + note,
            "content_state": "emotional_vague",
            "last_visited_node": "classify_content_state",
        }

    llm = _get_llm()
    structured_llm = llm.with_structured_output(ContentStateClassification)

    try:
        result = structured_llm.invoke(
            [
                {"role": "system", "content": CONST_PROMPTS["classify_content_state"]},
                {"role": "user", "content": last_message},
            ]
        )
    except Exception as exc:
        return {
            "internal_audit_log": existing_log
            + f"\nWARNING: content state classification failed: {exc}",
            "last_visited_node": "classify_content_state",
        }

    return {
        "internal_audit_log": existing_log + "\n" + result.reasoning,
        "content_state": result.state,
        "last_visited_node": "classify_content_state",
    }


def ask_direction(state: GraphState) -> dict:
    content_state = state.get("content_state")
    existing_log = state.get("internal_audit_log", "")

    if content_state == "practical_clear":
        response_text = MESSAGES["ask_direction_practical_check"]
        note = "[ask_direction] Triggered by content_state=practical_clear (fixed check, no LLM call)."
        return {
            "messages": [AIMessage(content=response_text)],
            "internal_audit_log": existing_log + "\n" + note,
            "last_visited_node": "ask_direction_practical_check",
        }

    user_messages = _human_messages(state["messages"])
    last_message = user_messages[-1].content if user_messages else ""

    if content_state == "emotional_vague":
        system_prompt = PROMPTS["ask_direction"]["emotional_vague"]
        note = "[ask_direction] Triggered by content_state=emotional_vague."
    else:
        system_prompt = PROMPTS["ask_direction"]["dual"]
        note = "[ask_direction] Triggered by content_state=dual."

    llm = _get_llm()
    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": last_message},
        ]
    )
    response_text = _content_text(response)

    return {
        "messages": [AIMessage(content=response_text)],
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "ask_direction",
    }


def route_after_content_state(state: GraphState) -> str:
    if state.get("content_state") in ("emotional_vague", "dual", "practical_clear"):
        return "ask_direction"
    if state.get("content_state") == "emotional_clear":
        return "retrieve_expressions_content"
    return "end"


def present_success_analysis_intro(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    note = "[present_success_analysis_intro] Presented the success-analysis intro and asked for consent."

    return {
        "messages": [AIMessage(content=MESSAGES["present_success_analysis_intro"])],
        "success_analysis_start_index": len(state.get("messages") or []),
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "present_success_analysis_intro",
    }


def classify_success_consent(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    user_messages = _human_messages(state["messages"])
    if not user_messages:
        return {
            "internal_audit_log": existing_log
            + "\nWARNING: no user message found to classify success consent.",
            "last_visited_node": "classify_success_consent",
        }

    last_message = user_messages[-1].content

    try:
        answer = classify_yes_no_other(last_message)
    except Exception as exc:
        return {
            "internal_audit_log": existing_log
            + f"\nWARNING: success consent classification failed: {exc}",
            "last_visited_node": "classify_success_consent",
        }

    note = f"[classify_success_consent] Client's reply classified as: {answer}."

    return {
        "success_consent_answer": answer,
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "classify_success_consent",
    }


def route_after_success_consent(state: GraphState) -> str:
    answer = state.get("success_consent_answer")
    if answer == "yes":
        return "invite_success_story"
    if answer == "no":
        return "explain_success_value"
    return "classify_scope_creep"  # "other" - they likely already started sharing content


def explain_success_value(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    user_messages = _human_messages(state["messages"])
    last_message = user_messages[-1].content if user_messages else ""

    llm = _get_llm()
    response = llm.invoke(
        [
            {"role": "system", "content": PROMPTS["explain_success_value"]},
            {"role": "user", "content": last_message},
        ]
    )
    response_text = _content_text(response)

    note = "[explain_success_value] Explained the value of the process without pushing further."

    return {
        "messages": [AIMessage(content=response_text)],
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "explain_success_value",
    }


def invite_success_story(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    note = "[invite_success_story] Invited the client to share a success story."

    return {
        "messages": [AIMessage(content=MESSAGES["invite_success_story"])],
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "invite_success_story",
    }


def _format_conversation(messages) -> str:
    lines = []
    for m in messages:
        role = "Client" if isinstance(m, HumanMessage) else "Assistant"
        content = _content_text(m)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def classify_scope_creep(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    all_messages = state.get("messages") or []
    start_index = state.get("success_analysis_start_index")
    relevant_messages = all_messages[start_index:] if start_index is not None else all_messages

    if not relevant_messages:
        return {
            "internal_audit_log": existing_log
            + "\nWARNING: no conversation found to check for scope creep.",
            "last_visited_node": "classify_scope_creep",
        }

    conversation_text = _format_conversation(relevant_messages)

    llm = _get_llm()
    structured_llm = llm.with_structured_output(ScopeCreepResult)

    try:
        result = structured_llm.invoke(
            [
                {"role": "system", "content": CONST_PROMPTS["classify_scope_creep"]},
                {"role": "user", "content": conversation_text},
            ]
        )
    except Exception as exc:
        return {
            "internal_audit_log": existing_log
            + f"\nWARNING: scope creep classification failed: {exc}",
            "last_visited_node": "classify_scope_creep",
        }

    note = f"[classify_scope_creep] Status: {result.status}."

    return {
        "scope_creep_status": result.status,
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "classify_scope_creep",
    }


def route_after_scope_creep(state: GraphState) -> str:
    if state.get("scope_creep_status") == "drifting":
        return "pivot_to_deeper_process"
    return "success_analysis_conversation"


def pivot_to_deeper_process(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    note = "[pivot_to_deeper_process] Detected scope drift into identity/career-direction territory - pivoted."

    return {
        "messages": [AIMessage(content=MESSAGES["pivot_to_deeper_process"])],
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "pivot_to_deeper_process",
    }


def classify_pivot_consent(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    user_messages = _human_messages(state["messages"])
    if not user_messages:
        return {
            "internal_audit_log": existing_log
            + "\nWARNING: no user message found to classify pivot consent.",
            "last_visited_node": "classify_pivot_consent",
        }

    last_message = user_messages[-1].content

    try:
        answer = classify_yes_no_other(last_message)
    except Exception as exc:
        return {
            "internal_audit_log": existing_log
            + f"\nWARNING: pivot consent classification failed: {exc}",
            "last_visited_node": "classify_pivot_consent",
        }

    note = f"[classify_pivot_consent] Client's reply classified as: {answer}."

    update = {
        "pivot_consent_answer": answer,
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "classify_pivot_consent",
    }
    if answer == "yes":
        # We already know this is emotional content - that's why the pivot fired in
        # the first place. Set content_state explicitly so downstream routing that
        # keys off it (e.g. classify_present_choice's branch) keeps working correctly.
        update["content_state"] = "emotional_clear"
    return update


def route_after_pivot_consent(state: GraphState) -> str:
    if state.get("pivot_consent_answer") == "yes":
        return "retrieve_expressions_content"
    return "success_analysis_conversation"  # "no" or "other" - stay in the success-analysis tool


def retrieve_success_analysis_context(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    all_messages = state.get("messages") or []
    start_index = state.get("success_analysis_start_index")
    relevant_messages = all_messages[start_index:] if start_index is not None else all_messages
    human_messages = _human_messages(relevant_messages)
    query_text = " ".join(
        _content_text(m)
        for m in human_messages
    ).strip()

    if not query_text:
        return {
            "success_analysis_rag_context": [],
            "success_analysis_rag_query": None,
            "internal_audit_log": existing_log
            + "\nWARNING: no user messages found to build success analysis RAG query.",
            "last_visited_node": "retrieve_success_analysis_context",
        }

    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    if not api_key or not index_name:
        return {
            "success_analysis_rag_context": [],
            "success_analysis_rag_query": query_text,
            "internal_audit_log": existing_log
            + "\nWARNING: PINECONE_API_KEY/PINECONE_INDEX_NAME not set - skipping success analysis context retrieval.",
            "last_visited_node": "retrieve_success_analysis_context",
        }

    try:
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)

        # Same namespace auto-detection as retrieve_expressions_content.
        namespace = _detect_namespace(index)

        results = index.search(
            namespace=namespace,
            query={
                "inputs": {"text": query_text},
                "top_k": SUCCESS_ANALYSIS_RAG_TOP_K,
                "filter": {"module": {"$eq": "success_stories_analysis"}, "retrievable": {"$eq": True}},
            },
        )
        hits = (results.get("result") or {}).get("hits") or []

        rag_context = []
        for hit in hits:
            fields = _extract_hit_value(hit, "fields") or {}
            hit_id = _extract_hit_value(hit, "_id", "id")
            rag_context.append(
                {
                    "id": hit_id,
                    "chunk_id": hit_id,
                    "score": _extract_hit_value(hit, "_score", "score"),
                    "text": fields.get("text"),
                    "chunk_title": fields.get("chunk_title"),
                }
            )
    except Exception as exc:
        return {
            "success_analysis_rag_context": [],
            "success_analysis_rag_query": query_text,
            "internal_audit_log": existing_log
            + f"\nWARNING: success analysis context retrieval failed: {exc}",
            "last_visited_node": "retrieve_success_analysis_context",
        }

    note = f"[retrieve_success_analysis_context] Retrieved {len(rag_context)} success-stories-analysis chunks."

    return {
        "success_analysis_rag_context": rag_context,
        "success_analysis_rag_query": query_text,
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "retrieve_success_analysis_context",
    }


def success_analysis_conversation(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    all_messages = state.get("messages") or []
    start_index = state.get("success_analysis_start_index")
    relevant_messages = all_messages[start_index:] if start_index is not None else all_messages
    conversation_text = _format_conversation(relevant_messages)

    rag_context = state.get("success_analysis_rag_context") or []
    user_content = conversation_text
    if rag_context:
        retrieved_text = "\n\n".join(chunk.get("text") or "" for chunk in rag_context)
        user_content = (
            CONST_PROMPTS["success_analysis_rag_preamble"]
            + f"[INTERNAL RETRIEVED KNOWLEDGE]\n{retrieved_text}\n[/INTERNAL RETRIEVED KNOWLEDGE]\n\n"
            f"{conversation_text}"
        )

    llm = _get_llm()
    response = llm.invoke(
        [
            {"role": "system", "content": PROMPTS["success_analysis_conversation"]},
            {"role": "user", "content": user_content},
        ]
    )
    response_text = _content_text(response)

    note = "[success_analysis_conversation] Continued the success-analysis conversation."

    return {
        "messages": [AIMessage(content=response_text)],
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "success_analysis_conversation",
    }


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
    human_messages = _human_messages(state["messages"])
    query_text = " ".join(
        _content_text(m)
        for m in human_messages
    ).strip()

    if not query_text:
        return {
            "internal_audit_log": existing_log
            + "\nWARNING: no user messages found to build expressions_content query.",
            "last_visited_node": "retrieve_expressions_content",
        }

    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    if not api_key or not index_name:
        return {
            "internal_audit_log": existing_log
            + "\nWARNING: PINECONE_API_KEY/PINECONE_INDEX_NAME not set - skipping expressions content retrieval.",
            "last_visited_node": "retrieve_expressions_content",
        }

    try:
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)

        # Auto-detect which namespace actually has data, same as /api/pinecone-query.
        # Not applying a module/doc_type filter for "bank" content yet - we haven't
        # confirmed the actual metadata values in this index, and a wrong filter
        # would silently return zero results rather than erroring.
        namespace = _detect_namespace(index)

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
            "last_visited_node": "retrieve_expressions_content",
        }

    note = f"[retrieve_expressions_content] Retrieved {len(expressions_content)} bank chunks for expression matching."
    return {
        "expressions_content": expressions_content,
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "retrieve_expressions_content",
    }


def build_expressions_table(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    human_messages = _human_messages(state["messages"])
    conversation_text = "\n".join(
        _content_text(m)
        for m in human_messages
    )

    expressions_content = state.get("expressions_content") or []
    bank_content_text = "\n\n".join(
        f"[{chunk.get('module')} - {chunk.get('chunk_title')}]\n{chunk.get('text')}"
        for chunk in expressions_content
    )

    llm = _get_llm()
    structured_llm = llm.with_structured_output(ExpressionsTableResult)

    user_content = (
        f"Client conversation:\n{conversation_text}\n\n"
        f"Retrieved bank content:\n{bank_content_text}"
    )

    try:
        result = structured_llm.invoke(
            [
                {"role": "system", "content": CONST_PROMPTS["build_expressions_table"]},
                {"role": "user", "content": user_content},
            ]
        )
    except Exception as exc:
        return {
            "internal_audit_log": existing_log
            + f"\nWARNING: build_expressions_table failed: {exc}",
            "last_visited_node": "build_expressions_table",
        }

    rows = [row.model_dump() for row in result.rows]
    bank_count = len({row["bank_name"] for row in rows})
    note = f"[build_expressions_table] Identified {len(rows)} expressions across {bank_count} banks."

    return {
        "expressions_table": rows,
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "build_expressions_table",
    }


def build_blocks(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    expressions_table = state.get("expressions_table") or []

    if not expressions_table:
        return {
            "internal_audit_log": existing_log
            + "\nWARNING: no expressions_table rows found to build blocks.",
            "last_visited_node": "build_blocks",
        }

    rows_text = "\n".join(
        f"{row.get('row_number')}. expression={row.get('expression')!r}, "
        f"expression_type={row.get('expression_type')}, bank_name={row.get('bank_name')}, "
        f"matched_expression={row.get('matched_expression')!r}, match_level={row.get('match_level')}"
        for row in expressions_table
    )

    llm = _get_llm()
    structured_llm = llm.with_structured_output(BuildBlocksResult)

    try:
        result = structured_llm.invoke(
            [
                {"role": "system", "content": CONST_PROMPTS["build_blocks"]},
                {"role": "user", "content": rows_text},
            ]
        )
    except Exception as exc:
        return {
            "internal_audit_log": existing_log
            + f"\nWARNING: build_blocks failed: {exc}",
            "last_visited_node": "build_blocks",
        }

    blocks = [block.model_dump() for block in result.blocks]
    rows = [row.model_dump() for row in result.rows]
    note = f"[build_blocks] Organized {len(rows)} rows into {len(blocks)} blocks."

    return {
        "blocks": blocks,
        "expressions_table": rows,
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "build_blocks",
    }


def color_blocks(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    blocks = state.get("blocks") or []
    expressions_table = state.get("expressions_table") or []

    if not blocks:
        return {
            "internal_audit_log": existing_log + "\nWARNING: no blocks found to color.",
            "last_visited_node": "color_blocks",
        }

    blocks_text = _format_blocks(blocks)
    expressions_text = "\n".join(
        f"{row.get('row_number')}. block_id={row.get('block_id')}, expression={row.get('expression')!r}"
        for row in expressions_table
    )

    llm = _get_llm()
    structured_llm = llm.with_structured_output(ColorBlocksResult)

    user_content = (
        f"Blocks:\n{blocks_text}\n\n"
        f"Expressions table (with block assignments):\n{expressions_text}"
    )

    try:
        result = structured_llm.invoke(
            [
                {"role": "system", "content": CONST_PROMPTS["color_blocks"]},
                {"role": "user", "content": user_content},
            ]
        )
    except Exception as exc:
        return {
            "internal_audit_log": existing_log + f"\nWARNING: color_blocks failed: {exc}",
            "last_visited_node": "color_blocks",
        }

    colors_by_block_id = {c.block_id: c.color for c in result.colors}
    updated_blocks = [
        {**block, "color": colors_by_block_id.get(block.get("block_id"), block.get("color"))}
        for block in blocks
    ]

    clear_count = sum(1 for c in result.colors if c.color != "לא חד משמעי")
    unclear_count = len(result.colors) - clear_count
    note = (
        f"[color_blocks] Assigned colors to {len(result.colors)} blocks "
        f"({clear_count} with a clear color, {unclear_count} marked לא חד משמעי)."
    )

    return {
        "blocks": updated_blocks,
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "color_blocks",
    }


def present_and_ask(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    blocks = state.get("blocks") or []

    if not blocks:
        return {
            "internal_audit_log": existing_log + "\nWARNING: no blocks found to present.",
            "last_visited_node": "present_and_ask",
        }

    blocks_text = _format_blocks(blocks, include_color=True)

    llm = _get_llm()
    response = llm.invoke(
        [
            {"role": "system", "content": PROMPTS["present_and_ask"]},
            {"role": "user", "content": blocks_text},
        ]
    )
    response_text = _content_text(response)

    note = f"[present_and_ask] Presented {len(blocks)} blocks and asked management question."

    return {
        "messages": [AIMessage(content=response_text)],
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "present_and_ask",
    }


def classify_present_choice(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    user_messages = _human_messages(state["messages"])
    if not user_messages:
        return {
            "internal_audit_log": existing_log
            + "\nWARNING: no user message found to classify present choice.",
            "last_visited_node": "classify_present_choice",
        }

    last_message = user_messages[-1].content
    blocks = state.get("blocks") or []
    blocks_text = _format_blocks(blocks)

    llm = _get_llm()
    structured_llm = llm.with_structured_output(PresentChoiceResult)

    try:
        result = structured_llm.invoke(
            [
                {"role": "system", "content": CONST_PROMPTS["classify_present_choice"]},
                {
                    "role": "user",
                    "content": f"Blocks:\n{blocks_text}\n\nClient reply:\n{last_message}",
                },
            ]
        )
    except Exception as exc:
        return {
            "internal_audit_log": existing_log
            + f"\nWARNING: present choice classification failed: {exc}",
            "last_visited_node": "classify_present_choice",
        }

    block_suffix = (
        f", targeting block {result.current_block_id}"
        if result.current_block_id is not None
        else ""
    )
    note = f"[classify_present_choice] Client chose {result.intent}{block_suffix}."

    return {
        "intent": result.intent,
        "current_block_id": result.current_block_id,
        "block_chosen_unprompted": result.current_block_id is not None,
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "classify_present_choice",
    }


def route_after_present_choice(state: GraphState) -> str:
    if state.get("intent") == "deepen" and state.get("current_block_id") is None:
        return "ask_which_block"
    return "end"


def ask_which_block(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    blocks = state.get("blocks") or []
    blocks_text = _format_blocks(blocks)

    llm = _get_llm()
    response = llm.invoke(
        [
            {"role": "system", "content": PROMPTS["ask_which_block"]},
            {"role": "user", "content": blocks_text},
        ]
    )
    response_text = _content_text(response)

    note = "[ask_which_block] Asked client to specify which block to deepen on."

    return {
        "messages": [AIMessage(content=response_text)],
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "ask_which_block",
    }


def classify_block_target(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    user_messages = _human_messages(state["messages"])
    if not user_messages:
        return {
            "internal_audit_log": existing_log
            + "\nWARNING: no user message found to classify block target.",
            "last_visited_node": "classify_block_target",
        }

    last_message = user_messages[-1].content
    blocks = state.get("blocks") or []
    blocks_text = _format_blocks(blocks)

    llm = _get_llm()
    structured_llm = llm.with_structured_output(BlockTargetResult)

    try:
        result = structured_llm.invoke(
            [
                {"role": "system", "content": CONST_PROMPTS["classify_block_target"]},
                {
                    "role": "user",
                    "content": f"Blocks:\n{blocks_text}\n\nClient reply:\n{last_message}",
                },
            ]
        )
    except Exception as exc:
        return {
            "internal_audit_log": existing_log
            + f"\nWARNING: block target classification failed: {exc}",
            "last_visited_node": "classify_block_target",
        }

    valid_block_ids = {block.get("block_id") for block in blocks}
    if result.current_block_id not in valid_block_ids:
        # The model didn't return a block_id that actually exists - fall back to
        # asking again, same wording/pattern as ask_which_block, rather than
        # silently accepting a hallucinated target.
        clarify_llm = _get_llm()
        clarify_response = clarify_llm.invoke(
            [
                {"role": "system", "content": PROMPTS["ask_which_block"]},
                {"role": "user", "content": blocks_text},
            ]
        )
        clarify_text = _content_text(clarify_response)
        note = (
            f"WARNING: classify_block_target returned block_id={result.current_block_id!r}, "
            "which doesn't match any known block - asking client to clarify."
        )
        return {
            "messages": [AIMessage(content=clarify_text)],
            "internal_audit_log": existing_log + "\n" + note,
            "last_visited_node": "classify_block_target",
        }

    note = f"[classify_block_target] Identified target block: {result.current_block_id}."

    return {
        "current_block_id": result.current_block_id,
        "block_chosen_unprompted": False,
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "classify_block_target",
    }


def route_after_block_target(state: GraphState) -> str:
    if state.get("current_block_id") is not None and (state.get("deepen_round_count") or 0) < 2:
        return "deepen_round"
    return "end"


def deepen_round(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    user_messages = _human_messages(state["messages"])
    if not user_messages:
        return {
            "internal_audit_log": existing_log
            + "\nWARNING: no user message found for deepen round.",
            "last_visited_node": "deepen_round",
        }

    last_message = user_messages[-1].content
    expressions_table = state.get("expressions_table") or []
    blocks = state.get("blocks") or []
    current_block_id = state.get("current_block_id")

    table_text = "\n".join(
        f"{row.get('row_number')}. block_id={row.get('block_id')}, expression={row.get('expression')!r}, "
        f"expression_type={row.get('expression_type')}, bank_name={row.get('bank_name')}, "
        f"matched_expression={row.get('matched_expression')!r}, match_level={row.get('match_level')}"
        for row in expressions_table
    )
    blocks_text = _format_blocks(blocks, include_color=True)

    llm = _get_llm()
    structured_llm = llm.with_structured_output(DeepenRoundResult)

    user_content = (
        f"Currently deepening on block_id={current_block_id}.\n\n"
        f"Existing expressions table:\n{table_text}\n\n"
        f"Existing blocks:\n{blocks_text}\n\n"
        f"Client's new message:\n{last_message}"
    )

    try:
        result = structured_llm.invoke(
            [
                {"role": "system", "content": CONST_PROMPTS["deepen_round"]},
                {"role": "user", "content": user_content},
            ]
        )
    except Exception as exc:
        return {
            "internal_audit_log": existing_log + f"\nWARNING: deepen_round failed: {exc}",
            "last_visited_node": "deepen_round",
        }

    next_row_number = len(expressions_table) + 1
    new_rows = []
    for offset, row in enumerate(result.new_rows):
        row_dict = row.model_dump()
        row_dict["row_number"] = next_row_number + offset
        new_rows.append(row_dict)

    block_updates = [block.model_dump() for block in result.block_updates]
    blocks_by_id = {block.get("block_id"): dict(block) for block in blocks}
    for block_dict in block_updates:
        blocks_by_id[block_dict["block_id"]] = block_dict
    updated_blocks = list(blocks_by_id.values())

    round_number = (state.get("deepen_round_count") or 0) + 1
    note = (
        f"[deepen_round] Round {round_number}: added {len(new_rows)} new expressions, "
        f"{len(block_updates)} block updates."
    )

    return {
        "expressions_table": expressions_table + new_rows,
        "blocks": updated_blocks,
        "deepen_round_new_rows": new_rows,
        "deepen_round_block_updates": block_updates,
        "deepen_round_count": round_number,
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "deepen_round",
    }


def route_after_deepen_round(state: GraphState) -> str:
    if (state.get("deepen_round_count") or 0) < 2:
        return "deepen_reply"
    return "focus_on_block"


def deepen_reply(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    round_count = state.get("deepen_round_count") or 0

    new_rows = state.get("deepen_round_new_rows") or []
    block_updates = state.get("deepen_round_block_updates") or []

    additions_text = "\n".join(f"- {row.get('expression')!r}" for row in new_rows)
    blocks_text = "\n".join(f"- {block.get('topic')!r}" for block in block_updates)

    llm = _get_llm()
    response = llm.invoke(
        [
            {"role": "system", "content": PROMPTS["deepen_reply"]},
            {
                "role": "user",
                "content": f"New expressions this round:\n{additions_text}\n\nAffected blocks:\n{blocks_text}",
            },
        ]
    )
    response_text = _content_text(response)

    note = f"[deepen_reply] Round {round_count}: reflected back {len(new_rows)} new expressions and invited more or moving on."

    return {
        "messages": [AIMessage(content=response_text)],
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "deepen_reply",
    }


def focus_on_block(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")

    if state.get("block_chosen_unprompted"):
        note = (
            f"[focus_on_block] Skipped - client already named block "
            f"{state.get('current_block_id')} unprompted via classify_present_choice."
        )
        return {
            "internal_audit_log": existing_log + "\n" + note,
            "last_visited_node": "focus_on_block",
        }

    blocks = state.get("blocks") or []
    if not blocks:
        return {
            "internal_audit_log": existing_log + "\nWARNING: no blocks found to focus on.",
            "last_visited_node": "focus_on_block",
        }

    blocks_text = _format_blocks(blocks)

    llm = _get_llm()
    response = llm.invoke(
        [
            {"role": "system", "content": PROMPTS["focus_on_block"]},
            {"role": "user", "content": blocks_text},
        ]
    )
    response_text = _content_text(response)

    note = "[focus_on_block] Asked client which block feels most emotionally significant now."

    return {
        "messages": [AIMessage(content=response_text)],
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "focus_on_block",
    }


def classify_focus_choice(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    user_messages = _human_messages(state["messages"])
    if not user_messages:
        return {
            "internal_audit_log": existing_log
            + "\nWARNING: no user message found to classify focus choice.",
            "last_visited_node": "classify_focus_choice",
        }

    last_message = user_messages[-1].content
    blocks = state.get("blocks") or []
    blocks_text = _format_blocks(blocks)

    llm = _get_llm()
    structured_llm = llm.with_structured_output(FocusChoiceResult)

    try:
        result = structured_llm.invoke(
            [
                {"role": "system", "content": CONST_PROMPTS["classify_focus_choice"]},
                {
                    "role": "user",
                    "content": f"Blocks:\n{blocks_text}\n\nClient reply:\n{last_message}",
                },
            ]
        )
    except Exception as exc:
        return {
            "internal_audit_log": existing_log
            + f"\nWARNING: focus choice classification failed: {exc}",
            "last_visited_node": "classify_focus_choice",
        }

    valid_block_ids = {block.get("block_id") for block in blocks}
    if result.current_block_id not in valid_block_ids:
        # The model didn't return a block_id that actually exists - fall back to
        # asking again, same wording/pattern as focus_on_block, rather than
        # silently accepting a hallucinated target.
        clarify_llm = _get_llm()
        clarify_response = clarify_llm.invoke(
            [
                {"role": "system", "content": PROMPTS["focus_on_block"]},
                {"role": "user", "content": blocks_text},
            ]
        )
        clarify_text = _content_text(clarify_response)
        note = (
            f"WARNING: classify_focus_choice returned block_id={result.current_block_id!r}, "
            "which doesn't match any known block - asking client to clarify."
        )
        return {
            "messages": [AIMessage(content=clarify_text)],
            "internal_audit_log": existing_log + "\n" + note,
            "last_visited_node": "classify_focus_choice",
        }

    note = f"[classify_focus_choice] Client focused on block {result.current_block_id}."

    return {
        "current_block_id": result.current_block_id,
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "classify_focus_choice",
    }


def route_after_focus_choice(state: GraphState) -> str:
    messages = state.get("messages") or []
    if messages and isinstance(messages[-1], AIMessage):
        return "end"  # fallback re-ask just happened - wait for the client's next reply
    return "classify_readiness"  # current_block_id was successfully updated - continue immediately


def classify_readiness(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    human_messages = _human_messages(state["messages"])
    if not human_messages:
        return {
            "internal_audit_log": existing_log
            + "\nWARNING: no user messages found to assess readiness.",
            "last_visited_node": "classify_readiness",
        }

    conversation_text = "\n".join(
        _content_text(m)
        for m in human_messages
    )

    llm = _get_llm()
    structured_llm = llm.with_structured_output(ReadinessResult)

    try:
        result = structured_llm.invoke(
            [
                {"role": "system", "content": CONST_PROMPTS["classify_readiness"]},
                {"role": "user", "content": conversation_text},
            ]
        )
    except Exception as exc:
        return {
            "internal_audit_log": existing_log
            + f"\nWARNING: readiness classification failed: {exc}",
            "last_visited_node": "classify_readiness",
        }

    note = f"[classify_readiness] Assessed readiness: {result.readiness}."

    return {
        "readiness": result.readiness,
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "classify_readiness",
    }


def route_after_readiness(state: GraphState) -> str:
    if state.get("readiness") == "not_ready":
        return "summarize_and_pivot"
    return "end"


def summarize_and_pivot(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    blocks = state.get("blocks") or []

    if not blocks:
        return {
            "internal_audit_log": existing_log + "\nWARNING: no blocks found to summarize.",
            "last_visited_node": "summarize_and_pivot",
        }

    blocks_text = _format_blocks(blocks, include_color=True)

    llm = _get_llm()
    response = llm.invoke(
        [
            {"role": "system", "content": PROMPTS["summarize_and_pivot"]},
            {"role": "user", "content": blocks_text},
        ]
    )
    response_text = _content_text(response)

    note = "[summarize_and_pivot] Summarized blocks/colors and pivoted toward practical work."

    return {
        "messages": [AIMessage(content=response_text)],
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "summarize_and_pivot",
    }


def present_practical_track_intro(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    note = "[present_practical_track_intro] Presented the practical-track intro and asked to begin."

    return {
        "messages": [AIMessage(content=MESSAGES["present_practical_track_intro"])],
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "present_practical_track_intro",
    }


def classify_practical_track_consent(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    user_messages = _human_messages(state["messages"])
    if not user_messages:
        return {
            "internal_audit_log": existing_log
            + "\nWARNING: no user message found to classify practical track consent.",
            "last_visited_node": "classify_practical_track_consent",
        }

    last_message = user_messages[-1].content

    try:
        answer = classify_yes_no_other(last_message)
    except Exception as exc:
        return {
            "internal_audit_log": existing_log
            + f"\nWARNING: practical track consent classification failed: {exc}",
            "last_visited_node": "classify_practical_track_consent",
        }

    note = f"[classify_practical_track_consent] Client's reply classified as: {answer}."

    return {
        "practical_track_consent_answer": answer,
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "classify_practical_track_consent",
    }


def route_after_practical_track_consent(state: GraphState) -> str:
    if state.get("practical_track_consent_answer") == "no":
        return "end"
    return "practical_track_conversation"  # "yes" or "other" - begin the open-ended dialogue


def practical_track_conversation(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    all_messages = state.get("messages") or []
    conversation_text = _format_conversation(all_messages)

    llm = _get_llm()
    response = llm.invoke(
        [
            {"role": "system", "content": PROMPTS["practical_track_conversation"]},
            {"role": "user", "content": conversation_text},
        ]
    )
    response_text = _content_text(response)

    note = "[practical_track_conversation] Continued the practical-track conversation."

    return {
        "messages": [AIMessage(content=response_text)],
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "practical_track_conversation",
    }


def classify_from_practical_track(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    all_messages = state.get("messages") or []

    if not all_messages:
        return {
            "internal_audit_log": existing_log
            + "\nWARNING: no conversation found to classify.",
            "last_visited_node": "classify_from_practical_track",
        }

    conversation_text = _format_conversation(all_messages)

    llm = _get_llm()
    structured_llm = llm.with_structured_output(FromPracticalTrackResult)

    try:
        result = structured_llm.invoke(
            [
                {"role": "system", "content": CONST_PROMPTS["classify_from_practical_track"]},
                {"role": "user", "content": conversation_text},
            ]
        )
    except Exception as exc:
        return {
            "internal_audit_log": existing_log
            + f"\nWARNING: from-practical-track classification failed: {exc}",
            "last_visited_node": "classify_from_practical_track",
        }

    note = f"[classify_from_practical_track] Status: {result.status}."

    return {
        "practical_track_pause_status": result.status,
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "classify_from_practical_track",
    }


def route_after_from_practical_track(state: GraphState) -> str:
    status = state.get("practical_track_pause_status")
    if status == "pausing":
        return "end"
    if status == "capability_doubt":
        return "pivot_practical_to_success"
    return "practical_track_conversation"  # "continuing"


def pivot_practical_to_success(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    note = "[pivot_practical_to_success] Detected capability doubt - offered to pivot to success-moment analysis."

    return {
        "messages": [AIMessage(content=MESSAGES["pivot_practical_to_success"])],
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "pivot_practical_to_success",
    }


def classify_practical_to_success_consent(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    user_messages = _human_messages(state["messages"])
    if not user_messages:
        return {
            "internal_audit_log": existing_log
            + "\nWARNING: no user message found to classify practical-to-success consent.",
            "last_visited_node": "classify_practical_to_success_consent",
        }

    last_message = user_messages[-1].content

    try:
        answer = classify_yes_no_other(last_message)
    except Exception as exc:
        return {
            "internal_audit_log": existing_log
            + f"\nWARNING: practical-to-success consent classification failed: {exc}",
            "last_visited_node": "classify_practical_to_success_consent",
        }

    note = f"[classify_practical_to_success_consent] Client's reply classified as: {answer}."

    return {
        "practical_to_success_consent_answer": answer,
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "classify_practical_to_success_consent",
    }


def route_after_practical_to_success_consent(state: GraphState) -> str:
    if state.get("practical_to_success_consent_answer") == "yes":
        return "present_success_analysis_intro"
    return "practical_track_conversation"  # "no" or "other" - stay in the practical track


def classify_direction_choice(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    user_messages = _human_messages(state["messages"])
    if not user_messages:
        return {
            "internal_audit_log": existing_log
            + "\nWARNING: no user message found to classify direction choice.",
            "last_visited_node": "classify_direction_choice",
        }

    last_message = user_messages[-1].content

    llm = _get_llm()
    structured_llm = llm.with_structured_output(DirectionChoiceClassification)

    try:
        result = structured_llm.invoke(
            [
                {"role": "system", "content": CONST_PROMPTS["classify_direction_choice"]},
                {"role": "user", "content": last_message},
            ]
        )
    except Exception as exc:
        return {
            "internal_audit_log": existing_log
            + f"\nWARNING: direction choice classification failed: {exc}",
            "last_visited_node": "classify_direction_choice",
        }

    return {
        "internal_audit_log": existing_log + "\n" + result.reasoning,
        "direction_choice": result.choice,
        "last_visited_node": "classify_direction_choice",
    }


def route_after_direction_choice(state: GraphState) -> str:
    if state.get("direction_choice") == "pause":
        return "invite_to_share"
    return "present_practical_track_intro"  # "continue"


def classify_professional_content(state: GraphState) -> dict:
    # Cross-cutting check that runs on every turn, before route_from_start's
    # per-stage dispatch - unlike every other node in this file, it does NOT
    # stamp last_visited_node in any return path. route_from_start is the only
    # reader of last_visited_node, and it only runs once per turn (right after
    # this node). If this node stamped its own name and then turned out to be
    # the last thing that ran this turn (e.g. the deterministic "יצאתי לחשוב"
    # pause inside practical_track_conversation, which can end a turn with no
    # further node executing), the NEXT turn's route_from_start would see
    # last_visited_node="classify_professional_content" instead of the real
    # stage the conversation was actually in, breaking its dispatch. Leaving
    # last_visited_node untouched here means whatever real stage node runs
    # afterward in the same turn stamps it exactly as it always has.
    existing_log = state.get("internal_audit_log", "")
    user_messages = _human_messages(state["messages"])
    if not user_messages:
        return {
            "internal_audit_log": existing_log
            + "\nWARNING: no user message found to classify professional content.",
        }

    last_message = user_messages[-1].content

    llm = _get_llm()
    structured_llm = llm.with_structured_output(ProfessionalContentClassification)

    try:
        result = structured_llm.invoke(
            [
                {"role": "system", "content": PROMPTS["is_considered_professional"]},
                {"role": "user", "content": last_message},
            ]
        )
    except Exception as exc:
        return {
            "internal_audit_log": existing_log
            + f"\nWARNING: professional content classification failed: {exc}",
        }

    note = f"[classify_professional_content] is_professional={result.is_professional}. {result.reasoning}"

    update = {
        "internal_audit_log": existing_log + "\n" + note,
    }
    if result.is_professional:
        update["messages"] = [AIMessage(content=MESSAGES["professional_content_disclaimer"])]
    return update


def classify_practical_check_choice(state: GraphState) -> dict:
    existing_log = state.get("internal_audit_log", "")
    user_messages = _human_messages(state["messages"])
    if not user_messages:
        return {
            "internal_audit_log": existing_log
            + "\nWARNING: no user message found to classify practical check choice.",
            "last_visited_node": "classify_practical_check_choice",
        }

    last_message = user_messages[-1].content

    try:
        answer = classify_yes_no_other(last_message)
    except Exception as exc:
        return {
            "internal_audit_log": existing_log
            + f"\nWARNING: practical check choice classification failed: {exc}",
            "last_visited_node": "classify_practical_check_choice",
        }

    note = f"[classify_practical_check_choice] Client's reply classified as: {answer}."

    return {
        "practical_check_answer": answer,
        "internal_audit_log": existing_log + "\n" + note,
        "last_visited_node": "classify_practical_check_choice",
    }


def route_after_practical_check_choice(state: GraphState) -> str:
    answer = state.get("practical_check_answer")
    if answer == "no":
        return "present_practical_track_intro"  # nothing beyond the practical - confirmed pure practical
    return "classify_content_state"  # "yes"/"other" - there's more, or unclear - reclassify on the new content


def route_from_start(state: GraphState) -> str:
    last = state.get("last_visited_node")

    if last == "ask_direction_practical_check":
        return "classify_practical_check_choice"  # practical check question was asked - classify the reply

    if last == "ask_direction":
        return "classify_direction_choice"  # ask_direction just asked a question - classify the reply

    if last == "respond_with_check":
        return "classify_respond_check_choice"  # respond_with_check asked "רוצה להמשיך?" - classify the reply

    if last == "invite_to_share":
        return "classify_content_state"  # invite_to_share asked for content - classify what the client shared

    if last in ("ask_which_block", "classify_block_target"):
        return "classify_block_target"  # asked (or re-asked) which block - classify the reply

    if last == "present_and_ask" and state.get("blocks"):
        return "classify_present_choice"  # present_and_ask just asked its question - classify the reply

    if last == "deepen_reply":
        # deepen_reply only ever runs when deepen_round_count was < 2 at the time
        # deepen_round ran (route_after_deepen_round skips straight to focus_on_block
        # otherwise, in the same turn) - so there's always another round to do here.
        return "deepen_round"

    if last == "classify_focus_choice":
        return "classify_focus_choice"  # invalid block match last time - re-ask and reclassify

    if last == "focus_on_block" and not state.get("block_chosen_unprompted"):
        return "classify_focus_choice"  # focus_on_block asked its question - classify the reply
        # (if block_chosen_unprompted is True, focus_on_block skipped silently and asked
        # nothing - falls through below, same as any other "nothing left to do yet" turn)

    if last == "present_success_analysis_intro":
        return "classify_success_consent"  # asked for consent to try the process - classify the reply

    if last in ("invite_success_story", "success_analysis_conversation"):
        return "classify_scope_creep"  # a new success-analysis message arrived - check scope every round

    if last == "pivot_to_deeper_process":
        return "classify_pivot_consent"  # pivot_to_deeper_process asked its question - classify the reply

    if last == "present_practical_track_intro":
        return "classify_practical_track_consent"  # asked to begin the practical track - classify the reply

    if last == "practical_track_conversation":
        user_messages = _human_messages(state["messages"])
        last_text = user_messages[-1].content if user_messages else ""
        last_text = last_text if isinstance(last_text, str) else str(last_text)
        if last_text.strip(PRACTICAL_TRACK_PAUSE_STRIP_CHARS) == PRACTICAL_TRACK_PAUSE_PHRASE:
            return "end"  # "יצאתי לחשוב" - deterministic pause, no LLM call needed
        return "classify_from_practical_track"  # check every round while inside this conversation

    if last == "pivot_practical_to_success":
        return "classify_practical_to_success_consent"  # asked to pivot - classify the reply

    # explain_success_value: dead end for now (future work) - deliberately no branch
    # here, falls through to the generic fallback below.

    if state.get("opening_status") is not None:
        return "classify_content_state"  # opening already handled in a previous turn - skip
    return "classify_opening"  # first turn - no opening_status yet


def build_graph_builder() -> StateGraph:
    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("classify_professional_content", classify_professional_content)
    graph_builder.add_node("classify_opening", classify_opening)
    graph_builder.add_node("respond_direct", respond_direct)
    graph_builder.add_node("respond_with_check", respond_with_check)
    graph_builder.add_node("classify_content_state", classify_content_state)
    graph_builder.add_node("ask_direction", ask_direction)
    graph_builder.add_node("classify_direction_choice", classify_direction_choice)
    graph_builder.add_node("present_practical_track_intro", present_practical_track_intro)
    graph_builder.add_node("classify_practical_track_consent", classify_practical_track_consent)
    graph_builder.add_node("classify_respond_check_choice", classify_respond_check_choice)
    graph_builder.add_node("classify_practical_check_choice", classify_practical_check_choice)
    graph_builder.add_node("invite_to_share", invite_to_share)
    graph_builder.add_node("retrieve_expressions_content", retrieve_expressions_content)
    graph_builder.add_node("build_expressions_table", build_expressions_table)
    graph_builder.add_node("build_blocks", build_blocks)
    graph_builder.add_node("color_blocks", color_blocks)
    graph_builder.add_node("present_and_ask", present_and_ask)
    graph_builder.add_node("classify_present_choice", classify_present_choice)
    graph_builder.add_node("ask_which_block", ask_which_block)
    graph_builder.add_node("classify_block_target", classify_block_target)
    graph_builder.add_node("deepen_round", deepen_round)
    graph_builder.add_node("deepen_reply", deepen_reply)
    graph_builder.add_node("focus_on_block", focus_on_block)
    graph_builder.add_node("classify_focus_choice", classify_focus_choice)
    graph_builder.add_node("classify_readiness", classify_readiness)
    graph_builder.add_node("summarize_and_pivot", summarize_and_pivot)
    graph_builder.add_node("present_success_analysis_intro", present_success_analysis_intro)
    graph_builder.add_node("classify_success_consent", classify_success_consent)
    graph_builder.add_node("explain_success_value", explain_success_value)
    graph_builder.add_node("invite_success_story", invite_success_story)
    graph_builder.add_node("classify_scope_creep", classify_scope_creep)
    graph_builder.add_node("pivot_to_deeper_process", pivot_to_deeper_process)
    graph_builder.add_node("classify_pivot_consent", classify_pivot_consent)
    graph_builder.add_node("retrieve_success_analysis_context", retrieve_success_analysis_context)
    graph_builder.add_node("success_analysis_conversation", success_analysis_conversation)
    graph_builder.add_node("practical_track_conversation", practical_track_conversation)
    graph_builder.add_node("classify_from_practical_track", classify_from_practical_track)
    graph_builder.add_node("pivot_practical_to_success", pivot_practical_to_success)
    graph_builder.add_node("classify_practical_to_success_consent", classify_practical_to_success_consent)

    graph_builder.add_edge(START, "classify_professional_content")
    graph_builder.add_conditional_edges(
        "classify_professional_content",
        route_from_start,
        {
            "classify_opening": "classify_opening",
            "classify_content_state": "classify_content_state",
            "classify_direction_choice": "classify_direction_choice",
            "classify_respond_check_choice": "classify_respond_check_choice",
            "classify_practical_check_choice": "classify_practical_check_choice",
            "classify_present_choice": "classify_present_choice",
            "classify_block_target": "classify_block_target",
            "deepen_round": "deepen_round",
            "focus_on_block": "focus_on_block",
            "classify_focus_choice": "classify_focus_choice",
            "classify_success_consent": "classify_success_consent",
            "classify_scope_creep": "classify_scope_creep",
            "classify_pivot_consent": "classify_pivot_consent",
            "classify_practical_track_consent": "classify_practical_track_consent",
            "classify_from_practical_track": "classify_from_practical_track",
            "classify_practical_to_success_consent": "classify_practical_to_success_consent",
            "end": END,
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
        "classify_respond_check_choice",
        route_after_respond_check_choice,
        {
            "invite_to_share": "invite_to_share",
            "classify_content_state": "classify_content_state",
            "end": END,
        },
    )
    graph_builder.add_edge("invite_to_share", END)
    graph_builder.add_conditional_edges(
        "classify_practical_check_choice",
        route_after_practical_check_choice,
        {
            "present_practical_track_intro": "present_practical_track_intro",
            "classify_content_state": "classify_content_state",
        },
    )
    graph_builder.add_conditional_edges(
        "classify_content_state",
        route_after_content_state,
        {
            "ask_direction": "ask_direction",
            "retrieve_expressions_content": "retrieve_expressions_content",
            "present_practical_track_intro": "present_practical_track_intro",
            "end": END,
        },
    )
    graph_builder.add_edge("ask_direction", END)
    graph_builder.add_edge("present_success_analysis_intro", END)
    graph_builder.add_conditional_edges(
        "classify_success_consent",
        route_after_success_consent,
        {
            "invite_success_story": "invite_success_story",
            "explain_success_value": "explain_success_value",
            "classify_scope_creep": "classify_scope_creep",
        },
    )
    graph_builder.add_edge("explain_success_value", END)
    graph_builder.add_edge("invite_success_story", END)
    graph_builder.add_conditional_edges(
        "classify_scope_creep",
        route_after_scope_creep,
        {
            "pivot_to_deeper_process": "pivot_to_deeper_process",
            "success_analysis_conversation": "retrieve_success_analysis_context",
        },
    )
    graph_builder.add_edge("pivot_to_deeper_process", END)
    graph_builder.add_edge("retrieve_success_analysis_context", "success_analysis_conversation")
    graph_builder.add_edge("success_analysis_conversation", END)
    graph_builder.add_conditional_edges(
        "classify_pivot_consent",
        route_after_pivot_consent,
        {
            "retrieve_expressions_content": "retrieve_expressions_content",
            "success_analysis_conversation": "retrieve_success_analysis_context",
        },
    )
    graph_builder.add_conditional_edges(
        "classify_direction_choice",
        route_after_direction_choice,
        {
            "invite_to_share": "invite_to_share",
            "present_practical_track_intro": "present_practical_track_intro",
        },
    )
    graph_builder.add_edge("present_practical_track_intro", END)
    graph_builder.add_conditional_edges(
        "classify_practical_track_consent",
        route_after_practical_track_consent,
        {
            "practical_track_conversation": "practical_track_conversation",
            "end": END,
        },
    )
    graph_builder.add_edge("practical_track_conversation", END)
    graph_builder.add_conditional_edges(
        "classify_from_practical_track",
        route_after_from_practical_track,
        {
            "practical_track_conversation": "practical_track_conversation",
            "pivot_practical_to_success": "pivot_practical_to_success",
            "end": END,
        },
    )
    graph_builder.add_edge("pivot_practical_to_success", END)
    graph_builder.add_conditional_edges(
        "classify_practical_to_success_consent",
        route_after_practical_to_success_consent,
        {
            "present_success_analysis_intro": "present_success_analysis_intro",
            "practical_track_conversation": "practical_track_conversation",
        },
    )
    graph_builder.add_edge("retrieve_expressions_content", "build_expressions_table")
    graph_builder.add_edge("build_expressions_table", "build_blocks")
    graph_builder.add_edge("build_blocks", "color_blocks")
    graph_builder.add_edge("color_blocks", "present_and_ask")
    graph_builder.add_edge("present_and_ask", END)
    graph_builder.add_conditional_edges(
        "classify_present_choice",
        route_after_present_choice,
        {
            "ask_which_block": "ask_which_block",
            "end": END,
        },
    )
    graph_builder.add_edge("ask_which_block", END)
    graph_builder.add_conditional_edges(
        "classify_block_target",
        route_after_block_target,
        {
            "deepen_round": "deepen_round",
            "end": END,
        },
    )
    graph_builder.add_conditional_edges(
        "deepen_round",
        route_after_deepen_round,
        {
            "deepen_reply": "deepen_reply",
            "focus_on_block": "focus_on_block",
        },
    )
    graph_builder.add_edge("deepen_reply", END)
    graph_builder.add_edge("focus_on_block", END)
    graph_builder.add_conditional_edges(
        "classify_focus_choice",
        route_after_focus_choice,
        {
            "classify_readiness": "classify_readiness",
            "end": END,
        },
    )
    graph_builder.add_conditional_edges(
        "classify_readiness",
        route_after_readiness,
        {
            "summarize_and_pivot": "summarize_and_pivot",
            "end": END,
        },
    )
    graph_builder.add_edge("summarize_and_pivot", END)

    return graph_builder


graph = build_graph_builder().compile()
