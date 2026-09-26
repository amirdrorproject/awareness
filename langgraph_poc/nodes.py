import os
import re
from typing import Literal

from langchain_core.messages import AIMessage
from pinecone import Pinecone

from langgraph_poc.common import (
    CONST_PROMPTS,
    MESSAGES,
    PROMPTS,
    SUCCESS_ANALYSIS_RAG_TOP_K,
    _content_text,
    _detect_namespace,
    _extract_hit_value,
    _format_blocks,
    _format_conversation,
    _get_llm,
    _human_messages,
)
from langgraph_poc.schemas import (
    BlockTargetResult,
    BuildBlocksResult,
    ColorBlocksResult,
    ContentStateClassification,
    DeepenRoundResult,
    DirectionChoiceClassification,
    ExpressionsTableResult,
    FocusChoiceResult,
    FromPracticalTrackResult,
    GraphState,
    OpeningClassification,
    PresentChoiceResult,
    ProfessionalContentClassification,
    ReadinessResult,
    ReflectionContext,
    ScopeCreepResult,
    YesNoOtherResult,
)


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
