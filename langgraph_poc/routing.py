from langchain_core.messages import AIMessage

from langgraph_poc.common import (
    PRACTICAL_TRACK_PAUSE_PHRASE,
    PRACTICAL_TRACK_PAUSE_STRIP_CHARS,
    _human_messages,
)
from langgraph_poc.schemas import GraphState


def route_after_respond_direct(state: GraphState) -> str:
    if state.get("opening_status") == 1:
        return "end"
    return "continue"


def route_after_respond_check_choice(state: GraphState) -> str:
    answer = state.get("respond_check_answer")
    if answer == "yes":
        return "invite_to_share"
    if answer == "no":
        return "end"
    return "classify_content_state"


def route_after_classification(state: GraphState) -> str:
    if state.get("opening_status") in (1, 3, 5):
        return "respond_direct"
    return "respond_with_check"


def route_after_content_state(state: GraphState) -> str:
    if state.get("content_state") in ("emotional_vague", "dual", "practical_clear"):
        return "ask_direction"
    if state.get("content_state") == "emotional_clear":
        return "retrieve_expressions_content"
    return "end"


def route_after_success_consent(state: GraphState) -> str:
    answer = state.get("success_consent_answer")
    if answer == "yes":
        return "invite_success_story"
    if answer == "no":
        return "explain_success_value"
    return "classify_scope_creep"  # "other" - they likely already started sharing content


def route_after_scope_creep(state: GraphState) -> str:
    if state.get("scope_creep_status") == "drifting":
        return "pivot_to_deeper_process"
    return "success_analysis_conversation"


def route_after_pivot_consent(state: GraphState) -> str:
    if state.get("pivot_consent_answer") == "yes":
        return "retrieve_expressions_content"
    return "success_analysis_conversation"  # "no" or "other" - stay in the success-analysis tool


def route_after_present_choice(state: GraphState) -> str:
    if state.get("intent") == "deepen" and state.get("current_block_id") is None:
        return "ask_which_block"
    return "end"


def route_after_block_target(state: GraphState) -> str:
    if state.get("current_block_id") is not None and (state.get("deepen_round_count") or 0) < 2:
        return "deepen_round"
    return "end"


def route_after_deepen_round(state: GraphState) -> str:
    if (state.get("deepen_round_count") or 0) < 2:
        return "deepen_reply"
    return "focus_on_block"


def route_after_focus_choice(state: GraphState) -> str:
    messages = state.get("messages") or []
    if messages and isinstance(messages[-1], AIMessage):
        return "end"  # fallback re-ask just happened - wait for the client's next reply
    return "classify_readiness"  # current_block_id was successfully updated - continue immediately


def route_after_readiness(state: GraphState) -> str:
    if state.get("readiness") == "not_ready":
        return "summarize_and_pivot"
    return "end"


def route_after_practical_track_consent(state: GraphState) -> str:
    if state.get("practical_track_consent_answer") == "no":
        return "end"
    return "practical_track_conversation"  # "yes" or "other" - begin the open-ended dialogue


def route_after_from_practical_track(state: GraphState) -> str:
    status = state.get("practical_track_pause_status")
    if status == "pausing":
        return "end"
    if status == "capability_doubt":
        return "pivot_practical_to_success"
    return "practical_track_conversation"  # "continuing"


def route_after_practical_to_success_consent(state: GraphState) -> str:
    if state.get("practical_to_success_consent_answer") == "yes":
        return "present_success_analysis_intro"
    return "practical_track_conversation"  # "no" or "other" - stay in the practical track


def route_after_direction_choice(state: GraphState) -> str:
    if state.get("direction_choice") == "pause":
        return "invite_to_share"
    return "present_practical_track_intro"  # "continue"


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
