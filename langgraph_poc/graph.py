from langgraph.graph import END, START, StateGraph

from langgraph_poc.nodes import (
    ask_direction,
    ask_which_block,
    build_blocks,
    build_expressions_table,
    classify_block_target,
    classify_content_state,
    classify_direction_choice,
    classify_focus_choice,
    classify_from_practical_track,
    classify_opening,
    classify_pivot_consent,
    classify_practical_check_choice,
    classify_practical_to_success_consent,
    classify_practical_track_consent,
    classify_present_choice,
    classify_professional_content,
    classify_readiness,
    classify_respond_check_choice,
    classify_scope_creep,
    classify_success_consent,
    color_blocks,
    deepen_reply,
    deepen_round,
    explain_success_value,
    focus_on_block,
    invite_success_story,
    invite_to_share,
    pivot_practical_to_success,
    pivot_to_deeper_process,
    practical_track_conversation,
    present_and_ask,
    present_practical_track_intro,
    present_success_analysis_intro,
    respond_direct,
    respond_with_check,
    retrieve_expressions_content,
    retrieve_success_analysis_context,
    success_analysis_conversation,
    summarize_and_pivot,
)
from langgraph_poc.routing import (
    route_after_block_target,
    route_after_classification,
    route_after_content_state,
    route_after_deepen_round,
    route_after_direction_choice,
    route_after_focus_choice,
    route_after_from_practical_track,
    route_after_pivot_consent,
    route_after_practical_check_choice,
    route_after_practical_to_success_consent,
    route_after_practical_track_consent,
    route_after_present_choice,
    route_after_readiness,
    route_after_respond_check_choice,
    route_after_respond_direct,
    route_after_scope_creep,
    route_after_success_consent,
    route_from_start,
)
from langgraph_poc.schemas import GraphState


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
