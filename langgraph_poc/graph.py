from langgraph.graph import END, START, StateGraph

from langgraph_poc import nodes, routing
from langgraph_poc.schemas import GraphState

# Each node is registered under its function's name.
NODES = [
    nodes.classify_professional_content,
    nodes.classify_opening,
    nodes.respond_direct,
    nodes.respond_with_check,
    nodes.classify_content_state,
    nodes.ask_direction,
    nodes.classify_direction_choice,
    nodes.present_practical_track_intro,
    nodes.classify_practical_track_consent,
    nodes.classify_respond_check_choice,
    nodes.classify_practical_check_choice,
    nodes.invite_to_share,
    nodes.retrieve_expressions_content,
    nodes.build_expressions_table,
    nodes.build_blocks,
    nodes.color_blocks,
    nodes.present_and_ask,
    nodes.classify_present_choice,
    nodes.ask_which_block,
    nodes.classify_block_target,
    nodes.deepen_round,
    nodes.deepen_reply,
    nodes.focus_on_block,
    nodes.classify_focus_choice,
    nodes.classify_readiness,
    nodes.summarize_and_pivot,
    nodes.present_success_analysis_intro,
    nodes.classify_success_consent,
    nodes.explain_success_value,
    nodes.invite_success_story,
    nodes.classify_scope_creep,
    nodes.pivot_to_deeper_process,
    nodes.classify_pivot_consent,
    nodes.retrieve_success_analysis_context,
    nodes.success_analysis_conversation,
    nodes.practical_track_conversation,
    nodes.classify_from_practical_track,
    nodes.pivot_practical_to_success,
    nodes.classify_practical_to_success_consent,
]

# (source, target)
EDGES = [
    (START, "classify_professional_content"),
    ("respond_with_check", END),
    ("invite_to_share", END),
    ("ask_direction", END),
    ("present_success_analysis_intro", END),
    ("explain_success_value", END),
    ("invite_success_story", END),
    ("pivot_to_deeper_process", END),
    ("retrieve_success_analysis_context", "success_analysis_conversation"),
    ("success_analysis_conversation", END),
    ("present_practical_track_intro", END),
    ("practical_track_conversation", END),
    ("pivot_practical_to_success", END),
    ("retrieve_expressions_content", "build_expressions_table"),
    ("build_expressions_table", "build_blocks"),
    ("build_blocks", "color_blocks"),
    ("color_blocks", "present_and_ask"),
    ("present_and_ask", END),
    ("ask_which_block", END),
    ("deepen_reply", END),
    ("focus_on_block", END),
    ("summarize_and_pivot", END),
]

# (source, router, outcomes). Each outcome is a router return value that leads to
# the node of the same name; "end" leads to END; a (value, node) pair maps a
# return value to a node with a different name.
CONDITIONAL_EDGES = [
    (
        "classify_professional_content",
        routing.route_from_start,
        [
            "classify_opening",
            "classify_content_state",
            "classify_direction_choice",
            "classify_respond_check_choice",
            "classify_practical_check_choice",
            "classify_present_choice",
            "classify_block_target",
            "deepen_round",
            "focus_on_block",
            "classify_focus_choice",
            "classify_success_consent",
            "classify_scope_creep",
            "classify_pivot_consent",
            "classify_practical_track_consent",
            "classify_from_practical_track",
            "classify_practical_to_success_consent",
            "end",
        ],
    ),
    (
        "classify_opening",
        routing.route_after_classification,
        ["respond_direct", "respond_with_check"],
    ),
    (
        "respond_direct",
        routing.route_after_respond_direct,
        [("continue", "classify_content_state"), "end"],
    ),
    (
        "classify_respond_check_choice",
        routing.route_after_respond_check_choice,
        ["invite_to_share", "classify_content_state", "end"],
    ),
    (
        "classify_practical_check_choice",
        routing.route_after_practical_check_choice,
        ["present_practical_track_intro", "classify_content_state"],
    ),
    (
        "classify_content_state",
        routing.route_after_content_state,
        ["ask_direction", "retrieve_expressions_content", "present_practical_track_intro", "end"],
    ),
    (
        "classify_success_consent",
        routing.route_after_success_consent,
        ["invite_success_story", "explain_success_value", "classify_scope_creep"],
    ),
    (
        "classify_scope_creep",
        routing.route_after_scope_creep,
        [
            "pivot_to_deeper_process",
            ("success_analysis_conversation", "retrieve_success_analysis_context"),
        ],
    ),
    (
        "classify_pivot_consent",
        routing.route_after_pivot_consent,
        [
            "retrieve_expressions_content",
            ("success_analysis_conversation", "retrieve_success_analysis_context"),
        ],
    ),
    (
        "classify_direction_choice",
        routing.route_after_direction_choice,
        ["invite_to_share", "present_practical_track_intro"],
    ),
    (
        "classify_practical_track_consent",
        routing.route_after_practical_track_consent,
        ["practical_track_conversation", "end"],
    ),
    (
        "classify_from_practical_track",
        routing.route_after_from_practical_track,
        ["practical_track_conversation", "pivot_practical_to_success", "end"],
    ),
    (
        "classify_practical_to_success_consent",
        routing.route_after_practical_to_success_consent,
        ["present_success_analysis_intro", "practical_track_conversation"],
    ),
    (
        "classify_present_choice",
        routing.route_after_present_choice,
        ["ask_which_block", "end"],
    ),
    (
        "classify_block_target",
        routing.route_after_block_target,
        ["deepen_round", "end"],
    ),
    (
        "deepen_round",
        routing.route_after_deepen_round,
        ["deepen_reply", "focus_on_block"],
    ),
    (
        "classify_focus_choice",
        routing.route_after_focus_choice,
        ["classify_readiness", "end"],
    ),
    (
        "classify_readiness",
        routing.route_after_readiness,
        ["summarize_and_pivot", "end"],
    ),
]


def _route_map(outcomes: list) -> dict:
    route_map = {}
    for outcome in outcomes:
        if isinstance(outcome, tuple):
            value, target = outcome
        elif outcome == "end":
            value, target = outcome, END
        else:
            value, target = outcome, outcome
        route_map[value] = target
    return route_map


def build_graph_builder() -> StateGraph:
    graph_builder = StateGraph(GraphState)
    for node in NODES:
        graph_builder.add_node(node.__name__, node)
    for source, target in EDGES:
        graph_builder.add_edge(source, target)
    for source, router, outcomes in CONDITIONAL_EDGES:
        graph_builder.add_conditional_edges(source, router, _route_map(outcomes))
    return graph_builder


graph = build_graph_builder().compile()
