import logging
import os

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver

from langgraph_poc.graph import build_graph_builder

logger = logging.getLogger("chat_langgraph")
logger.setLevel(logging.INFO)


def run_chat_turn(thread_id: str, user_message: str) -> dict:
    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        raise RuntimeError("SUPABASE_DB_URL environment variable is not set.")

    with PostgresSaver.from_conn_string(db_url) as checkpointer:
        checkpointer.setup()
        graph = build_graph_builder().compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": thread_id}}
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
        )

        # Diagnostic fingerprint only - no message content, just shape/state,
        # so a repeated identical line across requests is an immediate red flag
        # in Vercel's runtime logs without needing to reproduce with new prints.
        # Using logging (not print) - confirmed via Vercel logs that logging-module
        # output (e.g. httpx's request logs) is captured while plain print() was not.
        logger.info(
            "thread_id=%r messages_count=%d opening_status=%r content_state=%r "
            "direction_choice=%r audit_log_len=%d",
            thread_id,
            len(result.get("messages", [])),
            result.get("opening_status"),
            result.get("content_state"),
            result.get("direction_choice"),
            len(result.get("internal_audit_log") or ""),
        )

        return {
            "messages": result.get("messages", []),
            "internal_audit_log": result.get("internal_audit_log"),
            "opening_status": result.get("opening_status"),
            "content_state": result.get("content_state"),
        }


def get_last_assistant_message(messages: list) -> str | None:
    # Only return a reply if this turn actually produced one - i.e. the trailing
    # message is a fresh AIMessage, not just the human message we sent (which is
    # what happens when a turn only classifies/logs, like classify_direction_choice,
    # or reaches END directly, like emotional_clear/practical_clear). Otherwise this
    # would keep re-serving the last AIMessage from an earlier turn as if it were new.
    if not messages or not isinstance(messages[-1], AIMessage):
        return None
    return messages[-1].content
