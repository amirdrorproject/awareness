import os

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver

from langgraph_poc.graph import build_graph_builder


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

        return {
            "messages": result.get("messages", []),
            "internal_audit_log": result.get("internal_audit_log"),
            "opening_status": result.get("opening_status"),
            "content_state": result.get("content_state"),
        }


def get_last_assistant_message(messages: list) -> str | None:
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    return ai_messages[-1].content if ai_messages else None
