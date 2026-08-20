import logging
import os
from datetime import datetime, timezone

import anthropic
from fastapi import FastAPI
from pinecone import Pinecone
from pydantic import BaseModel

from .chat_langgraph import get_last_assistant_message, run_chat_turn
from .system_prompt import get_system_prompt, get_system_prompt_record, update_system_prompt

app = FastAPI()

logger = logging.getLogger("api.index")
logger.setLevel(logging.INFO)

CLAUDE_MODEL = "claude-sonnet-4-6"


@app.get("/api/time")
def get_time():
    return {"time": datetime.now(timezone.utc).isoformat()}


@app.get("/api/pinecone-test")
def pinecone_test():
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")

    if not api_key:
        return {"connected": False, "error": "PINECONE_API_KEY is not set."}
    if not index_name:
        return {"connected": False, "error": "PINECONE_INDEX_NAME is not set."}

    try:
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        return {
            "connected": True,
            "vector_count": stats.get("total_vector_count"),
            "dimension": stats.get("dimension"),
        }
    except Exception as exc:
        return {"connected": False, "error": f"Failed to connect to Pinecone: {exc}"}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


@app.post("/api/chat")
def chat(request: ChatRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {
            "role": "assistant",
            "content": "Server misconfiguration: ANTHROPIC_API_KEY is not set.",
        }

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=CLAUDE_MODEL,
            system=get_system_prompt(),
            max_tokens=1024,
            messages=[
                {"role": m.role, "content": m.content} for m in request.messages
            ],
        )
        content = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return {"role": "assistant", "content": content}
    except Exception as exc:
        return {
            "role": "assistant",
            "content": f"Failed to reach Claude: {exc}",
        }


class SystemPromptUpdateRequest(BaseModel):
    content: str


@app.get("/api/admin/system-prompt")
def get_system_prompt_admin():
    record = get_system_prompt_record()
    return {"content": record.get("content"), "updated_at": record.get("updated_at")}


@app.post("/api/admin/system-prompt")
def update_system_prompt_admin(request: SystemPromptUpdateRequest):
    try:
        record = update_system_prompt(request.content)
        return {"content": record.get("content"), "updated_at": record.get("updated_at")}
    except Exception as exc:
        return {"error": f"Failed to update system prompt: {exc}"}


class LangGraphChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


@app.post("/api/chat-langgraph")
def chat_langgraph(request: LangGraphChatRequest):
    if not request.thread_id:
        return {"error": "Missing thread_id in request body."}

    try:
        result = run_chat_turn(request.thread_id, request.message)
        return {
            "response": get_last_assistant_message(result["messages"]),
            "internal_audit_log": result.get("internal_audit_log"),
            "_debug": {
                "thread_id": request.thread_id,
                "messages_count": len(result.get("messages", [])),
                "opening_status": result.get("opening_status"),
                "content_state": result.get("content_state"),
                "direction_choice": result.get("direction_choice"),
            },
        }
    except Exception as exc:
        logger.exception(
            "chat_langgraph failed for thread_id=%r", request.thread_id
        )
        return {
            "error": f"Failed to run chat turn: {exc}",
            "_debug": {"thread_id": request.thread_id, "exception_type": type(exc).__name__},
        }
