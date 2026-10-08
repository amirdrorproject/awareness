import hmac
import logging
import os
from datetime import datetime, timezone
from typing import Literal

import anthropic
from fastapi import Depends, FastAPI, Header, HTTPException
from pinecone import Pinecone
from pydantic import BaseModel

from .chat_langgraph import get_last_assistant_message, get_new_assistant_messages, run_chat_turn
from .conversation_store import save_turn
from .engine import generate_reply
from .memory import format_memory_for_prompt, load_latest_memory, save_memory, summarize_conversation
from .prompts import PROMPTS, get_prompt_record, list_versions, save_version

app = FastAPI()

logger = logging.getLogger("api.index")
logger.setLevel(logging.INFO)


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


@app.get("/api/pinecone-query")
def pinecone_query(q: str):
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")

    if not api_key:
        return {"error": "PINECONE_API_KEY is not set."}
    if not index_name:
        return {"error": "PINECONE_INDEX_NAME is not set."}

    try:
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)

        # Auto-detect which namespace actually has data, rather than assuming -
        # prefer the default namespace if it's populated, else the first populated one.
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
            query={"inputs": {"text": q}, "top_k": 5},
        )
        hits = (results.get("result") or {}).get("hits") or []

        matches = []
        for hit in hits:
            fields = _extract_hit_value(hit, "fields") or {}
            matches.append(
                {
                    "id": _extract_hit_value(hit, "_id", "id"),
                    "score": _extract_hit_value(hit, "_score", "score"),
                    "text": fields.get("text"),
                    "module": fields.get("module"),
                    "chunk_title": fields.get("chunk_title"),
                    "doc_type": fields.get("doc_type"),
                }
            )

        return {"namespace": namespace, "matches": matches}
    except Exception as exc:
        return {"error": f"Failed to query Pinecone: {exc}"}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    conversation_id: str | None = None
    knowledge_mode: Literal["none", "search", "full"] = "none"
    # Who the client is, so their earlier conversations' memory can be loaded.
    client_name: str | None = None


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
        client_name = (request.client_name or "").strip()
        memory = load_latest_memory(client_name) if client_name else None
        reply = generate_reply(
            client,
            [{"role": m.role, "content": m.content} for m in request.messages],
            request.knowledge_mode,
            system_suffix=format_memory_for_prompt(memory) if memory else None,
        )

        if request.conversation_id and request.messages:
            save_turn(
                request.conversation_id,
                user_message=request.messages[-1].content,
                assistant_message=reply["content"],
                model=reply["model"],
                stop_reason=reply["stop_reason"],
                knowledge_mode=request.knowledge_mode,
                tool_calls=reply["tool_calls"],
                client_name=client_name or None,
            )

        return {
            "role": "assistant",
            "content": reply["content"],
            "tool_calls": reply["tool_calls"],
            "memory_loaded": memory is not None,
        }
    except Exception as exc:
        return {
            "role": "assistant",
            "content": f"Failed to reach Claude: {exc}",
        }


class EndConversationRequest(BaseModel):
    conversation_id: str
    client_name: str
    messages: list[ChatMessage]


@app.post("/api/conversation/end")
def end_conversation(request: EndConversationRequest):
    # Summarises what was established (on top of the client's earlier memory)
    # and saves it, so the client's next conversation can start from it.
    client_name = request.client_name.strip()
    if not client_name:
        return {"error": "A client name is needed to save memory."}
    if not request.messages:
        return {"error": "The conversation is empty - nothing to save."}
    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        previous = load_latest_memory(client_name)
        memory = summarize_conversation(
            client, [{"role": m.role, "content": m.content} for m in request.messages], previous
        )
        save_memory(client_name, request.conversation_id, memory)
        return {"memory": memory.model_dump()}
    except Exception as exc:
        logger.exception("end_conversation failed for client %r", client_name)
        return {"error": f"Failed to save memory: {exc}"}


def require_admin(x_admin_password: str | None = Header(default=None)) -> None:
    # One shared password (ADMIN_PASSWORD) guards every /api/admin route.
    # With no password configured, editing is off rather than open.
    expected = os.environ.get("ADMIN_PASSWORD")
    if not expected:
        raise HTTPException(status_code=503, detail="Admin is disabled: ADMIN_PASSWORD is not set.")
    if not x_admin_password or not hmac.compare_digest(x_admin_password.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Wrong password.")


def _check_prompt_name(name: str) -> None:
    if name not in PROMPTS:
        raise HTTPException(status_code=404, detail=f"Unknown prompt: {name}")


@app.get("/api/admin/prompts", dependencies=[Depends(require_admin)])
def admin_list_prompts():
    return {
        "prompts": [
            {"name": name, "label": label, **get_prompt_record(name)}
            for name, (_file, label) in PROMPTS.items()
        ]
    }


@app.get("/api/admin/prompts/{name}/versions", dependencies=[Depends(require_admin)])
def admin_prompt_versions(name: str):
    _check_prompt_name(name)
    return {"versions": list_versions(name)}


class PromptSaveRequest(BaseModel):
    content: str
    note: str | None = None


@app.post("/api/admin/prompts/{name}", dependencies=[Depends(require_admin)])
def admin_save_prompt(name: str, request: PromptSaveRequest):
    _check_prompt_name(name)
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="The prompt can't be empty.")
    try:
        return save_version(name, request.content, (request.note or "").strip())
    except Exception as exc:
        logger.exception("Saving prompt %r failed", name)
        raise HTTPException(status_code=500, detail=f"Failed to save: {exc}")


class LangGraphChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


@app.post("/api/chat-langgraph")
def chat_langgraph(request: LangGraphChatRequest):
    if not request.thread_id:
        return {"error": "Missing thread_id in request body."}

    try:
        result = run_chat_turn(request.thread_id, request.message)
        # responses holds every message this turn produced, in order (usually
        # one, but classify_professional_content can prepend a disclaimer
        # before the turn's normal reply). response is kept alongside it,
        # unchanged, for any caller still relying on the single-string shape.
        return {
            "response": get_last_assistant_message(result["messages"]),
            "responses": get_new_assistant_messages(result["messages"]),
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
