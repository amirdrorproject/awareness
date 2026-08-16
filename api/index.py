import os
import uuid
from datetime import datetime, timezone

import anthropic
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .chat_langgraph import get_last_assistant_message, run_chat_turn
from .system_prompt import get_system_prompt, get_system_prompt_record, update_system_prompt

app = FastAPI()

CLAUDE_MODEL = "claude-sonnet-4-6"


@app.get("/api/time")
def get_time():
    return {"time": datetime.now(timezone.utc).isoformat()}


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


THREAD_ID_COOKIE = "awareness_thread_id"
THREAD_ID_COOKIE_MAX_AGE = 60 * 60 * 24 * 365  # 1 year


class LangGraphChatRequest(BaseModel):
    message: str


@app.post("/api/chat-langgraph")
def chat_langgraph(request: LangGraphChatRequest, http_request: Request):
    thread_id = http_request.cookies.get(THREAD_ID_COOKIE)
    is_new_thread = thread_id is None
    if is_new_thread:
        thread_id = str(uuid.uuid4())

    try:
        result = run_chat_turn(thread_id, request.message)
        payload = {
            "response": get_last_assistant_message(result["messages"]),
            "internal_audit_log": result.get("internal_audit_log"),
        }
    except Exception as exc:
        payload = {"error": f"Failed to run chat turn: {exc}"}

    response = JSONResponse(content=payload)
    if is_new_thread:
        response.set_cookie(
            key=THREAD_ID_COOKIE,
            value=thread_id,
            max_age=THREAD_ID_COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
        )
    return response
