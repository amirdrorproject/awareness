import os
from datetime import datetime, timezone

import anthropic
from fastapi import FastAPI
from pydantic import BaseModel

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
