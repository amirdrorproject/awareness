import json
import os
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

CLASSIFICATION_MODEL = "claude-sonnet-4-6"

SUCCESS_ANALYSIS_RAG_TOP_K = 3

PRACTICAL_TRACK_PAUSE_PHRASE = "יצאתי לחשוב"

PRACTICAL_TRACK_PAUSE_STRIP_CHARS = " \t\n\r.,!?;:\"'״׳"

# Fixed response text sent to the client verbatim, with no LLM call involved in
# producing it, lives in messages.json. System prompts that guide LLM-generated
# text (never sent to the client as-is) live in prompts.json. Both are loaded
# once here at import time - nodes read from these dicts at call time, not from
# disk per call.
with open(Path(__file__).parent / "messages.json", "r", encoding="utf-8") as _messages_file:
    MESSAGES: dict = json.load(_messages_file)

with open(Path(__file__).parent / "prompts.json", "r", encoding="utf-8") as _prompts_file:
    PROMPTS: dict = json.load(_prompts_file)

# Developer-owned prompts (classifier/extraction system prompts and fixed prompt
# scaffolding). Unlike prompts.json/messages.json, not exposed in Keystatic.
with open(Path(__file__).parent / "const_prompts.json", "r", encoding="utf-8") as _const_prompts_file:
    CONST_PROMPTS: dict = json.load(_const_prompts_file)


def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=CLASSIFICATION_MODEL,
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )


def _content_text(message) -> str:
    return message.content if isinstance(message.content, str) else str(message.content)


def _human_messages(messages) -> list:
    return [m for m in messages if isinstance(m, HumanMessage)]


def _format_blocks(blocks: list[dict], include_color: bool = False) -> str:
    if include_color:
        return "\n".join(
            f"{block.get('block_id')}. topic={block.get('topic')!r}, color={block.get('color')!r}"
            for block in blocks
        )
    return "\n".join(
        f"{block.get('block_id')}. topic={block.get('topic')!r}" for block in blocks
    )


def _detect_namespace(index) -> str:
    # Auto-detect which namespace actually has data (same as /api/pinecone-query):
    # the default namespace if populated, otherwise the first populated one.
    stats = index.describe_index_stats()
    namespaces = stats.get("namespaces") or {}
    namespace = ""
    if not (namespaces.get("") or {}).get("vector_count"):
        populated = [
            name for name, info in namespaces.items() if (info or {}).get("vector_count")
        ]
        if populated:
            namespace = populated[0]
    return namespace

def _format_conversation(messages) -> str:
    lines = []
    for m in messages:
        role = "Client" if isinstance(m, HumanMessage) else "Assistant"
        content = _content_text(m)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


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
