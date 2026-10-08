import logging
from pathlib import Path

import anthropic

from .knowledge import TOOLS_BY_MODE, run_tool

logger = logging.getLogger("api.engine")

CLAUDE_MODEL = "claude-opus-5-5"
CLAUDE_EFFORT = "medium"
MAX_TOOL_ROUNDS = 3

# The engine prompt lives in the repo so every change to it is versioned.
# It's read on every request, so edits apply without restarting the server.
ENGINE_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "engine.md"


def get_engine_prompt() -> str:
    return ENGINE_PROMPT_PATH.read_text(encoding="utf-8")


def generate_reply(
    client: anthropic.Anthropic,
    messages: list[dict],
    knowledge_mode: str = "none",
    system_suffix: str | None = None,
) -> dict:
    """Runs one agent turn: the engine prompt + conversation, plus whichever
    knowledge tool the mode allows. Returns the reply text and every tool call.

    `system_suffix` appends text to the engine prompt - used by the test
    scripts to try a prompt variant without touching prompts/engine.md."""
    tools = TOOLS_BY_MODE[knowledge_mode]
    system = get_engine_prompt() + (f"\n\n{system_suffix}" if system_suffix else "")
    messages = list(messages)
    tool_calls = []

    # Claude decides whether to call a tool; each call is run here and its
    # result sent back, until Claude answers in text. On the last round
    # tools are switched off, so the turn always ends with a reply.
    for round_number in range(MAX_TOOL_ROUNDS + 1):
        tool_params = {}
        if tools:
            tool_params["tools"] = tools
            if round_number == MAX_TOOL_ROUNDS:
                tool_params["tool_choice"] = {"type": "none"}
        # fallbacks="default" re-runs a request the safety classifiers decline
        # on a fallback model inside the same call, instead of just stopping.
        response = client.beta.messages.create(
            model=CLAUDE_MODEL,
            system=system,
            max_tokens=16000,
            output_config={"effort": CLAUDE_EFFORT},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=messages,
            **tool_params,
        )
        if response.stop_reason != "tool_use":
            break

        # The full content (thinking blocks included) goes back unchanged.
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                result_text, record = run_tool(block.name, block.input)
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result_text})
                tool_calls.append({"name": block.name, "input": block.input, "output": record})
            except Exception as exc:
                logger.exception("Tool %s failed", block.name)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": f"Error: {exc}", "is_error": True}
                )
                tool_calls.append({"name": block.name, "input": block.input, "error": str(exc)})
        messages.append({"role": "user", "content": tool_results})

    if response.stop_reason == "refusal":
        content = "(Claude declined to answer this message.)"
    else:
        content = "".join(block.text for block in response.content if block.type == "text")

    return {
        "content": content,
        "model": response.model,
        "stop_reason": response.stop_reason,
        "tool_calls": tool_calls,
    }
