import logging
import time

import anthropic

from .knowledge import TOOLS_BY_MODE, run_tool
from .prompts import get_prompt, get_prompt_record

logger = logging.getLogger("api.engine")

CLAUDE_MODEL = "claude-opus-5-5"
CLAUDE_EFFORT = "medium"
MAX_TOOL_ROUNDS = 3

# USD per million tokens, for the per-turn cost estimate in the run trace:
# (input, output, cache read, cache write). A fallback model can answer a turn,
# so the model that actually ran picks the row; an unknown model has no estimate.
PRICES_PER_MTOK = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
    "claude-opus-5": (5.00, 25.00, 0.50, 6.25),
    "claude-opus-4-8": (5.00, 25.00, 0.50, 6.25),
    "claude-fable-5-1": (10.00, 50.00, 0.25, 12.50),
}


def get_engine_prompt() -> str:
    # The newest version saved from /admin, else prompts/engine.md. Read on
    # every request, so an edit applies to the very next message.
    return get_prompt("engine")


def _estimate_cost(model: str, usage: dict) -> float | None:
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        return None
    input_price, output_price, cache_read_price, cache_write_price = prices
    return round(
        (
            usage["input_tokens"] * input_price
            + usage["output_tokens"] * output_price
            + usage["cache_read_input_tokens"] * cache_read_price
            + usage["cache_creation_input_tokens"] * cache_write_price
        )
        / 1_000_000,
        5,
    )


def generate_reply(
    client: anthropic.Anthropic,
    messages: list[dict],
    knowledge_mode: str = "none",
    system_suffix: str | None = None,
) -> dict:
    """Runs one agent turn: the engine prompt + conversation, plus whichever
    knowledge tool the mode allows. Returns the reply text, every tool call,
    and a trace of how the reply came about (prompt version, thinking
    summaries, tokens, latency, cost) for the run-details panel.

    `system_suffix` is appended to the engine prompt - the client's memory in
    the chat, and prompt variants in the test scripts."""
    started = time.monotonic()
    tools = TOOLS_BY_MODE[knowledge_mode]
    prompt_record = get_prompt_record("engine")
    system = prompt_record["content"] + (f"\n\n{system_suffix}" if system_suffix else "")
    messages = list(messages)
    # How many conversation messages Claude was given (this one included),
    # before any tool round adds to the list.
    history_messages = len(messages)
    tool_calls = []
    thinking = []
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    rounds = 0

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
        # display="summarized" returns a readable summary of Claude's thinking
        # (the full reasoning is never exposed); it doesn't change what is billed.
        response = client.beta.messages.create(
            model=CLAUDE_MODEL,
            system=system,
            max_tokens=16000,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": CLAUDE_EFFORT},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=messages,
            **tool_params,
        )
        rounds += 1
        for key in usage:
            usage[key] += getattr(response.usage, key, None) or 0
        thinking += [block.thinking for block in response.content if block.type == "thinking" and block.thinking]
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

    trace = {
        "model": response.model,
        "effort": CLAUDE_EFFORT,
        "stop_reason": response.stop_reason,
        "prompt": {
            "source": prompt_record["source"],
            "created_at": prompt_record["created_at"],
            "note": prompt_record["note"],
        },
        "system_suffix": system_suffix,
        "knowledge_mode": knowledge_mode,
        "history_messages": history_messages,
        "api_calls": rounds,
        "thinking": thinking,
        "usage": usage,
        "cost_usd": _estimate_cost(response.model, usage),
        "latency_s": round(time.monotonic() - started, 2),
    }
    return {
        "content": content,
        "model": response.model,
        "stop_reason": response.stop_reason,
        "tool_calls": tool_calls,
        "trace": trace,
    }
