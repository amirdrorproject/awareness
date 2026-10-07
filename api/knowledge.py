import os
from functools import lru_cache

from pinecone import Pinecone

PINECONE_NAMESPACE = "default"
EXPRESSION_BANK_MODULE = "expression_banks"
EXPRESSION_BANK_ID_PREFIX = "rag_bank_"
EXPRESSION_BANK_TOP_K = 3

# How the agent gets at the expression banks. Three modes so the same
# conversation can be compared with no bank, with retrieval, and with the
# whole bank - the question being whether retrieval actually helps.
KNOWLEDGE_MODES = ("none", "search", "full")

# What the bank is and when to reach for it is shared by both tools, so the
# only difference between "search" and "full" is how the content comes back.
# The "specific to the method" framing matters: without it Claude assumed it
# already understood every expression and never called the tool.
_BANK_DESCRIPTION = (
    "The expression banks of Amir Dror's method: how clients phrase emotional states, "
    "doubts, being stuck, threats, needs and values, and which category of the method "
    "each phrasing belongs to. These categories are specific to the method, not general "
    "knowledge, so your own reading of an expression can differ from how the method "
    "classifies it. Use it when the client's words carry an emotional charge or a hint "
    "of one, before you offer a hypothesis about the emotional side. Don't use it for "
    "plain factual updates about the work. The results are background for your own "
    "judgment - never quote them to the client or name a category to them."
)

SEARCH_EXPRESSION_BANK_TOOL = {
    "name": "search_expression_bank",
    "description": "Searches " + _BANK_DESCRIPTION[0].lower() + _BANK_DESCRIPTION[1:]
    + " Returns the few bank entries closest to your query.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The client's own words (or a close paraphrase) that you want to identify.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": True,
}

GET_EXPRESSION_BANK_TOOL = {
    "name": "get_expression_bank",
    "description": "Returns all of " + _BANK_DESCRIPTION[0].lower() + _BANK_DESCRIPTION[1:]
    + " Returns every bank in full; there is nothing to search for.",
    "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    "strict": True,
}

TOOLS_BY_MODE = {
    "none": [],
    "search": [SEARCH_EXPRESSION_BANK_TOOL],
    "full": [GET_EXPRESSION_BANK_TOOL],
}


def _get_index():
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    if not api_key or not index_name:
        raise RuntimeError("Pinecone is not configured (missing PINECONE_API_KEY/PINECONE_INDEX_NAME).")
    return Pinecone(api_key=api_key).Index(index_name)


def search_expression_bank(query: str) -> dict:
    results = _get_index().search(
        namespace=PINECONE_NAMESPACE,
        query={
            "inputs": {"text": query},
            "top_k": EXPRESSION_BANK_TOP_K,
            "filter": {"module": {"$eq": EXPRESSION_BANK_MODULE}},
        },
        fields=["chunk_title", "text"],
    )
    hits = [
        {"id": hit.id, "score": round(hit.score, 3), "title": hit.fields.get("chunk_title"), "text": hit.fields.get("text")}
        for hit in results.result.hits
    ]
    return {"query": query, "hits": hits}


@lru_cache(maxsize=1)
def load_expression_bank() -> tuple:
    # The bank is small (~10 records) and doesn't change between requests, so
    # it's fetched once per server process.
    index = _get_index()
    ids = [
        item.id
        for page in index.list(prefix=EXPRESSION_BANK_ID_PREFIX, namespace=PINECONE_NAMESPACE)
        for item in page.vectors
    ]
    records = index.fetch(ids=ids, namespace=PINECONE_NAMESPACE).vectors.values()
    entries = [
        {"id": record.id, "title": record.metadata.get("chunk_title"), "text": record.metadata.get("text")}
        for record in records
        if (record.metadata or {}).get("module") == EXPRESSION_BANK_MODULE
    ]
    return tuple(sorted(entries, key=lambda entry: entry["id"]))


def _format_entries(entries) -> str:
    return "\n\n".join(f"[{entry['title']}]\n{entry['text']}" for entry in entries) or "No matches."


def run_tool(name: str, tool_input: dict) -> tuple[str, dict]:
    """Runs a tool call. Returns the text Claude gets back, plus a record for logging."""
    if name == "search_expression_bank":
        result = search_expression_bank(tool_input["query"])
        return _format_entries(result["hits"]), result
    if name == "get_expression_bank":
        entries = load_expression_bank()
        # The log only needs which banks were returned, not their full text again.
        return _format_entries(entries), {"hits": [{"title": entry["title"]} for entry in entries]}
    raise ValueError(f"Unknown tool: {name}")
