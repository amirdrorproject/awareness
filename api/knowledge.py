import os

from pinecone import Pinecone

PINECONE_NAMESPACE = "default"
EXPRESSION_BANK_MODULE = "expression_banks"
EXPRESSION_BANK_TOP_K = 3

# Claude decides on its own whether and when to call this - the tool
# description is what it reads to make that decision, so it says what the
# bank is for and when it is NOT worth calling.
SEARCH_EXPRESSION_BANK_TOOL = {
    "name": "search_expression_bank",
    "description": (
        "Searches the expression banks of Amir Dror's method: how clients phrase emotional "
        "states, doubts, being stuck, threats, needs and values, and which category of the "
        "method each phrasing belongs to. These categories are specific to the method, not "
        "general knowledge, so your own reading of an expression can differ from how the "
        "method classifies it. Use it when the client's words carry an emotional charge or "
        "a hint of one, before you offer a hypothesis about the emotional side. Don't use it "
        "for plain factual updates about the work. The results are background for your own "
        "judgment - never quote them to the client or name a category to them."
    ),
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

TOOLS = [SEARCH_EXPRESSION_BANK_TOOL]


def search_expression_bank(query: str) -> dict:
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    if not api_key or not index_name:
        raise RuntimeError("Pinecone is not configured (missing PINECONE_API_KEY/PINECONE_INDEX_NAME).")

    index = Pinecone(api_key=api_key).Index(index_name)
    results = index.search(
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


def run_tool(name: str, tool_input: dict) -> tuple[str, dict]:
    """Runs a tool call. Returns the text Claude gets back, plus a record for logging."""
    if name == "search_expression_bank":
        result = search_expression_bank(tool_input["query"])
        text = "\n\n".join(f"[{hit['title']}]\n{hit['text']}" for hit in result["hits"]) or "No matches."
        return text, result
    raise ValueError(f"Unknown tool: {name}")
