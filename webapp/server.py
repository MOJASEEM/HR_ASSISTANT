"""
FastAPI backend for the RAG bot web UI.

This does NOT reimplement any pipeline logic — it just calls your
existing graph.py (build_graph / app.invoke) and returns the resulting
state as JSON, so the frontend can show the answer + the pipeline trace
(router decision, retrieval count, grading result, hallucination check).

Run it with:  uvicorn webapp.server:app --reload
Then open:    http://127.0.0.1:8000
"""

import sys
from pathlib import Path

# Make the project root importable (so "from graph import build_graph" works
# no matter where uvicorn is launched from)
sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from graph import build_graph

app = FastAPI(title="HR Policy Assistant")

# Build the LangGraph pipeline once at startup, reuse across requests
pipeline = build_graph()

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ChatRequest(BaseModel):
    question: str


@app.post("/api/chat")
def chat(request: ChatRequest):
    result = pipeline.invoke({"question": request.question, "retries": 0})

    # Build a clean trace summary from whatever fields ended up populated
    # in state — different routes populate different fields, so we only
    # include what's actually present.
    trace = {"route": result.get("route")}

    if result.get("docs") is not None:
        trace["chunks_retrieved"] = len(result["docs"])
    if result.get("relevant_docs") is not None:
        trace["chunks_kept"] = len(result["relevant_docs"])
    if result.get("is_grounded") is not None:
        trace["grounded"] = result["is_grounded"]
    if result.get("search_query"):
        trace["rewritten_query"] = result["search_query"]

    sources = []
    for src in result.get("sources") or []:
        if hasattr(src, "page_content"):
            sources.append(src.page_content[:200])
        elif isinstance(src, dict):
            sources.append(src.get("title", "web result"))

    return {
        "answer": result.get("answer", "I wasn't able to generate an answer."),
        "trace": trace,
        "sources": sources,
    }


# Serve the frontend files (must come after the /api routes above)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
