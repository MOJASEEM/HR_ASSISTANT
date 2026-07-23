
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

from retreiver.rag import load_retreiver, get_llm, generate_answer, rewrite_query
from retreiver.grader import grade_chunks
from routing.routed import (
    classify_question,
    answer_direct,
    answer_from_web,
)

# --- The shared "State" that flows through every node ---
# Every node reads from this and returns updates to merge into it.
class GraphState(TypedDict):
    question: str
    route: Optional[str]
    docs: Optional[List]
    relevant_docs: Optional[List]
    search_query: Optional[str]
    answer: Optional[str]
    sources: Optional[List]
    is_grounded: Optional[bool]
    retries: int


# --- Set up shared resources once ---
retriever = load_retreiver()
llm = get_llm()


# ============ NODES ============
# Each node is just your existing function, wrapped to read/write state.

def router_node(state: GraphState) -> dict:
    route = classify_question(state["question"], llm)
    print(f"   [Router] → {route}")
    return {"route": route}


def retrieve_node(state: GraphState) -> dict:
    docs = retriever.invoke(state["question"])
    print(f"   [Retrieve] {len(docs)} chunk(s) fetched")
    return {"docs": docs}


def grade_node(state: GraphState) -> dict:
    relevant = grade_chunks(state["question"], state["docs"], llm)
    print(f"   [Grade] {len(relevant)}/{len(state['docs'])} chunk(s) kept")
    return {"relevant_docs": relevant}


def generate_node(state: GraphState) -> dict:
    answer, sources = generate_answer(state["question"], state["relevant_docs"], llm)
    print("   [Generate] answer drafted")
    return {"answer": answer, "sources": sources}


def rewrite_node(state: GraphState) -> dict:
    rewritten = rewrite_query(state["question"], llm)
    print(f"   [Rewrite] '{state['question']}' → '{rewritten}'")
    return {"search_query": rewritten}


def web_search_node(state: GraphState) -> dict:
    query = state.get("search_query") or state["question"]
    answer, sources = answer_from_web(query, llm)
    print("   [Web Search] answer drafted from web results")
    return {"answer": answer, "sources": sources}


def direct_node(state: GraphState) -> dict:
    answer, sources = answer_direct(state["question"], llm)
    print("   [Direct] answered without retrieval")
    return {"answer": answer, "sources": sources}


# CONDITIONAL EDGES 
# These are the "decision points" — they look at state and pick the next node.

def route_decision(state: GraphState) -> str:
    return {
        "VECTORSTORE": "retrieve",
        "WEB_SEARCH": "rewrite",
        "DIRECT": "direct",
    }[state["route"]]


def grade_decision(state: GraphState) -> str:
    return "generate" if state["relevant_docs"] else "rewrite"


# BUILD THE GRAPH 

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("generate", generate_node)

    graph.add_node("rewrite", rewrite_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("direct", direct_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges("router", route_decision, {
        "retrieve": "retrieve",
        "rewrite": "rewrite",
        "direct": "direct",
    })

    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges("grade", grade_decision, {
        "generate": "generate",
        "rewrite": "rewrite",
    })

    graph.add_edge("rewrite", "web_search")
    graph.add_edge("web_search", END)
    graph.add_edge("direct", END)

    return graph.compile()


def main():
    app = build_graph()
    print("LangGraph RAG bot ready! Type a question, or 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        result = app.invoke({"question": question, "retries": 0})
        print(f"\nBot: {result['answer']}\n")


if __name__ == "__main__":
    main()