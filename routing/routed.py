from dotenv import load_dotenv
from ddgs import DDGS
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from retreiver.rag import load_retreiver, get_llm, generate_answer
from retreiver.grader import grade_chunks

load_dotenv()


ROUTER_PROMPT = """You are a routing assistant. Classify the user's question
into exactly ONE of these three categories:

- VECTORSTORE: the question is about internal HR policy, company rules,
  leave, benefits, or anything that would be found in company documents.
- WEB_SEARCH: the question needs current/real-time information, recent
  news, or facts not likely to be in internal HR documents.
- DIRECT: the question is a greeting, small talk, or a general question
  that doesn't need any lookup at all (e.g. "hi", "thanks", "what can you do?").

Reply with exactly one word: VECTORSTORE, WEB_SEARCH, or DIRECT.

Question: {question}"""


def classify_question(question, llm):
    prompt = ROUTER_PROMPT.format(question=question)
    response = llm.invoke(prompt)
    route = response.content.strip().upper()

    # Safety net: if the LLM replies with something unexpected,
    # default to VECTORSTORE since that's our "safest" fallback.
    if route not in ("VECTORSTORE", "WEB_SEARCH", "DIRECT"):
        route = "VECTORSTORE"

    return route


def answer_direct(question, llm):
    """No retrieval at all — just answer normally."""
    response = llm.invoke(question)
    return response.content, []


def answer_from_web(question, llm, max_results=3):
    """Do a live web search, then answer using those results as context."""
    print("   Searching the web...")
    with DDGS() as ddgs:
        results = list(ddgs.text(question, max_results=max_results))

    if not results:
        return "I couldn't find anything on the web for that.", []

    context = "\n\n".join(
        f"{r['title']}: {r['body']}" for r in results
    )

    prompt = f"""Answer the question using the web search results below.

Web results:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    return response.content, results


def answer_from_vectorstore(question, retriever, llm):
    """Same flow as grader.py: retrieve, grade, then generate."""
    docs = retriever.invoke(question)
    print("   Grading chunks for relevance...")
    relevant_docs = grade_chunks(question, docs, llm)

    if not relevant_docs:
        return "I don't have enough policy context to answer this.", []

    answer, sources = generate_answer(question, relevant_docs, llm)
    return answer, sources


def route_and_answer(question, retriever, llm):
    route = classify_question(question, llm)
    print(f"   Router decision: {route}")

    if route == "VECTORSTORE":
        return answer_from_vectorstore(question, retriever, llm)
    elif route == "WEB_SEARCH":
        return answer_from_web(question, llm)
    else:  # DIRECT
        return answer_direct(question, llm)


def main():
    retriever = load_retreiver()
    llm = get_llm()

    print("Routed RAG bot ready! Type a question, or 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        answer, sources = route_and_answer(question, retriever, llm)

        print(f"\nBot: {answer}\n")
        if sources:
            print("--- Sources used ---")
            for i, src in enumerate(sources, 1):
                if hasattr(src, "page_content"):
                    snippet = src.page_content[:100].replace("\n", " ")
                else:
                    snippet = src.get("title", "web result")
                print(f"[{i}] {snippet}")
        print()


if __name__ == "__main__":
    main()