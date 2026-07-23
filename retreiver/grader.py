from dotenv import load_dotenv
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from retreiver.rag import load_retreiver, get_llm, generate_answer, rewrite_query
load_dotenv()


def grade_chunk(question, chunk_text, llm):
    grading_prompt = f"""You are an expert retrieval relevance grader assessing whether a retrieved document chunk is useful for answering a user question.

Document chunk:
{chunk_text}

Question: {question}

### EVALUATION CRITERIA:
1. **Direct Answer:** Does the chunk explicitly state the answer or part of the answer?
2. **Context & Background:** Does the chunk provide necessary context, definitions, prerequisites, or partial information that helps formulate a complete answer?
3. **Intent Match:** Does the chunk address the underlying intent or topic of the question, even if the exact keywords differ?

### RULES:
- Grade as **YES** if the chunk contains *any* relevant facts, clues, partial answers, or background information related to the question.
- Grade as **NO** ONLY if the chunk is completely unrelated, off-topic, or provides zero useful context for the question.
- Do NOT require the chunk to answer the entire question on its own—partial relevance is sufficient for **YES**.

### INSTRUCTIONS:
First, briefly state your reasoning in 1 sentence. Then on a new line, provide your final grade as either "GRADE: YES" or "GRADE: NO".

Reasoning:"""

    response = llm.invoke(grading_prompt)
    text = response.content.strip().upper()
    return "GRADE: YES" in text or "YES" in text.splitlines()[-1]


def grade_chunks(question, docs, llm):
    relevant_docs = []
    for i, doc in enumerate(docs, 1):
        # Ensure grade_chunk returns a clean string or bool
        res = grade_chunk(question, doc.page_content, llm)
        
        # Handle string return ('YES' / 'NO')
        if isinstance(res, str):
            is_relevant = "YES" in res.strip().upper()
        else:
            is_relevant = bool(res)
            
        print(f"   [chunk {i}] {'RELEVANT' if is_relevant else 'discarded'}")
        if is_relevant:
            relevant_docs.append(doc)
            
    return relevant_docs


def answer_with_grading(question, retriever, llm):
    docs = retriever.invoke(question)
    print("Grading chunks for relevance...")
    relevant_docs = grade_chunks(question, docs, llm)

    if not relevant_docs:
        print("   No relevant chunks found — falling back to web search...")
        rewritten = rewrite_query(question, llm)
        print(f"   Rewritten query: {rewritten}")
        from routing.routed import answer_from_web
        return answer_from_web(rewritten, llm)

    return generate_answer(question, relevant_docs, llm)


def main():
    retriever = load_retreiver()
    llm = get_llm()

    print("Graded RAG bot ready! Type a question, or 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        answer, sources = answer_with_grading(question, retriever, llm)
        print(f"\nBot: {answer}\n")


if __name__ == "__main__":
    main()