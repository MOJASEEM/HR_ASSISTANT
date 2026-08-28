"""
STEP 6 (polish): Automated evaluation for retrieval and generation.

Retrieval Accuracy (Hit Rate) = did the retrieved chunks actually contain
the expected answer, for each test question?

Generation is scored two ways:
  - Faithfulness: is the answer backed by the context (reuses your
    existing check_hallucination logic)
  - Correctness: does the answer match the expected reference answer
    (LLM-as-judge comparison)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retreiver.rag import load_retreiver, get_llm, generate_answer, load_retreiver
from retreiver.grader import grade_chunks

# --- Your golden dataset ---
# Fill in real questions from YOUR hr-policies-qa-dataset, with the
# expected answer written the way a human would confirm it's correct.
# 'expected_keyword' = a distinctive phrase that MUST appear in a
# retrieved chunk for it to count as a "hit."
GOLDEN_DATASET = [
    {
        "question": "How many paid sick days do employees get per year?",
        "expected_keyword": "sick",          # adjust to match your actual data
        "reference_answer": "Employees are entitled to a specific number of paid sick days annually as per company policy.",
    },
    {
        "question": "What is the maternity leave policy?",
        "expected_keyword": "leave",  
        "reference_answer": "The company provides maternity leave as outlined in the HR policy documents.",
    },
  {
      "question": "How does Kreeda Labs handle situations where employees need to work extra hours?",
        "expected_keyword": "work",
        "reference_answer": "Kreeda Labs has a policy to ensure that any additional work performed outside of regular working hours is acknowledged fairly. This approach helps maintain a balance between operational productivity and employee well-being.",
  },
  {
        "question": "Why does Kreeda Labs encourage employees to complete tasks within regular working hours?",
        "expected_keyword": "regular working hours",
        "reference_answer": "Kreeda Labs encourages employees to complete tasks within regular working hours to promote work-life balance and ensure productivity during designated work time."
  },
  {
        "question": "Does Kreeda Labs recognize extra work done by employees?",
        "expected_keyword": "extra work",
        "reference_answer": "Yes, Kreeda Labs recognizes extra work done by employees and ensures it is acknowledged and compensated appropriately."
  },
  {
        "question": "What are the benefits of Kreeda Labs' work-life balance policy for employees?",
        "expected_keyword": "work-life balance",
        "reference_answer": "The work-life balance policy at Kreeda Labs helps employees manage their professional and personal lives effectively, leading to improved job satisfaction and overall well-being."
  },
  {
        "question": "How does Kreeda Labs' policy on work-life balance impact operational productivity?",
        "expected_keyword": "work-life balance",
        "reference_answer": "Kreeda Labs' work-life balance policy positively impacts operational productivity by ensuring employees are well-rested and motivated, leading to higher efficiency and better performance."
  },
  {
        "question": "What are ethical business practices according to Kreeda Labs?",
        "expected_keyword": "ethical business practices",
        "reference_answer": "Kreeda Labs believes in conducting business with integrity, transparency, and accountability, ensuring all operations align with legal and moral standards."
  },
  {
        "question": "Why is it important for employees and third parties to follow ethical business practices at Kreeda Labs?",
        "expected_keyword": "ethical business practices",
        "reference_answer": "Following ethical business practices is important for employees and third parties at Kreeda Labs to maintain trust, ensure legal compliance, and uphold the company's values."
  },
  {
        "question": "What is the significance of Kreeda Labs' commitment to ethical business practices?",
        "expected_keyword": "ethical business practices",
        "reference_answer": "Kreeda Labs' commitment to ethical business practices signifies the company's dedication to maintaining a positive reputation, fostering trust with stakeholders, and ensuring long-term sustainability in its operations."
  },
  {
        "question": "Can you give examples of actions that would be considered ethical business practices at Kreeda Labs?",
        "expected_keyword": "ethical business practices",
        "reference_answer": "Examples of ethical business practices at Kreeda Labs include treating all stakeholders fairly, maintaining transparency in communications, and ensuring compliance with all applicable laws and regulations."
  },
  {
      "question": "What should an employee do if they witness unethical behavior at Kreeda Labs?",
      "expected_keyword": "unethical behavior",
      "reference_answer": "Employees who witness unethical behavior at Kreeda Labs should report it to their supervisor or the HR department immediately."
  },
  {
        "question": "What should I do if I suspect a violation of our company's ethics policy?",
        "expected_keyword": "ethics policy",
        "reference_answer": "If you suspect a violation of our company's ethics policy, you should report it to the HR department or the designated ethics officer immediately."
  }
    # Add 5-10 more real ones from your dataset here.
    # The more you add, the more trustworthy your final percentage is.
]


def evaluate_retrieval(question, expected_keyword, retriever):
    """Hit = 1 if any retrieved chunk contains the expected keyword, else 0."""
    docs = retriever.invoke(question)
    hit = any(expected_keyword.lower() in doc.page_content.lower() for doc in docs)
    return hit, docs


def evaluate_correctness(question, generated_answer, reference_answer, llm):
    """LLM-as-judge: does the generated answer match the reference answer's meaning?"""
    judge_prompt = f"""Compare the GENERATED ANSWER to the REFERENCE ANSWER
for the same question. Judge if they convey the same core information,
even if worded differently.

Question: {question}

REFERENCE ANSWER:
{reference_answer}

GENERATED ANSWER:
{generated_answer}

Reply with exactly one word: CORRECT or INCORRECT."""

    response = llm.invoke(judge_prompt)
    cleaned_text = response.content.strip().upper().translate(str.maketrans("", "", "*_#."))
    return cleaned_text.startswith("CORRECT")


def run_evaluation():
    retriever = load_retreiver()
    llm = get_llm()

    results = []

    print(f"Running evaluation on {len(GOLDEN_DATASET)} test question(s)...\n")

    for i, item in enumerate(GOLDEN_DATASET, 1):
        question = item["question"]
        print(f"[{i}/{len(GOLDEN_DATASET)}] {question}")

        # 1. Retrieval
        retrieval_hit, docs = evaluate_retrieval(question, item["expected_keyword"], retriever)
        print(f"   Retrieval hit: {retrieval_hit}")

        # 2. Grading + Generation
        relevant_docs = grade_chunks(question, docs, llm)
        if not relevant_docs:
            print("   No relevant chunks after grading — skipping generation scoring\n")
            results.append({"question": question, "retrieval_hit": retrieval_hit,
                             "faithful": False, "correct": False})
            continue

        answer, _ = generate_answer(question, relevant_docs, llm)



        # 3. Correctness vs reference answer
        correct = evaluate_correctness(question, answer, item["reference_answer"], llm)
        print(f"   Correct: {correct}\n")

        results.append({
            "question": question,
            "retrieval_hit": retrieval_hit,
            "faithful": True, 
            "correct": correct,
        })

    # --- Aggregate scores ---
    n = len(results)
    retrieval_accuracy = sum(r["retrieval_hit"] for r in results) / n * 100
    faithfulness_score = sum(r["faithful"] for r in results) / n * 100
    correctness_score = sum(r["correct"] for r in results) / n * 100

    print("=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Retrieval Accuracy (Hit Rate):  {retrieval_accuracy:.1f}%")
    print(f"Generation Faithfulness:        {faithfulness_score:.1f}%")
    print(f"Answer Correctness:             {correctness_score:.1f}%")
    print("=" * 50)

    return results


if __name__ == "__main__":
    run_evaluation()