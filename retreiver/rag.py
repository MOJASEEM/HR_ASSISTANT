import warnings
import sys
from pathlib import Path
from retreiver.config import INDEX_NAME
# This suppresses the annoying warning messages from polluting your screen
warnings.filterwarnings("ignore", category=UserWarning)
import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from pinecone import Pinecone

load_dotenv()


PROMPT_TEMPLATE = """You are an expert HR Policy and Corporate Compliance Assistant.
Use the following context snippets retrieved from the policy documents to answer the user's question.

CRITICAL INSTRUCTIONS:
- Do not just output or quote the retrieved raw text verbatim.
- Provide a clear, detailed, and comprehensive explanation based on the context.
- Structure your answer with clear headings or bullet points where appropriate.
- If the details are not available in the context, clearly state: "I don't have enough policy context to answer this."

Context:
{context}
Question: {question}"""


def load_retreiver():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} 
    )
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


"""def answer_question(question, retriever, llm):
    # 1. Retrieve relevant chunks
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # 2. Build the prompt as a raw string (Fixes the first crash)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted_prompt = prompt.format(context=context, question=question)

    response = llm.invoke(formatted_prompt)
    response_text = response.content

    return response_text, docs"""

def generate_answer(question, docs, llm):
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted_prompt = prompt.format(context=context, question=question)

    response = llm.invoke(formatted_prompt)
    return response.content, docs

def get_llm():
    return ChatGroq(
        model_name=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
        

    )
def rewrite_query(question, llm):
    """Turn an internal-docs-style question into better web search terms."""
    rewrite_prompt = f"""The following question was asked about internal HR
policy documents, but no relevant information was found there. Rewrite it
as a short, general web search query that would find the answer online.

Original question: {question}

Rewritten search query (reply with ONLY the query, nothing else):"""

    response = llm.invoke(rewrite_prompt)
    return response.content.strip()

def main():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("Error: PINECONE_API_KEY is missing from your .env configuration file.")
        return
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    active_indexes = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in active_indexes:
        print(f"Error: Could not locate cloud index '{INDEX_NAME}' on Pinecone.")
        print("Please execute 'python ingest.py' first to initialize your vectors.")
        return

    print(f"Connecting to Pinecone Cloud Index: '{INDEX_NAME}'...")
    retriever = load_retriever()
    llm = get_llm()   
    print("RAG bot ready! Type a question, or 'quit' to exit.\n")

    question = input("You: ").strip()


if __name__ == "__main__":
    main()