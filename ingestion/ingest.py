
import os
from dotenv import load_dotenv
import sys
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Load the HF_TOKEN from the .env file
load_dotenv()
from config import INDEX_NAME
def main():
        # Load the HR Policies dataset from Hugging Face
    dataset = load_dataset("strova-ai/hr-policies-qa-dataset", split="train")

    hr_documents = []
    total_bytes = 0
    print("Processing HR Policies dataset...")

    for idx, row in enumerate(dataset):
        # Extract multi-turn conversation or Q&A fields
        messages = row.get("messages", [])
        if messages:
            # Format conversation into readable context
            content = "\n".join([f"{m.get('role', '').capitalize()}: {m.get('content', '')}" for m in messages])
        else:
            content = str(row)
        total_bytes += sys.getsizeof(content)

        doc = Document(
            page_content=content,
            metadata={
                "source": "strova-ai-hr-policies",
                "doc_id": f"hr_qa_{idx}"
            }
        )
        hr_documents.append(doc)

    print(f"Loaded {len(hr_documents)} document records.")
    print(f"Current raw memory size: {total_bytes / 1024:.2f} KB")
    INDEX_NAME = "ragproject"  # Ensure this matches the index name used in rag.py
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)
    print("2. Splitting documents into small chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # roughly how many characters per chunk
        chunk_overlap=50,    # a little overlap so we don't cut off context mid-sentence
    )
    chunks = splitter.split_documents(hr_documents)
    print(f"   Created {len(chunks)} chunk(s).")

    print("3. Creating embeddings and saving to Pinecone vector store...")
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'} # Force it to run on CPU; change to 'cuda' if you have an Nvidia GPU setup
)
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,  # Name of the Pinecone index
        )
    

    print(f"\nDone! Your vector database is saved in ./{INDEX_NAME}/")
    print("Next step: run  python rag.py  and start asking questions.")


if __name__ == "__main__":
    main()
