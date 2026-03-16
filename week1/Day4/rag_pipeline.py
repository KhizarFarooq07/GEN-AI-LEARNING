"""
Day 4: Simple RAG Pipeline
- Data Source: pdf_1_rag_guide.txt
- Vector DB: ChromaDB
- LLM: llama-3.3-70b-versatile (via Groq)
"""

import os
import chromadb
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "Day3", "pdf_1_rag_guide.txt")
COLLECTION_NAME = "rag_guide"
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50
TOP_K = 5
LLM_MODEL = "llama-3.3-70b-versatile"


def load_and_chunk(filepath: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Load text file and split into overlapping chunks."""
    with open(filepath, "r") as f:
        text = f.read()

    # Remove decorative separator lines
    lines = text.splitlines()
    cleaned = "\n".join(line for line in lines if not line.startswith("==="))
    cleaned = cleaned.strip()

    chunks = []
    start = 0
    while start < len(cleaned):
        end = start + chunk_size
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    print(f"Loaded {len(chunks)} chunks from {filepath}")
    return chunks


def build_vector_db(chunks: list[str]) -> chromadb.Collection:
    """Create an in-memory ChromaDB collection and add chunks."""
    client = chromadb.Client()  # in-memory

    # Delete collection if it already exists (for re-runs)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)

    print(f"ChromaDB collection '{COLLECTION_NAME}' created with {collection.count()} documents")
    return collection


def retrieve(collection: chromadb.Collection, query: str, top_k: int = TOP_K) -> list[str]:
    """Retrieve the top-k most relevant chunks for a query."""
    results = collection.query(query_texts=[query], n_results=top_k)
    return results["documents"][0]


def ask(client: Groq, query: str, context_chunks: list[str]) -> str:
    """Send query + retrieved context to the LLM and return the answer."""
    context = "\n---\n".join(context_chunks)

    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}"
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content


def main():
    # 1. Load & chunk the document
    chunks = load_and_chunk(DATA_FILE)

    # 2. Build vector database
    collection = build_vector_db(chunks)

    # 3. Init Groq client
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # 4. Interactive Q&A loop
    print("\n--- RAG Pipeline Ready ---")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("Ask a question: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break

        # Retrieve relevant chunks
        relevant_chunks = retrieve(collection, query)
        print(f"\n[Retrieved {len(relevant_chunks)} chunks]")

        # Generate answer
        answer = ask(groq_client, query, relevant_chunks)
        print(f"\nAnswer:\n{answer}\n")


if __name__ == "__main__":
    main()
