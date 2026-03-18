import sys
import os
import glob
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document


# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(SCRIPT_DIR, "documents")          # Put your PDFs and files here
CHROMA_DIR = os.path.join(SCRIPT_DIR, "chroma_db")        # Where vectors are persisted
EMBED_MODEL = "all-MiniLM-L6-v2"  # HuggingFace embedding model (no Ollama needed)
CHAT_MODEL = "llama3.1:8b"        # Chat model (requires Ollama running)
CHUNK_SIZE = 1000                 # Characters per chunk


def build_vector_store():
    """Load documents, chunk, embed, and store in ChromaDB."""

    print(f"Loading documents from {DOCS_DIR}/ ...")

    documents = []
    
    # Load all text and markdown files directly
    
    # Supported file patterns
    file_patterns = ["**/*.txt", "**/*.md"]
    
    for pattern in file_patterns:
        for filepath in glob.glob(os.path.join(DOCS_DIR, pattern), recursive=True):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Create a document object with metadata
                    doc = Document(
                        page_content=content,
                        metadata={"source": filepath, "filename": os.path.basename(filepath)}
                    )
                    documents.append(doc)
                    print(f"  ✓ Loaded: {os.path.basename(filepath)} ({len(content)} chars)")
            except Exception as e:
                print(f"  Error loading {filepath}: {e}")

    print(f"\nLoaded {len(documents)} documents.")

    if not documents:
        print("No documents found. Add files to the documents/ directory.")
        return None

    # Chunk the documents - simple fixed-size chunking
    chunks = []
    for doc in documents:
        content = doc.page_content
        for i in range(0, len(content), CHUNK_SIZE):
            chunk_text = content[i:i + CHUNK_SIZE]
            chunk_doc = Document(
                page_content=chunk_text,
                metadata=doc.metadata
            )
            chunks.append(chunk_doc)
    print(f"Split into {len(chunks)} chunks.")

    # Embed and store
    print("Embedding chunks and building vector store...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=HuggingFaceEmbeddings(model_name=EMBED_MODEL),
        collection_name="my_documents",
        persist_directory=CHROMA_DIR,
    )
    print("✓ Vector store built and persisted.")
    return vector_store


def load_vector_store():
    """Load an existing vector store from disk."""
    return Chroma(
        collection_name="my_documents",
        embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL),
        persist_directory=CHROMA_DIR,
    )


def query_documents(vector_store, question, use_full_document=False):
    """Run a RAG query against the vector store.
    
    Args:
        vector_store: ChromaDB vector store
        question: User question
        use_full_document: If True, pass all documents; if False, use retrieval (top-5)
    """

    # Set up the LLM
    llm = ChatOllama(model=CHAT_MODEL)

    # The prompt template
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer the question using ONLY the "
        "provided context. If the context doesn't contain enough information "
        "to answer, say so.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )

    if use_full_document:
        # Mode 1: Get ALL documents/chunks (no retrieval filtering)
        print("[Mode: FULL DOCUMENT]")
        all_docs = vector_store._collection.get(include=['documents'])
        full_context = "\n---\n".join(all_docs['documents'])
        print("Pfull_context:", full_context[:500], "...")  # Print the first 500 chars of context for verification
        # Pass full context directly without retriever
        response = prompt.format(context=full_context, question=question)
        chain = llm | StrOutputParser()
        return chain.invoke(response)
    else:
        # Mode 2: Retrieve top-5 most relevant chunks (default RAG)
        print("[Mode: RETRIEVAL-AUGMENTED]")
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 5}
        )
        
        # Build the RAG chain with retriever
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain.invoke(question)


if __name__ == "__main__":
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"Created {DOCS_DIR}/ directory. Add your documents there.")
        print("Then run: python rag.py --build")
        sys.exit(0)

    if "--build" in sys.argv:
        build_vector_store()
    else:
        # Load existing store and enter query loop
        store = load_vector_store()
        
        # Determine mode based on command-line flag
        use_full_doc = "--full" in sys.argv
        mode_name = "FULL DOCUMENT" if use_full_doc else "RETRIEVAL (Top-5)"
        
        print(f"\nRAG system ready. Using {CHAT_MODEL} for answers.")
        print(f"Mode: {mode_name}")
        print("Commands:")
        print("  - Type a question to ask")
        print("  - Type 'mode' to toggle between retrieval and full document")
        print("  - Type 'exit' or Ctrl+C to quit\n")

        while True:
            try:
                question = input("Q: ").strip()
                if not question:
                    continue
                
                # Handle special commands
                if question.lower() == "exit":
                    print("Goodbye.")
                    break
                elif question.lower() == "mode":
                    use_full_doc = not use_full_doc
                    mode_name = "FULL DOCUMENT" if use_full_doc else "RETRIEVAL (Top-5)"
                    print(f"Switched to: {mode_name}\n")
                    continue
                
                print("\nSearching documents and generating answer...\n")
                answer = query_documents(store, question, use_full_document=use_full_doc)
                print(f"A: {answer}\n")
            except KeyboardInterrupt:
                print("\nGoodbye.")
                break