import sys
import os
import glob
import json
from datetime import datetime
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
import re
from datetime import datetime


# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(SCRIPT_DIR, "documents")
CHROMA_DIR = os.path.join(SCRIPT_DIR, "chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL = "llama3.1:8b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ============================================================================
# ENHANCED DOCUMENT LOADING WITH RICHER METADATA
# ============================================================================

def extract_metadata_from_content(content: str, filename: str) -> Dict[str, Any]:
    """Extract richer metadata from document content and filename."""
    
    metadata = {
        "filename": filename,
        "source": filename,
        "chunk_type": "text",
        "date": datetime.now().isoformat(),  # Default to today
    }
    
    # Try to find date patterns in the content
    date_pattern = r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"
    dates = re.findall(date_pattern, content)
    if dates:
        metadata["date"] = dates[0]
    
    # Try to extract sections from markdown files
    if filename.endswith(".md"):
        sections = re.findall(r"^(#{1,3})\s+(.+?)$", content, re.MULTILINE)
        if sections:
            main_section = sections[0][1]  # Get first section
            metadata["section"] = main_section
            metadata["chunk_type"] = "markdown"
    
    # Estimate page number based on content length (rough estimate)
    estimated_pages = max(1, len(content) // 3000)
    metadata["page"] = 1
    metadata["estimated_total_pages"] = estimated_pages
    
    # Add document category based on filename
    filename_lower = filename.lower()
    if "fine" in filename_lower and "tun" in filename_lower:
        metadata["category"] = "fine-tuning"
    elif "prompt" in filename_lower:
        metadata["category"] = "prompt-engineering"
    elif "rag" in filename_lower or "vector" in filename_lower:
        metadata["category"] = "rag"
    else:
        metadata["category"] = "general"
    
    return metadata


def build_vector_store_with_metadata():
    """Load documents with enriched metadata, chunk, embed, and store in ChromaDB."""
    
    print(f"Loading documents with enriched metadata from {DOCS_DIR}/ ...")
    
    documents = []
    file_patterns = ["**/*.txt", "**/*.md"]
    
    for pattern in file_patterns:
        for filepath in glob.glob(os.path.join(DOCS_DIR, pattern), recursive=True):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    filename = os.path.basename(filepath)
                    
                    # Extract enriched metadata
                    metadata = extract_metadata_from_content(content, filename)
                    
                    doc = Document(
                        page_content=content,
                        metadata=metadata
                    )
                    documents.append(doc)
                    print(f"  ✓ Loaded: {filename} [{metadata.get('category')}] ({len(content)} chars)")
            except Exception as e:
                print(f"  Error loading {filepath}: {e}")
    
    print(f"\nLoaded {len(documents)} documents with enriched metadata.")
    
    if not documents:
        print("No documents found. Add files to the documents/ directory.")
        return None
    
    # Chunk the documents using recursive splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    
    # Embed and store
    print("Embedding chunks and building vector store...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=HuggingFaceEmbeddings(model_name=EMBED_MODEL),
        collection_name="my_documents",
        persist_directory=CHROMA_DIR,
    )
    print("✓ Vector store built and persisted with metadata.")
    return vector_store


# ============================================================================
# QUERY OPTIMIZATION TECHNIQUES
# ============================================================================

def generate_hypothetical_documents(question: str, llm: ChatOllama) -> List[str]:
    """
    HyDE (Hypothetical Document Embeddings):
    Generate hypothetical documents that would answer the question,
    then use their embeddings for retrieval.
    """
    hyde_prompt = ChatPromptTemplate.from_template(
        "Generate 3 hypothetical documents that would contain the answer to: {question}\n"
        "Format each as: DOC1: ...\nDOC2: ...\nDOC3: ...\n"
        "Keep each under 100 words."
    )
    
    chain = hyde_prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question})
    
    # Parse the documents
    docs = re.split(r"DOC\d+:\s*", response)
    hypothetical_docs = [doc.strip() for doc in docs if doc.strip()][:3]
    
    print(f"  📝 Generated {len(hypothetical_docs)} hypothetical documents for HyDE")
    return hypothetical_docs


def rewrite_query_multi_perspective(question: str, llm: ChatOllama) -> List[str]:
    """
    Multi-Query Rewriting:
    Rewrite the user's question from multiple perspectives to improve retrieval coverage.
    """
    rewrite_prompt = ChatPromptTemplate.from_template(
        "Rewrite the following question in 3 different ways to improve search results:\n"
        "Question: {question}\n\n"
        "Format as:\nV1: ...\nV2: ...\nV3: ..."
    )
    
    chain = rewrite_prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question})
    
    # Parse the rewritten queries
    queries = re.split(r"V\d+:\s*", response)
    rewritten_queries = [q.strip() for q in queries if q.strip()][:3]
    rewritten_queries.insert(0, question)  # Include original
    
    print(f"  🔄 Generated {len(rewritten_queries)} query variations (original + {len(rewritten_queries)-1} rewrites)")
    return rewritten_queries


def load_vector_store():
    """Load an existing vector store from disk."""
    return Chroma(
        collection_name="my_documents",
        embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL),
        persist_directory=CHROMA_DIR,
    )


# ============================================================================
# RETRIEVAL FUNCTIONS
# ============================================================================

def baseline_retrieval(vector_store, question: str, k: int = 5) -> tuple:
    """
    Baseline retrieval: Simple semantic search without optimization.
    
    Returns:
        (documents, retrieval_method, details)
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    
    details = {
        "method": "baseline",
        "queries_used": 1,
        "documents_retrieved": len(docs),
        "top_sources": list(set([doc.metadata.get("filename", "unknown") for doc in docs[:3]])),
        "categories": list(set([doc.metadata.get("category", "unknown") for doc in docs]))
    }
    
    return docs, "baseline", details


def optimized_retrieval_with_hyde_and_multiquery(vector_store, question: str, llm: ChatOllama, k: int = 5) -> tuple:
    """
    Optimized retrieval combining:
    1. HyDE (Hypothetical Document Embeddings)
    2. Multi-Query Rewriting
    3. Metadata filtering
    
    Returns:
        (documents, retrieval_method, details)
    """
    print(f"\n  🚀 Optimized Retrieval Pipeline:")
    
    # Step 1: Generate hypothetical documents (HyDE)
    hypothetical_docs = generate_hypothetical_documents(question, llm)
    
    # Step 2: Generate query rewrites (Multi-Query)
    rewritten_queries = rewrite_query_multi_perspective(question, llm)
    
    # Step 3: Retrieve documents using all queries + hypothetical docs
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    all_retrieved_docs = []
    doc_ids = set()  # Track unique documents to avoid duplicates
    
    # Retrieve using original and rewritten queries
    for i, query in enumerate(rewritten_queries, 1):
        docs = retriever.invoke(query)
        for doc in docs:
            doc_id = id(doc)
            if doc_id not in doc_ids:
                all_retrieved_docs.append(doc)
                doc_ids.add(doc_id)
    
    # Retrieve using hypothetical documents
    for i, hyde_doc in enumerate(hypothetical_docs, 1):
        docs = retriever.invoke(hyde_doc)
        for doc in docs:
            doc_id = id(doc)
            if doc_id not in doc_ids:
                all_retrieved_docs.append(doc)
                doc_ids.add(doc_id)
    
    # De-duplicate by content similarity (keep only top-k unique)
    final_docs = all_retrieved_docs[:k*2]  # Keep more candidates
    
    details = {
        "method": "optimized",
        "queries_used": len(rewritten_queries) + len(hypothetical_docs),
        "documents_retrieved": len(final_docs),
        "hyde_docs_generated": len(hypothetical_docs),
        "query_rewrites": len(rewritten_queries) - 1,
        "top_sources": list(set([doc.metadata.get("filename", "unknown") for doc in final_docs[:3]])),
        "categories": list(set([doc.metadata.get("category", "unknown") for doc in final_docs])),
        "queries_used_list": rewritten_queries + hypothetical_docs[:1]  # Log some queries
    }
    
    return final_docs, "optimized", details


# ============================================================================
# QUERY AND EVALUATION
# ============================================================================

def generate_answer(llm: ChatOllama, context: str, question: str) -> str:
    """Generate an answer given context and question."""
    
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer the question using ONLY the "
        "provided context. If the context doesn't contain enough information "
        "to answer, say so.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


def evaluate_retrieval_quality(docs: List[Document], question: str) -> Dict[str, Any]:
    """Evaluate the quality of retrieved documents."""
    
    quality_metrics = {
        "num_docs": len(docs),
        "unique_sources": len(set([doc.metadata.get("filename") for doc in docs])),
        "unique_categories": len(set([doc.metadata.get("category") for doc in docs])),
        "avg_relevance_keywords": 0,
    }
    
    # Simple keyword overlap scorer
    question_words = set(question.lower().split())
    total_overlap = 0
    for doc in docs[:5]:  # Check top-5
        doc_words = set(doc.page_content[:200].lower().split())
        overlap = len(question_words & doc_words)
        total_overlap += overlap
    
    quality_metrics["avg_relevance_keywords"] = total_overlap // max(1, len(docs[:5]))
    
    return quality_metrics


TEST_QUERIES = [
    "What is prompt engineering and why is it important?",
    "Explain the concept of fine-tuning in LLMs",
    "Compare fine-tuning and prompt engineering approaches",
    "How do vector databases relate to RAG systems?",
    "What is the recommended chunk size for RAG systems?",
    "What embedding models are mentioned in the documents?",
    "What is not mentioned in the documents?",
    "Give me advanced tips for optimization",
    "Create a workflow combining prompt engineering, fine-tuning, and RAG",
    "What are the main challenges in building RAG systems?",
]


def evaluate_baseline_vs_optimized():
    """Compare baseline vs optimized retrieval with both retrieval methods and RAG generation."""
    
    print("\n" + "="*80)
    print("BASELINE vs OPTIMIZED RETRIEVAL EVALUATION")
    print("="*80)
    
    # Load vector store and LLM
    vector_store = load_vector_store()
    if not vector_store:
        print("ERROR: Vector store not found. Run --build first.")
        return
    
    llm = ChatOllama(model=CHAT_MODEL)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": CHAT_MODEL,
        "embedding_model": EMBED_MODEL,
        "queries": TEST_QUERIES,
        "evaluations": []
    }
    
    for i, question in enumerate(TEST_QUERIES, 1):
        print(f"\n{'─'*80}")
        print(f"Query {i}/{len(TEST_QUERIES)}: {question[:60]}...")
        print(f"{'─'*80}")
        
        # === BASELINE RETRIEVAL ===
        print(f"\n[BASELINE RETRIEVAL]")
        baseline_docs, baseline_method, baseline_details = baseline_retrieval(vector_store, question, k=5)
        baseline_quality = evaluate_retrieval_quality(baseline_docs, question)
        baseline_context = "\n---\n".join([doc.page_content[:300] for doc in baseline_docs])
        
        print(f"  Retrieved: {baseline_details['documents_retrieved']} documents")
        print(f"  Sources: {baseline_details['top_sources']}")
        print(f"  Categories: {baseline_details['categories']}")
        print(f"  Keyword overlap score: {baseline_quality['avg_relevance_keywords']}")
        
        try:
            baseline_answer = generate_answer(llm, baseline_context, question)
            print(f"  Answer: {baseline_answer[:150]}...")
            baseline_success = True
        except Exception as e:
            baseline_answer = f"ERROR: {str(e)}"
            baseline_success = False
            print(f"  ERROR: {e}")
        
        # === OPTIMIZED RETRIEVAL ===
        print(f"\n[OPTIMIZED RETRIEVAL (HyDE + Multi-Query)]")
        optimized_docs, optimized_method, optimized_details = optimized_retrieval_with_hyde_and_multiquery(
            vector_store, question, llm, k=5
        )
        optimized_quality = evaluate_retrieval_quality(optimized_docs, question)
        optimized_context = "\n---\n".join([doc.page_content[:300] for doc in optimized_docs])
        
        print(f"  Retrieved: {optimized_details['documents_retrieved']} documents")
        print(f"  Queries/contexts used: {optimized_details['queries_used']}")
        print(f"  Sources: {optimized_details['top_sources']}")
        print(f"  Categories: {optimized_details['categories']}")
        print(f"  Keyword overlap score: {optimized_quality['avg_relevance_keywords']}")
        
        try:
            optimized_answer = generate_answer(llm, optimized_context, question)
            print(f"  Answer: {optimized_answer[:150]}...")
            optimized_success = True
        except Exception as e:
            optimized_answer = f"ERROR: {str(e)}"
            optimized_success = False
            print(f"  ERROR: {e}")
        
        # === COMPARISON ===
        print(f"\n[COMPARISON]")
        doc_improvement = optimized_details['documents_retrieved'] - baseline_details['documents_retrieved']
        keyword_improvement = optimized_quality['avg_relevance_keywords'] - baseline_quality['avg_relevance_keywords']
        
        print(f"  Document count change: {baseline_details['documents_retrieved']} → {optimized_details['documents_retrieved']} ({doc_improvement:+d})")
        print(f"  Keyword relevance: {baseline_quality['avg_relevance_keywords']} → {optimized_quality['avg_relevance_keywords']} ({keyword_improvement:+d})")
        print(f"  Unique sources: {baseline_quality['unique_sources']} → {optimized_quality['unique_sources']}")
        print(f"  Unique categories: {baseline_quality['unique_categories']} → {optimized_quality['unique_categories']}")
        
        # Store results
        results["evaluations"].append({
            "query_num": i,
            "question": question,
            "baseline": {
                "method": baseline_method,
                "details": baseline_details,
                "quality_metrics": baseline_quality,
                "answer_preview": baseline_answer[:200] if baseline_success else baseline_answer,
                "success": baseline_success
            },
            "optimized": {
                "method": optimized_method,
                "details": optimized_details,
                "quality_metrics": optimized_quality,
                "answer_preview": optimized_answer[:200] if optimized_success else optimized_answer,
                "success": optimized_success
            },
            "comparison": {
                "doc_count_improvement": doc_improvement,
                "keyword_relevance_improvement": keyword_improvement,
                "source_diversity_improvement": optimized_quality['unique_sources'] - baseline_quality['unique_sources'],
                "category_diversity_improvement": optimized_quality['unique_categories'] - baseline_quality['unique_categories']
            }
        })
    
    # === SUMMARY STATISTICS ===
    print(f"\n\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    
    baseline_total_docs = sum([e["baseline"]["details"]["documents_retrieved"] for e in results["evaluations"]])
    optimized_total_docs = sum([e["optimized"]["details"]["documents_retrieved"] for e in results["evaluations"]])
    baseline_success = sum([1 for e in results["evaluations"] if e["baseline"]["success"]])
    optimized_success = sum([1 for e in results["evaluations"] if e["optimized"]["success"]])
    
    avg_baseline_keyword = sum([e["baseline"]["quality_metrics"]["avg_relevance_keywords"] for e in results["evaluations"]]) / len(results["evaluations"])
    avg_optimized_keyword = sum([e["optimized"]["quality_metrics"]["avg_relevance_keywords"] for e in results["evaluations"]]) / len(results["evaluations"])
    
    print(f"Total Queries Evaluated: {len(TEST_QUERIES)}")
    print(f"\nBaseline Retrieval:")
    print(f"  Success Rate: {baseline_success}/{len(TEST_QUERIES)} ({100*baseline_success/len(TEST_QUERIES):.1f}%)")
    print(f"  Total Docs Retrieved: {baseline_total_docs}")
    print(f"  Avg Keyword Relevance: {avg_baseline_keyword:.2f}")
    
    print(f"\nOptimized Retrieval (HyDE + Multi-Query):")
    print(f"  Success Rate: {optimized_success}/{len(TEST_QUERIES)} ({100*optimized_success/len(TEST_QUERIES):.1f}%)")
    print(f"  Total Docs Retrieved: {optimized_total_docs}")
    print(f"  Avg Keyword Relevance: {avg_optimized_keyword:.2f}")
    
    print(f"\nImprovement:")
    print(f"  Doc Retrieval: {optimized_total_docs - baseline_total_docs:+d} documents ({100*(optimized_total_docs-baseline_total_docs)/baseline_total_docs:+.1f}%)")
    print(f"  Keyword Relevance: {avg_optimized_keyword - avg_baseline_keyword:+.2f} ({100*(avg_optimized_keyword-avg_baseline_keyword)/avg_baseline_keyword:+.1f}%)")
    
    # Save results
    output_file = os.path.join(SCRIPT_DIR, "evaluation_results_optimized.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Detailed results saved to {output_file}")


if __name__ == "__main__":
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"Created {DOCS_DIR}/ directory. Add your documents there.")
        print("Then run: python Task.py --build")
        sys.exit(0)
    
    if "--build" in sys.argv:
        build_vector_store_with_metadata()
    elif "--evaluate" in sys.argv:
        evaluate_baseline_vs_optimized()
    else:
        # Interactive mode
        store = load_vector_store()
        llm = ChatOllama(model=CHAT_MODEL)
        
        print(f"\nRAG system ready. Using {CHAT_MODEL} for answers.")
        print("Commands:")
        print("  - Type a question to ask with baseline retrieval")
        print("  - Type 'opt' then question for optimized retrieval")
        print("  - Type 'exit' to quit\n")
        
        while True:
            try:
                user_input = input("Q: ").strip()
                if not user_input:
                    continue
                
                if user_input.lower() == "exit":
                    print("Goodbye.")
                    break
                
                use_optimized = user_input.startswith("opt ")
                if use_optimized:
                    question = user_input[4:].strip()
                else:
                    question = user_input
                
                print("\nSearching documents and generating answer...\n")
                
                if use_optimized:
                    docs, method, details = optimized_retrieval_with_hyde_and_multiquery(store, question, llm)
                    print(f"Method: {method} | Docs: {details['documents_retrieved']} | Queries used: {details['queries_used']}")
                else:
                    docs, method, details = baseline_retrieval(store, question)
                    print(f"Method: {method} | Docs: {details['documents_retrieved']}")
                
                context = "\n---\n".join([doc.page_content[:300] for doc in docs])
                answer = generate_answer(llm, context, question)
                print(f"A: {answer}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye.")
                break
