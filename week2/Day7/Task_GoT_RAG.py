"""
Game of Thrones RAG System with Query Optimization
Uses Tuana/game-of-thrones dataset from Hugging Face
Implements: HyDE + Multi-Query Rewriting with Baseline Comparison
"""

import sys
import os
import json
import re
from datetime import datetime
from typing import List, Dict, Any
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document


# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(SCRIPT_DIR, "chroma_db_got")
EMBED_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL = "llama3.1:8b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


print("Loading Game of Thrones dataset from Hugging Face...")
# ============================================================================
# LOAD DATASET FROM HUGGING FACE
# ============================================================================

def load_got_dataset():
    """Load Game of Thrones dataset from Hugging Face."""
    try:
        dataset = load_dataset("Tuana/game-of-thrones", split="train")
        print(f"✓ Loaded Game of Thrones dataset: {len(dataset)} records")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def extract_rich_metadata(record: Dict[str, Any], record_id: int) -> Dict[str, Any]:
    """
    Extract enriched metadata from Game of Thrones dataset record.
    
    Expected fields in dataset:
    - content: Main text content
    - meta: Dictionary with metadata (name, etc.)
    - content_type: Type of content
    """
    
    metadata = {
        "source": "game-of-thrones",
        "record_id": record_id,
        "date": datetime.now().isoformat(),
    }
    
    # Extract name/title from meta dict
    if "meta" in record and isinstance(record["meta"], dict):
        if "name" in record["meta"]:
            metadata["name"] = str(record["meta"]["name"])
            metadata["filename"] = str(record["meta"]["name"])
        else:
            metadata["name"] = f"record_{record_id}"
            metadata["filename"] = f"record_{record_id}"
    else:
        metadata["name"] = f"record_{record_id}"
        metadata["filename"] = f"record_{record_id}"
    
    # Extract content type
    if "content_type" in record:
        metadata["content_type"] = record["content_type"]
    
    # Auto-categorize based on content
    content = ""
    if "content" in record:
        content = str(record["content"]).lower()
    
    if any(word in content for word in ["battle", "war", "fight", "army", "sword"]):
        metadata["category"] = "battles"
    elif any(word in content for word in ["king", "throne", "lord", "rule", "power"]):
        metadata["category"] = "politics"
    elif any(word in content for word in ["love", "marriage", "family", "house"]):
        metadata["category"] = "family"
    elif any(word in content for word in ["dragon", "magic", "magic", "white walker"]):
        metadata["category"] = "magic"
    elif any(word in content for word in ["winter", "wall", "north", "south", "king's landing"]):
        metadata["category"] = "locations"
    else:
        metadata["category"] = "general"
    
    # Extract main location mentions
    locations = ["north", "king's landing", "riverlands", "reach", "dorne", "vale", "stormlands", "westerlands", "crownlands"]
    found_locations = [loc for loc in locations if loc in content]
    if found_locations:
        metadata["locations"] = found_locations[:3]
    
    # Estimate reading length
    text_length = len(content)
    metadata["text_length"] = text_length
    metadata["chunk_type"] = "narrative" if text_length > 500 else "dialogue"
    
    return metadata


def build_got_vector_store():
    """Load Game of Thrones dataset, process with metadata, and build vector store."""
    
    print(f"Building vector store from Game of Thrones dataset...")
    
    # Load dataset
    dataset = load_got_dataset()
    if not dataset:
        print("Failed to load dataset")
        return None
    
    # Process into Document objects with metadata
    documents = []
    for idx, record in enumerate(dataset):
        try:
            # Extract text content
            if "content" in record and record["content"]:
                content = str(record["content"])
            else:
                continue
            
            # Skip very short entries
            if len(content) < 100:
                continue
            
            # Extract metadata
            metadata = extract_rich_metadata(record, idx)
            
            # Create document
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
            
            if (idx + 1) % 50 == 0:
                print(f"  ✓ Processed {idx + 1} records...")
                
        except Exception as e:
            print(f"  Error processing record {idx}: {e}")
            continue
    
    print(f"\nLoaded {len(documents)} documents from Game of Thrones dataset")
    
    if not documents:
        print("No documents found in dataset")
        return None
    
    # Chunk the documents
    print("Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    
    # Build and persist vector store
    print("Embedding and building vector store...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=HuggingFaceEmbeddings(model_name=EMBED_MODEL),
        collection_name="game_of_thrones",
        persist_directory=CHROMA_DIR,
    )
    print("✓ Vector store built and persisted")
    return vector_store


# ============================================================================
# QUERY OPTIMIZATION TECHNIQUES
# ============================================================================

def generate_hypothetical_documents(question: str, llm: ChatOllama) -> List[str]:
    """
    HyDE (Hypothetical Document Embeddings):
    Generate hypothetical Game of Thrones passages that would answer the question.
    """
    hyde_prompt = ChatPromptTemplate.from_template(
        "You are familiar with Game of Thrones. Generate 3 hypothetical passages from "
        "the Game of Thrones universe that would contain the answer to: {question}\n"
        "Format as:\nDOC1: [passage]\nDOC2: [passage]\nDOC3: [passage]\n"
        "Keep each under 100 words and in the style of Game of Thrones."
    )
    
    chain = hyde_prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question})
    
    # Parse documents
    docs = re.split(r"DOC\d+:\s*", response)
    hypothetical_docs = [doc.strip() for doc in docs if doc.strip()][:3]
    
    print(f"  📝 Generated {len(hypothetical_docs)} hypothetical Game of Thrones passages for HyDE")
    return hypothetical_docs


def rewrite_query_multi_perspective(question: str, llm: ChatOllama) -> List[str]:
    """
    Multi-Query Rewriting:
    Rewrite the user's question about Game of Thrones from multiple angles.
    """
    rewrite_prompt = ChatPromptTemplate.from_template(
        "Rewrite the following Game of Thrones question in 3 different ways "
        "to improve search coverage:\n"
        "Question: {question}\n\n"
        "Format as:\nV1: [rephrasing]\nV2: [rephrasing]\nV3: [rephrasing]"
    )
    
    chain = rewrite_prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question})
    
    # Parse queries
    queries = re.split(r"V\d+:\s*", response)
    rewritten_queries = [q.strip() for q in queries if q.strip()][:3]
    rewritten_queries.insert(0, question)  # Include original
    
    print(f"  🔄 Generated {len(rewritten_queries)} query variations")
    return rewritten_queries


def load_vector_store():
    """Load existing vector store from disk."""
    return Chroma(
        collection_name="game_of_thrones",
        embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL),
        persist_directory=CHROMA_DIR,
    )


# ============================================================================
# RETRIEVAL FUNCTIONS
# ============================================================================

def baseline_retrieval(vector_store, question: str, k: int = 5) -> tuple:
    """
    Baseline retrieval: Simple semantic search.
    
    Returns:
        (documents, retrieval_method, details)
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    
    details = {
        "method": "baseline",
        "queries_used": 1,
        "documents_retrieved": len(docs),
        "top_sources": list(set([doc.metadata.get("name", "unknown") for doc in docs[:3]])),
        "categories": list(set([doc.metadata.get("category", "unknown") for doc in docs]))
    }
    
    return docs, "baseline", details


def optimized_retrieval_with_hyde_and_multiquery(vector_store, question: str, llm: ChatOllama, k: int = 5) -> tuple:
    """
    Optimized retrieval combining HyDE and Multi-Query Rewriting.
    
    Returns:
        (documents, retrieval_method, details)
    """
    print(f"\n  🚀 Optimized Retrieval (HyDE + Multi-Query):")
    
    # Step 1: HyDE
    hypothetical_docs = generate_hypothetical_documents(question, llm)
    
    # Step 2: Multi-Query
    rewritten_queries = rewrite_query_multi_perspective(question, llm)
    
    # Step 3: Retrieve with all queries
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    all_retrieved_docs = []
    doc_ids = set()
    
    # Retrieve using rewritten queries
    for query in rewritten_queries:
        docs = retriever.invoke(query)
        for doc in docs:
            doc_id = id(doc)
            if doc_id not in doc_ids:
                all_retrieved_docs.append(doc)
                doc_ids.add(doc_id)
    
    # Retrieve using hypothetical docs
    for hyde_doc in hypothetical_docs:
        docs = retriever.invoke(hyde_doc)
        for doc in docs:
            doc_id = id(doc)
            if doc_id not in doc_ids:
                all_retrieved_docs.append(doc)
                doc_ids.add(doc_id)
    
    final_docs = all_retrieved_docs[:k*2]
    
    details = {
        "method": "optimized",
        "queries_used": len(rewritten_queries) + len(hypothetical_docs),
        "documents_retrieved": len(final_docs),
        "hyde_docs_generated": len(hypothetical_docs),
        "query_rewrites": len(rewritten_queries) - 1,
        "top_sources": list(set([doc.metadata.get("name", "unknown") for doc in final_docs[:3]])),
        "categories": list(set([doc.metadata.get("category", "unknown") for doc in final_docs]))
    }
    
    return final_docs, "optimized", details


# ============================================================================
# ANSWER GENERATION & EVALUATION
# ============================================================================

def generate_answer(llm: ChatOllama, context: str, question: str) -> str:
    """Generate answer from context and question."""
    
    prompt = ChatPromptTemplate.from_template(
        "You are a Game of Thrones expert. Answer the question using ONLY the "
        "provided context from Game of Thrones. If the context doesn't contain "
        "enough information to answer, say so.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


def evaluate_retrieval_quality(docs: List[Document], question: str) -> Dict[str, Any]:
    """Evaluate quality of retrieved documents."""
    
    quality_metrics = {
        "num_docs": len(docs),
        "unique_sources": len(set([doc.metadata.get("name") for doc in docs])),
        "unique_categories": len(set([doc.metadata.get("category") for doc in docs])),
        "avg_relevance_keywords": 0,
    }
    
    # Simple keyword overlap
    question_words = set(question.lower().split())
    total_overlap = 0
    for doc in docs[:5]:
        doc_words = set(doc.page_content[:200].lower().split())
        overlap = len(question_words & doc_words)
        total_overlap += overlap
    
    quality_metrics["avg_relevance_keywords"] = total_overlap // max(1, len(docs[:5]))
    
    return quality_metrics


# Game of Thrones focused test queries
GOT_TEST_QUERIES = [
    "Who is the rightful heir to the Iron Throne?",
    "Describe the major battles in Game of Thrones",
    "What are the main houses and their sigils?",
    "Tell me about the White Walkers and the threat from the North",
    "How did Ned Stark die and what were the consequences?",
    "Explain the relationship between Jon Snow and Daenerys Targaryen",
    "What happened to the dragons and how were they used?",
    "Describe the political intrigue in King's Landing",
    "What were the major plot twists in Game of Thrones?",
    "How did the series end for the main characters?",
]


def evaluate_baseline_vs_optimized_got():
    """Comprehensive evaluation of baseline vs optimized retrieval for Game of Thrones."""
    
    print("\n" + "="*80)
    print("GAME OF THRONES: BASELINE vs OPTIMIZED RETRIEVAL")
    print("="*80)
    
    # Load vector store and LLM
    vector_store = load_vector_store()
    if not vector_store:
        print("ERROR: Vector store not found. Run --build first.")
        return
    
    llm = ChatOllama(model=CHAT_MODEL)
    
    results = {
        "dataset": "game-of-thrones",
        "timestamp": datetime.now().isoformat(),
        "model": CHAT_MODEL,
        "embedding_model": EMBED_MODEL,
        "total_queries": len(GOT_TEST_QUERIES),
        "evaluations": []
    }
    
    for i, question in enumerate(GOT_TEST_QUERIES, 1):
        print(f"\n{'─'*80}")
        print(f"Query {i}/{len(GOT_TEST_QUERIES)}: {question[:70]}...")
        print(f"{'─'*80}")
        
        # === BASELINE ===
        print(f"\n[BASELINE RETRIEVAL]")
        baseline_docs, _, baseline_details = baseline_retrieval(vector_store, question, k=5)
        baseline_quality = evaluate_retrieval_quality(baseline_docs, question)
        baseline_context = "\n---\n".join([doc.page_content[:300] for doc in baseline_docs])
        
        print(f"  Retrieved: {baseline_details['documents_retrieved']} documents")
        print(f"  Sources: {baseline_details['top_sources']}")
        print(f"  Categories: {baseline_details['categories']}")
        print(f"  Keyword overlap: {baseline_quality['avg_relevance_keywords']}")
        
        try:
            baseline_answer = generate_answer(llm, baseline_context, question)
            baseline_success = True
            print(f"  Answer: {baseline_answer[:120]}...")
        except Exception as e:
            baseline_answer = f"ERROR: {str(e)}"
            baseline_success = False
            print(f"  ERROR: {e}")
        
        # === OPTIMIZED ===
        print(f"\n[OPTIMIZED RETRIEVAL]")
        optimized_docs, _, optimized_details = optimized_retrieval_with_hyde_and_multiquery(
            vector_store, question, llm, k=5
        )
        optimized_quality = evaluate_retrieval_quality(optimized_docs, question)
        optimized_context = "\n---\n".join([doc.page_content[:300] for doc in optimized_docs])
        
        print(f"  Retrieved: {optimized_details['documents_retrieved']} documents")
        print(f"  Queries used: {optimized_details['queries_used']}")
        print(f"  Sources: {optimized_details['top_sources']}")
        print(f"  Categories: {optimized_details['categories']}")
        print(f"  Keyword overlap: {optimized_quality['avg_relevance_keywords']}")
        
        try:
            optimized_answer = generate_answer(llm, optimized_context, question)
            optimized_success = True
            print(f"  Answer: {optimized_answer[:120]}...")
        except Exception as e:
            optimized_answer = f"ERROR: {str(e)}"
            optimized_success = False
            print(f"  ERROR: {e}")
        
        # === COMPARISON ===
        print(f"\n[COMPARISON]")
        doc_improvement = optimized_details['documents_retrieved'] - baseline_details['documents_retrieved']
        keyword_improvement = optimized_quality['avg_relevance_keywords'] - baseline_quality['avg_relevance_keywords']
        
        print(f"  Doc count: {baseline_details['documents_retrieved']} → {optimized_details['documents_retrieved']} ({doc_improvement:+d})")
        print(f"  Keyword relevance: {baseline_quality['avg_relevance_keywords']} → {optimized_quality['avg_relevance_keywords']} ({keyword_improvement:+d})")
        print(f"  Source diversity: {baseline_quality['unique_sources']} → {optimized_quality['unique_sources']}")
        
        # Store result
        results["evaluations"].append({
            "query_num": i,
            "question": question,
            "baseline": {
                "details": baseline_details,
                "quality_metrics": baseline_quality,
                "answer_preview": baseline_answer[:250] if baseline_success else baseline_answer,
                "success": baseline_success
            },
            "optimized": {
                "details": optimized_details,
                "quality_metrics": optimized_quality,
                "answer_preview": optimized_answer[:250] if optimized_success else optimized_answer,
                "success": optimized_success
            },
            "improvement": {
                "doc_count": doc_improvement,
                "keyword_relevance": keyword_improvement,
                "source_diversity": optimized_quality['unique_sources'] - baseline_quality['unique_sources']
            }
        })
    
    # === SUMMARY ===
    print(f"\n\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    
    baseline_total_docs = sum([e["baseline"]["details"]["documents_retrieved"] for e in results["evaluations"]])
    optimized_total_docs = sum([e["optimized"]["details"]["documents_retrieved"] for e in results["evaluations"]])
    baseline_success_count = sum([1 for e in results["evaluations"] if e["baseline"]["success"]])
    optimized_success_count = sum([1 for e in results["evaluations"] if e["optimized"]["success"]])
    
    avg_baseline_keyword = sum([e["baseline"]["quality_metrics"]["avg_relevance_keywords"] for e in results["evaluations"]]) / len(results["evaluations"])
    avg_optimized_keyword = sum([e["optimized"]["quality_metrics"]["avg_relevance_keywords"] for e in results["evaluations"]]) / len(results["evaluations"])
    
    print(f"Total Queries: {len(GOT_TEST_QUERIES)}")
    print(f"\nBaseline Retrieval:")
    print(f"  Success Rate: {baseline_success_count}/{len(GOT_TEST_QUERIES)} ({100*baseline_success_count/len(GOT_TEST_QUERIES):.1f}%)")
    print(f"  Total Docs: {baseline_total_docs}")
    print(f"  Avg Keyword Relevance: {avg_baseline_keyword:.2f}")
    
    print(f"\nOptimized Retrieval (HyDE + Multi-Query):")
    print(f"  Success Rate: {optimized_success_count}/{len(GOT_TEST_QUERIES)} ({100*optimized_success_count/len(GOT_TEST_QUERIES):.1f}%)")
    print(f"  Total Docs: {optimized_total_docs}")
    print(f"  Avg Keyword Relevance: {avg_optimized_keyword:.2f}")
    
    print(f"\nImprovement:")
    print(f"  Docs: {optimized_total_docs - baseline_total_docs:+d} ({100*(optimized_total_docs-baseline_total_docs)/baseline_total_docs:+.1f}%)")
    print(f"  Keyword Relevance: {avg_optimized_keyword - avg_baseline_keyword:+.2f}")
    
    # Save results
    output_file = os.path.join(SCRIPT_DIR, "got_evaluation_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if "--build" in sys.argv:
        build_got_vector_store()
    elif "--evaluate" in sys.argv:
        evaluate_baseline_vs_optimized_got()
    else:
        # Interactive mode
        store = load_vector_store()
        llm = ChatOllama(model=CHAT_MODEL)
        
        print(f"\n🐉 Game of Thrones RAG System Ready")
        print(f"Using {CHAT_MODEL} for answers")
        print("Commands:")
        print("  - Type a question about Game of Thrones")
        print("  - Type 'opt [question]' for optimized retrieval")
        print("  - Type 'exit' to quit\n")
        
        while True:
            try:
                user_input = input("Q: ").strip()
                if not user_input:
                    continue
                
                if user_input.lower() == "exit":
                    print("Goodbye!")
                    break
                
                use_optimized = user_input.startswith("opt ")
                question = user_input[4:].strip() if use_optimized else user_input
                
                print("\nSearching Game of Thrones documents...")
                
                if use_optimized:
                    docs, method, details = optimized_retrieval_with_hyde_and_multiquery(store, question, llm)
                    print(f"[Optimized] Docs: {details['documents_retrieved']} | Queries: {details['queries_used']}")
                else:
                    docs, method, details = baseline_retrieval(store, question)
                    print(f"[Baseline] Docs: {details['documents_retrieved']}")
                
                context = "\n---\n".join([doc.page_content[:300] for doc in docs])
                answer = generate_answer(llm, context, question)
                print(f"\nA: {answer}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
