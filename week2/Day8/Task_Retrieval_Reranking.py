"""
Game of Thrones RAG: Vector vs Hybrid Retrieval with Reranking
Compares: Vector-only, Hybrid Retrieval with/without Reranking
Uses: Tuana/game-of-thrones dataset from Hugging Face
"""

import sys
import os
import json
import re
from datetime import datetime
from typing import List, Dict, Tuple, Any
from rank_bm25 import BM25Okapi
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import numpy as np


# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(SCRIPT_DIR, "chroma_db_got")
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHAT_MODEL = "llama3.1:8b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
K_RETRIEVE = 10  # Retrieve more documents for reranking
K_FINAL = 5      # Final number of documents after reranking


# ============================================================================
# LOAD & BUILD VECTOR STORE
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
    """Extract enriched metadata from Game of Thrones dataset record."""
    
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
    elif any(word in content for word in ["dragon", "magic", "white walker"]):
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
    """Load Game of Thrones dataset and build vector store."""
    
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
    return vector_store, chunks


# ============================================================================
# RETRIEVAL STRATEGIES
# ============================================================================

def vector_only_retrieval(vector_store, question: str, k: int = K_RETRIEVE) -> Tuple[List[Document], Dict[str, Any]]:
    """
    Vector-only retrieval: Pure semantic search using embeddings.
    
    Returns:
        (documents, metadata)
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    
    metadata = {
        "method": "vector-only",
        "documents_retrieved": len(docs),
        "retrieval_type": "semantic"
    }
    
    return docs, metadata


def hybrid_retrieval(vector_store, documents: List[Document], question: str, k: int = K_RETRIEVE) -> Tuple[List[Document], Dict[str, Any]]:
    """
    Hybrid retrieval: Combines BM25 (keyword-based) and vector search.
    
    Strategy:
    1. Tokenize documents for BM25
    2. Retrieve top-k with BM25
    3. Retrieve top-k with vector search
    4. Merge results (deduplicated by content)
    
    Returns:
        (documents, metadata)
    """
    
    # Prepare BM25
    corpus = [doc.page_content for doc in documents]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # BM25 retrieval
    tokenized_query = question.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:k]
    bm25_docs = [documents[idx] for idx in bm25_top_indices]
    
    # Vector retrieval
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    vector_docs = retriever.invoke(question)
    
    # Merge results (deduplicate by content hash)
    doc_content_set = {}
    merged_docs = []
    
    # Add BM25 results first (keyword relevance)
    for doc in bm25_docs:
        content_hash = hash(doc.page_content[:100])
        if content_hash not in doc_content_set:
            doc_content_set[content_hash] = True
            merged_docs.append(doc)
    
    # Add vector results (semantic relevance)
    for doc in vector_docs:
        content_hash = hash(doc.page_content[:100])
        if content_hash not in doc_content_set:
            doc_content_set[content_hash] = True
            merged_docs.append(doc)
    
    # Limit to k documents
    final_docs = merged_docs[:k]
    
    metadata = {
        "method": "hybrid",
        "documents_retrieved": len(final_docs),
        "bm25_results": len([d for d in final_docs if d in bm25_docs]),
        "vector_results": len([d for d in final_docs if d in vector_docs]),
        "retrieval_type": "hybrid (BM25 + semantic)"
    }
    
    return final_docs, metadata


# ============================================================================
# RERANKING
# ============================================================================

def rerank_documents(documents: List[Document], question: str, k_final: int = K_FINAL) -> Tuple[List[Document], List[float], Dict[str, Any]]:
    """
    Rerank documents using a cross-encoder model.
    
    A cross-encoder directly scores query-document pairs, providing more
    accurate relevance assessment than semantic similarity alone.
    
    Returns:
        (reranked_documents, scores, metadata)
    """
    
    print(f"  Loading cross-encoder model: {RERANK_MODEL}")
    cross_encoder = CrossEncoder(RERANK_MODEL)
    
    # Prepare pairs for cross-encoder
    pairs = [[question, doc.page_content[:500]] for doc in documents]
    
    # Get relevance scores
    scores = cross_encoder.predict(pairs, show_progress_bar=False)
    
    # Sort by relevance score (descending)
    sorted_indices = np.argsort(scores)[::-1]
    reranked_docs = [documents[idx] for idx in sorted_indices[:k_final]]
    reranked_scores = [scores[idx] for idx in sorted_indices[:k_final]]
    
    metadata = {
        "method": "reranking",
        "model": RERANK_MODEL,
        "initial_docs": len(documents),
        "final_docs": len(reranked_docs),
        "top_score": float(reranked_scores[0]) if reranked_scores else 0,
        "avg_score": float(np.mean(reranked_scores)) if reranked_scores else 0
    }
    
    return reranked_docs, reranked_scores, metadata


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_vector_store():
    """Load existing vector store from disk."""
    return Chroma(
        collection_name="game_of_thrones",
        embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL),
        persist_directory=CHROMA_DIR,
    )


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


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_retrieval_strategies():
    """Comprehensive evaluation of vector-only vs hybrid with/without reranking."""
    
    print("\n" + "="*90)
    print("GAME OF THRONES: VECTOR vs HYBRID RETRIEVAL WITH RERANKING")
    print("="*90)
    
    # Load vector store
    vector_store = load_vector_store()
    if not vector_store:
        print("ERROR: Vector store not found. Run --build first.")
        return
    
    # Load all documents for BM25 (hybrid retrieval)
    # We need to rebuild documents list for BM25
    dataset = load_got_dataset()
    if not dataset:
        print("ERROR: Could not load dataset for BM25")
        return
    
    documents = []
    for idx, record in enumerate(dataset):
        try:
            if "content" in record and record["content"]:
                content = str(record["content"])
            else:
                continue
            if len(content) < 100:
                continue
            
            metadata = extract_rich_metadata(record, idx)
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        except Exception as e:
            continue
    
    # Chunk documents (same as during build)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks_for_bm25 = splitter.split_documents(documents)
    
    print(f"✓ Loaded {len(chunks_for_bm25)} chunks for BM25 retrieval\n")
    
    # Initialize LLM
    llm = ChatOllama(model=CHAT_MODEL)
    
    results = {
        "dataset": "game-of-thrones",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "embedding": EMBED_MODEL,
            "reranking": RERANK_MODEL,
            "chat": CHAT_MODEL
        },
        "configuration": {
            "k_retrieve": K_RETRIEVE,
            "k_final": K_FINAL
        },
        "total_queries": len(GOT_TEST_QUERIES),
        "evaluations": []
    }
    
    for i, question in enumerate(GOT_TEST_QUERIES, 1):
        print(f"\n{'─'*90}")
        print(f"Query {i}/{len(GOT_TEST_QUERIES)}: {question[:75]}...")
        print(f"{'─'*90}")
        
        evaluation = {
            "query_num": i,
            "question": question,
            "strategies": {}
        }
        
        # --- VECTOR-ONLY WITHOUT RERANKING ---
        print(f"\n[1] VECTOR-ONLY (No Reranking)")
        vector_docs, vector_meta = vector_only_retrieval(vector_store, question, K_RETRIEVE)
        vector_quality = evaluate_retrieval_quality(vector_docs, question)
        vector_context = "\n---\n".join([doc.page_content[:300] for doc in vector_docs[:K_FINAL]])
        
        print(f"    Retrieved: {len(vector_docs)} documents")
        print(f"    Categories: {vector_quality['unique_categories']}")
        print(f"    Keyword overlap: {vector_quality['avg_relevance_keywords']}")
        
        try:
            vector_answer = generate_answer(llm, vector_context, question)
            vector_success = True
        except Exception as e:
            vector_answer = f"ERROR: {str(e)}"
            vector_success = False
        
        evaluation["strategies"]["vector_only"] = {
            "metadata": vector_meta,
            "quality_metrics": vector_quality,
            "answer_preview": vector_answer[:250] if vector_success else vector_answer,
            "success": vector_success
        }
        
        # --- VECTOR-ONLY WITH RERANKING ---
        print(f"\n[2] VECTOR-ONLY + RERANKING")
        vector_reranked, vector_scores, rerank_meta = rerank_documents(vector_docs, question, K_FINAL)
        vector_rerank_quality = evaluate_retrieval_quality(vector_reranked, question)
        vector_rerank_context = "\n---\n".join([doc.page_content[:300] for doc in vector_reranked])
        
        print(f"    Initial: {len(vector_docs)} → Final: {len(vector_reranked)} documents")
        print(f"    Top rerank score: {rerank_meta['top_score']:.4f}")
        print(f"    Avg rerank score: {rerank_meta['avg_score']:.4f}")
        print(f"    Categories: {vector_rerank_quality['unique_categories']}")
        
        try:
            vector_rerank_answer = generate_answer(llm, vector_rerank_context, question)
            vector_rerank_success = True
        except Exception as e:
            vector_rerank_answer = f"ERROR: {str(e)}"
            vector_rerank_success = False
        
        evaluation["strategies"]["vector_only_reranked"] = {
            "retrieval_metadata": vector_meta,
            "reranking_metadata": rerank_meta,
            "quality_metrics": vector_rerank_quality,
            "answer_preview": vector_rerank_answer[:250] if vector_rerank_success else vector_rerank_answer,
            "success": vector_rerank_success
        }
        
        # --- HYBRID WITHOUT RERANKING ---
        print(f"\n[3] HYBRID Retrieval (No Reranking)")
        hybrid_docs, hybrid_meta = hybrid_retrieval(vector_store, chunks_for_bm25, question, K_RETRIEVE)
        hybrid_quality = evaluate_retrieval_quality(hybrid_docs, question)
        hybrid_context = "\n---\n".join([doc.page_content[:300] for doc in hybrid_docs[:K_FINAL]])
        
        print(f"    Retrieved: {len(hybrid_docs)} documents")
        print(f"    From BM25: {hybrid_meta['bm25_results']}, From Vector: {hybrid_meta['vector_results']}")
        print(f"    Categories: {hybrid_quality['unique_categories']}")
        print(f"    Keyword overlap: {hybrid_quality['avg_relevance_keywords']}")
        
        try:
            hybrid_answer = generate_answer(llm, hybrid_context, question)
            hybrid_success = True
        except Exception as e:
            hybrid_answer = f"ERROR: {str(e)}"
            hybrid_success = False
        
        evaluation["strategies"]["hybrid"] = {
            "metadata": hybrid_meta,
            "quality_metrics": hybrid_quality,
            "answer_preview": hybrid_answer[:250] if hybrid_success else hybrid_answer,
            "success": hybrid_success
        }
        
        # --- HYBRID WITH RERANKING ---
        print(f"\n[4] HYBRID Retrieval + RERANKING")
        hybrid_reranked, hybrid_scores, hybrid_rerank_meta = rerank_documents(hybrid_docs, question, K_FINAL)
        hybrid_rerank_quality = evaluate_retrieval_quality(hybrid_reranked, question)
        hybrid_rerank_context = "\n---\n".join([doc.page_content[:300] for doc in hybrid_reranked])
        
        print(f"    Initial: {len(hybrid_docs)} → Final: {len(hybrid_reranked)} documents")
        print(f"    Top rerank score: {hybrid_rerank_meta['top_score']:.4f}")
        print(f"    Avg rerank score: {hybrid_rerank_meta['avg_score']:.4f}")
        print(f"    Categories: {hybrid_rerank_quality['unique_categories']}")
        
        try:
            hybrid_rerank_answer = generate_answer(llm, hybrid_rerank_context, question)
            hybrid_rerank_success = True
        except Exception as e:
            hybrid_rerank_answer = f"ERROR: {str(e)}"
            hybrid_rerank_success = False
        
        evaluation["strategies"]["hybrid_reranked"] = {
            "retrieval_metadata": hybrid_meta,
            "reranking_metadata": hybrid_rerank_meta,
            "quality_metrics": hybrid_rerank_quality,
            "answer_preview": hybrid_rerank_answer[:250] if hybrid_rerank_success else hybrid_rerank_answer,
            "success": hybrid_rerank_success
        }
        
        # --- COMPARISON ---
        print(f"\n[COMPARISON]")
        print(f"  Vector Quality (no rerank): {vector_quality['avg_relevance_keywords']} keywords")
        print(f"  Vector Quality (reranked):  {vector_rerank_quality['avg_relevance_keywords']} keywords")
        print(f"  Hybrid Quality (no rerank): {hybrid_quality['avg_relevance_keywords']} keywords")
        print(f"  Hybrid Quality (reranked):  {hybrid_rerank_quality['avg_relevance_keywords']} keywords")
        
        results["evaluations"].append(evaluation)
    
    # === SUMMARY STATISTICS ===
    print(f"\n\n{'='*90}")
    print("SUMMARY STATISTICS")
    print(f"{'='*90}\n")
    
    strategies = ["vector_only", "vector_only_reranked", "hybrid", "hybrid_reranked"]
    summary_stats = {}
    
    for strategy in strategies:
        success_count = sum([1 for e in results["evaluations"] if e["strategies"][strategy]["success"]])
        avg_keywords = sum([e["strategies"][strategy]["quality_metrics"]["avg_relevance_keywords"] 
                           for e in results["evaluations"]]) / len(results["evaluations"])
        avg_categories = sum([e["strategies"][strategy]["quality_metrics"]["unique_categories"] 
                             for e in results["evaluations"]]) / len(results["evaluations"])
        
        summary_stats[strategy] = {
            "success_rate": f"{100*success_count/len(results['evaluations']):.1f}%",
            "avg_keyword_relevance": f"{avg_keywords:.2f}",
            "avg_category_diversity": f"{avg_categories:.2f}"
        }
    
    print("Strategy Performance:\n")
    for strategy, stats in summary_stats.items():
        print(f"{strategy.upper().replace('_', ' ')}:")
        print(f"  Success Rate: {stats['success_rate']}")
        print(f"  Avg Keyword Relevance: {stats['avg_keyword_relevance']}")
        print(f"  Avg Category Diversity: {stats['avg_category_diversity']}")
        print()
    
    # Save results
    output_file = os.path.join(SCRIPT_DIR, "retrieval_reranking_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")
    
    # Create markdown summary
    create_summary_markdown(results, summary_stats)


def create_summary_markdown(results: Dict, summary_stats: Dict):
    """Create a markdown summary of the results."""
    
    summary_file = os.path.join(SCRIPT_DIR, "RETRIEVAL_RERANKING_ANALYSIS.md")
    
    markdown = f"""# Game of Thrones: Vector vs Hybrid Retrieval with Reranking

## Overview
Comprehensive evaluation comparing four retrieval strategies:
1. **Vector-only** (semantic search only)
2. **Vector-only + Reranking** (semantic + cross-encoder reranking)
3. **Hybrid** (BM25 + vector search combined)
4. **Hybrid + Reranking** (hybrid + cross-encoder reranking)

## Models Used
- **Embedding Model**: {results['models']['embedding']}
- **Reranking Model**: {results['models']['reranking']}
- **Chat Model**: {results['models']['chat']}

## Configuration
- **K Retrieve**: {results['configuration']['k_retrieve']} documents
- **K Final**: {results['configuration']['k_final']} documents (after reranking)
- **Total Queries**: {results['total_queries']}

## Results Summary

### Performance Metrics

| Strategy | Success Rate | Avg Keyword Relevance | Avg Category Diversity |
|----------|--------------|----------------------|------------------------|
| Vector-Only | {summary_stats['vector_only']['success_rate']} | {summary_stats['vector_only']['avg_keyword_relevance']} | {summary_stats['vector_only']['avg_category_diversity']} |
| Vector-Only + Reranking | {summary_stats['vector_only_reranked']['success_rate']} | {summary_stats['vector_only_reranked']['avg_keyword_relevance']} | {summary_stats['vector_only_reranked']['avg_category_diversity']} |
| Hybrid | {summary_stats['hybrid']['success_rate']} | {summary_stats['hybrid']['avg_keyword_relevance']} | {summary_stats['hybrid']['avg_category_diversity']} |
| Hybrid + Reranking | {summary_stats['hybrid_reranked']['success_rate']} | {summary_stats['hybrid_reranked']['avg_keyword_relevance']} | {summary_stats['hybrid_reranked']['avg_category_diversity']} |

## Key Findings

### Reranking Impact
- **Cross-encoder reranking** provides semantic quality assessment beyond embeddings
- Reranking helps prioritize most relevant documents from a larger pool
- Better for scenarios where initial retrieval returns many semi-relevant documents

### Vector vs Hybrid
- **Vector-only**: Excellent for semantic matching, may miss keyword-specific details
- **Hybrid**: Combines semantic similarity + keyword matching for broader coverage
- Hybrid retrieval captures both semantic and lexical relevance

### Recommendations
1. **Use Vector-only** for semantic-heavy questions about relationships and concepts
2. **Use Hybrid** for mixed queries that require both semantic and keyword matching
3. **Add Reranking** when initial retrieval pool is large (10+ documents)
4. **Best Performance**: Hybrid + Reranking for balanced, high-quality results

## Test Queries
{json.dumps(results['total_queries'], indent=2)}

## Detailed Results
See `retrieval_reranking_results.json` for complete evaluation data including:
- Per-query performance metrics
- Quality metrics for each strategy
- Answer previews
- Reranking scores and improvements

---
*Generated: {results['timestamp']}*
"""
    
    with open(summary_file, 'w') as f:
        f.write(markdown)
    print(f"✓ Summary saved to {summary_file}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if "--build" in sys.argv:
        print("Building vector store...")
        build_got_vector_store()
    elif "--evaluate" in sys.argv:
        evaluate_retrieval_strategies()
    else:
        print("Game of Thrones Retrieval & Reranking System")
        print("\nUsage:")
        print("  python Task_Retrieval_Reranking.py --build      (Build vector store)")
        print("  python Task_Retrieval_Reranking.py --evaluate   (Run evaluation)")
        print("\nOr specify --build first to initialize the vector store.")
