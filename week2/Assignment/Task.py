"""
Travel RAG Assistant with Preferences
Combines web content ingestion, preference extraction, retrieval + re-ranking, 
quality checks, and grounded answer generation using Groq.

Pattern: query → preference extraction → retrieval + re-ranking → quality check → answer
"""

import sys
import os
import json
import re
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_DIR = os.path.join(SCRIPT_DIR, "faiss_index")
URLS_FILE = os.path.join(SCRIPT_DIR, "travel_urls.json")

# Models
EMBED_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")  # Groq model
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
K_RETRIEVE = 5  # Top K documents to retrieve

# Quality thresholds
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for "good" context
QUALITY_CHECK_ENABLED = True


# ============================================================================
# DEFAULT TRAVEL URLS & METADATA
# ============================================================================

DEFAULT_TRAVEL_URLS = [
    {
        "url": "https://example.com/berlin-guide",
        "city": "Berlin",
        "category": "sightseeing",
        "price_level": "cheap",
        "title": "Complete Berlin Travel Guide"
    },
    {
        "url": "https://example.com/berlin-food",
        "city": "Berlin",
        "category": "food",
        "price_level": "cheap",
        "title": "Best Budget Food in Berlin"
    },
    {
        "url": "https://example.com/berlin-museums",
        "city": "Berlin",
        "category": "art",
        "price_level": "medium",
        "title": "Berlin Museum Guide"
    },
    # TODO: Add your real travel URLs here
]


# ============================================================================
# UTIL FUNCTIONS
# ============================================================================

def load_or_create_urls_file():
    """Load URLs from JSON file or create with defaults."""
    if os.path.exists(URLS_FILE):
        with open(URLS_FILE, 'r') as f:
            return json.load(f)
    else:
        with open(URLS_FILE, 'w') as f:
            json.dump(DEFAULT_TRAVEL_URLS, f, indent=2)
        print(f"Created {URLS_FILE} with placeholders. Fill in your travel URLs.")
        return DEFAULT_TRAVEL_URLS


def fetch_and_clean_text(url: str) -> Optional[str]:
    """
    Fetch content from URL and extract clean text.
    
    Returns:
        Clean text content or None if fetch fails
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text if len(text) > 100 else None
        
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


# ============================================================================
# SOURCE INGESTION
# ============================================================================

def ingest_travel_content() -> List[Document]:
    """
    Fetch and chunk travel content from URLs.
    
    Returns:
        List of Document objects with metadata
    """
    print("Starting travel content ingestion...")
    
    urls_data = load_or_create_urls_file()
    documents = []
    
    for url_item in urls_data:
        try:
            url = url_item.get("url")
            if not url:
                print(f"Skipping invalid URL entry: {url_item}")
                continue
            
            print(f"  Fetching: {url_item.get('title', url[:50])}")
            
            # Fetch content
            content = fetch_and_clean_text(url)
            if not content:
                print(f"    → No usable content")
                continue
            
            # Create Document with metadata
            metadata = {
                "url": url,
                "city": url_item.get("city", "unknown").lower(),
                "category": url_item.get("category", "general").lower(),
                "price_level": url_item.get("price_level", "medium").lower(),
                "title": url_item.get("title", ""),
                "source": "travel_web",
                "fetch_date": datetime.now().isoformat()
            }
            
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
            print(f"    ✓ Ingested {len(content)} chars")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            continue
    
    print(f"Ingested {len(documents)} documents total")
    return documents


def build_vector_store() -> FAISS:
    """
    Ingest travel content and build FAISS vector store.
    
    Returns:
        FAISS vector store
    """
    print("\n" + "="*80)
    print("BUILDING VECTOR STORE")
    print("="*80)
    
    # Ingest documents
    documents = ingest_travel_content()
    if not documents:
        print("ERROR: No documents ingested")
        return None
    
    # Chunk documents
    print(f"\nChunking {len(documents)} documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Embed and build FAISS index
    print(f"\nEmbedding chunks with {EMBED_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    # Save to disk
    print(f"Saving FAISS index to {FAISS_INDEX_DIR}...")
    vector_store.save_local(FAISS_INDEX_DIR)
    
    print("✓ Vector store built successfully")
    return vector_store


def load_vector_store() -> Optional[FAISS]:
    """Load existing FAISS vector store from disk."""
    if not os.path.exists(FAISS_INDEX_DIR):
        print("Vector store not found. Building...")
        return build_vector_store()
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vector_store = FAISS.load_local(
            FAISS_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✓ Loaded FAISS vector store from {FAISS_INDEX_DIR}")
        return vector_store
    except Exception as e:
        print(f"Error loading vector store: {e}. Rebuilding...")
        return build_vector_store()


# ============================================================================
# PREFERENCE EXTRACTION
# ============================================================================

def extract_preferences(user_query: str, llm: ChatGroq) -> Dict[str, Any]:
    """
    Extract travel preferences from user query using Groq.
    
    Returns:
        JSON dict with budget, interests, city, etc.
    """
    
    prompt = ChatPromptTemplate.from_template(
        """Extract travel preferences from the user query.
Return ONLY a valid JSON object (no markdown, no extra text).

User query: {query}

JSON format:
{{
    "cities": ["city1", "city2"],
    "budget": "cheap" | "medium" | "expensive",
    "interests": ["food", "art", "sightseeing", "history", "adventure"],
    "duration_days": integer,
    "other_keywords": ["keyword1", "keyword2"]
}}

Extract what you can. Use reasonable defaults if not specified.
If no specific city mentioned, use empty array for cities.
If budget unclear, default to "medium".
"""
    )
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"query": user_query})
    
    # Parse JSON from response (robustly)
    prefs = extract_json_from_response(response, json_type="object") or {}
    
    # Ensure required fields
    prefs.setdefault("cities", [])
    prefs.setdefault("budget", "medium")
    prefs.setdefault("interests", [])
    prefs.setdefault("duration_days", 3)
    prefs.setdefault("other_keywords", [])
    
    return prefs


# ============================================================================
# RETRIEVAL + RE-RANKING
# ============================================================================

def apply_metadata_filters(
    docs: List[Document],
    preferences: Dict[str, Any]
) -> List[Document]:
    """
    Filter retrieved documents by preference metadata.
    
    Filters:
    - city (if specified in preferences)
    - price_level (if budget specified)
    - category (if interests specified)
    """
    
    filtered = docs
    
    # Filter by city
    if preferences.get("cities"):
        city_filter = [c.lower() for c in preferences["cities"]]
        filtered = [
            d for d in filtered 
            if any(city in d.metadata.get("city", "").lower() for city in city_filter)
        ]
        if not filtered and docs:  # If no matches, allow all
            filtered = docs
    
    # Filter by price level
    budget = preferences.get("budget", "medium")
    if budget in ["cheap", "medium", "expensive"]:
        # Map budget to allowed price levels (cheap → cheap only, medium → cheap+medium, expensive → all)
        if budget == "cheap":
            allowed_prices = ["cheap"]
        elif budget == "medium":
            allowed_prices = ["cheap", "medium"]
        else:
            allowed_prices = ["cheap", "medium", "expensive"]
        
        filtered = [
            d for d in filtered 
            if d.metadata.get("price_level", "medium") in allowed_prices
        ]
    
    # Filter by interests/category
    interests = preferences.get("interests", [])
    if interests:
        interests_lower = [i.lower() for i in interests]
        filtered = [
            d for d in filtered
            if any(interest in d.metadata.get("category", "").lower() for interest in interests_lower)
        ]
    
    return filtered


def rerank_documents(
    docs: List[Document],
    query: str,
    llm: ChatGroq,
    preferences: Dict[str, Any]
) -> List[Document]:
    """
    Re-rank retrieved documents using Groq as a simple judge.
    
    Score each document's relevance to the query given the preferences.
    """
    
    if not docs:
        return docs
    
    # Create a simple scoring prompt
    prompt = ChatPromptTemplate.from_template(
        """Score the relevance of each document to the user's travel query.

User query: {query}
Preferences: {preferences}

Documents to score:
{docs_text}

Return ONLY a JSON array with object format:
[
    {{"doc_index": 0, "score": 0.85, "reason": "..."}}
]

Score from 0 to 1. Higher = more relevant.
"""
    )
    
    # Format documents for scoring
    docs_text = "\n---\n".join([
        f"[{i}] {doc.page_content[:300]}...  (city: {doc.metadata.get('city')}, category: {doc.metadata.get('category')})"
        for i, doc in enumerate(docs)
    ])
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "query": query,
        "preferences": json.dumps(preferences),
        "docs_text": docs_text
    })
    
    # Parse scores
    try:
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
            
            # Sort by score (descending)
            sorted_indices = sorted(
                range(len(docs)),
                key=lambda i: next((s["score"] for s in scores if s.get("doc_index") == i), 0),
                reverse=True
            )
            
            return [docs[i] for i in sorted_indices]
    except Exception as e:
        print(f"Warning: Re-ranking failed: {e}")
    
    return docs


def retrieve_and_rerank(
    vector_store: FAISS,
    query: str,
    preferences: Dict[str, Any],
    llm: ChatGroq,
    k: int = K_RETRIEVE
) -> Tuple[List[Document], Dict[str, Any]]:
    """
    Retrieve documents and apply filtering + re-ranking.
    
    Returns:
        (docs, metadata)
    """
    
    # Semantic search
    retriever = vector_store.as_retriever(search_kwargs={"k": k * 2})  # Get extra to filter
    initial_docs = retriever.invoke(query)
    
    # Apply metadata filters
    filtered_docs = apply_metadata_filters(initial_docs, preferences)
    
    # If too many filtered out, use all
    if not filtered_docs:
        filtered_docs = initial_docs
    
    # Re-rank
    reranked_docs = rerank_documents(filtered_docs[:k], query, llm, preferences)
    
    metadata = {
        "initial_retrieved": len(initial_docs),
        "after_filter": len(filtered_docs),
        "final_returned": len(reranked_docs),
        "urls_used": [d.metadata.get("url") for d in reranked_docs],
        "cities": list(set(d.metadata.get("city") for d in reranked_docs if d.metadata.get("city")))
    }
    
    return reranked_docs, metadata


# ============================================================================
# CONTEXT QUALITY CHECK
# ============================================================================

def check_context_quality(
    docs: List[Document],
    query: str,
    llm: ChatGroq
) -> Tuple[str, float, str]:
    """
    Lightweight judge: is the retrieved context good enough?
    
    Returns:
        (quality: "good"|"insufficient", confidence: 0-1, reason: str)
    """
    
    if not docs:
        return "insufficient", 0.0, "No documents retrieved"
    
    context_text = "\n---\n".join([d.page_content[:500] for d in docs])
    
    prompt = ChatPromptTemplate.from_template(
        """Judge if the retrieved context is sufficient to answer the user's travel query.

Query: {query}

Retrieved context:
{context}

Return ONLY JSON (no markdown):
{{
    "quality": "good" | "insufficient",
    "confidence": 0.0-1.0,
    "reason": "..."
}}

"good" if context clearly addresses the query.
"insufficient" if missing key info (e.g., no food options, no budget info, etc.)
Confidence: how confident in this judgment.
"""
    )
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"query": query, "context": context_text})
    
    result = extract_json_from_response(response, json_type="object")
    if result:
        return (
            result.get("quality", "insufficient"),
            result.get("confidence", 0.5),
            result.get("reason", "")
        )
    
    return "insufficient", 0.5, "Quality check error"


# ============================================================================
# ANSWER GENERATION
# ============================================================================

def extract_json_from_response(response: str, json_type: str = "object") -> Optional[Dict | List]:
    """
    Robustly extract JSON from LLM response (handles markdown, extra text, etc).
    
    Args:
        response: Raw LLM response
        json_type: "object" for {}, "array" for []
    
    Returns:
        Parsed JSON dict/list or None
    """
    if not response:
        return None
    
    # Try 1: Remove markdown code blocks
    cleaned = re.sub(r'```(?:json)?\s*', '', response)
    
    # Try 2: Extract JSON using regex (non-greedy)
    if json_type == "object":
        # For objects: find balanced {} pairs
        json_match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', cleaned, re.DOTALL)
    else:
        # For arrays: find [ ... ]
        json_match = re.search(r'\[.*?\]', cleaned, re.DOTALL)
    
    if not json_match:
        return None
    
    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError:
        return None


def generate_grounded_answer(
    llm: ChatGroq,
    query: str,
    docs: List[Document],
    preferences: Dict[str, Any],
    quality_context: Tuple[str, float, str]
) -> Dict[str, Any]:
    """
    Generate final travel answer grounded in retrieved context.
    
    Returns:
        Dict with answer, citations, confidence, etc.
    """
    
    quality, quality_confidence, quality_reason = quality_context
    
    if quality == "insufficient" and quality_confidence > CONFIDENCE_THRESHOLD:
        return {
            "answer": f"I don't have enough information to answer: {quality_reason}. "
                     f"Please ask more specifically about a particular city, activity, or budget.",
            "citations": [],
            "confidence": 0.0,
            "quality_status": "insufficient",
            "valid": True
        }
    
    # Build context
    context_text = "\n---\n".join([
        f"[{i}] {doc.page_content}\n   (source: {doc.metadata.get('url')})"
        for i, doc in enumerate(docs)
    ])
    
    prompt = ChatPromptTemplate.from_template(
        """You are a travel assistant. Answer the user's travel question using ONLY the provided context.
Provide a practical, actionable answer. Light citations encouraged (reference [N] for source).

User query: {query}
User preferences: {preferences}

Context:
{context}

Return JSON (no markdown):
{{
    "answer": "Your grounded answer here",
    "citations": ["context_quote_1", "context_quote_2"],
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of how you arrived at this answer"
}}

confidence: 0=not confident, 1=very confident in the answer.
"""
    )
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "query": query,
        "preferences": json.dumps(preferences),
        "context": context_text
    })
    
    # Parse and validate
    parsed = extract_json_from_response(response, json_type="object")
    
    if parsed and all(k in parsed for k in ["answer", "citations", "confidence"]):
        parsed["valid"] = True
        parsed["quality_status"] = quality
        return parsed
    
    # Fallback: still better than nothing
    return {
        "answer": response[:300] if response else "Unable to generate answer",
        "citations": [],
        "confidence": 0.0,
        "valid": False,
        "error": "Failed to parse structured output",
        "quality_status": quality
    }


# ============================================================================
# FULL PIPELINE
# ============================================================================

def run_travel_query(
    vector_store: FAISS,
    llm: ChatGroq,
    user_query: str
) -> Dict[str, Any]:
    """
    Full pipeline: query → preference extraction → retrieval → quality check → answer
    
    Returns:
        Complete result with intermediate steps
    """
    
    print(f"\nQuery: {user_query}")
    print("-" * 80)
    
    result = {
        "query": user_query,
        "timestamp": datetime.now().isoformat(),
        "steps": {}
    }
    
    # Step 1: Preference Extraction
    print("\n[1] PREFERENCE EXTRACTION")
    preferences = extract_preferences(user_query, llm)
    print(f"  Extracted: {json.dumps(preferences, indent=2)}")
    result["steps"]["preferences"] = preferences
    
    # Step 2: Retrieval + Re-ranking
    print("\n[2] RETRIEVAL + RE-RANKING")
    docs, retrieval_meta = retrieve_and_rerank(vector_store, user_query, preferences, llm)
    print(f"  Initial retrieved: {retrieval_meta['initial_retrieved']}")
    print(f"  After filtering: {retrieval_meta['after_filter']}")
    print(f"  Final reranked: {retrieval_meta['final_returned']}")
    print(f"  Cities targeted: {retrieval_meta['cities']}")
    
    # Store both metadata AND documents
    retrieval_meta["documents"] = [
        {
            "content": doc.page_content[:500],  # First 500 chars
            "metadata": doc.metadata,
            "full_content": doc.page_content
        }
        for doc in docs
    ]
    result["steps"]["retrieval"] = retrieval_meta
    
    # Step 3: Quality Check
    print("\n[3] CONTEXT QUALITY CHECK")
    quality_status, quality_score, quality_reason = check_context_quality(docs, user_query, llm)
    print(f"  Status: {quality_status}")
    print(f"  Confidence: {quality_score:.2f}")
    print(f"  Reason: {quality_reason}")
    result["steps"]["quality_check"] = {
        "status": quality_status,
        "confidence": quality_score,
        "reason": quality_reason
    }
    
    # Step 4: Answer Generation
    print("\n[4] ANSWER GENERATION")
    answer_result = generate_grounded_answer(
        llm, user_query, docs, preferences, 
        (quality_status, quality_score, quality_reason)
    )
    print(f"  Answer: {answer_result['answer'][:150]}...")
    print(f"  Confidence: {answer_result['confidence']:.2f}")
    print(f"  Valid: {answer_result['valid']}")
    result["answer"] = answer_result
    
    return result


# ============================================================================
# DEMO/EVALUATION
# ============================================================================

def run_demo():
    """Run demo queries through the system."""
    
    print("\n" + "="*80)
    print("TRAVEL RAG ASSISTANT DEMO")
    print("="*80)
    
    # Check if GROQ API key is set
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        print("ERROR: GROQ_API_KEY not set!")
        print("Set your Groq API key:")
        print("  export GROQ_API_KEY='your-actual-key-here'")
        print("\nGet a free key at: https://console.groq.com/")
        return
    
    # Load vector store
    vector_store = load_vector_store()
    if not vector_store:
        print("Failed to load vector store")
        return
    
    # Initialize LLM
    llm = ChatGroq(
        model=CHAT_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.3
    )
    
    # Demo queries
    demo_queries = [
        # "I want a 3-day trip to Berlin with cheap food and lots of art galleries.",
        # "Where can I find expensive restaurants in Berlin?",
        # "Tell me about free sightseeing in Berlin",
        # NOT IN DOCS
        "Best budget hotels in Bangkok for backpackers",
        "Where to find vegan food in Barcelona for under €10?",
        "Top 5 Michelin-starred restaurants in Tokyo",
        "Hiking trails near Dubai with water sports activities",
    ]
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "embedding_model": EMBED_MODEL,
            "chat_model": CHAT_MODEL,
            "vector_db": "FAISS",
            "chunk_size": CHUNK_SIZE
        },
        "demo_runs": []
    }
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n\n{'='*80}")
        print(f"DEMO QUERY {i}/{len(demo_queries)}")
        print(f"{'='*80}")
        
        result = run_travel_query(vector_store, llm, query)
        results["demo_runs"].append(result)
    
    # Save results
    output_file = os.path.join(SCRIPT_DIR, "demo_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Demo results saved to {output_file}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if "--build" in sys.argv:
        print("Building vector store...")
        build_vector_store()
    
    elif "--demo" in sys.argv:
        run_demo()
    
    else:
        print("Travel RAG Assistant")
        print("\nUsage:")
        print("  python Task.py --build    (Build vector store from travel URLs)")
        print("  python Task.py --demo     (Run demo queries)")
        print("  streamlit run app.py      (Launch interactive UI)")
        print("\nBefore running:")
        print("  1. Set GROQ_API_KEY environment variable (or in .env file)")
        print("  2. Fill in your travel URLs in travel_urls.json")
