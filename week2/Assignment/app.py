"""
Streamlit UI for Travel RAG Assistant

Run with: streamlit run app.py
"""

import streamlit as st
import json
import os
from datetime import datetime
from typing import Dict, Any

from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Import functions from Task.py
from Task import (
    load_vector_store,
    run_travel_query,
    GROQ_API_KEY,
    CHAT_MODEL,
    EMBED_MODEL
)

# Load environment variables
load_dotenv()

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Travel RAG Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# STYLING
# ============================================================================

st.markdown("""
    <style>
    .main {
        max-width: 1200px;
    }
    .query-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .answer-box {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin-bottom: 20px;
    }
    .debug-box {
        background-color: #e7d4f5;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

st.title("✈️ Travel RAG Assistant")
st.markdown("""
Preference-aware RAG system that answers travel questions using semantic search, 
metadata filtering, re-ranking, and quality checks.

**Pattern:** Query → Preferences → Retrieval + Re-ranking → Quality Check → Answer
""")

# ============================================================================
# SIDEBAR - CONFIG & INFO
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Display config
    st.subheader("System Config")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Embedding Model", EMBED_MODEL.split("/")[-1])
    with col2:
        st.metric("Chat Model", CHAT_MODEL.split("-")[0])
    
    st.divider()
    
    # Display tech stack
    st.subheader("Tech Stack")
    st.markdown("""
    - **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
    - **Vector DB**: FAISS (in-memory, 384-dim vectors)
    - **LLM**: Groq (llama-3.1-8b-instant)
    
    **Why these choices?**
    - Small embedding model (22MB) → fast inference
    - FAISS → simple, local, no external service
    - Groq → free tier, very fast, good quality
    """)
    
    st.divider()
    
    # Help section
    st.subheader("📖 How it Works")
    st.markdown("""
    1. **Extract Preferences**: LLM parses your query for cities, budget, interests
    2. **Semantic Search**: Find top-K relevant documents using embeddings
    3. **Filter & Re-rank**: Apply metadata filters, then LLM scores relevance
    4. **Quality Check**: LLM judges if context is sufficient
    5. **Generate Answer**: LLM creates grounded answer with citations
    """)

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Initialize session state for caching results
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# Check if API key is set
if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
    st.error("❌ GROQ_API_KEY not set!")
    st.info("""
    Please set your Groq API key:
    1. Get a free key at https://console.groq.com/
    2. Add to your .env file:
       ```
       GROQ_API_KEY=your-key-here
       ```
    """)
    st.stop()

# Load vector store
@st.cache_resource
def get_vector_store():
    """Cache the vector store to avoid reloading on each run."""
    try:
        vs = load_vector_store()
        return vs
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
        st.info("Run: `python Task.py --build` to build the vector store first")
        st.stop()

# Initialize LLM
@st.cache_resource
def get_llm():
    """Cache the LLM to avoid reloading on each run."""
    return ChatGroq(
        model=CHAT_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.3
    )

vector_store = get_vector_store()
llm = get_llm()

# Query input section
st.markdown("### 🔍 Enter Your Travel Query")

# Initialize query in session state if not present
if "query_text" not in st.session_state:
    st.session_state.query_text = ""

user_query = st.text_area(
    label="What travel advice are you looking for?",
    placeholder="E.g., '3-day Berlin trip with cheap food and art galleries'",
    height=100,
    value=st.session_state.query_text
)

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    run_button = st.button("🚀 Get Answer", use_container_width=True)
with col2:
    clear_button = st.button("🔄 Clear", use_container_width=True)
with col3:
    st.write("")  # Spacer

if clear_button:
    st.session_state.last_query = None
    st.session_state.last_result = None
    st.rerun()

# Process query
if run_button and user_query:
    if user_query == st.session_state.last_query and st.session_state.last_result:
        # Use cached result
        result = st.session_state.last_result
    else:
        # Run the pipeline
        with st.spinner("🔄 Processing your query..."):
            result = run_travel_query(vector_store, llm, user_query)
            st.session_state.last_query = user_query
            st.session_state.last_result = result
    
    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    
    st.divider()
    
    # Main Answer Section
    st.markdown("### 📝 Answer")
    
    answer_data = result.get("answer", {})
    
    if answer_data.get("valid"):
        # Display answer
        answer_text = answer_data.get("answer", "No answer generated")
        st.markdown(f"""
        <div class="answer-box">
        {answer_text}
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence and status
        col1, col2, col3 = st.columns(3)
        with col1:
            confidence = answer_data.get("confidence", 0)
            color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
            st.metric("Confidence", f"{confidence:.2f}", delta=f"{color}")
        
        with col2:
            quality = answer_data.get("quality_status", "unknown")
            st.metric("Context Quality", quality.title())
        
        with col3:
            st.metric("Generated", "✓")
    else:
        st.error(f"❌ Failed to generate valid answer: {answer_data.get('error', 'Unknown error')}")
    
    st.divider()
    
    # Debug Panel with Tabs
    st.markdown("### 🔧 Debug Panel")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📍 Preferences", "📚 Retrieved Chunks", "🌐 URLs Used", "⚖️ Quality Check", "📊 Raw JSON"]
    )
    
    # Tab 1: Extracted Preferences
    with tab1:
        preferences = result.get("steps", {}).get("preferences", {})
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Cities", ", ".join(preferences.get("cities", [])) or "Any")
            st.metric("Budget", preferences.get("budget", "N/A").title())
            st.metric("Duration", f"{preferences.get('duration_days', 'N/A')} days")
        
        with col2:
            interests = preferences.get("interests", [])
            st.metric("Interests", ", ".join(interests) if interests else "General")
            keywords = preferences.get("other_keywords", [])
            st.metric("Keywords", ", ".join(keywords) if keywords else "None")
        
        st.json(preferences)
    
    # Tab 2: Retrieved Chunks
    with tab2:
        retrieval_meta = result.get("steps", {}).get("retrieval", {})
        
        # Retrieval stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Initial Retrieved", retrieval_meta.get("initial_retrieved", 0))
        with col2:
            st.metric("After Filter", retrieval_meta.get("after_filter", 0))
        with col3:
            st.metric("Re-ranked", retrieval_meta.get("final_returned", 0))
        with col4:
            cities = retrieval_meta.get("cities", [])
            st.metric("Cities Found", ", ".join(cities) if cities else "None")
        
        st.subheader("Top Retrieved Chunks")
        
        # Get documents from retrieval metadata
        documents = retrieval_meta.get("documents", [])
        
        if documents:
            # Show each chunk
            for i, doc in enumerate(documents, 1):
                with st.expander(f"🔍 Chunk {i} - {doc['metadata'].get('city', 'Unknown').title()} ({doc['metadata'].get('category', 'general').title()})"):
                    # Show metadata
                    st.markdown(f"**Source:** [{doc['metadata'].get('url', 'Unknown')}]({doc['metadata'].get('url', '')})")
                    st.markdown(f"**City:** {doc['metadata'].get('city', 'Unknown').title()}")
                    st.markdown(f"**Category:** {doc['metadata'].get('category', 'Unknown').title()}")
                    st.markdown(f"**Price Level:** {doc['metadata'].get('price_level', 'Unknown').title()}")
                    st.divider()
                    # Show content preview
                    st.markdown("**Content Preview:**")
                    st.write(doc['content'])
        else:
            st.info("No chunks found in retrieval results.")
        
        st.info("💡 Tip: These chunks were retrieved, filtered, and re-ranked based on your preferences.")
    
    # Tab 3: URLs Used
    with tab3:
        urls_used = retrieval_meta.get("urls_used", [])
        
        if urls_used:
            st.subheader("Source URLs")
            for i, url in enumerate(urls_used, 1):
                st.markdown(f"{i}. [{url}]({url})")
        else:
            st.info("No URLs were used in the final answer.")
    
    # Tab 4: Quality Check
    with tab4:
        quality_check = result.get("steps", {}).get("quality_check", {})
        
        col1, col2 = st.columns(2)
        with col1:
            status = quality_check.get("status", "unknown")
            color = "🟢" if status == "good" else "🔴"
            st.metric("Quality Status", f"{color} {status.title()}")
        
        with col2:
            conf = quality_check.get("confidence", 0)
            st.metric("Confidence", f"{conf:.2f}")
        
        reason = quality_check.get("reason", "N/A")
        st.write(f"**Reason:** {reason}")
    
    # Tab 5: Raw JSON
    with tab5:
        st.json(result)
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        timestamp = result.get("timestamp", "N/A")
        st.caption(f"⏱️ Generated: {timestamp}")
    
    with col2:
        st.caption("💾 Results auto-saved to demo_results.json")
    
    with col3:
        if st.button("📥 Download Results", use_container_width=True):
            st.download_button(
                label="Download JSON",
                data=json.dumps(result, indent=2),
                file_name=f"travel_rag_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

elif run_button:
    st.warning("⚠️ Please enter a query first!")

# ============================================================================
# EXAMPLE QUERIES
# ============================================================================

st.divider()

st.markdown("### 💡 Example Queries")

example_queries = [
    "I want a 3-day trip to Berlin with cheap food and lots of art galleries.",
    "Where can I find budget-friendly sightseeing in Paris?",
    "Tell me about museums in Amsterdam",
    "Best affordable places to visit in Barcelona",
]

cols = st.columns(2)
for i, query in enumerate(example_queries):
    with cols[i % 2]:
        if st.button(f'"{query}"', use_container_width=True, key=f"example_{i}"):
            st.session_state.query_text = query
            st.rerun()

# ============================================================================
# FOOTER INFO
# ============================================================================

st.divider()

st.markdown("""
---
**Travel RAG Assistant** | Assignment 2 - Preference-Aware Travel RAG

**Stack:**
- Embeddings: Sentence-Transformers (all-MiniLM-L6-v2)
- Vector DB: FAISS
- LLM: Groq (llama-3.1-8b-instant)
- UI: Streamlit

**Core Capabilities:**
1. ✅ Source Ingestion (Web scraping + chunking)
2. ✅ Preference Extraction (LLM-based parsing)
3. ✅ Retrieval + Re-ranking (Semantic + metadata + LLM judge)
4. ✅ Context Quality Check (LLM quality assessment)
5. ✅ Answer Generation (Grounded + cited)

[GitHub](https://github.com) | [Documentation](README.md)
""")
