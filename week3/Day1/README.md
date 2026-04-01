# Travel RAG Assistant - Architecture Decisions

## 1. Embedding Model: `all-MiniLM-L6-v2`

**Why we chose it:**

- **Sentence-BERT (SBERT) based** → Purpose-built for semantic similarity, not just text encoding
- **Lightweight** (22MB) → Fast inference on CPU, no GPU needed
- **384 dimensions** → Sufficient for travel domain, good memory footprint
- **Battle-tested** → Pre-trained on millions of sentence pairs → understands semantic relationships
- **Free & offline** → No API dependencies, no rate limits, works anywhere

**Trade-off:** Not state-of-the-art quality, but excellent for prototyping and resource constraints.

**Alternatives considered:**
- `all-mpnet-base-v2`: Better quality but 20x larger (438MB)
- `BAAI/bge-small-en-v1.5`: Modern alternative, slightly better quality, only 11MB larger
- OpenAI `text-embedding-3-small`: Highest quality but requires paid API

---

## 2. Vector Database: FAISS (in-memory, local)

**Why we chose it:**

- **No external service** → Entire RAG system runs locally, no infrastructure needed
- **Fast similarity search** → Uses optimized indexing algorithms (LSH, HNSW)
- **Persistent storage** → Save/load indexes to disk between sessions
- **Simple API** → LangChain integration is trivial
- **Perfect for dev/testing** → Low latency, easy debugging

**Trade-off:** Not suitable for massive datasets (>millions of vectors) or distributed deployments.

**Alternatives considered:**
- Pinecone/Weaviate: Managed cloud, scalable but requires API
- Milvus/Qdrant: Self-hosted, more features but more complex
- PostgreSQL pgvector: Good for hybrid query + vectors

---

## 3. Observed Failure Case: Out-of-Domain Queries

### The Problem

When users query about cities **not in the training documents**, the RAG system retrieves irrelevant content but still generates answers using that bad context.

### Concrete Example

```
Query: "Best budget hotels in Bangkok for backpackers?"
Documents: Only contain Berlin travel guides

What went wrong:
✗ FAISS retrieves Berlin hotel content (highest similarity by accident)
✗ Metadata filter (city="bangkok") finds nothing
✗ Returns ALL Berlin docs as fallback
✗ LLM grounds answer in irrelevant Berlin content
✗ User gets hallucinated Bangkok advice based on Berlin data
```

### How We Detect It

Added **quality check** before answer generation:

```python
quality_status, confidence, reason = check_context_quality(docs, query, llm)
# Returns: ("insufficient", 0.92, "Query about Bangkok but all docs are about Berlin")

if quality_status == "insufficient":
    return "I don't have information about Bangkok. Please ask about [cities in docs]"
```

### Lesson Learned

**Never fully trust vector similarity.** Always:
1. Verify metadata matches user intent
2. Check context quality before answering
3. Fail gracefully with honest "I don't know" instead of hallucinating

---

## 4. Observed Failure Case: Multi-City Queries with Missing Data

### The Problem

When users query **multiple cities** where some cities are **NOT in the documents**, the system mixes results and hallucinate answers about the missing cities.

### Concrete Example

```
Query: "I want a 3-day trip to Berlin and Qatar with cheap food and lots of art galleries."
Documents: Only contain Berlin travel guides (no Qatar data)

What went wrong:
✗ Preference extraction identifies 2 cities: ["berlin", "qatar"]
✗ FAISS retrieves documents, but only Berlin docs exist
✗ Metadata filter searches for both cities, finds only Berlin
✗ LLM sees query mentions Qatar & art galleries
✗ LLM generates answer mixing:
   - Real Berlin data (grounded)
   - HALLUCINATED Qatar art gallery info (NOT grounded)
✗ User gets plausible-sounding but false Qatar recommendations
```

### Why This Happens

1. **Greedy LLM** → Generates complete answer even with partial data
2. **Weak metadata filtering** → Fallback to all docs if no matches found
3. **No city-level confidence** → Quality check passes even though half the query is hallucinated


## System Architecture

📊 **Interactive Diagram:** [View on Figma Board](https://www.figma.com/board/7NJZvLQ7jlXI5V6nWyp8EF/Travel-RAG-Architecture?node-id=0-1&p=f)

