"""
Day 4: Evaluate RAG Pipeline using test_set.json
- Runs all 25 questions through the RAG pipeline
- Scores each answer using same formulas as Day 3 (evaluate_llms.py):
  Relevance, Completeness, Hallucination Risk, Answer Quality
"""

import json
import os
import re
import chromadb
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# --- Config ---
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "Day3", "pdf_1_rag_guide.txt")
TEST_SET_FILE = os.path.join(os.path.dirname(__file__), "..", "Day3", "test_set.json")
COLLECTION_NAME = "rag_guide_eval"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
LLM_MODEL = "llama-3.3-70b-versatile"


# ---------- RAG Pipeline helpers ----------

def load_and_chunk(filepath):
    with open(filepath, "r") as f:
        text = f.read()
    lines = text.splitlines()
    cleaned = "\n".join(line for line in lines if not line.startswith("===")).strip()
    chunks, start = [], 0
    while start < len(cleaned):
        chunk = cleaned[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_vector_db(chunks):
    client = chromadb.Client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    collection.add(documents=chunks, ids=[f"chunk_{i}" for i in range(len(chunks))])
    return collection


def retrieve(collection, query):
    results = collection.query(query_texts=[query], n_results=TOP_K)
    return results["documents"][0]


def ask(client, query, context_chunks):
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


# ---------- Same scoring formulas as evaluate_llms.py ----------

def calculate_relevance_score(response, ground_truth):
    """Keyword overlap + length match. Score: 0-100"""
    if not response or not ground_truth:
        return 0.0

    truth_keywords = set(ground_truth.lower().split())
    response_keywords = set(response.lower().split())

    if len(truth_keywords) == 0:
        return 50.0

    overlap = len(truth_keywords & response_keywords)
    keyword_match = (overlap / len(truth_keywords)) * 100

    length_ratio = min(
        len(response) / max(len(ground_truth), 1),
        len(ground_truth) / max(len(response), 1),
    ) * 100

    relevance = keyword_match * 0.6 + length_ratio * 0.4
    return min(100.0, max(0.0, relevance))


def calculate_completeness_score(response, expected_keywords):
    """Coverage of expected keywords. Score: 0-100"""
    if not response or not expected_keywords:
        return 50.0

    response_lower = response.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
    return min(100.0, max(0.0, (found / len(expected_keywords)) * 100))


def calculate_hallucination_risk(response, source_type):
    """Heuristic hallucination risk. Score: 0-100 (lower is better)"""
    if not response:
        return 50.0

    risk_score = 0.0
    response_lower = response.lower()

    uncertain_phrases = [
        "i don't know", "i'm not sure", "it's not mentioned",
        "not in the document", "not specified", "outside the scope",
    ]

    if any(phrase in response_lower for phrase in uncertain_phrases):
        if source_type == "NOT in PDFs":
            risk_score = 10.0
        else:
            risk_score = 40.0
    else:
        number_count = len(re.findall(r'\b\d+\b', response))
        if number_count > 5:
            risk_score += 15
        if len(response) < 30:
            risk_score += 20
        elif len(response) > 1500:
            risk_score += 10

    return min(100.0, max(0.0, risk_score))


def check_citation_present(response, context_chunks):
    """Check if the answer references or quotes the retrieved context. Returns yes/no."""
    if not response or not context_chunks:
        return False

    response_lower = response.lower()

    for chunk in context_chunks:
        # Check if a meaningful substring (6+ words) from any chunk appears in the answer
        words = chunk.lower().split()
        for i in range(len(words) - 5):
            phrase = " ".join(words[i:i + 6])
            if phrase in response_lower:
                return True

    return False


def calculate_answer_quality(relevance, completeness, hallucination):
    """Combined quality: relevance*0.4 + completeness*0.4 + (100-hallucination)*0.2"""
    quality = relevance * 0.4 + completeness * 0.4 + (100 - hallucination) * 0.2
    return min(100.0, max(0.0, quality))


# ---------- Main evaluation ----------

def main():
    # Load test set
    with open(TEST_SET_FILE, "r") as f:
        questions = json.load(f)["test_set"]["questions"]

    # Build RAG pipeline
    print("Building RAG pipeline...")
    chunks = load_and_chunk(DATA_FILE)
    collection = build_vector_db(chunks)
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    print(f"Pipeline ready — {len(chunks)} chunks indexed\n")

    # Evaluate
    results = []
    print(f"{'Q#':<4} {'Relevance':<11} {'Complete':<10} {'Halluc':<9} {'Quality':<9} {'Pass':<6} {'Cited':<7} {'Question'}")
    print("-" * 120)

    for q in questions:
        qid = q["id"]
        question = q["question"]
        ground_truth = q["ground_truth_summary"]
        expected_keywords = q["expected_keywords"]
        source_type = q["source"]

        # RAG: retrieve + generate
        context_chunks = retrieve(collection, question)
        answer = ask(groq_client, question, context_chunks)

        # Score using same formulas as evaluate_llms.py
        relevance = calculate_relevance_score(answer, ground_truth)
        completeness = calculate_completeness_score(answer, expected_keywords)
        hallucination = calculate_hallucination_risk(answer, source_type)
        quality = calculate_answer_quality(relevance, completeness, hallucination)
        passed = quality >= 70
        citation = check_citation_present(answer, context_chunks)

        results.append({
            "id": qid,
            "question": question,
            "source": source_type,
            "answer": answer,
            "ground_truth": ground_truth,
            "relevance_score": round(relevance, 2),
            "completeness_score": round(completeness, 2),
            "hallucination_risk": round(hallucination, 2),
            "answer_quality": round(quality, 2),
            "passed_quality_gate": passed,
            "citation_present": citation,
        })

        cited_label = "yes" if citation else "no"
        print(f"{qid:<4} {relevance:<11.1f} {completeness:<10.1f} {hallucination:<9.1f} {quality:<9.1f} {'✓' if passed else '✗':<6} {cited_label:<7} {question[:50]}")

    # --- Summary ---
    all_quality = [r["answer_quality"] for r in results]
    pdf_quality = [r["answer_quality"] for r in results if r["source"] == "PDF-based"]
    non_pdf_quality = [r["answer_quality"] for r in results if r["source"] == "NOT in PDFs"]
    pass_count = sum(1 for r in results if r["passed_quality_gate"])

    citation_count = sum(1 for r in results if r["citation_present"])

    avg_all = sum(all_quality) / len(all_quality)
    avg_pdf = sum(pdf_quality) / len(pdf_quality) if pdf_quality else 0
    avg_non = sum(non_pdf_quality) / len(non_pdf_quality) if non_pdf_quality else 0

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY  (RAG Pipeline)")
    print("=" * 70)
    print(f"Model:                  {LLM_MODEL}")
    print(f"Total questions:        {len(results)}")
    print(f"Avg Quality (all):      {avg_all:.2f} / 100")
    print(f"Avg Quality (PDF):      {avg_pdf:.2f} / 100  ({len(pdf_quality)} questions)")
    print(f"Avg Quality (non-PDF):  {avg_non:.2f} / 100  ({len(non_pdf_quality)} questions)")
    print(f"Pass rate (>= 70):      {pass_count}/{len(results)} ({pass_count/len(results)*100:.1f}%)")
    print(f"Citation present:       {citation_count}/{len(results)} ({citation_count/len(results)*100:.1f}%)")
    print("=" * 70)

    # Save
    output_file = os.path.join(os.path.dirname(__file__), "rag_eval_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "evaluation_timestamp": datetime.now().isoformat(),
            "model": LLM_MODEL,
            "total_questions": len(results),
            "avg_quality": round(avg_all, 2),
            "avg_quality_pdf": round(avg_pdf, 2),
            "avg_quality_non_pdf": round(avg_non, 2),
            "pass_rate": round(pass_count / len(results) * 100, 1),
            "citation_rate": round(citation_count / len(results) * 100, 1),
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
