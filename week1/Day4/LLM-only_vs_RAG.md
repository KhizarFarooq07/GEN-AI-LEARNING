# LLM-only vs RAG Comparison

Both using **llama-3.3-70b-versatile**, same scoring formulas (from evaluate_llms.py).

## Overall

| Metric | LLM-only (Day 3) | RAG (Day 4) | Delta |
|---|---|---|---|
| Avg Quality | **56.22** | 48.97 | -7.25 |
| Pass rate (>=70) | 0% | **8%** (2/25) | +8% |
| Avg Hallucination Risk | 20.40 | **~13** | Lower |

RAG scored **lower overall** but had **2 passes** (Q5, Q6) vs 0 for LLM-only, and lower hallucination risk across the board.

---

## Where RAG Helped (Quality Improved)

| Q# | Topic | LLM-only | RAG | Delta | Why |
|---|---|---|---|---|---|
| **Q6** | Ground truth curation | 65.79 | **75.95** | **+10.16** | Retrieved exact curation steps from doc |
| **Q2** | Eval metrics vs traditional ML | 44.42 | **54.65** | **+10.23** | Retrieved BLEU/ROUGE/Similarity sections |
| **Q11** | Retrieval vs answer quality | 59.64 | **68.65** | **+9.01** | Retrieved RAG pipeline description |
| **Q9** | Precision & recall in LLM eval | 58.40 | **65.65** | **+7.25** | Retrieved metric definitions verbatim |
| **Q12** | Domain-specific failures | 45.55 | **50.55** | **+5.00** | Retrieved failure categorization chapter |
| **Q5** | Temperature for eval | 68.02 | **71.63** | **+3.61** | Retrieved temperature=0 guidance; **passed** |

**Pattern**: RAG excelled when the question closely mapped to a specific section in the document. The retrieved chunks gave the model precise, on-topic context.

---

## Where RAG Hurt (Quality Dropped)

### PDF-based questions where RAG hurt

| Q# | Topic | LLM-only | RAG | Delta | Why |
|---|---|---|---|---|---|
| **Q19** | Rubric best practices | 63.06 | 35.59 | **-27.47** | Topic only lightly covered; model still answered but poorly |
| **Q18** | Multiple metrics purpose | 51.42 | 28.92 | **-22.50** | Said "I don't know" — topic not explicit in chunks |
| **Q13** | Ambiguous ground truth | 32.05 | 13.22 | **-18.83** | Said "I don't know" — topic absent from doc |
| **Q15** | Response latency | 60.60 | 44.26 | **-16.34** | Said "I don't know" — barely covered in doc |
| **Q20** | Model size vs performance | 51.91 | 38.44 | **-13.47** | Said "I don't know" — not in doc |
| **Q4** | Factuality vs hallucination | 64.04 | 50.12 | **-13.92** | Shorter, less complete than LLM's full knowledge |

### Non-PDF questions (Q21-Q25)

| Q# | Topic | LLM-only | RAG | Delta |
|---|---|---|---|---|
| **Q21** | Capital of Mongolia | **67.36** | 19.56 | **-47.80** |
| **Q23** | Top 3 economies by GDP | **49.23** | 20.24 | **-28.99** |
| **Q22** | Haiku about AI | **29.32** | 19.64 | **-9.68** |

RAG correctly said "I don't know" for these (honoring the system prompt), which tanked relevance/completeness scores even though it's arguably the *right behavior* for a grounded RAG system.

---

## Key Takeaways

1. **RAG improved answers when the document had strong coverage** — questions about RAG principles, evaluation metrics, temperature, and failure categories all benefited from retrieved context.

2. **RAG hurt when topics were absent or thin in the document** — the "only use context" constraint made the model refuse to answer, even when its parametric knowledge would have been correct. This caused 6 PDF-based questions to drop significantly.

3. **RAG devastated out-of-domain questions** — by design. The system prompt constrains it to retrieved context, so general knowledge questions (Mongolia, GDP, haiku) got "I don't know" responses.

4. **Hallucination risk dropped dramatically with RAG** — 0.0 on many PDF questions (vs 25.0 for LLM-only), because answers were grounded in actual retrieved text.

5. **The scoring formula penalizes RAG's conservative behavior** — saying "I don't know" gets 40.0 hallucination risk for PDF-based questions, even though it's often better than hallucinating. The evaluation framework wasn't designed with RAG's constrained behavior in mind.

**Bottom line**: RAG trades breadth for precision. It's better when the answer is *in* the documents, worse when it's not. For a production system, you'd want a fallback mechanism that lets the model use general knowledge when retrieval confidence is low.
