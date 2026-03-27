"""
Game of Thrones: Answer Generation with Citations
Compares free-form vs citation-enforced answer generation
Uses: Vector semantic search + Structured JSON output
Dataset: Tuana/game-of-thrones from Hugging Face
"""

import sys
import os
import json
import re
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(SCRIPT_DIR, "chroma_db_got")
EMBED_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL = "llama3.1:8b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
K_RETRIEVE = 5


# ============================================================================
# DATASET & VECTOR STORE SETUP
# ============================================================================

def load_got_dataset():
    """Load Game of Thrones dataset from Hugging Face."""
    try:
        dataset = load_dataset("Tuana/game-of-thrones", split="train")
        print(f"Loaded Game of Thrones dataset: {len(dataset)} records")
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
        else:
            metadata["name"] = f"record_{record_id}"
    else:
        metadata["name"] = f"record_{record_id}"
    
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
    
    return metadata


def build_got_vector_store():
    """Load Game of Thrones dataset and build vector store."""
    
    print("Building vector store from Game of Thrones dataset...")
    
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
                print(f"  Processed {idx + 1} records...")
                
        except Exception as e:
            print(f"Error processing record {idx}: {e}")
            continue
    
    print(f"Loaded {len(documents)} documents from Game of Thrones dataset")
    
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
    print("Vector store built and persisted")
    return vector_store


def load_vector_store():
    """Load existing vector store from disk."""
    return Chroma(
        collection_name="game_of_thrones",
        embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL),
        persist_directory=CHROMA_DIR,
    )


# ============================================================================
# VECTOR RETRIEVAL
# ============================================================================

def vector_retrieval(vector_store, question: str, k: int = K_RETRIEVE) -> List[Document]:
    """Vector semantic search retrieval."""
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    return docs


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

def get_free_form_prompt() -> ChatPromptTemplate:
    """
    Free-form answer prompt with no citation constraints.
    Allows the LLM to answer naturally without strict citation requirements.
    """
    return ChatPromptTemplate.from_template(
        """You are a Game of Thrones expert. Answer the question using the provided context.

Context from Game of Thrones:
{context}

Question: {question}

Provide your answer in the following JSON format:
{{
    "answer": "Your comprehensive answer here",
    "citations": ["relevant_detail_1", "relevant_detail_2"],
    "confidence": 0.0
}}

Where:
- "answer": A natural, flowing answer to the question
- "citations": Key supporting details or quotes extracted from the context
- "confidence": A number between 0 and 1 indicating how confident you are (0=not confident, 1=very confident)
"""
    )


def get_citation_enforced_prompt() -> ChatPromptTemplate:
    """
    Citation-enforced answer prompt.
    Requires every statement to be backed by citations from the context.
    """
    return ChatPromptTemplate.from_template(
        """You are a Game of Thrones expert. Answer the question using ONLY the provided context, 
and EVERY statement must be supported by citations.

Context from Game of Thrones:
{context}

Question: {question}

Provide your answer in the following JSON format:
{{
    "answer": "Your answer with [CITATION_1], [CITATION_2] markers for each supporting detail",
    "citations": {{"[CITATION_1]": "exact quote from context", "[CITATION_2]": "exact quote from context"}},
    "confidence": 0.0
}}

Where:
- "answer": Answer with [CITATION_N] markers. EVERY factual claim MUST be cited.
- "citations": Dictionary mapping citation markers to exact quotes from the context
- "confidence": A number between 0 and 1

IMPORTANT: If the context doesn't contain enough information to answer confidently, 
set confidence to 0.0 and state what information is missing.
"""
    )


# ============================================================================
# JSON VALIDATION & PARSING
# ============================================================================

def validate_json_output(response: str, prompt_type: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Validate and parse JSON output from LLM.
    
    Returns:
        (parsed_dict, error_message) - Either (dict, "") for success or (None, error_msg)
    """
    
    # Try to extract JSON from response
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if not json_match:
        return None, "No JSON object found in response"
    
    try:
        parsed = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {str(e)}"
    
    # Validate required fields
    required_fields = {"answer", "citations", "confidence"}
    missing_fields = required_fields - set(parsed.keys())
    if missing_fields:
        return None, f"Missing required fields: {missing_fields}"
    
    # Validate field types
    if not isinstance(parsed["answer"], str) or len(parsed["answer"]) == 0:
        return None, "answer must be a non-empty string"
    
    if not isinstance(parsed["confidence"], (int, float)):
        return None, "confidence must be a number"
    
    if not (0 <= parsed["confidence"] <= 1):
        return None, f"confidence must be between 0 and 1, got {parsed['confidence']}"
    
    # Validate citations field based on prompt type
    if prompt_type == "free-form":
        if not isinstance(parsed["citations"], list):
            return None, "citations must be a list for free-form prompt"
        for citation in parsed["citations"]:
            if not isinstance(citation, str):
                return None, "All citations must be strings"
    
    elif prompt_type == "citation-enforced":
        if not isinstance(parsed["citations"], dict):
            return None, "citations must be a dictionary for citation-enforced prompt"
        
        # Check that answer contains citation markers
        citation_markers = set(parsed["citations"].keys())
        for marker in citation_markers:
            if marker not in parsed["answer"]:
                return None, f"Citation marker {marker} referenced but not used in answer"
    
    return parsed, ""


# ============================================================================
# ANSWER GENERATION
# ============================================================================

def generate_answer_with_validation(
    llm: ChatOllama,
    vector_store,
    question: str,
    prompt_type: str
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate structured answer with validation.
    
    Returns:
        (parsed_output, metadata)
    """
    
    # Retrieve context
    docs = vector_retrieval(vector_store, question, K_RETRIEVE)
    context = "\n---\n".join([doc.page_content for doc in docs])
    
    # Get appropriate prompt
    if prompt_type == "free-form":
        prompt = get_free_form_prompt()
    elif prompt_type == "citation-enforced":
        prompt = get_citation_enforced_prompt()
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    # Generate answer
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context, "question": question})
    
    # Validate and parse
    parsed, error_msg = validate_json_output(response, prompt_type)
    
    metadata = {
        "prompt_type": prompt_type,
        "documents_retrieved": len(docs),
        "raw_response_length": len(response),
        "valid": parsed is not None,
        "error": error_msg if error_msg else None,
        "raw_response": response
    }
    
    return parsed, metadata


# ============================================================================
# EVALUATION
# ============================================================================

def run_evaluation():
    """Comprehensive evaluation of both answer generation prompts."""
    
    print("\n" + "="*90)
    print("GAME OF THRONES: ANSWER GENERATION WITH CITATIONS")
    print("="*90)
    
    # Check if vector store exists, build if needed
    if not os.path.exists(CHROMA_DIR):
        print("Vector store not found. Building...")
        build_got_vector_store()
    
    # Load vector store
    vector_store = load_vector_store()
    if not vector_store:
        print("Failed to load vector store")
        return
    
    print("Vector store loaded successfully")
    
    # Initialize LLM
    llm = ChatOllama(model=CHAT_MODEL)
    
    # Test queries (same as in Task_Retrieval_Reranking.py)
    test_queries = [
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
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "game-of-thrones",
        "models": {
            "embedding": EMBED_MODEL,
            "chat": CHAT_MODEL
        },
        "configuration": {
            "k_retrieve": K_RETRIEVE,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP
        },
        "total_queries": len(test_queries),
        "evaluations": []
    }
    
    invalid_outputs_log = {
        "timestamp": datetime.now().isoformat(),
        "invalid_responses": []
    }
    
    for i, question in enumerate(test_queries, 1):
        print(f"\n{'─'*90}")
        print(f"Query {i}/{len(test_queries)}: {question[:75]}...")
        print(f"{'─'*90}")
        
        evaluation = {
            "query_num": i,
            "question": question,
            "responses": {}
        }
        
        # --- FREE-FORM PROMPT ---
        print(f"\n[1] FREE-FORM ANSWER")
        freeform_output, freeform_meta = generate_answer_with_validation(
            llm, vector_store, question, "free-form"
        )
        
        if freeform_output:
            print(f"    ✓ Valid JSON output")
            print(f"    Answer: {freeform_output['answer'][:150]}...")
            print(f"    Confidence: {freeform_output['confidence']:.2f}")
            print(f"    Citations: {len(freeform_output['citations'])} items")
        else:
            print(f"    ✗ Invalid output: {freeform_meta['error']}")
            invalid_outputs_log["invalid_responses"].append({
                "query_num": i,
                "question": question,
                "prompt_type": "free-form",
                "error": freeform_meta["error"],
                "raw_response_preview": freeform_meta["raw_response"][:300]
            })
            print(f"WARNING: Query {i} free-form: {freeform_meta['error']}")
        
        evaluation["responses"]["free-form"] = {
            "output": freeform_output,
            "metadata": freeform_meta
        }
        
        # --- CITATION-ENFORCED PROMPT ---
        print(f"\n[2] CITATION-ENFORCED ANSWER")
        citation_output, citation_meta = generate_answer_with_validation(
            llm, vector_store, question, "citation-enforced"
        )
        
        if citation_output:
            print(f"    ✓ Valid JSON output")
            print(f"    Answer: {citation_output['answer'][:150]}...")
            print(f"    Confidence: {citation_output['confidence']:.2f}")
            print(f"    Citations: {len(citation_output['citations'])} items")
        else:
            print(f"    ✗ Invalid output: {citation_meta['error']}")
            invalid_outputs_log["invalid_responses"].append({
                "query_num": i,
                "question": question,
                "prompt_type": "citation-enforced",
                "error": citation_meta["error"],
                "raw_response_preview": citation_meta["raw_response"][:300]
            })
            print(f"WARNING: Query {i} citation-enforced: {citation_meta['error']}")
        
        evaluation["responses"]["citation-enforced"] = {
            "output": citation_output,
            "metadata": citation_meta
        }
        
        # Compare
        print(f"\n[COMPARISON]")
        if freeform_output and citation_output:
            print(f"  Free-form confidence: {freeform_output['confidence']:.2f}")
            print(f"  Citation confidence: {citation_output['confidence']:.2f}")
            print(f"  Free-form citations: {len(freeform_output['citations'])}")
            print(f"  Citation-enforced citations: {len(citation_output['citations'])}")
        
        results["evaluations"].append(evaluation)
    
    # === SUMMARY STATISTICS ===
    print(f"\n\n{'='*90}")
    print("SUMMARY STATISTICS")
    print(f"{'='*90}\n")
    
    freeform_valid = sum([1 for e in results["evaluations"] if e["responses"]["free-form"]["output"] is not None])
    citation_valid = sum([1 for e in results["evaluations"] if e["responses"]["citation-enforced"]["output"] is not None])
    
    print(f"FREE-FORM PROMPT:")
    print(f"  Valid responses: {freeform_valid}/{len(test_queries)} ({100*freeform_valid/len(test_queries):.1f}%)")
    
    if freeform_valid > 0:
        avg_confidence = sum([e["responses"]["free-form"]["output"]["confidence"] 
                             for e in results["evaluations"] 
                             if e["responses"]["free-form"]["output"]]) / freeform_valid
        print(f"  Avg confidence: {avg_confidence:.3f}")
    
    print(f"\nCITATION-ENFORCED PROMPT:")
    print(f"  Valid responses: {citation_valid}/{len(test_queries)} ({100*citation_valid/len(test_queries):.1f}%)")
    
    if citation_valid > 0:
        avg_confidence = sum([e["responses"]["citation-enforced"]["output"]["confidence"] 
                             for e in results["evaluations"] 
                             if e["responses"]["citation-enforced"]["output"]]) / citation_valid
        print(f"  Avg confidence: {avg_confidence:.3f}")
    
    print(f"\nINVALID OUTPUTS:")
    print(f"  Total invalid: {len(invalid_outputs_log['invalid_responses'])}")
    if len(invalid_outputs_log['invalid_responses']) > 0:
        error_types = {}
        for inv in invalid_outputs_log['invalid_responses']:
            error = inv['error']
            error_types[error] = error_types.get(error, 0) + 1
        
        print(f"  Error breakdown:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"    - {error_type}: {count}")
    
    # Save results
    output_file = os.path.join(SCRIPT_DIR, "answer_generation_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")
    
    # Save invalid outputs log
    invalid_log_file = os.path.join(SCRIPT_DIR, "invalid_outputs.json")
    with open(invalid_log_file, 'w') as f:
        json.dump(invalid_outputs_log, f, indent=2)
    print(f"✓ Invalid outputs log saved to {invalid_log_file}")
    print("Evaluation complete")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if "--build" in sys.argv:
        print("Building vector store...")
        build_got_vector_store()
    elif "--evaluate" in sys.argv:
        run_evaluation()
    else:
        print("Game of Thrones: Answer Generation with Citations")
        print("\nUsage:")
        print("  python Task.py --build      (Build vector store)")
        print("  python Task.py --evaluate   (Run evaluation)")
        print("\nNote: Run --build first to initialize the vector store.")
