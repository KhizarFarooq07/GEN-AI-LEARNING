"""
Game of Thrones: Agent Loop with Memory
Simple agent that decides actions (retrieve/tool/answer/refuse) for multi-turn queries.
Maintains lightweight memory of conversation history.
"""

import os
import sys
import json
import re
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional, Literal
from dataclasses import dataclass, asdict
from collections import deque

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
MAX_MEMORY_TURNS = 4  # Keep last N turns in memory


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Turn:
    """Single turn in the conversation."""
    turn_num: int
    question: str
    action: Literal["retrieve", "tool", "answer", "refuse"]
    source: Literal["vector_store", "memory", "refused"]  # Where did the answer come from?
    docs_retrieved: int
    answer_text: str
    confidence: float
    summary: str  # Brief summary for memory
    
    def to_memory_string(self) -> str:
        """Convert turn to compact memory string."""
        return f"Q{self.turn_num}: {self.question[:60]}... → {self.action}[{self.source}] (conf: {self.confidence:.2f}) | {self.summary[:100]}"


@dataclass
class AgentMemory:
    """Lightweight agent memory tracking conversation history."""
    turns: deque  # deque of Turn objects (max MAX_MEMORY_TURNS)
    topics: List[str]  # Topics mentioned so far
    
    def __init__(self):
        self.turns = deque(maxlen=MAX_MEMORY_TURNS)
        self.topics = []
    
    def add_turn(self, turn: Turn):
        """Add a new turn to memory."""
        self.turns.append(turn)
        # Extract topics from question
        topics = extract_topics(turn.question)
        self.topics.extend(topics)
        self.topics = list(set(self.topics))  # Deduplicate
    
    def get_context_summary(self) -> str:
        """Get summary of conversation so far."""
        if not self.turns:
            return "No previous context."
        
        summaries = [turn.to_memory_string() for turn in self.turns]
        return "\n".join(summaries)
    
    def get_topics_summary(self) -> str:
        """Get summary of topics discussed."""
        if not self.topics:
            return "No topics yet."
        return ", ".join(self.topics[:5])


# ============================================================================
# VECTOR STORE SETUP
# ============================================================================

def build_got_vector_store():
    """Load Game of Thrones dataset and build vector store."""
    print("Building vector store from Game of Thrones dataset...")
    
    try:
        dataset = load_dataset("Tuana/game-of-thrones", split="train")
        print(f"Loaded {len(dataset)} records")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # Process into Document objects
    documents = []
    for idx, record in enumerate(dataset):
        try:
            if "content" in record and record["content"] and len(record["content"]) >= 100:
                metadata = {
                    "source": "game-of-thrones",
                    "record_id": idx,
                }
                doc = Document(page_content=str(record["content"]), metadata=metadata)
                documents.append(doc)
                
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1} records...")
        except Exception as e:
            continue
    
    print(f"Loaded {len(documents)} documents")
    
    # Chunk documents
    print("Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    
    # Build vector store
    print("Building vector store...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=HuggingFaceEmbeddings(model_name=EMBED_MODEL),
        collection_name="game_of_thrones",
        persist_directory=CHROMA_DIR,
    )
    print("Vector store ready")
    return vector_store


def load_vector_store():
    """Load existing vector store."""
    return Chroma(
        collection_name="game_of_thrones",
        embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL),
        persist_directory=CHROMA_DIR,
    )


# ============================================================================
# RETRIEVAL & ANSWER GENERATION
# ============================================================================

def retrieve_docs(vector_store, query: str, k: int = K_RETRIEVE) -> List[Document]:
    """Retrieve relevant documents."""
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def extract_topics(text: str) -> List[str]:
    """Extract key topics/entities from text (simple heuristic)."""
    topics = []
    keywords = [
        "jon snow", "daenerys", "stark", "lannister", "iron throne",
        "battle", "dragon", "white walker", "king's landing", "winterfell",
        "ned", "cersei", "tyrion", "arya", "sansa"
    ]
    text_lower = text.lower()
    for keyword in keywords:
        if keyword in text_lower:
            topics.append(keyword)
    return topics


def generate_structured_answer(
    llm: ChatOllama,
    question: str,
    context: str,
    memory_summary: str
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Generate structured JSON answer. Returns (response_dict, error_msg)."""
    
    prompt = ChatPromptTemplate.from_template(
        """You are a Game of Thrones expert. Answer the question using the provided context.

Previous conversation context:
{memory_summary}

Current context from Game of Thrones:
{context}

Question: {question}

Provide your answer in JSON format:
{{
    "answer": "Your answer here",
    "confidence": 0.85,
    "key_topics": ["topic1", "topic2"]
}}

- confidence: 0.0 to 1.0
- key_topics: Main topics mentioned in your answer
"""
    )
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "memory_summary": memory_summary,
        "context": context,
        "question": question
    })
    
    # Parse JSON
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            return {}, "No JSON found in response"
        
        parsed = json.loads(json_match.group())
        
        # Validate
        if "answer" not in parsed or "confidence" not in parsed:
            return {}, "Missing required fields"
        
        if not isinstance(parsed["confidence"], (int, float)) or not (0 <= parsed["confidence"] <= 1):
            return {}, "Invalid confidence value"
        
        return parsed, None
    except json.JSONDecodeError as e:
        return {}, f"JSON parse error: {str(e)}"


# ============================================================================
# AGENT LOOP
# ============================================================================

def decide_action(
    question: str,
    memory: AgentMemory,
    vector_store,
    llm: ChatOllama
) -> Literal["retrieve", "tool", "answer", "refuse"]:
    """
    Decide what action to take next based on multiple factors.
    
    Actions:
    - retrieve: Search vector store for relevant documents (new knowledge)
    - answer: Use memory context directly (already discussed)
    - tool: Would require external tool (future enhancement)
    - refuse: Cannot process question
    """
    
    # ===== Rule 1: Question Quality =====
    question_length = len(question.strip())
    if question_length < 5:
        return "refuse"  # Too short to be meaningful
    
    # ===== Rule 2: Memory Context Check =====
    question_lower = question.lower()
    topics_in_question = extract_topics(question)
    
    # Check if question references previous discussion
    if any(word in question_lower for word in ["earlier", "mentioned", "discussed", "before", "initially", "previously"]):
        # Question is explicitly referencing previous context
        if len(memory.turns) > 0 and len(memory.topics) > 0:
            return "answer"  # Likely answerable from memory
    
    # Check if question asks about already-discussed topics with high confidence
    if topics_in_question:
        matching_topics = [t for t in memory.topics if any(topic in t for topic in topics_in_question)]
        if matching_topics:
            # Topics were discussed before
            # Check if previous turns had high confidence
            previous_confidences = [turn.confidence for turn in memory.turns 
                                   if turn.action == "answer" and any(t in turn.summary.lower() for t in matching_topics)]
            if previous_confidences and sum(previous_confidences) / len(previous_confidences) > 0.85:
                return "answer"  # High confidence knowledge, use memory
    
    # ===== Rule 3: Question Characteristics =====
    # Check if it's asking for definitions, relationships, connections
    question_keywords = ["what is", "who is", "describe", "tell me about", "explain", 
                        "how does", "what about", "relationship"]
    is_definition_query = any(keyword in question_lower for keyword in question_keywords)
    
    # ===== Rule 4: Multi-topic correlation (reference to multiple known things) =====
    if "relate" in question_lower or "connection" in question_lower or "compare" in question_lower:
        if len(memory.topics) > 1:
            return "retrieve"  # Need fresh data to make connections
    
    # ===== Rule 5: Default - Retrieve for well-formed questions =====
    if question_length > 10 and is_definition_query:
        return "retrieve"  # Standard case: retrieve from vector store
    
    # ===== Rule 6: Fallback =====
    return "refuse"


def execute_agent_loop(
    llm: ChatOllama,
    vector_store,
    turn_num: int,
    question: str,
    memory: AgentMemory
) -> Tuple[Turn, AgentMemory]:
    """
    Execute single turn of agent loop:
    1. Decide action
    2. Execute action
    3. Record in memory
    """
    
    print(f"\n{'='*80}")
    print(f"TURN {turn_num}: {question}")
    print(f"{'='*80}")
    
    # Get memory context
    memory_summary = memory.get_context_summary()
    
    # Decide action
    action = decide_action(question, memory, vector_store, llm)
    print(f"[ACTION] {action.upper()}")
    
    answer_text = ""
    confidence = 0.0
    docs_retrieved = 0
    summary = ""
    
    if action == "retrieve":
        # Retrieve documents from vector store
        print(f"[RETRIEVE] Searching knowledge base...")
        docs = retrieve_docs(vector_store, question, K_RETRIEVE)
        docs_retrieved = len(docs)
        
        if docs:
            context = "\n---\n".join([doc.page_content[:300] for doc in docs])
            print(f"[RETRIEVE] Found {docs_retrieved} documents")
            
            # Generate answer
            print(f"[ANSWER] Generating response...")
            response, error = generate_structured_answer(llm, question, context, memory_summary)
            
            if not error:
                answer_text = response.get("answer", "")
                confidence = response.get("confidence", 0.0)
                action = "answer"
                source = "vector_store"  # ← NEW: Track source
                summary = answer_text[:100].replace("\n", " ")
                
                print(f"[ANSWER] Confidence: {confidence:.2f}")
                print(f"[ANSWER] {answer_text[:150]}...")
            else:
                action = "refuse"
                source = "refused"  # ← NEW
                summary = f"Error: {error}"
                print(f"[REFUSE] {error}")
        else:
            action = "refuse"
            source = "refused"  # ← NEW
            summary = "No relevant documents found"
            print(f"[REFUSE] No relevant documents found")
    
    elif action == "answer":
        # Answer using memory context only (no new retrieval)
        print(f"[ANSWER_FROM_MEMORY] Using conversation context...")
        print(f"[MEMORY] Context: {memory_summary[:200]}...")
        
        # Use only memory context, no retrieval
        context = f"Previous conversation context provides sufficient information."
        response, error = generate_structured_answer(llm, question, context, memory_summary)
        
        if not error:
            answer_text = response.get("answer", "")
            confidence = response.get("confidence", 0.0)
            source = "memory"  # ← NEW: Track source
            summary = answer_text[:100].replace("\n", " ")
            
            print(f"[ANSWER_FROM_MEMORY] Confidence: {confidence:.2f}")
            print(f"[ANSWER_FROM_MEMORY] {answer_text[:150]}...")
        else:
            action = "refuse"
            source = "refused"  # ← NEW
            summary = f"Error: {error}"
            print(f"[REFUSE] {error}")
        
        docs_retrieved = 0
    
    elif action == "tool":
        # Placeholder for tool use (external APIs, calculations, etc.)
        print(f"[TOOL] Would use external tool/capability")
        print(f"[TOOL] Not yet implemented")
        action = "refuse"
        source = "refused"  # ← NEW
        summary = "Tool capability not implemented"
    
    elif action == "refuse":
        source = "refused"  # ← NEW
        summary = "Could not process question"
        print(f"[REFUSE] Cannot process question")
    
    # Create turn record
    turn = Turn(
        turn_num=turn_num,
        question=question,
        action=action,
        source=source,  # ← NEW: Include source field
        docs_retrieved=docs_retrieved,
        answer_text=answer_text,
        confidence=confidence,
        summary=summary
    )
    
    # Add to memory
    memory.add_turn(turn)
    
    print(f"[MEMORY] Topics discussed: {memory.get_topics_summary()}")
    
    return turn, memory


# ============================================================================
# TESTING: MULTI-TURN CONVERSATION
# ============================================================================

def test_multi_turn_queries():
    """Test agent with multi-turn queries where context matters."""
    
    print("\n" + "="*80)
    print("GAME OF THRONES: AGENT LOOP WITH MEMORY")
    print("="*80)
    
    # Load vector store
    if not os.path.exists(CHROMA_DIR):
        print("\nVector store not found. Building...")
        build_got_vector_store()
    
    vector_store = load_vector_store()
    if not vector_store:
        print("Failed to load vector store")
        return
    
    print("Vector store loaded ✓")
    
    # Initialize LLM
    llm = ChatOllama(model=CHAT_MODEL)
    
    # 5 multi-turn test scenarios: 3 correct decisions + 2 edge cases with memory errors
    # NOTE: decisions now include source: "answer[vector_store]", "answer[memory]", or "refuse[refused]"
    
    test_scenarios = [
        {
            "name": "Scenario 1: Straight Forward Memory Usage",
            "description": "Good case: Agent correctly uses memory for follow-up questions",
            "queries": [
                "Who is Jon Snow?",
                "What's his relationship to the other characters mentioned earlier?",
                "Tell me about House Stark.",
                "How does Jon connect to this house we just discussed?",
            ],
            "expected_decisions": ["answer[vector_store]", "answer[memory]", "answer[vector_store]", "answer[memory]"],
            "expected_docs_retrieved": [5, 0, 5, 0],
            "expected_outcome": "✅ CORRECT - Alternates between retrieval & memory"
        },
        {
            "name": "Scenario 2: Accumulating Context Over Turns",
            "description": "Good case: Memory accumulates and informs decisions",
            "queries": [
                "Who is Daenerys Targaryen?",
                "What dragons did she have?",
                "Tell me about the Targaryen family.",
                "How were the Targaryens connected to dragons initially?",
            ],
            "expected_decisions": ["answer[vector_store]", "answer[vector_store]", "answer[vector_store]", "answer[memory]"],
            "expected_docs_retrieved": [5, 5, 5, 0],
            "expected_outcome": "✅ CORRECT - Topics accumulate, later questions use memory"
        },
        {
            "name": "Scenario 3: Explicit Memory References",
            "description": "Good case: Agent detects explicit references to previous discussions",
            "queries": [
                "Who rules the Seven Kingdoms?",
                "What was discussed earlier about rulers?",
                "Compare this to what we initially mentioned.",
                "Has anything we discussed before apply here?",
            ],
            "expected_decisions": ["answer[vector_store]", "answer[memory]", "answer[memory]", "answer[memory]"],
            "expected_docs_retrieved": [5, 0, 0, 0],
            "expected_outcome": "✅ CORRECT - Keywords trigger memory-only responses"
        },
        {
            "name": "Scenario 4: Memory False Positive (Edge Case)",
            "description": "⚠️ EDGE CASE: Agent reuses memory for DIFFERENT entity with same name",
            "queries": [
                "Who is Jon Snow?",
                "What about Jon the blacksmith? (different Jon)",
                "Tell me about northern characters.",
                "Does Jon the blacksmith relate to the north?",
            ],
            "expected_decisions": ["answer[vector_store]", "answer[vector_store]", "answer[vector_store]", "refuse[refused]"],
            "expected_docs_retrieved": [5, 5, 5, 0],
            "expected_outcome": "⚠️ SHOWS LIMITATION - Same entity name causes confusion"
        },
        {
            "name": "Scenario 5: Memory Overgeneralization (Edge Case)",
            "description": "⚠️ EDGE CASE: High confidence on one query type transfers to different types",
            "queries": [
                "Who is Tyrion Lannister?",
                "Is Tyrion evil?",
                "What are Lannister family values?",
                "Would you say Tyrion fits these values?",
            ],
            "expected_decisions": ["answer[vector_store]", "answer[memory]", "answer[memory]", "answer[memory]"],
            "expected_docs_retrieved": [5, 0, 0, 0],
            "expected_outcome": "⚠️ SHOWS LIMITATION - High confidence overgeneralizes across question types"
        }
    ]
    
    # Run all scenarios
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "scenarios": []
    }
    
    # Initialize agent memory
    memory = AgentMemory()
    
    # Track results
    results = {
        "timestamp": datetime.now().isoformat(),
        "conversation": []
    }
    
    # Run all scenarios
    scenario_results = []
    
    for scenario_idx, scenario in enumerate(test_scenarios, 1):
        print(f"\n\n{'#'*80}")
        print(f"# TEST SCENARIO {scenario_idx}: {scenario['name']}")
        print(f"{'#'*80}")
        print(f"Description: {scenario['description']}")
        print(f"Expected Outcome: {scenario['expected_outcome']}\n")
        
        # Reset memory for new scenario
        memory = AgentMemory()
        scenario_conversation = []
        
        # Run each query in the scenario
        for turn_num, question in enumerate(scenario['queries'], 1):
            turn, memory = execute_agent_loop(llm, vector_store, turn_num, question, memory)
            scenario_conversation.append(asdict(turn))
        
        # Analyze scenario
        actual_decisions = [f"{turn.action}[{turn.source}]" for turn in memory.turns]
        actual_docs = [turn.docs_retrieved for turn in memory.turns]
        decisions_match = actual_decisions == scenario['expected_decisions']
        
        print(f"\n--- SCENARIO ANALYSIS ---")
        print(f"Expected decisions: {scenario['expected_decisions']}")
        print(f"Actual decisions:   {actual_decisions}")
        print(f"Expected docs:      {scenario['expected_docs_retrieved']}")
        print(f"Actual docs:        {actual_docs}")
        print(f"Match: {'✅ YES' if decisions_match else '❌ MISMATCH'}")
        
        scenario_results.append({
            "scenario_num": scenario_idx,
            "name": scenario['name'],
            "description": scenario['description'],
            "expected_outcome": scenario['expected_outcome'],
            "expected_decisions": scenario['expected_decisions'],
            "actual_decisions": actual_decisions,
            "expected_docs_retrieved": scenario['expected_docs_retrieved'],
            "actual_docs_retrieved": actual_docs,
            "match": decisions_match,
            "conversation": scenario_conversation,
            "queries": scenario['queries']
        })
    
    # === SUMMARY ===
    print(f"\n\n{'='*80}")
    print("COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*80}\n")
    
    # === SUMMARY ===
    print(f"\n\n{'='*80}")
    print("COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*80}\n")
    
    correct_scenarios = sum([1 for s in scenario_results if "✅" in s['expected_outcome']])
    edge_case_scenarios = sum([1 for s in scenario_results if "⚠️" in s['expected_outcome']])
    
    matching_scenarios = sum([1 for s in scenario_results if s['match']])
    mismatching_scenarios = sum([1 for s in scenario_results if not s['match']])
    
    print(f"Test Scenarios: {len(scenario_results)}")
    print(f"  ✅ Correct scenarios (expected to pass): {correct_scenarios}")
    print(f"  ⚠️  Edge case scenarios (showing limitations): {edge_case_scenarios}")
    print()
    print(f"Decision Matching: {matching_scenarios}/{len(scenario_results)}")
    print(f"  ✅ Passed (decisions match expected): {matching_scenarios}")
    print(f"  ❌ Failed (decisions differ from expected): {mismatching_scenarios}")
    
    # Save to JSON
    output_file = os.path.join(SCRIPT_DIR, "agent_loop_results.json")
    with open(output_file, 'w') as f:
        json.dump(scenario_results, f, indent=2)
    print(f"\n✓ JSON results saved to {output_file}")
    
    return scenario_results


# ============================================================================
# TESTING: MULTI-TURN CONVERSATION
# ============================================================================

def test_multi_turn_queries_old():
    """Old test function - replaced by comprehensive scenario testing."""


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if "--build" in sys.argv:
        print("Building vector store...")
        build_got_vector_store()
    else:
        test_multi_turn_queries()
