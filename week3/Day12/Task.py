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
    docs_retrieved: int
    answer_text: str
    confidence: float
    summary: str  # Brief summary for memory
    
    def to_memory_string(self) -> str:
        """Convert turn to compact memory string."""
        return f"Q{self.turn_num}: {self.question[:60]}... → {self.action} (conf: {self.confidence:.2f}) | {self.summary[:100]}"


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
) -> Literal["retrieve", "refuse"]:
    """Decide what action to take next."""
    
    # Check if we should retrieve documents
    # Simple heuristic: if question has enough content and topics, retrieve
    if len(question) > 10:
        return "retrieve"
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
        # Retrieve documents
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
                summary = answer_text[:100].replace("\n", " ")
                
                print(f"[ANSWER] Confidence: {confidence:.2f}")
                print(f"[ANSWER] {answer_text[:150]}...")
            else:
                action = "refuse"
                summary = f"Error: {error}"
                print(f"[REFUSE] {error}")
        else:
            action = "refuse"
            summary = "No relevant documents found"
            print(f"[REFUSE] No relevant documents found")
    
    elif action == "refuse":
        summary = "Could not process question"
        print(f"[REFUSE] Cannot process question")
    
    # Create turn record
    turn = Turn(
        turn_num=turn_num,
        question=question,
        action=action,
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
    
    # Multi-turn conversation where later queries depend on earlier context
    multi_turn_queries = [
        # Turn 1: Introduce main character
        "Who is Jon Snow?",
        
        # Turn 2: Follow-up about the same character
        "What's his relationship to the other characters mentioned earlier?",
        
        # Turn 3: Related topic
        "Tell me about the Stark family.",
        
        # Turn 4: Connect back to earlier topics
        "How does this relate to what we discussed about Jon Snow initially?",
    ]
    
    # Initialize agent memory
    memory = AgentMemory()
    
    # Track results
    results = {
        "timestamp": datetime.now().isoformat(),
        "conversation": []
    }
    
    # Run multi-turn loop
    for turn_num, question in enumerate(multi_turn_queries, 1):
        turn, memory = execute_agent_loop(llm, vector_store, turn_num, question, memory)
        results["conversation"].append(asdict(turn))
    
    # === SUMMARY ===
    print(f"\n\n{'='*80}")
    print("CONVERSATION SUMMARY")
    print(f"{'='*80}\n")
    
    print("Memory Summary:")
    print(memory.get_context_summary())
    
    print(f"\nTopics discussed: {memory.get_topics_summary()}")
    
    print(f"\nTurns completed: {len(memory.turns)}")
    
    successful_answers = sum([1 for turn in memory.turns if turn.action == "answer"])
    print(f"Successful answers: {successful_answers}/{len(memory.turns)}")
    
    if successful_answers > 0:
        avg_confidence = sum([turn.confidence for turn in memory.turns if turn.action == "answer"]) / successful_answers
        print(f"Average confidence: {avg_confidence:.2f}")
    
    # Save results
    output_file = os.path.join(SCRIPT_DIR, "agent_loop_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if "--build" in sys.argv:
        print("Building vector store...")
        build_got_vector_store()
    else:
        test_multi_turn_queries()
