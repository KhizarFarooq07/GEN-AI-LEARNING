"""
Game of Thrones: ReAct Agent with Llama Index
Uses Llama Index's ReAct agent framework with tools for retrieval and reasoning.
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque

from datasets import load_dataset
from llama_index.core import Document, Settings
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import ReActAgent, AgentStream, ToolCallResult
from llama_index.core.workflow import Context
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(SCRIPT_DIR, "chroma_db_got")
EMBED_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL = "llama3.1:8b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
K_RETRIEVE = 5
MAX_MEMORY_TURNS = 4


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Turn:
    """Single turn in the conversation."""
    turn_num: int
    question: str
    action: str  # retrieve, tool, answer, refuse
    source: str  # vector_store, memory, tool, refused
    docs_retrieved: int
    answer_text: str
    confidence: float
    summary: str
    reasoning: str  # Agent's reasoning
    
    def to_memory_string(self) -> str:
        return f"Q{self.turn_num}: {self.question[:60]}... → {self.action}[{self.source}] (conf: {self.confidence:.2f}) | {self.summary[:100]}"


@dataclass
class ConversationMemory:
    """Lightweight memory for conversation history."""
    turns: deque
    topics: List[str]
    
    def __init__(self):
        self.turns = deque(maxlen=MAX_MEMORY_TURNS)
        self.topics = []
    
    def add_turn(self, turn: Turn):
        self.turns.append(turn)
        topics = extract_topics(turn.question)
        self.topics.extend(topics)
        self.topics = list(set(self.topics))
    
    def get_context_summary(self) -> str:
        if not self.turns:
            return "No previous context."
        summaries = [turn.to_memory_string() for turn in self.turns]
        return "\n".join(summaries)
    
    def get_topics_summary(self) -> str:
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


def retrieve_docs(vector_store, query: str, k: int = K_RETRIEVE):
    """Retrieve relevant documents from vector store."""
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    return docs


def extract_topics(text: str) -> List[str]:
    """Extract key topics/entities from text."""
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


# ============================================================================
# TOOL FUNCTIONS
# ============================================================================

# Global references for tools
_vector_store = None
_memory = None


def retrieve_from_knowledge_base(query: str = "") -> str:
    """Retrieve documents from Game of Thrones knowledge base."""
    if _vector_store is None:
        return "Knowledge base not available."
    
    docs = retrieve_docs(_vector_store, query if query else "Game of Thrones", K_RETRIEVE)
    if not docs:
        return "No relevant documents found in knowledge base."
    
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"[Document {i}]:\n{doc.page_content[:300]}")
    
    return "\n---\n".join(context_parts)


def consult_conversation_memory() -> str:
    """Consult previous conversation history and discussed topics."""
    if _memory is None:
        return "No memory available."
    
    return _memory.get_context_summary()


def calculate_confidence(text: str) -> float:
    """
    Calculate confidence score (simple heuristic).
    
    Args:
        text: Response text
    
    Returns:
        Confidence score 0-1
    """
    if not text:
        return 0.0
    
    # Simple heuristics: longer, more detailed responses = higher confidence
    word_count = len(text.split())
    if word_count > 100:
        return 0.85
    elif word_count > 50:
        return 0.70
    elif word_count > 20:
        return 0.60
    else:
        return 0.40


# ============================================================================
# REACT AGENT WITH TOOLS
# ============================================================================

def create_react_agent(vector_store, memory: ConversationMemory):
    """
    Create a Llama Index ReAct agent with tools for retrieval and memory consultation.
    
    Args:
        vector_store: Vector store for retrieval
        memory: Conversation memory
    
    Returns:
        ReActAgent configured with tools
    """
    
    # Store global references for tool functions
    global _vector_store, _memory
    _vector_store = vector_store
    _memory = memory
    
    # Setup LLM
    llm = Ollama(model=CHAT_MODEL, request_timeout=120.0)
    
    # Setup embedding model
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    # System prompt to constrain agent behavior
    system_prompt = """You are an expert Game of Thrones assistant. You MUST follow these rules strictly:

1. ONLY respond based on information from the Game of Thrones knowledge base, retrieved documents, or previous conversation memory.
2. Use the following strategy:
   a) First, check the "consult_memory" tool to see if relevant information from previous turns can help answer the question.
   b) Then use the "retrieve_knowledge_base" tool to search for relevant documents about the topic.
   c) Combine information from BOTH sources (memory + documents) when available.
3. If NEITHER the knowledge base NOR memory contain relevant information, respond with: "I don't know - this information is not available in my knowledge base or conversation history."
4. NEVER make up information, guesses, or use general knowledge not in retrieved documents or memory.
5. Always cite the source of your information (whether it's from documents or previous conversation).
6. Be honest about the limitations of your knowledge - if information is incomplete or missing, state that clearly.
7. If a question builds on previous context, prioritize memory to show conversation continuity.

Remember: It is better to say "I don't know" than to provide incorrect information."""
    
    # Define tools
    tools = [
        FunctionTool.from_defaults(
            fn=retrieve_from_knowledge_base,
            name="retrieve_knowledge_base",
            description="Search the Game of Thrones knowledge base for information about characters, events, houses, and relationships. MUST be called before answering any question.",
        ),
        FunctionTool.from_defaults(
            fn=consult_conversation_memory,
            name="consult_memory",
            description="Check previous conversation history and discussed topics to understand context from earlier turns",
        ),
    ]
    
    # Create ReAct agent with system prompt
    agent = ReActAgent(
        tools=tools, 
        llm=llm, 
        verbose=True, 
        max_iterations=5,
        system_prompt=system_prompt
    )
    
    return agent


# ============================================================================
# AGENT EXECUTION
# ============================================================================

async def execute_agent_turn_async(
    agent,
    question: str,
    turn_num: int,
    memory: ConversationMemory,
    vector_store
) -> tuple[Turn, ConversationMemory]:
    """Execute a single turn of the ReAct agent asynchronously."""
    
    print(f"\n{'='*80}")
    print(f"TURN {turn_num}: {question}")
    print(f"{'='*80}")
    
    try:
        # Create context for this agent run
        ctx = Context(agent)
        
        # Run the agent asynchronously
        handler = agent.run(question, ctx=ctx)
        
        # Collect output
        output = []
        async for ev in handler.stream_events():
            if isinstance(ev, AgentStream):
                output.append(ev.delta)
                print(ev.delta, end="", flush=True)
        
        # Get final response
        response = await handler
        
        answer_text = str(response)
        confidence = calculate_confidence(answer_text)
        
        action = "answer"
        source = "vector_store"
        docs_retrieved = K_RETRIEVE
        reasoning = ""
        
        summary = answer_text[:100].replace("\n", " ")
        
        print(f"\n[ANSWER] Confidence: {confidence:.2f}")
        print(f"[ANSWER] {answer_text[:200]}...\n")
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        action = "refuse"
        source = "refused"
        answer_text = f"Error: {str(e)}"
        confidence = 0.0
        docs_retrieved = 0
        summary = "Error occurred"
        reasoning = str(e)
    
    # Create turn record
    turn = Turn(
        turn_num=turn_num,
        question=question,
        action=action,
        source=source,
        docs_retrieved=docs_retrieved,
        answer_text=answer_text,
        confidence=confidence,
        summary=summary,
        reasoning=reasoning
    )
    
    # Update memory
    memory.add_turn(turn)
    
    print(f"[MEMORY] Topics discussed: {memory.get_topics_summary()}")
    
    return turn, memory


# ============================================================================
# TEST SCENARIOS
# ============================================================================

async def test_react_agent_async():
    """Test ReAct agent with multi-turn Game of Thrones queries."""
    
    print("\n" + "="*80)
    print("GAME OF THRONES: REACT AGENT WITH LLAMA INDEX")
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
    
    # Initialize memory
    memory = ConversationMemory()
    
    # Create agent
    print("\nInitializing ReAct agent...")
    agent = create_react_agent(vector_store, memory)
    print("Agent ready ✓\n")
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Scenario 1: Character Information",
            "queries": [
                "Who is Jon Snow?",
                "What houses is he connected to?",
                "Tell me about the Stark family.",
            ]
        },
        {
            "name": "Scenario 2: Dynasty Knowledge",
            "queries": [
                "Who is Daenerys Targaryen?",
                "What dragons did she have?",
                "Describe the Targaryen dynasty.",
            ]
        },
        {
            "name": "Scenario 3: Relationships and Conflicts",
            "queries": [
                "Who rules the Seven Kingdoms?",
                "What conflicts existed between major houses?",
                "How do the Lannisters relate to other houses?",
            ]
        },
    ]
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "scenarios": []
    }
    
    # Run scenarios
    for scenario_idx, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'#'*80}")
        print(f"# TEST SCENARIO {scenario_idx}: {scenario['name']}")
        print(f"{'#'*80}\n")
        
        # Reset memory for each scenario
        memory = ConversationMemory()
        agent = create_react_agent(vector_store, memory)
        
        scenario_conversation = []
        
        # Run each query asynchronously
        for turn_num, question in enumerate(scenario['queries'], 1):
            turn, memory = await execute_agent_turn_async(agent, question, turn_num, memory, vector_store)
            scenario_conversation.append(asdict(turn))
        
        # Scenario summary
        print(f"\n--- SCENARIO {scenario_idx} SUMMARY ---")
        print(f"Queries processed: {len(scenario['queries'])}")
        print(f"Topics accumulated: {memory.get_topics_summary()}")
        
        scenario_result = {
            "scenario_num": scenario_idx,
            "name": scenario['name'],
            "queries": scenario['queries'],
            "conversation": scenario_conversation,
            "topics_discussed": memory.topics
        }
        all_results["scenarios"].append(scenario_result)
    
    # Final summary
    print(f"\n\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total scenarios tested: {len(test_scenarios)}")
    print(f"Total queries processed: {sum(len(s['queries']) for s in test_scenarios)}")
    
    # Save results
    output_file = os.path.join(SCRIPT_DIR, "react_agent_results.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")
    
    return all_results


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

async def interactive_mode():
    """Interactive mode for user queries."""
    
    print("\n" + "="*80)
    print("GAME OF THRONES: REACT AGENT - INTERACTIVE MODE")
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
    
    # Initialize memory
    memory = ConversationMemory()
    
    # Create agent
    print("Initializing ReAct agent...")
    agent = create_react_agent(vector_store, memory)
    print("Agent ready ✓\n")
    
    print("Type your questions about Game of Thrones (type 'quit' to exit, 'clear' to reset memory)\n")
    
    turn_num = 0
    conversation_log = []
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n✓ Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                memory = ConversationMemory()
                print("✓ Memory cleared")
                continue
            
            turn_num += 1
            
            # Execute agent turn
            turn, memory = await execute_agent_turn_async(agent, user_input, turn_num, memory, vector_store)
            
            # Add to conversation log
            conversation_log.append(asdict(turn))
            
        except KeyboardInterrupt:
            print("\n\n✓ Goodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")
            continue
    
    # Save conversation log
    if conversation_log:
        output_file = os.path.join(SCRIPT_DIR, "interactive_conversation.json")
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "turns": conversation_log,
                "final_topics": memory.topics
            }, f, indent=2)
        print(f"\n✓ Conversation saved to {output_file}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if "--build" in sys.argv:
        print("Building vector store...")
        build_got_vector_store()
    elif "--interactive" in sys.argv or "-i" in sys.argv:
        asyncio.run(interactive_mode())
    else:
        asyncio.run(test_react_agent_async())
