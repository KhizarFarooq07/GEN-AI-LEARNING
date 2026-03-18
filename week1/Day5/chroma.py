from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
# Initialize embedding model and vector store
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    collection_name="rag_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
# print(vector_store.get())  # Clear existing data for a clean test run
# Add documents
docs = [
    Document(page_content="ChromaDB stores vector embeddings", metadata={"source": "doc1"}),
    Document(page_content="LangChain simplifies LLM application development", metadata={"source": "doc2"}),
]
vector_store.add_documents(docs)


# Similarity search
results = vector_store.similarity_search("LangChain simplifies LLM application development", k=2)
for doc in results:
    print(doc.page_content)