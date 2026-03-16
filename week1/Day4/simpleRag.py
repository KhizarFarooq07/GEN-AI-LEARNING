import ollama

dataset = []
with open('cat-facts.txt', 'r') as file:
  dataset = file.readlines()
  print(f'Loaded {len(dataset)} entries')


EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# Each element in the VECTOR_DB will be a tuple (chunk, embedding)
# The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]
VECTOR_DB = [] #in memory vector database for simple RAG implementation

def add_chunk_to_database(chunk):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  VECTOR_DB.append((chunk, embedding))

for i, chunk in enumerate(dataset):
  add_chunk_to_database(chunk)
  if (i + 1) % 100 == 0:
    print(f'Processed {i + 1} chunks')

def cosine_similarity(vec1, vec2):
  dot_product = sum(a * b for a, b in zip(vec1, vec2))
  magnitude_vec1 = sum(a ** 2 for a in vec1) ** 0.5
  magnitude_vec2 = sum(b ** 2 for b in vec2) ** 0.5
  if magnitude_vec1 == 0 or magnitude_vec2 == 0:
    return 0.0
  return dot_product / (magnitude_vec1 * magnitude_vec2)


def retrive(query, top_n=3):
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  similarities=[]

  for chunk, embedding in VECTOR_DB:
    similarity = cosine_similarity(query_embedding, embedding)
    similarities.append((chunk, similarity))

  similarities.sort(key=lambda x: x[1], reverse=True)
  return [chunk for chunk, _ in similarities[:top_n]]

input_query = input('Ask me a question: ')
retrieved_knowledge = retrive(input_query)
print("Retrieved knowledge:")
for i, chunk in enumerate(retrieved_knowledge, 1):
  print(f"{i}. {chunk}")


instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{'\n'.join([f' - {chunk}' for chunk in retrieved_knowledge])}
'''


stream = ollama.chat(
  model=LANGUAGE_MODEL,
  messages=[
    {'role': 'system', 'content': instruction_prompt},
    {'role': 'user', 'content': input_query},
  ],
  stream=True,
)

# print the response from the chatbot in real-time
print('Chatbot response:')
for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)

