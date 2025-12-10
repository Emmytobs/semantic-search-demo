import chromadb
from sentence_transformers import SentenceTransformer

import json

# 1. Setup Data
with open('documents.json', 'r') as f:
    documents = json.load(f)

ids = [str(i) for i in range(len(documents))]
metadatas = [{'source': 'blog'} for _ in range(len(documents))]

# 2. Initialize
print("Initializing ChromaDB and Model...")
client = chromadb.Client()
collection_name = "test_blog_posts"
try:
    client.delete_collection(name=collection_name)
except:
    pass
collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Index
print("Indexing documents...")
embeddings = model.encode(documents).tolist()
collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

# 4. Search
def test_query(query):
    print(f"\nQuery: '{query}'")
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=1
    )
    best_doc = results['documents'][0][0]
    distance = results['distances'][0][0]
    similarity = 1 - distance
    if similarity >= 0.5:
        print(f"Top Result (Sim: {similarity:.4f}): {best_doc[:100]}...")
    else:
        print(f"Top Result (Sim: {similarity:.4f}) [BELOW THRESHOLD]: {best_doc[:100]}...")

def search_keyword(query):
    print(f"\nKeyword Search for: '{query}'")
    stop_words = set(["a", "an", "the", "in", "on", "of", "and", "is", "to", "with", "for", "it", "that", "this", "by", "at"])
    query_words = [w for w in query.lower().split() if w not in stop_words]
    
    if not query_words:
        print("Query contains only stop words.")
        return

    scores = []
    for doc in documents:
        score = 0
        doc_lower = doc.lower()
        for word in query_words:
            if word in doc_lower:
                score += 1
        scores.append(score)
    
    max_score = max(scores)
    if max_score > 0:
        best_idx = scores.index(max_score)
        print(f"Top Result (Matches: {max_score}): {documents[best_idx][:100]}...")
    else:
        print("No matches found.")

print("\n--- COMPARISON ---")
queries = ["neural networks", "making bread", "soccer", "playing a sport"]
for q in queries:
    print(f"\nQuery: {q}")
    test_query(q)
    search_keyword(q)
