import gradio as gr
import chromadb
from sentence_transformers import SentenceTransformer
import json

# --- 1. Initialization (Global State) ---
print("Initializing models and data...")

# Load documents
with open('documents.json', 'r') as f:
    documents = json.load(f)

ids = [str(i) for i in range(len(documents))]
metadatas = [{'source': 'blog'} for _ in range(len(documents))]

# Init Chroma
client = chromadb.Client()
collection_name = "blog_posts"
try:
    client.delete_collection(name=collection_name)
except:
    pass
collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

# Load Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Index Documents
embeddings = model.encode(documents).tolist()
collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)
print("Initialization complete.")

# --- 2. Search Logic ---
def search_semantic(query, n_results=3):
    if not query.strip():
        return "Please enter a query."
        
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    output = []
    found_any = False
    for i in range(n_results):
        if i < len(results['documents'][0]):
            doc = results['documents'][0][i]
            distance = results['distances'][0][i]
            similarity = 1 - distance
            
            if similarity >= 0.3:
                output.append(f"### Result {i+1} (Similarity: {similarity:.4f})\n>{doc}")
                found_any = True
            else:
                break
    
    if not found_any:
        return "No results found with similarity >= 30%."
    
    return "\n\n".join(output)

def search_keyword(query, n_results=3):
    if not query.strip():
        return ""

    stop_words = set(["a", "an", "the", "in", "on", "of", "and", "is", "to", "with", "for", "it", "that", "this", "by", "at"])
    query_words = [w for w in query.lower().split() if w not in stop_words]
    
    if not query_words:
        return "Query contains only stop words."

    scores = []
    for doc in documents:
        score = 0
        doc_lower = doc.lower()
        for word in query_words:
            if word in doc_lower:
                score += 1
        scores.append(score)
    
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    output = []
    found_any = False
    for i in range(min(n_results, len(documents))):
        idx = sorted_indices[i]
        if scores[idx] > 0:
            output.append(f"### Result {i+1} (Matches: {scores[idx]})\n>{documents[idx]}")
            found_any = True
        else:
            break
    
    if not found_any:
        return "No keyword matches found."
    
    return "\n\n".join(output)

def compare_search(query):
    return search_semantic(query), search_keyword(query)

# --- 3. Gradio UI ---
css = """
.container {
    max-width: 900px;
    margin: auto;
    padding-top: 20px;
}
"""
with gr.Blocks(title="Semantic Search Demo", css=css) as demo:
    with gr.Column(elem_classes="container"):
        gr.Markdown("# Semantic Search Demo")
        gr.Markdown("Compare **Semantic Search** (Vector Embeddings) vs **Keyword Search**.")
        
        query_input = gr.Textbox(label="Enter Search Query", placeholder="e.g., making bread, soccer, neural networks")
        search_btn = gr.Button("Search", variant="primary")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Semantic Search Results")
                semantic_output = gr.Markdown()
            
            with gr.Column():
                gr.Markdown("## Keyword Search Results")
                keyword_output = gr.Markdown()
                
        search_btn.click(fn=compare_search, inputs=query_input, outputs=[semantic_output, keyword_output])
        query_input.submit(fn=compare_search, inputs=query_input, outputs=[semantic_output, keyword_output])

if __name__ == "__main__":
    demo.launch()
