import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load local embedding model
embedding_model = SentenceTransformer("/scratch/user/rheasudheer19/hybrid_selfrag/models/all-MiniLM-L6-v2")

def build_dynamic_faiss(texts):
    """Build a temporary FAISS index from a list of texts."""
    if not texts:
        return None, None

    embeddings = embedding_model.encode(texts, convert_to_tensor=False)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, texts

def search_faiss(index, texts, query, top_k=2):
    """Search FAISS index with query and return top-k texts."""
    if index is None or not texts:
        return []

    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = [texts[i] for i in indices[0] if i < len(texts)]
    return results
