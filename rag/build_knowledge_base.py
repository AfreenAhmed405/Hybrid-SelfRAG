import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Paths
FAISS_INDEX_PATH = "/scratch/user/rheasudheer19/hybrid_selfrag/rag/knowledge_base.faiss"
EMBEDDINGS_PATH = "/scratch/user/rheasudheer19/hybrid_selfrag/rag/knowledge_texts.pkl"

# Example knowledge (Expand this later)
knowledge_texts = [
    "The Hubble Space Telescope helped discover dark energy.",
    "Titan is Saturn's largest moon and has a thick atmosphere.",
    "The Declaration of Independence was signed on July 4, 1776.",
    "CRISPR-Cas9 allows for precise genome editing.",
    "Climate change is mainly caused by greenhouse gas emissions."
]

# Load embedding model
embedding_model = SentenceTransformer("/scratch/user/rheasudheer19/hybrid_selfrag/models/all-MiniLM-L6-v2")

# Create embeddings
embeddings = embedding_model.encode(knowledge_texts, convert_to_tensor=False)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save
faiss.write_index(index, FAISS_INDEX_PATH)
with open(EMBEDDINGS_PATH, "wb") as f:
    pickle.dump(knowledge_texts, f)

print(f"✅ Knowledge Base Created with {len(knowledge_texts)} entries!")
