import faiss
import pickle
import requests
import feedparser
from sentence_transformers import SentenceTransformer
from validators.utils import truncate_text

# ===== Paths =====
FAISS_INDEX_PATH = "/scratch/user/rheasudheer19/hybrid_selfrag/rag/knowledge_base.faiss"
EMBEDDINGS_PATH = "/scratch/user/rheasudheer19/hybrid_selfrag/rag/knowledge_texts.pkl"
EMBEDDING_MODEL_PATH = "/scratch/user/rheasudheer19/hybrid_selfrag/models/all-MiniLM-L6-v2"

# ===== Load models and indexes =====
embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
index = faiss.read_index(FAISS_INDEX_PATH)
with open(EMBEDDINGS_PATH, "rb") as f:
    knowledge_texts = pickle.load(f)

# ===== Wikipedia Retrieval =====
def search_wikipedia(query):
    try:
        url = f"https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json"
        }
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        results = response.json().get("query", {}).get("search", [])
        if results:
            return results[0]["snippet"]
        else:
            return ""
    except Exception as e:
        print(f"Warning: Wikipedia retrieval failed: {e}")
        return ""

# ===== Arxiv Retrieval =====
def search_arxiv(query):
    try:
        query = query.replace('\n', ' ').replace(' ', '+')
        base_url = 'http://export.arxiv.org/api/query?search_query=all:'
        query_url = f"{base_url}{query}&start=0&max_results=5"
        parsed = feedparser.parse(query_url)
        return [entry.summary for entry in parsed.entries]
    except Exception as e:
        print(f"Warning: Arxiv retrieval failed: {e}")
        return []

# ===== FAISS Local KB Retrieval =====
def retrieve_best_evidence(query, top_k=2):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = [knowledge_texts[idx] for idx in indices[0] if idx < len(knowledge_texts)]
    return results

# ===== Combined Retrieval from All Sources =====
def retrieve_from_sources(query, top_k=2):
    """Retrieve relevant evidence from Wikipedia + Arxiv + Local Knowledge Base."""
    wiki_context = search_wikipedia(query)
    arxiv_contexts = search_arxiv(query)
    arxiv_context = arxiv_contexts[0] if arxiv_contexts else ""
    kb_contexts = retrieve_best_evidence(query, top_k=top_k)
    kb_context = " ".join(kb_contexts)

    # Combine all
    combined_context = " ".join([wiki_context, arxiv_context, kb_context])

    # Truncate final text to avoid token overflow
    return truncate_text(combined_context, max_tokens=512)
