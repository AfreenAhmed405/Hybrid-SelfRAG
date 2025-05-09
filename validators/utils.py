from sentence_transformers import SentenceTransformer, util

# Load the embedding model
embedding_model = SentenceTransformer("/scratch/user/rheasudheer19/hybrid_selfrag/models/all-MiniLM-L6-v2")

def score_claims_against_all_evidence(claims, evidences):
    """Score claims against evidences robustly."""
    if not claims or not evidences:
        return 0.0

    filtered_pairs = [(claim, evidence) for claim, evidence in zip(claims, evidences) if evidence.strip()]
    if not filtered_pairs:
        return 0.0

    filtered_claims, filtered_evidences = zip(*filtered_pairs)

    claim_embeddings = embedding_model.encode(list(filtered_claims), convert_to_tensor=True)
    evidence_embeddings = embedding_model.encode(list(filtered_evidences), convert_to_tensor=True)

    cosine_scores = util.cos_sim(claim_embeddings, evidence_embeddings)
    avg_score = cosine_scores.diag().mean().item()

    return avg_score

def truncate_text(text, max_tokens=512):
    """Simple function to truncate text if too long."""
    words = text.split()
    if len(words) > max_tokens:
        words = words[:max_tokens]
    return ' '.join(words)
