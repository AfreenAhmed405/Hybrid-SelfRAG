import feedparser
import spacy
from functools import lru_cache
from validators.utils import score_claims_against_all_evidence

nlp = spacy.load("en_core_web_sm")

@lru_cache(maxsize=128)
def search_arxiv(query):
    query = query.replace('\n', ' ').replace(' ', '+')
    base_url = 'http://export.arxiv.org/api/query?search_query=all:'
    query_url = f"{base_url}{query}&start=0&max_results=5"
    parsed = feedparser.parse(query_url)
    return [entry.summary for entry in parsed.entries]

def score_arxiv(response):
    claims = [sent.text.strip() for sent in nlp(response).sents]
    evidences = []
    for claim in claims:
        evs = search_arxiv(claim)
        evidences.append(evs[0] if evs else "")
    score = score_claims_against_all_evidence(claims, evidences)
    return round(score, 2) if score else 0.0
