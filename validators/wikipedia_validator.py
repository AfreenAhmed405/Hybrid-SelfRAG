import wikipedia
import spacy
from functools import lru_cache
from validators.utils import score_claims_against_all_evidence

nlp = spacy.load("en_core_web_sm")

@lru_cache(maxsize=128)
def search_wikipedia(query):
    try:
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except:
        return ""

def score_wikipedia(response):
    claims = [sent.text.strip() for sent in nlp(response).sents]
    evidences = [search_wikipedia(claim) for claim in claims]
    score = score_claims_against_all_evidence(claims, evidences)
    return round(score, 2) if score else 0.0
