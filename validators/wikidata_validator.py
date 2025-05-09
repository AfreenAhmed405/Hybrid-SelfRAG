import requests
import spacy
from functools import lru_cache
from validators.utils import score_claims_against_all_evidence

nlp = spacy.load("en_core_web_sm")

@lru_cache(maxsize=128)
def search_wikidata(query):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "language": "en",
        "format": "json",
        "search": query
    }
    try:
        response = requests.get(url, params=params)
        results = response.json().get("search", [])
        return results[0]["description"] if results else ""
    except:
        return ""

def score_wikidata(response):
    claims = [sent.text.strip() for sent in nlp(response).sents]
    evidences = [search_wikidata(claim) for claim in claims]
    score = score_claims_against_all_evidence(claims, evidences)
    return round(score, 2) if score else 0.0
