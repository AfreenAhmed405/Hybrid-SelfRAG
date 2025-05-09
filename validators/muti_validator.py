from validators.arxiv_validator import score_arxiv
from validators.wikipedia_validator import score_wikipedia
from validators.wikidata_validator import score_wikidata

class MultiValidator:
    def score_all(self, response):
        return {
            "ArXiv": score_arxiv(response),
            "Wikipedia": score_wikipedia(response),
            "Wikidata": score_wikidata(response),
        }
