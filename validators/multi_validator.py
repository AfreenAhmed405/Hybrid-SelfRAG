from validators.arxiv_validator import score_arxiv
from validators.wikipedia_validator import score_wikipedia

class MultiValidator:
    def score_all(self, response):
        return {
            "Wikipedia": score_wikipedia(response),
            "ArXiv": score_arxiv(response),
        }

    def score_wikipedia(self, response):
        return score_wikipedia(response)

    def score_arxiv(self, response):
        return score_arxiv(response)

