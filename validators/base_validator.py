import spacy
nlp = spacy.load("en_core_web_sm")

class BaseValidator:
    def extract_claims(self, text):
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]

    def score(self, claims):
        raise NotImplementedError("Each validator must implement its own scoring method.")
