import csv
import os

def init_log_file(path):
    if not os.path.exists(path):
        with open(path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["query", "context", "response", "ArXiv", "Wikipedia", "Wikidata"])

def log_example(path, query, context, response, scores):
    with open(path, "a") as f:
        writer = csv.writer(f)
        writer.writerow([
            query,
            context,
            response,
            round(scores.get("ArXiv", 0.0), 2),
            round(scores.get("Wikipedia", 0.0), 2),
            round(scores.get("Wikidata", 0.0), 2)
        ])
