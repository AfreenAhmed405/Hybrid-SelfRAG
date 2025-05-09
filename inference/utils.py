import csv

def format_prompt(query, context=None):
    prompt = f"### Instruction:\n{query.strip()}\n\n"
    if context:
        prompt += f"[Retrieval]<paragraph>{context.strip()}</paragraph>\n"
    return prompt + "### Response:"

def log_to_csv(filename, query, context, response, scores):
    with open(filename, "a") as f:
        writer = csv.writer(f)
        writer.writerow([
            query, context, response,
            scores.get("ArXiv", 0.0),
            scores.get("Wikipedia", 0.0),
            scores.get("Wikidata", 0.0)
        ])
