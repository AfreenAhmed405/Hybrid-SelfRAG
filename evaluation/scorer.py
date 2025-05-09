import pandas as pd

def evaluate_scores(csv_path, source_names=["ArXiv", "Wikipedia", "Wikidata"]):
    df = pd.read_csv(csv_path)

    results = {}
    total = len(df)

    for source in source_names:
        score_col = source
        supported = df[score_col] >= 0.66
        zero_score = df[score_col] == 0.0

        results[source] = {
            "Factual Accuracy": supported.sum() / total,
            "Hallucination Rate": zero_score.sum() / total
        }

    # Print results
    print(f"📊 Evaluation Results from {csv_path}")
    for source, metrics in results.items():
        print(f"\n✅ Source: {source}")
        print(f"   - Factual Accuracy: {metrics['Factual Accuracy']:.2f}")
        print(f"   - Hallucination Rate: {metrics['Hallucination Rate']:.2f}")

    return results
