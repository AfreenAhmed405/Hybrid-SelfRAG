import sys
import os
import json
import torch
from tqdm import tqdm

# Set environment variable to avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vllm import LLM, SamplingParams
from validators.multi_validator import MultiValidator
from rag.retrieve import retrieve_from_sources

# CONFIG
DATA_PATH = "data/triviaqa_sample.json"
MODEL_PATH = "/scratch/user/rheasudheer19/hybrid_selfrag/models/selfrag_llama2_7b"
CACHE_PATH = "/scratch/user/rheasudheer19/hybrid_selfrag/model_cache"
THRESHOLD = 0.6  # Fact score threshold
TOP_K_RETRIEVAL = 3
LOG_PATH = "results/final_outputs.csv"

# Initialize LLM
model = LLM(
    model=MODEL_PATH,
    download_dir=CACHE_PATH,
    dtype="half",
    max_model_len=2048,
    gpu_memory_utilization=0.6
)

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=256
)

# Initialize Validator
validator = MultiValidator()

# Helpers
def format_prompt(query, context=None):
    prompt = f"### Instruction:\n{query.strip()}\n\n"
    if context:
        prompt += f"[Retrieved Knowledge]\n{context.strip()}\n"
    return prompt + "### Response:"

def generate_response(query, context=None):
    prompt = format_prompt(query, context)
    outputs = model.generate(prompt, sampling_params)
    return outputs[0].outputs[0].text.strip()

def load_queries(path):
    with open(path, "r") as f:
        return json.load(f)

# Main
if __name__ == "__main__":
    print(f"Loading queries from: {DATA_PATH}")
    dataset = load_queries(DATA_PATH)

    results = []

    for idx, item in enumerate(tqdm(dataset, desc="Processing prompts")):
        query = item.get("query")
        context = item.get("context", "")

        print(f"\n[{idx+1}] Query: {query}")

        # Step 1: Generate Initial Response
        initial_response = generate_response(query, context)
        print(f"Initial Response:\n{initial_response}")

        # Step 2: Score Initial Response
        initial_scores = validator.score_all(initial_response)
        print(f"Initial Fact Scores: {initial_scores}")

        improved_scores = initial_scores.copy()
        final_response = initial_response

        # Step 3: Check if any source score is below threshold
        if any(score < THRESHOLD for score in initial_scores.values()):
            print("Some fact score is low. Triggering retrieval...")

            # Retrieve additional evidence
            retrieved_context = retrieve_from_sources(query, top_k=TOP_K_RETRIEVAL)

            # Re-generate response with retrieval context
            regenerated_response = generate_response(query, retrieved_context)
            print(f"Regenerated Response:\n{regenerated_response}")

            # Step 4: Score new response
            regenerated_scores = validator.score_all(regenerated_response)

            # Step 5: Update scores only if improvement
            for source in initial_scores.keys():
                if initial_scores[source] < THRESHOLD:
                    if regenerated_scores[source] > initial_scores[source]:
                        improved_scores[source] = regenerated_scores[source]
                    else:
                        improved_scores[source] = initial_scores[source]
                else:
                    improved_scores[source] = initial_scores[source]

            final_response = regenerated_response
        else:
            print("All fact scores are good. No retrieval needed.")

        # Step 6: Calculate Delta Score
        delta_scores = {source: round(improved_scores[source] - initial_scores[source], 3)
                        for source in initial_scores.keys()}
        print(f"Delta Scores: {delta_scores}")

        # Step 7: Save results
        results.append({
            "query": query,
            "initial_response": initial_response,
            "final_response": final_response,
            "initial_scores": initial_scores,
            "improved_scores": improved_scores,
            "delta_scores": delta_scores
        })

    # Save to CSV (optional, we can improve this later)
    import csv
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "initial_response", "final_response", "initial_scores", "improved_scores", "delta_scores"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nAll results written to {LOG_PATH}")
