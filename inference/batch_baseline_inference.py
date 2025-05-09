import json
import os
import csv
from vllm import LLM, SamplingParams

# ======== CONFIG ========
DATA_PATH = "data/triviaqa_sample.json"          # Change to fever_sample.json or pubmedqa_sample.json
LOG_PATH = "results/baseline_outputs.csv"
MODEL_PATH = "/scratch/user/rheasudheer19/rag_inference/selfrag_model"
CACHE_PATH = "/scratch/user/rheasudheer19/rag_inference/model_cache"

# ======== INIT MODEL ========
model = LLM(
    model=MODEL_PATH,
    download_dir=CACHE_PATH,
    dtype="half",
    device="cuda",
    max_model_len=2048,
    gpu_memory_utilization=0.9
)

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=256
)

# ======== HELPERS ========
def format_prompt(query, context=None):
    prompt = f"### Instruction:\n{query.strip()}\n\n"
    if context:
        prompt += f"[Retrieval]<paragraph>{context.strip()}</paragraph>\n"
    return prompt + "### Response:"

def generate_response(query, context=None):
    prompt = format_prompt(query, context)
    outputs = model.generate(prompt, sampling_params)
    return outputs[0].outputs[0].text.strip()

def load_queries(path):
    with open(path, "r") as f:
        return json.load(f)

def init_log_file(path):
    if not os.path.exists(path):
        with open(path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["query", "context", "response"])

def log_to_csv(path, query, context, response):
    with open(path, "a") as f:
        writer = csv.writer(f)
        writer.writerow([query, context, response])

# ======== MAIN LOOP ========
if __name__ == "__main__":
    print(f"\n📂 Loading queries from: {DATA_PATH}")
    dataset = load_queries(DATA_PATH)
    init_log_file(LOG_PATH)

    for idx, item in enumerate(dataset, start=1):
        query = item.get("query")
        context = item.get("context", "")

        print(f"\n🔹 [{idx}] Query: {query}")
        response = generate_response(query, context)
        print(f"📄 Response:\n{response}")

        log_to_csv(LOG_PATH, query, context, response)
