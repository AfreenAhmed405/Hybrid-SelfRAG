from vllm import LLM, SamplingParams

# ============ CONFIG ============ #
model = LLM(
    model="/scratch/user/rheasudheer19/rag_inference/selfrag_model",
    download_dir="/scratch/user/rheasudheer19/rag_inference/model_cache",
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

# ============ FUNCTIONS ============ #
def format_prompt(query, context=None):
    prompt = f"### Instruction:\n{query.strip()}\n\n"
    if context:
        prompt += f"[Retrieval]<paragraph>{context.strip()}</paragraph>\n"
    return prompt + "### Response:"

def generate_response(query, context=None):
    prompt = format_prompt(query, context)
    outputs = model.generate(prompt, sampling_params)
    return outputs[0].outputs[0].text.strip()

# ============ MAIN ============ #
if __name__ == "__main__":
    query = "Do transformer models require labeled data for training?"
    context = "Transformer models like BERT and GPT are typically pre-trained on large unlabeled corpora and later fine-tuned on labeled datasets."

    response = generate_response(query, context)
    print("\n🔎 Baseline Query:", query)
    print("\n📄 Baseline Response:\n", response)
