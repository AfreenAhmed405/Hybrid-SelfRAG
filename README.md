# Hybrid Self-RAG with External Knowledge Validation to Prevent Hallucination

## Overview

Large Language Models (LLMs) have revolutionized natural language generation but suffer from hallucinations — plausible-sounding yet factually incorrect outputs — which limit their reliability in critical domains like medicine, law, and science. This project proposes a **Hybrid Self-RAG (Retrieval-Augmented Generation)** framework that enhances factual accuracy by combining:

- Self-reflection and self-critique of generated content,
- External knowledge validation using trusted live APIs (Wikipedia and ArXiv),
- A confidence-based FactScore mechanism to evaluate factual consistency,
- Fallback dense retrieval using an internally curated FAISS-indexed knowledge base to correct low-confidence responses.

This multi-layered approach significantly reduces hallucinations and improves alignment with verifiable evidence, enabling more trustworthy AI-generated content.

---

## Key Features

- **Self-Reflection**: Multiple candidate outputs are generated and internally critiqued to improve coherence.
- **External Validation**: Factual claims are validated in real-time using Wikipedia and ArXiv APIs.
- **FactScore Metric**: A semantic similarity score between claims and retrieved evidence quantifies factual reliability.
- **Fallback Retrieval**: If external validation confidence is low, a dense retriever queries an internal knowledge base to regenerate responses.
- **Modular Pipeline**: Combines retrieval, generation, critique, and validation in a streamlined workflow.

---

## Architecture

![System Architecture](./docs/hybrid_self_rag_architecture.png)

1. **Initial Generation**: LLM (GPT-4) generates an initial response.
2. **Self-Critique**: Produces multiple outputs and selects the most consistent version.
3. **Claim Extraction**: NLP parsing identifies factual claims from the response.
4. **External Validation**: Claims are validated against Wikipedia and ArXiv via live API calls.
5. **FactScore Computation**: Semantic similarity between claims and evidence is calculated.
6. **Fallback Retrieval**: If FactScore < threshold (0.5), a FAISS-indexed internal knowledge base is queried.
7. **Regeneration**: LLM regenerates response using retrieved fallback evidence.
8. **Final Output**: The factually validated, coherent response is returned.

---

## Methodology

- **Claim Extraction:** Dependency-based NLP parsing isolates factual claims for validation.
- **Embedding Model:** SentenceTransformers `all-MiniLM-L6-v2` used to embed claims and evidence.
- **FactScore:** Average cosine similarity between claims and matched evidence embeddings.
- **Dense Retriever:** FAISS used to index fallback corpus sourced from Wikipedia and ArXiv data dumps.
- **Regeneration Trigger:** Low FactScore responses automatically trigger retrieval-augmented regeneration.

---

## Setup & Installation

### Requirements

- Python 3.8+
- GPU with CUDA support (recommended for embedding computations)
- Libraries:
  - `transformers`
  - `sentence-transformers`
  - `faiss-cpu` or `faiss-gpu`
  - `spaCy`
  - `requests`
  - `vLLM` (optional, for optimized LLM inference)

### Installation

```bash
git clone https://github.com/AfreenAhmed405/Hybrid-SelfRAG.git
cd hybrid-self-rag
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

1. Start the API clients for Wikipedia and ArXiv.
2. Run the inference script to generate and validate responses:

```bash
python run_hybrid_self_rag.py --prompt "Explain the significance of CRISPR technology"
```
3. The system outputs a factually validated response along with FactScore and any fallback retrieval information if used.

## Evaluation

The system was evaluated using the following metrics:

- **Hallucination Rate:** Percentage of generated responses containing unsupported or unverifiable claims, identified through low FactScores.
- **Average FactScore:** Mean cosine similarity between embedded claims and their matched external evidence, indicating factual alignment.
- **External Validation Success Rate:** Proportion of responses passing validation (FactScore ≥ 0.5) without requiring fallback dense retrieval.
- **Before vs. After Analysis:** Comparison of FactScores and hallucination rates before and after applying external validation and fallback retrieval to demonstrate their impact.

### Results Summary

| Metric                 | Baseline Self-RAG | Hybrid Self-RAG |
|------------------------|-------------------|-----------------|
| Factual Accuracy       | 81%               | 89%             |
| Hallucination Rate     | 19%               | 11%             |
| Average FactScore      | 0.65              | 0.87            |

---

## Challenges & Limitations

- Increased latency due to external API calls, embedding computations, and similarity calculations.
- Dependence on external sources (Wikipedia, ArXiv) that may have biases or outdated information.
- FactScore threshold tuning requires balancing between factual strictness and computational cost.
- Lack of standardized, large-scale benchmarks for claim-level verification limits comprehensive evaluation.

---

## Future Work

- Adaptive FactScore thresholding based on query type, domain sensitivity, or user preferences.
- Integration of additional trusted sources like PubMed or Semantic Scholar for specialized domains.
- Use of trained claim extraction models to automate fact-checking pipelines.
- Incorporation of user feedback mechanisms for continuous improvement.
- Extension to multimodal validation including charts, tables, and images.
- Creation or adoption of large annotated datasets to enable rigorous, reproducible evaluations.

---

## Acknowledgments

We thank Professor Kuan-Hao Huang and TA Rahul Baid for their guidance. This work was completed as part of CSCE 638 at Texas A&M University. Experiments used an NVIDIA A100 GPU provided by the Texas A&M High Performance Research Computing facility.

---

## Contact

For inquiries, please contact:

- Afreen Ahmed — afreen04@tamu.edu  
- Hitha Magadi Vijayanad — hoshi_1996@tamu.edu  
- Rhea Sudheer — rheasudheer19@tamu.edu  
- Sai Aakarsh Padma — saiaakarsh@tamu.edu

