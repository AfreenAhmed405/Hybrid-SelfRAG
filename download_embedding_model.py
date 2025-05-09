from sentence_transformers import SentenceTransformer

# ✅ Set output path to the target location in your project
output_path = "/scratch/user/rheasudheer19/hybrid_selfrag/models/all-MiniLM-L6-v2"

# Load model from Huggingface
print("🔄 Downloading 'all-MiniLM-L6-v2' from HuggingFace...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Save model locally at the desired path
model.save(output_path)
print(f"✅ Model saved successfully to: {output_path}")
