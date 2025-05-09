from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="selfrag/selfrag_llama2_7b",
    local_dir="models/selfrag_llama2_7b",
    local_dir_use_symlinks=False
)
