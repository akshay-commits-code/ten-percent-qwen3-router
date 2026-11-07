"""CPU job: publish the frozen artifact (root) + the proof folder to a public Hugging Face model repo."""
import modal
REPO = "akshay-commits-code/qwen3-30b-a3b-10pct-voice-router"
vol = modal.Volume.from_name("tpc-vol")
P = "/tmp/claude-0/-home-claude/be3816a6-a293-558b-9ccc-9dacac55abb1/scratchpad"
image = (modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub[hf_xet]==0.35.3")
         .add_local_dir(f"{P}/proof", "/root/proof").add_local_file(f"{P}/hf_README.md", "/root/README.md"))
app = modal.App("ten-percent-upload", image=image)


@app.function(volumes={"/vol": vol}, secrets=[modal.Secret.from_name("huggingface")], timeout=3600, cpu=4, memory=8192)
def upload():
    import os
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=REPO, folder_path="/vol/artifacts5/C1", commit_message="Artifact (frozen before the exam was opened)")
    api.upload_folder(repo_id=REPO, folder_path="/root/proof", path_in_repo="proof", commit_message="Proof: code, locks, protocol, git history, results")
    api.upload_file(repo_id=REPO, path_or_fileobj="/root/README.md", path_in_repo="README.md", commit_message="Model card")
    files = api.list_repo_tree(REPO, recursive=False)
    sizes = {f.path: getattr(f, "size", None) for f in files}
    print(sizes)
    return sizes
