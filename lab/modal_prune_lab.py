"""Modal design lab for prune-only feedback. Sees build_data/ only (task, calib, dev, teach2 pool + dev): no locked file."""
import json, os, sys
import modal

sys.path.insert(0, "/root/app")
vol = modal.Volume.from_name("tpc-vol")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", "transformers==4.57.1", "accelerate==1.10.1", "safetensors==0.6.2", "numpy")
         .env({"TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
         .add_local_file("core.py", "/root/app/core.py").add_local_file("artifact.py", "/root/app/artifact.py")
         .add_local_file("prune.py", "/root/app/prune.py").add_local_file("prune_lab.py", "/root/app/prune_lab.py")
         .add_local_dir("build_data", "/root/app/data"))
app = modal.App("ten-percent-prune-lab", image=image)

CFG = {"n_feedback": 24, "N": 300, "M": 80, "tau": 0.5, "cap": 3.0, "floor": -8.0, "max_accept": 40, "min_keep": 16,
       "checkpoints": [5, 10, 20], "seed": 0, "n_guard": 64, "n_en": 400}


@app.function(volumes={"/vol": vol}, gpu=["A100-80GB", "L40S", "H100", "A100-40GB"], timeout=2 * 3600, cpu=4, memory=40960)
def lab(loc: str, tag: str = "a", cfg_json: str = "{}"):
    import torch, artifact, prune, prune_lab
    cfg = dict(CFG); cfg.update(json.loads(cfg_json)); cfg["loc"] = loc
    rd = lambda p: [json.loads(l) for l in open(f"/root/app/data/{p}.jsonl")]
    pool, dev, en_dev = rd(f"teach2/{loc}/pool"), rd(f"teach2/{loc}/dev"), rd("dev")[:cfg["n_en"]]
    guard = [x["text"] for x in rd("calib") if x["source"] == "massive"][:cfg["n_guard"]]
    m, tok = artifact.load("/vol/artifacts5/C1")
    pr = prune.Pruner(m, tok)
    os.makedirs("/vol/work7", exist_ok=True)
    def save(R, Tn):
        json.dump(R, open(f"/vol/work7/lab_{tag}_{loc}.json", "w")); torch.save(Tn, f"/vol/work7/lab_{tag}_{loc}.pt"); vol.commit()
    R = prune_lab.run_lab(pr, pool, dev, en_dev, guard, cfg, save)
    return {k: v for k, v in R.items() if k in ("pool_acc", "dev_acc0", "en_acc0", "secs_total")} | {
        n: {k: v for k, v in R[n].items() if k not in ("log", "dev_pred")} for n in ("real", "shuf")}


@app.local_entrypoint()
def main(loc: str = "hi-IN", tag: str = "a", cfg_json: str = "{}"):
    print(json.dumps(lab.remote(loc, tag, cfg_json), indent=1))
