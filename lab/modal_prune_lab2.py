"""Lab 2 (pool/dev only): numerics check, global top-k sweep, and a corrected greedy prune on one locale."""
import json, os, sys, time
import modal

sys.path.insert(0, "/root/app")
vol = modal.Volume.from_name("tpc-vol")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", "transformers==4.57.1", "accelerate==1.10.1", "safetensors==0.6.2", "numpy")
         .env({"TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
         .add_local_file("core.py", "/root/app/core.py").add_local_file("artifact.py", "/root/app/artifact.py")
         .add_local_file("prune.py", "/root/app/prune.py").add_local_file("prune_lab.py", "/root/app/prune_lab.py")
         .add_local_dir("build_data", "/root/app/data"))
app = modal.App("ten-percent-prune-lab2", image=image)


@app.function(volumes={"/vol": vol}, gpu=["A100-80GB", "L40S", "H100", "A100-40GB"], timeout=3600, cpu=4, memory=40960)
def lab2(locs: str = "hi-IN,tr-TR,ar-SA", ks: str = "4,6,12,16"):
    import torch, artifact, prune
    from prune import preds, margins, LAB
    rd = lambda p: [json.loads(l) for l in open(f"/root/app/data/{p}.jsonl")]
    m, tok = artifact.load("/vol/artifacts5/C1"); pr = prune.Pruner(m, tok)
    acc = lambda S, xs: round(100 * sum(p == x["label"] for p, x in zip(preds(S), xs)) / len(xs), 2)
    R = {}; save = lambda: (json.dump(R, open("/vol/work7/lab2.json", "w")), vol.commit())
    en = rd("dev")[:400]; ids_en = pr.encode([x["text"] for x in en])
    sets = {loc: rd(f"teach2/{loc}/dev") for loc in locs.split(",")}
    ids = {loc: pr.encode([x["text"] for x in xs]) for loc, xs in sets.items()}
    # 1. numerics: same items, two batch compositions
    a = pr.score(ids["hi-IN"][:96], bs=96); b = pr.score(ids["hi-IN"][:96], bs=7)
    R["noise_hi"] = {"max": (a - b).abs().max().item(), "std": (a - b).std().item()}
    a = pr.score(ids_en[:96], bs=96); b = pr.score(ids_en[:96], bs=7)
    R["noise_en"] = {"max": (a - b).abs().max().item(), "std": (a - b).std().item()}
    print(R, flush=True)
    # 2. global top-k sweep
    S_en0 = pr.score(ids_en); en0 = preds(S_en0)
    R["k8"] = {"en_acc": acc(S_en0, en)} | {loc: acc(pr.score(ids[loc]), sets[loc]) for loc in sets}
    print("k8", R["k8"], flush=True)
    for k in [int(x) for x in ks.split(",")]:
        for blk in pr.blocks:
            blk.top_k = k
        Se = pr.score(ids_en)
        R[f"k{k}"] = {"en_acc": acc(Se, en), "en_same": sum(x == y for x, y in zip(preds(Se), en0))} | {loc: acc(pr.score(ids[loc]), sets[loc]) for loc in sets}
        print(f"k{k}", R[f"k{k}"], flush=True); save()
    for blk in pr.blocks:
        blk.top_k = 8
    save()
    return R


@app.local_entrypoint()
def main():
    print(json.dumps(lab2.remote(), indent=1))
