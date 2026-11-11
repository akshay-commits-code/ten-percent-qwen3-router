"""Gate lab (pool only): per-item scores of the annotated items before / after putting back the top block, so the
acceptance test can be studied offline under many label shuffles. No dev or locked file is read."""
import json, os, sys
import modal

sys.path.insert(0, "/root/app")
MDIR = "/vol/models/qwen3-30b-a3b-instruct-2507"
vol = modal.Volume.from_name("tpc-vol")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", "transformers==4.57.1", "accelerate==1.10.1", "safetensors==0.6.2", "numpy")
         .env({"TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
for f in ("core.py", "artifact.py", "grow.py", "answer.py", "feedback.py", "stage.py", "keep.json", "growth_cfg.json"):
    image = image.add_local_file(f, f"/root/app/{f}")
image = image.add_local_dir("build_data", "/root/app/data")
app = modal.App("ten-percent-gate-lab", image=image)


@app.function(volumes={"/vol": vol}, gpu=["A100-80GB", "L40S", "H100", "A100-40GB"], timeout=3600, cpu=4, memory=49152)
def gatelab(locs: str = "hi-IN,tr-TR,ar-SA"):
    import torch, core, artifact, grow, stage
    d = stage.stage("/vol/artifacts5/C1", "/tmp/art"); m, tok = artifact.load(d); sc = grow.Scorer(m, tok); base = grow.Base(MDIR)
    L = len(core.moe_blocks(m)); gates = [base.tensor(f"model.layers.{l}.mlp.gate.weight").to(sc.dev, torch.bfloat16) for l in range(L)]
    keep = [set(k) for k in json.load(open(f"{d}/keep.json"))["keep"]]
    guard = [json.loads(x)["text"] for x in open(f"{d}/guard.jsonl")]; ids_g = sc.encode(guard); Sg0 = sc.score(ids_g)
    R = {}
    for loc in locs.split(","):
        pool = [json.loads(l) for l in open(f"/root/app/data/teach2/{loc}/pool.jsonl")]
        pp = core.run(m, tok, [x["text"] for x in pool])[0]
        wrong = [i for i, x in enumerate(pool) if pp[i] != x["label"]][:24]
        ids = sc.encode([pool[i]["text"] for i in wrong]); S0 = sc.score(ids); S0b = sc.score(ids, bs=7)
        mass = grow.would_be_mass(sc, ids, gates)
        cand = sorted(((mass[l][e].item(), l, e) for l in range(L) for e in range(128) if e not in keep[l]), reverse=True)[:70]
        r = {"labels": [pool[i]["label"] for i in wrong], "said": [pp[i] for i in wrong], "S0": S0.tolist(), "S0_other_batching": S0b.tolist()}
        for name, blk in (("S1", cand[:35]), ("S2", cand[35:70])):
            for _, l, e in blk:
                st, W = grow.round_expert(base, l, e, sc.dev); grow.attach(m, l, W, gates[l][e])
            r[name] = sc.score(ids).tolist(); r[name + "_guard_changed"] = sum(a != b for a, b in zip(grow.preds(sc.score(ids_g)), grow.preds(Sg0)))
            for _, l, e in reversed(blk):
                grow.detach(m, l)
        R[loc] = r; print(loc, r["labels"], r["S1_guard_changed"], r["S2_guard_changed"], flush=True)
        json.dump(R, open("/vol/work7/gatelab.json", "w")); vol.commit()
    return "ok"


@app.local_entrypoint()
def main():
    print(gatelab.remote())
