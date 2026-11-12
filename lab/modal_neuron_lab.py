"""Modal wrapper for lab/neuron_lab.py (pool / dev files only; run from the repo root)."""
import json, os, sys
import modal

sys.path.insert(0, "/root/app")
vol = modal.Volume.from_name("tpc-vol")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", "transformers==4.57.1", "accelerate==1.10.1", "safetensors==0.6.2", "numpy")
         .env({"TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
for f in ("core.py", "artifact.py", "grow.py", "answer.py", "feedback.py", "stage.py", "keep.json", "growth_cfg.json"):
    image = image.add_local_file(f, f"/root/app/{f}")
image = image.add_local_file("lab/neuron_lab.py", "/root/app/neuron_lab.py").add_local_dir("build_data", "/root/app/data")
app = modal.App("ten-percent-neuron-lab", image=image)


@app.function(volumes={"/vol": vol}, gpu=["A100-80GB", "H100", "L40S"], timeout=2 * 3600, cpu=4, memory=49152)
def lab(ks: str = "256,1024,4096,16384,65536", lam: float = 1.0, which: str = "hi,skills"):
    import torch, core, artifact, stage, neuron_lab
    rd = lambda p: [json.loads(l) for l in open(f"/root/app/data/{p}.jsonl")]
    d = stage.stage("/vol/artifacts5/C1", "/tmp/art"); m, tok = artifact.load(d)
    nl = neuron_lab.NeuronLab(m, tok, core.BASE_LABELS + ["recipe", "music"])
    en = rd("dev")[:400]; guard = [x["text"] for x in rd("guard")]; Ks = [int(k) for k in ks.split(",")]
    R = {"gpu": torch.cuda.get_device_name(0)}; os.makedirs("/vol/work7", exist_ok=True)
    def save():
        json.dump(R, open("/vol/work7/neuron_lab.json", "w")); vol.commit()
    def pick_hi(pool, p):
        w = [i for i, x in enumerate(pool) if p[i] != x["label"]]
        return w[:24], w[24:48]
    def pick_skills(pool, p):
        a, b = [], []
        for y in ("recipe", "music"):
            w = [i for i, x in enumerate(pool) if x["label"] == y and p[i] != y]
            a += w[:12]; b += w[12:24]
        return a, b
    log = lambda s: print(s, flush=True)
    if "hi" in which:
        R["hi-IN"] = neuron_lab.experiment(nl, "hi-IN", rd("teach2/hi-IN/pool"), rd("teach2/hi-IN/dev"), en, guard, pick_hi, Ks, lam, log); save()
    if "skills" in which:
        R["skills"] = neuron_lab.experiment(nl, "skills", rd("teach_pool"), rd("teach_dev"), en, guard, pick_skills, Ks, lam, log); save()
    if "v2" in which:
        big_guard = [x["text"] for x in rd("calib") if x["source"] == "massive"][:256]
        cfgs = [{"lam": 5.0, "c": 0.0, "q": 0.5}, {"lam": 5.0, "c": 0.5, "q": 0.5}, {"lam": 20.0, "c": 0.0, "q": 0.25}, {"lam": 20.0, "c": 0.5, "q": 0.25}]
        R["skills_v2"] = neuron_lab.experiment2(nl, rd("teach_pool"), rd("teach_dev"), en, big_guard, pick_skills, Ks, cfgs, log); save()
    return "ok"


@app.local_entrypoint()
def main(ks: str = "256,1024,4096,16384,65536", lam: float = 1.0, which: str = "hi,skills"):
    print(lab.remote(ks, lam, which))
