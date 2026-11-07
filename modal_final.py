"""The ONE run that opens the locked exam and teach set. Everything it uses was frozen before (artifact + feedback.json)."""
import hashlib, json, os, shutil, sys, time
import modal

sys.path.insert(0, "/root/app")
MDIR = "/vol/models/qwen3-30b-a3b-instruct-2507"
vol = modal.Volume.from_name("tpc-vol")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", "transformers==4.57.1", "accelerate==1.10.1", "safetensors==0.6.2", "numpy")
         .env({"TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
         .add_local_file("core.py", "/root/app/core.py").add_local_file("artifact.py", "/root/app/artifact.py")
         .add_local_file("LOCK.json", "/root/app/LOCK.json")
         .add_local_dir("data", "/root/app/data"))
app = modal.App("ten-percent-final-3", image=image)


@app.function(volumes={"/vol": vol}, gpu="A100-80GB", timeout=3 * 3600, cpu=4, memory=49152)
def final():
    import random, torch, core, artifact
    from transformers import AutoTokenizer, AutoModelForCausalLM
    lock = json.load(open("/root/app/LOCK.json"))
    for n, h in lock["locked_files"].items():
        assert hashlib.sha256(open(f"/root/app/data/{n}.jsonl", "rb").read()).hexdigest() == h, n
    rd = lambda n: [json.loads(l) for l in open(f"/root/app/data/{n}.jsonl")]
    exam, teach, pool = rd("exam"), rd("teach_locked"), rd("teach_pool")
    T = lambda xs: [x["text"] for x in xs]
    R = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    save = lambda: (json.dump(R, open("/vol/final3/raw.json", "w")), vol.commit())
    os.makedirs("/vol/final3", exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MDIR)
    m = AutoModelForCausalLM.from_pretrained(MDIR, dtype=torch.bfloat16, device_map="cuda").eval()
    R["base_exam"] = core.run(m, tok, T(exam))[0]; R["base_teach"] = core.run(m, tok, T(teach))[0]; save()
    import gc; del m; gc.collect(); torch.cuda.empty_cache()

    src = "/vol/artifacts3/C1"
    for v in ("real", "shuffled", "empty"):
        shutil.rmtree(f"/tmp/{v}", ignore_errors=True); shutil.copytree(src, f"/tmp/{v}")
    R["artifact_bytes"] = artifact.disk_bytes(src)
    m, tok = artifact.load("/tmp/real")
    R["small_exam"] = core.run(m, tok, T(exam))[0]; R["small_teach"] = core.run(m, tok, T(teach))[0]
    pp = core.run(m, tok, T(pool))[0]; save()
    idx = []
    for y in ("ham", "spam"):
        idx += [i for i, x in enumerate(pool) if x["label"] == y and pp[i] != y][:12]
    items = [(pool[i]["text"], pool[i]["label"]) for i in idx]
    labels = [y for _, y in items]; sh = labels[:]; random.Random(0).shuffle(sh)
    variants = {"real": items, "shuffled": [(t, y) for (t, _), y in zip(items, sh)], "empty": []}
    R["feedback_items"] = [{"id": pool[i]["id"], "label": pool[i]["label"]} for i in idx]
    R["shuffled_labels"] = sh
    orig_rows = m.lm_head.weight.data.clone()
    for v, its in variants.items():
        m.lm_head.weight.data.copy_(orig_rows)
        R[f"fb_{v}_log"] = artifact.apply_feedback(f"/tmp/{v}", m, tok, its)
        R[f"fb_{v}_teach"] = core.run(m, tok, T(teach))[0]
        R[f"fb_{v}_exam"] = core.run(m, tok, T(exam))[0]
        R[f"fb_{v}_bytes"] = artifact.disk_bytes(f"/tmp/{v}"); save()
    shutil.rmtree("/vol/final3/C1_after_real_feedback", ignore_errors=True)
    shutil.copytree("/tmp/real", "/vol/final3/C1_after_real_feedback")
    R["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); save()
    return "ok"
