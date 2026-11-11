"""The ONE run of the revised step 5 (PROTOCOL.md amendments 4-5).

It opens the chosen locale's locked teach file for the first time, and re-opens the exam (disclosed in advance) only to
check the model after feedback. Everything it uses was frozen before: the artifact folder (weights byte-identical to
the first submission, hashes in ARTIFACT.json), growth_cfg.json and the code in this commit.

  modal run modal_final2.py::freeze   # CPU: assemble /vol/artifacts6/C1 and print its file hashes (before the run)
  modal run modal_final2.py::final    # the one run
  TPC_REHEARSAL=1 modal run modal_final2.py::final --rehearsal   # same script on dev files only (hi-IN dev as the
                                                                  # teach set, English dev as the exam, no base, no locked file)
"""
import hashlib, json, os, random, shutil, subprocess, sys, time
import modal

sys.path.insert(0, "/root/app")
MDIR = "/vol/models/qwen3-30b-a3b-instruct-2507"
LOC = "hi-IN"
ART = "/vol/artifacts6/C1"
vol = modal.Volume.from_name("tpc-vol")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", "transformers==4.57.1", "accelerate==1.10.1", "safetensors==0.6.2", "numpy")
         .env({"TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
for f in ("core.py", "artifact.py", "grow.py", "answer.py", "feedback.py", "stage.py", "keep.json", "growth_cfg.json", "LOCK.json", "LOCK2.json"):
    image = image.add_local_file(f, f"/root/app/{f}")
REHEARSAL = os.environ.get("TPC_REHEARSAL") == "1"
# final_data: task, guard, exam, and ONLY the chosen locale's pool + locked.  build_data (rehearsal): dev files only.
image = image.add_local_dir("build_data" if REHEARSAL else "final_data", "/root/app/data")
image = image.add_local_file("ARTIFACT.json", "/root/app/ARTIFACT.json")
app = modal.App("ten-percent-final-6", image=image)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


@app.function(volumes={"/vol": vol}, timeout=3600, cpu=4, memory=8192)
def freeze():
    import stage, artifact
    stage.stage("/vol/artifacts5/C1", ART, link=False)
    out = {"files": {f: {"bytes": os.path.getsize(f"{ART}/{f}"), "sha256": sha(f"{ART}/{f}")} for f in sorted(os.listdir(ART))}}
    out["artifact_bytes"] = artifact.disk_bytes(ART)
    out["weights_identical_to_first_submission"] = all(sha(f"/vol/artifacts5/C1/{f}") == out["files"][f]["sha256"] for f in stage.BIG)
    json.dump(out, open("/vol/artifacts6/ARTIFACT.json", "w"), indent=1); vol.commit()
    print(json.dumps(out, indent=1))
    return out


@app.function(volumes={"/vol": vol}, gpu=["A100-80GB", "H100"], timeout=3 * 3600, cpu=4, memory=49152)  # both have 80 GB
def final(rehearsal: bool = False):
    import gc, torch, core, artifact, grow, stage
    from transformers import AutoTokenizer, AutoModelForCausalLM
    lock, lock2, frozen = json.load(open("/root/app/LOCK.json")), json.load(open("/root/app/LOCK2.json")), json.load(open("/root/app/ARTIFACT.json"))
    for f, v in frozen["files"].items():
        assert sha(f"{ART}/{f}") == v["sha256"], f
    rd = lambda n: [json.loads(l) for l in open(f"/root/app/data/{n}.jsonl")]
    if rehearsal:   # dev files only: no locked file exists in this image
        assert not os.path.exists("/root/app/data/exam.jsonl") and not os.path.exists(f"/root/app/data/teach2/{LOC}/locked.jsonl")
        exam, teach, pool, OUT = rd("dev"), rd(f"teach2/{LOC}/dev"), rd(f"teach2/{LOC}/pool"), "/vol/final6_rehearsal"
    else:
        assert sha("/root/app/data/exam.jsonl") == lock["locked_files"]["exam"]
        assert sha(f"/root/app/data/teach2/{LOC}/locked.jsonl") == lock2["locked_files"][f"teach2/{LOC}/locked.jsonl"]
        exam, teach, pool, OUT = rd("exam"), rd(f"teach2/{LOC}/locked"), rd(f"teach2/{LOC}/pool"), "/vol/final6"
    T = lambda xs: [x["text"] for x in xs]
    os.makedirs(OUT, exist_ok=True)
    R = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "locale": LOC, "rehearsal": rehearsal, "artifact_bytes": artifact.disk_bytes(ART),
         "gpu": torch.cuda.get_device_name(0)}
    save = lambda: (json.dump(R, open(f"{OUT}/raw.json", "w")), vol.commit())

    if not rehearsal:
        tok = AutoTokenizer.from_pretrained(MDIR)
        m = AutoModelForCausalLM.from_pretrained(MDIR, dtype=torch.bfloat16, device_map="cuda").eval()
        R["base_exam"] = core.run(m, tok, T(exam))[0]; R["base_teach"] = core.run(m, tok, T(teach))[0]; save()
        del m; gc.collect(); torch.cuda.empty_cache()

    def check(d):   # the staged folder is exactly the frozen artifact (weights are symlinks to the files checked above)
        for f, v in frozen["files"].items():
            assert f in stage.BIG or sha(f"{d}/{f}") == v["sha256"], f
        return d
    cfg = json.load(open(f"{ART}/growth_cfg.json"))
    d = check(stage.stage(ART, "/tmp/work"))
    m, tok = artifact.load(d)
    R["small_exam"] = core.run(m, tok, T(exam))[0]; R["small_teach"] = core.run(m, tok, T(teach))[0]
    pp = core.run(m, tok, T(pool))[0]; save()
    idx = [i for i, x in enumerate(pool) if pp[i] != x["label"]][:24]          # the first 24 real mistakes, in file order
    items = [(pool[i]["text"], pool[i]["label"]) for i in idx]
    R["feedback_items"] = [{"id": pool[i]["id"], "label": pool[i]["label"], "model_said": pp[i]} for i in idx]
    variants = {"empty": []}
    for seed in (1, 2, 3, 4, 5, 6):   # 1-3 as pre-registered; 4-6 added after the dev rehearsal had already run 1-3 (amendment 5)
        sh = [y for _, y in items]; random.Random(seed).shuffle(sh)
        variants[f"shuffled{seed}"] = [(t, y) for (t, _), y in zip(items, sh)]; R[f"shuffled{seed}_labels"] = sh
    hub = lambda: grow.Base(cfg["base"], cfg["revision"])                        # the shipped default: fetch from the public base repo
    for v, its in variants.items():                                            # controls, in process
        log = grow.apply_feedback(d, m, tok, its, base=hub(), cfg=cfg, grown=[])
        R[f"fb_{v}_log"] = log; R[f"fb_{v}_teach"] = core.run(m, tok, T(teach))[0]
        R[f"fb_{v}_exam"] = core.run(m, tok, T(exam))[0] if log["restored"] else None  # nothing restored = the model is unchanged
        R[f"fb_{v}_bytes"] = artifact.disk_bytes(d); save()
        for l, _ in reversed(log["restored"]):
            grow.detach(m, l)
        grow.save_growth(d, [])
    del m; gc.collect(); torch.cuda.empty_cache()

    # real feedback through the shipped commands, exactly as a user would run them
    d = check(stage.stage(ART, "/tmp/real"))
    with open("/tmp/annotations.jsonl", "w") as f:
        for t, y in items:
            f.write(json.dumps({"text": t, "label": y}, ensure_ascii=False) + "\n")
    t0 = time.time()
    p = subprocess.run([sys.executable, f"{d}/feedback.py", d, "/tmp/annotations.jsonl"], capture_output=True, text=True)
    R["fb_real_cli"] = {"returncode": p.returncode, "stdout": p.stdout[-6000:], "stderr": p.stderr[-3000:], "secs": round(time.time() - t0)}; save()
    assert p.returncode == 0, p.stderr[-2000:]
    R["fb_real_log"] = json.loads(p.stdout[p.stdout.index("{\n"):])
    m, tok = artifact.load(d)                                                  # fresh load of the artifact after feedback
    R["fb_real_teach"] = core.run(m, tok, T(teach))[0]; R["fb_real_exam"] = core.run(m, tok, T(exam))[0]
    R["fb_real_bytes"] = artifact.disk_bytes(d); R["growth"] = json.load(open(f"{d}/growth.json"))
    R["growth_sha256"] = sha(f"{d}/growth.safetensors"); save()
    del m; gc.collect(); torch.cuda.empty_cache()
    with open("/tmp/sample.jsonl", "w") as f:                                  # the answer command on 20 teach items
        for x in teach[:20]:
            f.write(json.dumps({"text": x["text"]}, ensure_ascii=False) + "\n")
    p = subprocess.run([sys.executable, f"{d}/answer.py", d, "/tmp/sample.jsonl"], capture_output=True, text=True)
    R["answer_cli"] = {"returncode": p.returncode, "out": p.stdout.split()[-20:], "same_as_in_process": p.stdout.split()[-20:] == R["fb_real_teach"][:20]}
    os.makedirs(f"{OUT}/after_real_feedback", exist_ok=True)
    for f in ("growth.safetensors", "growth.json"):
        shutil.copy(f"{d}/{f}", f"{OUT}/after_real_feedback/{f}")
    R["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); save()
    return "ok"
