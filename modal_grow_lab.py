"""Design lab for the regrow feedback step. Sees build_data/ only (task, calib, dev, guard, teach2 pool + dev): no locked file."""
import json, os, random, sys, time
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
app = modal.App("ten-percent-grow-lab", image=image)


@app.function(volumes={"/vol": vol}, gpu=["A100-80GB", "L40S", "H100", "A100-40GB"], timeout=2 * 3600, cpu=4, memory=49152)
def lab(loc: str = "hi-IN", tag: str = "a", cfgs_json: str = "[{}]", hub_first: bool = False, en_n: int = 1400, shuf: bool = True):
    import torch, core, artifact, grow, stage
    rd = lambda p: [json.loads(l) for l in open(f"/root/app/data/{p}.jsonl")]
    T = lambda xs: [x["text"] for x in xs]
    acc = lambda p, xs: round(100 * sum(a == x["label"] for a, x in zip(p, xs)) / len(xs), 2)
    d = stage.stage("/vol/artifacts5/C1", "/tmp/art"); cfg0 = json.load(open(f"{d}/growth_cfg.json"))
    R = {"loc": loc, "artifact_bytes": artifact.disk_bytes(d), "limit": cfg0["limit_bytes"]}
    out = f"/vol/work7/growlab_{tag}_{loc}.json"; os.makedirs("/vol/work7", exist_ok=True)
    save = lambda: (json.dump(R, open(out, "w")), vol.commit())
    m, tok = artifact.load(d); R["expert_bytes"] = grow.expert_bytes(m)
    pool, dev, en = rd(f"teach2/{loc}/pool"), rd(f"teach2/{loc}/dev"), rd("dev")[:en_n]
    base_en = json.load(open("/root/app/data/base_preds.json"))["dev"]["pred"][:en_n]
    t = time.time(); p_pool = core.run(m, tok, T(pool))[0]; dev0 = core.run(m, tok, T(dev))[0]; en0 = core.run(m, tok, T(en))[0]
    bc = [i for i, (b, x) in enumerate(zip(base_en, en)) if b == x["label"]]
    faith = lambda p: round(100 * sum(p[i] == en[i]["label"] for i in bc) / len(bc), 2)
    wrong = [i for i, (p, x) in enumerate(zip(p_pool, pool)) if p != x["label"]][:24]
    items = [(pool[i]["text"], pool[i]["label"]) for i in wrong]
    R.update(pool_acc=acc(p_pool, pool), dev_acc0=acc(dev0, dev), en_acc0=acc(en0, en), en_faith0=faith(en0), feedback_ids=[pool[i]["id"] for i in wrong],
             feedback_labels=[y for _, y in items], secs_base=round(time.time() - t), room=(cfg0["limit_bytes"] - R["artifact_bytes"] - cfg0["reserve_bytes"]) // R["expert_bytes"])
    print(json.dumps({k: v for k, v in R.items() if k != "feedback_ids"}), flush=True); save()
    local = grow.Base(MDIR); R["runs"] = {}
    def reset():
        for g in reversed(m.tpc_grown):
            grow.detach(m, g["layer"])
        m.tpc_grown = []; grow.save_growth(d, [])
    for ci, c in enumerate(json.loads(cfgs_json)):
        cfg = dict(cfg0); cfg.update(c); name = c.get("name", f"c{ci}")
        variants = [("real", items)]
        if shuf:
            sh = [y for _, y in items]; random.Random(0).shuffle(sh); variants.append(("shuf0", [(t_, y) for (t_, _), y in zip(items, sh)]))
        for vn, its in variants:
            reset(); t = time.time()
            base = grow.Base(cfg["base"], cfg["revision"]) if (hub_first and ci == 0 and vn == "real") else local
            log = grow.apply_feedback(d, m, tok, its, base=base, cfg=cfg, grown=[], log=lambda s: print("  ", s, flush=True))
            m.tpc_grown = [{"layer": l} for l, _ in log["restored"]]
            r = {"log": log, "secs_feedback": round(time.time() - t), "dev_pred": core.run(m, tok, T(dev))[0]}
            r["dev_acc"] = acc(r["dev_pred"], dev)
            if vn == "real":
                e1 = core.run(m, tok, T(en))[0]
                r.update(en_acc=acc(e1, en), en_faith=faith(e1), en_changed=sum(a != b for a, b in zip(e1, en0)))
                if torch.cuda.get_device_properties(0).total_memory > 70e9:  # a second copy of the model needs an 80 GB card
                    m2, tok2 = artifact.load(d); r["fresh_load_same_dev"] = sum(a == b for a, b in zip(core.run(m2, tok2, T(dev))[0], r["dev_pred"]))
                    del m2; torch.cuda.empty_cache()
            R["runs"][f"{name}/{vn}"] = r
            print(name, vn, json.dumps({k: v for k, v in r.items() if k not in ("log", "dev_pred")}), "restored", len(log["restored"]),
                  "bytes_after", log.get("bytes_after"), "fetched", log.get("fetched_bytes"), flush=True); save()
    reset()
    return {k: ({a: b for a, b in v.items() if a not in ("log", "dev_pred")} if isinstance(v, dict) else v) for k, v in R["runs"].items()}


@app.local_entrypoint()
def main(loc: str = "hi-IN", tag: str = "a", cfgs_json: str = "[{}]", hub_first: bool = False, en_n: int = 1400, shuf: bool = True):
    print(json.dumps(lab.remote(loc, tag, cfgs_json, hub_first, en_n, shuf), indent=1))
