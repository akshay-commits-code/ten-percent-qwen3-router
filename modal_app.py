"""Modal build jobs. They only ever see build_data/ (dev, calib, teach_dev, teach_pool, task.json).
The locked exam is uploaded only by modal_final.py."""
import json, os, sys, time
import modal

sys.path.insert(0, "/root/app")

MDIR = "/vol/models/qwen3-30b-a3b-instruct-2507"
vol = modal.Volume.from_name("tpc-vol", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", "transformers==4.57.1", "accelerate==1.10.1", "safetensors==0.6.2", "numpy")
         .env({"TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
         .add_local_file("core.py", "/root/app/core.py")
         .add_local_file("artifact.py", "/root/app/artifact.py")
         .add_local_file("answer.py", "/root/app/answer.py")
         .add_local_file("feedback.py", "/root/app/feedback.py")
         .add_local_dir("build_data", "/root/app/data"))
app = modal.App("ten-percent-5", image=image)
GPU = dict(volumes={"/vol": vol}, gpu="A100-80GB", timeout=3 * 3600, cpu=4, memory=49152)


def _load():
    import sys; sys.path.insert(0, "/root/app")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MDIR)
    m = AutoModelForCausalLM.from_pretrained(MDIR, dtype=torch.bfloat16, device_map="cuda").eval()
    return m, tok


def _read(name):
    return [json.loads(l) for l in open(f"/root/app/data/{name}.jsonl")]


def _eval_sets(m, tok, tag):
    import torch, core
    out = {}
    for name in ["dev", "teach_dev", "teach_pool", "calib"]:
        xs = _read(name); t = time.time()
        lab, raw, ex = core.run(m, tok, [x["text"] for x in xs], want_hidden=True)
        out[name] = {"pred": lab, "raw": raw, "secs": round(time.time() - t)}
        torch.save({"h": ex["h"].half(), "top": ex["top"]}, f"/vol/work5/{tag}_{name}_h.pt")
        json.dump(out, open(f"/vol/work5/{tag}_preds.json", "w")); vol.commit()
        print(tag, name, out[name]["secs"], "s", flush=True)
    ids = {w: core.label_token(tok, w) for w in core.BASE_LABELS}
    torch.save({"ids": ids, "rows": m.lm_head.weight.data[list(ids.values())].float().cpu()}, f"/vol/work5/{tag}_label_rows.pt")
    vol.commit()
    return out


@app.function(**GPU)
def base_stage():
    import torch, core
    os.makedirs("/vol/work", exist_ok=True)
    t = time.time(); m, tok = _load(); load_s = round(time.time() - t); print("load", load_s, flush=True)
    t = time.time()
    mass = core.routing_mass(m, tok, [x["text"] for x in _read("calib")])
    torch.save(mass, "/vol/work5/routing_mass.pt"); vol.commit(); print("routing", round(time.time() - t), flush=True)
    _eval_sets(m, tok, "base")
    return {"load_s": load_s}


@app.function(**GPU)
def candidate_stage(k: int, bits: int):
    K = k
    import torch, core, artifact
    m, tok = _load()
    mass = torch.load("/vol/work5/routing_mass.pt"); keep = core.keep_from_mass(mass, K); store = {}
    t = time.time(); core.compress_inplace(m, keep, bits, 4, store=store); print("compress", round(time.time() - t), flush=True)
    tag = f"K{K}b{bits}"
    out = _eval_sets(m, tok, tag)
    d = f"/vol/artifacts5/{tag}"
    files = ["/root/app/core.py", "/root/app/artifact.py", "/root/app/data/task.json"]
    t = time.time(); artifact.write(store, m, keep, MDIR, d, files); vol.commit(); print("pack", round(time.time() - t), flush=True)
    del m, store; torch.cuda.empty_cache()
    m2, tok2 = artifact.load(d)
    xs = _read("dev")[:300]
    _, raw2, _ = core.run(m2, tok2, [x["text"] for x in xs])
    same = sum(a == b for a, b in zip(raw2, out["dev"]["raw"][:300]))
    res = {"tag": tag, "artifact_bytes_so_far": artifact.disk_bytes(d), "roundtrip_same_of_300": same}
    json.dump(res, open(f"/vol/work5/{tag}_pack.json", "w")); vol.commit()
    return res


LAB = dict(volumes={"/vol": vol}, gpu="A100-80GB", timeout=3 * 3600, cpu=8, memory=90112)


@app.function(**LAB)
def lab_stage(configs_json: str):
    """Dev-only lab: load the base once, then evaluate many (pruning mask, rounding) configs on dev.
    Pruning is done by masking router logits of dropped experts to -inf (identical to slicing the router)."""
    import torch, core
    import torch.nn.functional as F
    m, tok = _load(); blocks = core.moe_blocks(m)
    cfgs = json.loads(configs_json)
    # routing mass split: system-prompt tokens vs. the rest (message + assistant header), and last position only
    os.makedirs("/vol/work5", exist_ok=True)
    calib = [x["text"] for x in _read("calib")] if not os.path.exists("/vol/work5/routing_mass_split.pt") else []
    P = len(tok(core.prompts(tok, [""])[0].split("<|im_start|>user")[0], add_special_tokens=False).input_ids)
    mass = {k: torch.zeros(len(blocks), 128, dtype=torch.float64) for k in ("sys", "msg", "last")}
    cur = {}
    def hook(mod, inp, out, li):
        w = F.softmax(out[1].float(), -1); v, sidx = torch.topk(w, 8, -1); v = v / v.sum(-1, keepdim=True)
        for k in mass:
            vv = v * cur[k][:, None]; mass[k][li].index_add_(0, sidx.flatten().cpu(), vv.flatten().double().cpu())
    hs = [b.register_forward_hook(lambda mod, i, o, li=li: hook(mod, i, o, li)) for li, b in enumerate(blocks)]
    tok.padding_side = "left"
    with torch.no_grad():
        for i in range(0, len(calib), 16):
            enc = tok(core.prompts(tok, calib[i:i + 16]), return_tensors="pt", padding=True).to("cuda")
            am = enc.attention_mask; pad = (am == 0).sum(1, keepdim=True)
            pos = torch.arange(am.shape[1], device="cuda")[None] - pad
            cur["sys"] = ((pos >= 0) & (pos < P)).flatten().float(); cur["msg"] = (pos >= P).flatten().float()
            last = torch.zeros_like(am); last[:, -1] = 1; cur["last"] = last.flatten().float()
            m(**enc)
    for h in hs:
        h.remove()
    if calib:
        torch.save(mass, "/vol/work5/routing_mass_split.pt"); vol.commit()
    else:
        mass = torch.load("/vol/work5/routing_mass_split.pt")
    # backups of every rounded weight (CPU) so configs can be applied and undone without reloading
    lin = {n: mod for n, mod in m.named_modules() if isinstance(mod, (torch.nn.Linear, torch.nn.Embedding)) and not n.endswith("mlp.gate")}
    backup = {n: mod.weight.data.to("cpu", copy=True) for n, mod in lin.items()}
    SYS0 = core.TASK["system"]
    xs = _read("dev"); rpath = f"/vol/work5/lab_{cfgs[0].get('run', 'x')}.json"
    results = json.load(open(rpath)) if os.path.exists(rpath) else {}
    for c in cfgs:
        name = c["name"]
        if name in results:
            continue
        mk = mass[c["mass"]] if c.get("mass") in mass else (mass["sys"] * c.get("w_sys", 1.0) + mass["msg"] + mass["last"] * c.get("w_last", 0.0))
        if c.get("uniform"):
            keep = core.keep_from_mass(mk, c["uniform"])
        elif c.get("total"):
            keep = core.keep_variable(mk, c["total"])
        else:
            keep = [list(range(128))] * len(blocks)
        gh = []
        for li, b in enumerate(blocks):
            bias = torch.full((128,), float("-inf"), device="cuda", dtype=torch.bfloat16); bias[keep[li]] = 0
            gh.append(b.gate.register_forward_hook(lambda mod, i, o, bias=bias: o + bias))
        bits = c.get("bits", {})
        with torch.no_grad():
            for n, mod in lin.items():
                nb = bits.get(core.part(n), 16); mode = c.get("qmode", {}).get(core.part(n), "clip")
                if nb < 16:
                    if mode == "asym":
                        mod.weight.data.copy_(core.rtn_asym(mod.weight.data, nb).to(mod.weight.dtype))
                    else:
                        q, sc = core.rtn(mod.weight.data, nb, grid=(1.0,) if mode == "absmax" else (1.0, 0.95, 0.9, 0.85, 0.8, 0.75))
                        mod.weight.data.copy_(core.dequant(q, sc, 128, mod.weight.dtype))
        core.TASK["system"] = c.get("system", SYS0)
        t = time.time(); lab, raw, _ = core.run(m, tok, [x["text"] for x in xs])
        results[name] = {"pred": lab, "keep_counts": [len(k) for k in keep], "secs": round(time.time() - t)}
        json.dump(results, open(rpath, "w")); vol.commit(); print(name, results[name]["secs"], flush=True)
        for h in gh:
            h.remove()
        with torch.no_grad():
            for n, mod in lin.items():
                if bits.get(core.part(n), 16) < 16:
                    mod.weight.data.copy_(backup[n].to("cuda"))
    return list(results)


C1 = {"total": 2650, "w_sys": 0.05, "w_last": 20.0, "bits": {"experts": 3, "attn": 6, "embed": 4, "head": 4},
      "modes": {"experts": "asym", "embed": "absmax", "attn": "clip", "head": "clip"}}


@app.function(**GPU)
def build_final():
    """Build the chosen config (C1) as the real artifact, save its states for the feedback design, verify round trip."""
    import torch, core, artifact
    from safetensors.torch import save_file
    m, tok = _load()
    ms = torch.load("/vol/work5/routing_mass_split.pt")
    mass = ms["sys"] * C1["w_sys"] + ms["msg"] + ms["last"] * C1["w_last"]
    keep = core.keep_variable(mass, C1["total"]); store = {}
    core.compress_inplace(m, keep, C1["bits"], C1["modes"], store=store)
    out = _eval_sets(m, tok, "C1")
    d = "/vol/artifacts5/C1"
    files = ["/root/app/" + f for f in ("core.py", "artifact.py", "answer.py", "feedback.py")] + ["/root/app/data/task.json"]
    artifact.write(store, m, keep, MDIR, d, files)
    S = torch.load("/vol/work5/C1_calib_h.pt")
    st = core.stats_from(S["h"].float(), out["calib"]["pred"])
    save_file({k: v.half().contiguous() for k, v in st.items()}, f"{d}/feedback_stats.safetensors")
    json.dump({"keep": keep}, open("/vol/work5/C1_keep.json", "w")); vol.commit()
    del m, store; torch.cuda.empty_cache()
    m2, tok2 = artifact.load(d)
    xs = _read("dev")[:300]
    _, raw2, _ = core.run(m2, tok2, [x["text"] for x in xs])
    same = sum(a == b for a, b in zip(raw2, out["dev"]["raw"][:300]))
    res = {"artifact_bytes": artifact.disk_bytes(d), "roundtrip_same_of_300": same}
    json.dump(res, open("/vol/work5/C1_pack.json", "w")); vol.commit()
    return res


@app.function(**GPU)
def verify_c1():
    """Fresh container: load the shipped artifact exactly as a user would and compare with the build-time answers."""
    import torch, core, artifact
    d = "/vol/artifacts5/C1"
    m, tok = artifact.load(d)
    P = json.load(open("/vol/work5/C1_preds.json"))
    xs = _read("dev")[:400]
    _, raw, _ = core.run(m, tok, [x["text"] for x in xs])
    same = sum(a == b for a, b in zip(raw, P["dev"]["raw"][:400]))
    res = {"artifact_bytes": artifact.disk_bytes(d), "files": {f: os.path.getsize(f"{d}/{f}") for f in sorted(os.listdir(d))},
           "roundtrip_same_of_400": same}
    json.dump(res, open("/vol/work5/C1_pack.json", "w")); vol.commit()
    return res
