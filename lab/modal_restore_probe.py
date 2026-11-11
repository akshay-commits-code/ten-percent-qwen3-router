"""Probe (pool/dev only): how much does putting back the base's own pruned experts help one locale?
Reads single expert tensors straight from the base files; the base model is never loaded."""
import json, os, sys, time
import modal

sys.path.insert(0, "/root/app")
vol = modal.Volume.from_name("tpc-vol")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", "transformers==4.57.1", "accelerate==1.10.1", "safetensors==0.6.2", "numpy")
         .env({"TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
         .add_local_file("core.py", "/root/app/core.py").add_local_file("artifact.py", "/root/app/artifact.py")
         .add_local_file("prune.py", "/root/app/prune.py")
         .add_local_dir("build_data", "/root/app/data"))
app = modal.App("ten-percent-restore-probe", image=image)
MDIR = "/vol/models/qwen3-30b-a3b-instruct-2507"


@app.function(volumes={"/vol": vol}, gpu=["A100-80GB", "L40S", "H100", "A100-40GB"], timeout=3600, cpu=4, memory=49152)
def probe(loc: str = "hi-IN", n_add: int = 272, ks: str = "0,8,16,34,68,136,272", nf: int = 24):
    import copy, random, torch, artifact, prune, core
    from safetensors import safe_open
    from prune import preds, margins, LAB
    t0 = time.time()
    keep = json.load(open("/vol/work5/C1_keep.json"))["keep"]
    wmap = json.load(open(f"{MDIR}/model.safetensors.index.json"))["weight_map"]; handles = {}
    def base_tensor(name):
        f = wmap[name]
        if f not in handles:
            handles[f] = safe_open(f"{MDIR}/{f}", "pt", device="cpu")
        return handles[f].get_tensor(name)
    rd = lambda p: [json.loads(l) for l in open(f"/root/app/data/{p}.jsonl")]
    m, tok = artifact.load("/vol/artifacts5/C1"); pr = prune.Pruner(m, tok); L = len(pr.blocks)
    gates = [base_tensor(f"model.layers.{l}.mlp.gate.weight").to("cuda", torch.bfloat16) for l in range(L)]
    R = {"loc": loc, "gate_rows_match": all(bool((pr.blocks[l].gate.weight.data == gates[l][keep[l]]).all()) for l in range(L))}
    W = base_tensor(f"model.layers.3.mlp.experts.{keep[3][2]}.up_proj.weight").to("cuda")
    cd, s, z = core.quantize(W, 3, "asym", 128)
    R["kept_expert_rederived_equal"] = bool((core.dequantize(cd, s, z, 3, 128, torch.bfloat16) == pr.blocks[3].experts[2].up_proj.weight.data).all())
    print(R, flush=True)
    acc = lambda S, xs: round(100 * sum(p == x["label"] for p, x in zip(preds(S), xs)) / len(xs), 2)
    Y = lambda xs: torch.tensor([LAB.index(x["label"]) for x in xs])
    pool, dev, en = rd(f"teach2/{loc}/pool"), rd(f"teach2/{loc}/dev"), rd("dev")[:400]
    ids_pool, ids_dev, ids_en = (pr.encode([x["text"] for x in xs]) for xs in (pool, dev, en))
    S_pool = pr.score(ids_pool); en0 = preds(pr.score(ids_en))
    wrong = [i for i, (p, x) in enumerate(zip(preds(S_pool), pool)) if p != x["label"]]
    F1, F2 = wrong[:nf], wrong[nf:2 * nf]; ids_f = [ids_pool[i] for i in F1 + F2]
    y1 = Y([pool[i] for i in F1]); sh = y1.tolist(); random.Random(0).shuffle(sh); ysh = torch.tensor(sh)
    # routing the base router would give to pruned experts on the annotated items (message part only)
    mass = [torch.zeros(128, dtype=torch.float64, device="cuda") for _ in range(L)]; on = {"v": False}
    def mk(l):
        def pre(mod, inp):
            if not on["v"]:
                return
            h = inp[0].reshape(-1, inp[0].shape[-1])
            w = torch.softmax((h @ gates[l].T).float(), -1); v, si = torch.topk(w, 8, -1); v = v / v.sum(-1, keepdim=True)
            mass[l].index_add_(0, si.flatten(), (v * pr.pos_w[:, None]).flatten().double())
        return pre
    hooks = [b.register_forward_pre_hook(mk(l)) for l, b in enumerate(pr.blocks)]
    on["v"] = True; pr.routing([ids_pool[i] for i in F1]); on["v"] = False
    for h in hooks:
        h.remove()
    tot = sum(x.sum().item() for x in mass)
    cand = sorted(((mass[l][e].item(), l, e) for l in range(L) for e in range(128) if e not in set(keep[l])), reverse=True)
    R["pruned_share_of_routing"] = round(sum(c[0] for c in cand) / tot, 4)
    R["share_covered_by_top"] = {k: round(sum(c[0] for c in cand[:k]) / tot, 4) for k in (8, 16, 34, 68, 136, 272)}
    cand = cand[:n_add]; R["cands"] = [[l, e, round(ms, 2)] for ms, l, e in cand]
    print({k: R[k] for k in ("pruned_share_of_routing", "share_covered_by_top")}, "layers of top34:", sorted(l for _, l, _ in cand[:34]), flush=True)
    added = []
    for ms, l, e in cand:  # put the expert back, rounded by the same fixed rule as every kept expert
        b = pr.blocks[l]; ex = copy.deepcopy(b.experts[0])
        for p in ("gate_proj", "up_proj", "down_proj"):
            cd, s, z = core.quantize(base_tensor(f"model.layers.{l}.mlp.experts.{e}.{p}.weight").to("cuda"), 3, "asym", 128)
            getattr(ex, p).weight.data = core.dequantize(cd, s, z, 3, 128, torch.bfloat16)
        b.experts.append(ex)
        b.gate.weight = torch.nn.Parameter(torch.cat([b.gate.weight.data, gates[l][e:e + 1]]), requires_grad=False)
        b.gate.out_features += 1; b.num_experts += 1; pr.E[l] += 1; added.append((l, b.num_experts - 1))
    R["secs_setup"] = round(time.time() - t0); print("setup", R["secs_setup"], flush=True)
    R["k"] = {}
    for K in [int(x) for x in ks.split(",")]:
        pr.removed = [set() for _ in range(L)]
        for l, j in added[K:]:
            pr.removed[l].add(j)
        Sd, Se, Sf = pr.score(ids_dev), pr.score(ids_en), pr.score(ids_f)
        R["k"][K] = {"dev_acc": acc(Sd, dev), "en_acc": acc(Se, en), "en_same": sum(a == b for a, b in zip(preds(Se), en0)),
                     "f1_fixed": sum(p == pool[i]["label"] for p, i in zip(preds(Sf[:nf]), F1)),
                     "f2_fixed": sum(p == pool[i]["label"] for p, i in zip(preds(Sf[nf:]), F2)),
                     "f1_margin_real": round(margins(Sf[:nf], y1).clamp(-8, 3).mean().item(), 3),
                     "f1_margin_shuf": round(margins(Sf[:nf], ysh).clamp(-8, 3).mean().item(), 3),
                     "dev_by_label": {y: acc(Sd[[i for i, x in enumerate(dev) if x["label"] == y]], [x for x in dev if x["label"] == y]) for y in LAB}}
        print(K, json.dumps(R["k"][K]), flush=True)
        json.dump(R, open(f"/vol/work7/restore_probe_{loc}.json", "w")); vol.commit()
    return R


@app.local_entrypoint()
def main(loc: str = "hi-IN"):
    R = probe.remote(loc)
    print(json.dumps({k: v for k, v in R.items() if k != "cands"}, indent=1))
