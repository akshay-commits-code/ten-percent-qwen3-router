"""Feedback by regrowth.

The small model is the original model with most of its experts pruned away. Feedback puts some of the original
model's own pruned experts back. It never changes an existing weight value and never creates a new one: a restored
expert is a slice of the original model, rounded by the same fixed rule as every expert that was kept, and its router
row is the original router's row. Nothing is trained, nothing is looked up at answer time, nothing goes in the prompt.

How the annotated items are used:
  1. texts  -> which pruned experts the original router would send these requests to (candidates, in that order);
  2. labels -> a group of candidates stays only if the annotated items move toward their annotated labels
               (most of them individually, and on average by a set amount) and the guard answers hold.
Growth stops before everything shipped exceeds 10% of the original model's fp16 bytes.

Forward passes only: no gradients, no optimiser.
"""
import copy, json, math, os, struct, time
import torch
from safetensors.torch import save_file, load_file
import core
from artifact import pack_bits, unpack_bits, disk_bytes, GROUP, PROJ

LAB = core.BASE_LABELS
BITS, MODE = 3, "asym"  # the rule every kept expert was rounded with


class Base:
    """Read-only access to single tensors of the original model: a local folder holding its safetensors files, or its
    public Hugging Face repo (only the bytes of the requested tensors are fetched, by HTTP range requests)."""

    def __init__(self, src, revision=None):
        self.src, self.rev, self.local = src, revision, os.path.isdir(src)
        self._hdr, self._open, self._sess, self.fetched = {}, {}, None, 0
        if self.local:
            p = f"{src}/model.safetensors.index.json"
        else:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(src, "model.safetensors.index.json", revision=revision)
        self.wmap = json.load(open(p))["weight_map"]

    def _range(self, fname, a, b):
        import requests
        from huggingface_hub import hf_hub_url
        self._sess = self._sess or requests.Session()
        url = hf_hub_url(self.src, fname, revision=self.rev)
        for attempt in range(6):
            try:
                r = self._sess.get(url, headers={"Range": f"bytes={a}-{b - 1}"}, timeout=180)
                if r.status_code == 206 and len(r.content) == b - a:
                    self.fetched += b - a
                    return r.content
            except Exception:
                pass
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"could not fetch {fname} bytes {a}-{b}")

    def tensor(self, name):
        f = self.wmap[name]
        if self.local:
            from safetensors import safe_open
            if f not in self._open:
                self._open[f] = safe_open(f"{self.src}/{f}", "pt", device="cpu")
            return self._open[f].get_tensor(name)
        if f not in self._hdr:
            n = struct.unpack("<Q", self._range(f, 0, 8))[0]
            self._hdr[f] = (json.loads(self._range(f, 8, 8 + n)), 8 + n)
        h, off = self._hdr[f]; e = h[name]; a, b = e["data_offsets"]
        assert e["dtype"] == "BF16", e["dtype"]
        return torch.frombuffer(bytearray(self._range(f, off + a, off + b)), dtype=torch.bfloat16).reshape(e["shape"]).clone()


class Scorer:
    """First-answer-token scores of the 7 skill names for a batch of requests (one forward pass each)."""

    def __init__(self, model, tok):
        self.m, self.tok, self.dev = model, tok, model.lm_head.weight.device
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.label_ids = torch.tensor([core.label_token(tok, y) for y in LAB], device=self.dev)
        self.P = len(tok(core.prompts(tok, [""])[0].split("<|im_start|>user")[0]).input_ids)  # system-prompt tokens
        self.pos_w = None

    def encode(self, texts):
        return [self.tok(p).input_ids for p in core.prompts(self.tok, texts)]

    @torch.no_grad()
    def score(self, ids, bs=64):
        """-> [n, 8]: logits of the 7 skill names at the first answer position, then the best logit of any other token."""
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        out = torch.empty(len(ids), len(LAB) + 1)
        for a in range(0, len(order), bs):
            chunk = order[a:a + bs]; T = max(len(ids[i]) for i in chunk)
            inp = torch.full((len(chunk), T), self.tok.pad_token_id, dtype=torch.long); att = torch.zeros_like(inp)
            for k, i in enumerate(chunk):
                inp[k, T - len(ids[i]):] = torch.tensor(ids[i]); att[k, T - len(ids[i]):] = 1
            pos = (att.cumsum(1) - 1).clamp(min=0)  # real tokens sit at positions 0..n-1 whatever the padding
            self.pos_w = ((pos >= self.P) & (att == 1)).flatten().float().to(self.dev)  # message part of each prompt
            h = self.m.model(input_ids=inp.to(self.dev), attention_mask=att.to(self.dev), position_ids=pos.to(self.dev),
                             use_cache=False).last_hidden_state[:, -1]
            lg = self.m.lm_head(h).float()
            lab = lg[:, self.label_ids].clone(); lg[:, self.label_ids] = float("-inf")
            out[torch.tensor(chunk)] = torch.cat([lab, lg.max(-1, keepdim=True).values], 1).cpu()
        return out


def margins(S, y):
    """Logit of the annotated label minus the best competing token."""
    own = S.gather(1, y[:, None])[:, 0]
    oth = S.clone(); oth.scatter_(1, y[:, None], float("-inf"))
    return own - oth.max(1).values


def preds(S):
    return [LAB[i] if i < len(LAB) else "?" for i in S.argmax(1).tolist()]


def would_be_mass(sc, ids, gates):
    """Top-8 routing weight the ORIGINAL router (all of its experts) would give each expert on the message part of
    these requests, read at the small model's own hidden states. One forward pass."""
    blocks = core.moe_blocks(sc.m)
    mass = [torch.zeros(g.shape[0], dtype=torch.float64, device=sc.dev) for g in gates]
    def mk(l):
        def pre(mod, inp):
            h = inp[0].reshape(-1, inp[0].shape[-1])
            w = torch.softmax((h @ gates[l].T).float(), -1); v, s = torch.topk(w, mod.top_k, -1); v = v / v.sum(-1, keepdim=True)
            mass[l].index_add_(0, s.flatten(), (v * sc.pos_w[:, None]).flatten().double())
        return pre
    hooks = [b.register_forward_pre_hook(mk(l)) for l, b in enumerate(blocks)]
    try:
        sc.score(ids)
    finally:
        for h in hooks:
            h.remove()
    return [x.cpu() for x in mass]


def round_expert(base, l, e, device):
    """The original model's expert (layer l, index e), rounded by the fixed rule. -> (stored form, usable weights)"""
    st, W = {}, {}
    for p in PROJ:
        w = base.tensor(f"model.layers.{l}.mlp.experts.{e}.{p}.weight").to(device)
        cd, s, z = core.quantize(w, BITS, MODE, GROUP)
        st[p] = (pack_bits(cd, BITS).cpu(), s.cpu(), z.cpu())
        W[p] = core.dequantize(cd, s, z, BITS, GROUP, torch.bfloat16)
    return st, W


def attach(model, l, W, row):
    b = core.moe_blocks(model)[l]; ex = copy.deepcopy(b.experts[0])
    for p in PROJ:
        getattr(ex, p).weight.data = W[p].to(b.gate.weight.device)
    b.experts.append(ex)
    b.gate.weight = torch.nn.Parameter(torch.cat([b.gate.weight.data, row[None].to(b.gate.weight)]), requires_grad=False)
    b.gate.out_features += 1; b.num_experts += 1


def detach(model, l):
    b = core.moe_blocks(model)[l]
    del b.experts[-1]
    b.gate.weight = torch.nn.Parameter(b.gate.weight.data[:-1].clone(), requires_grad=False)
    b.gate.out_features -= 1; b.num_experts -= 1


def expert_bytes(model):
    ex = core.moe_blocks(model)[0].experts[0]; n = 0
    for p in PROJ:
        o, i = getattr(ex, p).weight.shape
        n += math.ceil(o * i * BITS / 8) + 3 * (o * i // GROUP)  # codes + fp16 scale + int8 zero point per group
    return n + 2 * model.config.hidden_size  # + its router row (bf16)


def save_growth(d, grown):
    """grown: [{"layer", "expert", "st": {proj: (codes, scales, zeros)}, "row"}] -> growth.safetensors + growth.json"""
    for f in ("growth.safetensors", "growth.json"):
        if os.path.exists(f"{d}/{f}"):
            os.remove(f"{d}/{f}")
    if not grown:
        return
    t = {}
    for i, g in enumerate(grown):
        for p in PROJ:
            t[f"{i}.{p}::q"], t[f"{i}.{p}::s"], t[f"{i}.{p}::z"] = g["st"][p]
        t[f"{i}.router"] = g["row"].to(torch.bfloat16).cpu().contiguous()
    save_file(t, f"{d}/growth.safetensors")
    json.dump({"experts": [[g["layer"], g["expert"]] for g in grown]}, open(f"{d}/growth.json", "w"))


def load_growth(d, model):
    """Attach the experts restored by earlier feedback (called by artifact.load)."""
    if not os.path.exists(f"{d}/growth.json"):
        return []
    meta = json.load(open(f"{d}/growth.json"))["experts"]; t = load_file(f"{d}/growth.safetensors"); grown = []
    dev = model.lm_head.weight.device
    for i, (l, e) in enumerate(meta):
        st, W = {}, {}
        for p in PROJ:
            shape = getattr(core.moe_blocks(model)[l].experts[0], p).weight.shape
            q, s, z = t[f"{i}.{p}::q"], t[f"{i}.{p}::s"], t[f"{i}.{p}::z"]; st[p] = (q, s, z)
            cd = unpack_bits(q.to(dev), BITS, shape[0] * shape[1]).view(*shape)
            W[p] = core.dequantize(cd, s.to(dev), z.to(dev), BITS, GROUP, torch.bfloat16)
        attach(model, l, W, t[f"{i}.router"].to(dev))
        grown.append({"layer": l, "expert": e, "st": st, "row": t[f"{i}.router"]})
    return grown


def apply_feedback(d, model, tok, items, base=None, cfg=None, grown=None, log=print):
    """items: [(text, correct skill)], at most 25, marking mistakes. Restores pruned experts of the original model
    into `model` (in place) and into the artifact folder `d` (growth.safetensors, growth.json)."""
    cfg = cfg or json.load(open(f"{d}/growth_cfg.json")); grown = list(grown or [])
    out = {"annotated": len(items), "mistakes_used": 0, "groups": [], "restored": [], "bytes_before": disk_bytes(d)}
    items = [(t, y) for t, y in items if y in LAB]
    sc = Scorer(model, tok)
    ids = sc.encode([t for t, _ in items]) if items else []
    S0 = sc.score(ids) if items else None
    F = [i for i, (p, (_, y)) in enumerate(zip(preds(S0), items)) if p != y] if items else []  # feedback marks mistakes
    out["mistakes_used"] = len(F)
    room = (cfg["limit_bytes"] - out["bytes_before"] - cfg["reserve_bytes"]) // expert_bytes(model)
    if not F or room <= 0:
        out["bytes_after"] = out["bytes_before"]; return out
    base = base or Base(cfg["base"], cfg.get("revision"))
    ids_f = [ids[i] for i in F]; y = torch.tensor([LAB.index(items[i][1]) for i in F])
    L = len(core.moe_blocks(model))
    gates = [base.tensor(f"model.layers.{l}.mlp.gate.weight").to(sc.dev, torch.bfloat16) for l in range(L)]
    present = [set(k) for k in json.load(open(f"{d}/keep.json"))["keep"]]
    for g in grown:
        present[g["layer"]].add(g["expert"])
    mass = would_be_mass(sc, ids_f, gates)
    cand = sorted(((mass[l][e].item(), l, e) for l in range(L) for e in range(len(mass[l])) if e not in present[l]), reverse=True)
    cand = [(l, e) for ms, l, e in cand if ms > 0][:cfg["n_cand"]]
    guard = [json.loads(x)["text"] for x in open(f"{d}/guard.jsonl")] if os.path.exists(f"{d}/guard.jsonl") else []
    ids_g = sc.encode(guard); g0 = preds(sc.score(ids_g)) if guard else []
    cur_m = margins(sc.score(ids_f), y); misses = 0; G = cfg["group"]; pos = 0
    while pos < len(cand) and room > 0 and misses < cfg["patience"]:
        grp = cand[pos:pos + min(G, room)]; pos += len(grp); new = []
        for l, e in grp:
            st, W = round_expert(base, l, e, sc.dev)
            attach(model, l, W, gates[l][e]); new.append({"layer": l, "expert": e, "st": st, "row": gates[l][e].cpu()})
        new_m = margins(sc.score(ids_f), y)
        gain = (new_m.clamp(cfg["floor"], cfg["cap"]) - cur_m.clamp(cfg["floor"], cfg["cap"])).mean().item()  # average clipped margin gain
        moved = ((new_m - cur_m) > cfg["delta"]).float().mean().item()   # share of annotated items that move toward their label
        changed = sum(a != b for a, b in zip(preds(sc.score(ids_g)), g0)) if guard else 0
        ok = gain >= cfg["tau"] and moved >= cfg["share"] and changed <= cfg["guard_max_changed"]
        out["groups"].append({"experts": grp, "gain_per_item": round(gain, 3), "moved_share": round(moved, 3), "guard_changed": changed, "kept": ok})
        log(json.dumps(out["groups"][-1]))
        if ok:
            grown += new; cur_m = new_m; room -= len(grp); misses = 0; out["restored"] += [list(x) for x in grp]
        else:
            for l, _ in reversed(grp):
                detach(model, l)
            misses += 1
    save_growth(d, grown)
    out["bytes_after"] = disk_bytes(d); out["fetched_bytes"] = base.fetched
    out["fixed_of_mistakes"] = sum(p == items[i][1] for p, i in zip(preds(sc.score(ids_f)), F))
    assert out["bytes_after"] <= cfg["limit_bytes"], "size limit"
    return out
