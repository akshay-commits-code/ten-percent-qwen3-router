"""Shared logic: prompting, prediction, routing statistics, compression (prune + round), and the feedback edit."""
import json, math, os, re
import torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
_tp = os.path.join(HERE, "task.json")
TASK = json.load(open(_tp if os.path.exists(_tp) else os.path.join(HERE, "data", "task.json")))
BASE_LABELS = list(TASK["labels"])


def system_prompt():
    return TASK["system"]


def prompts(tok, texts):
    sp = system_prompt()
    return [tok.apply_chat_template([{"role": "system", "content": sp}, {"role": "user", "content": t}],
                                    tokenize=False, add_generation_prompt=True) for t in texts]


def parse(s):
    m = re.match(r"\W*([A-Za-z_]+)", s)
    return m.group(1).lower() if m else ""


@torch.no_grad()
def run(model, tok, texts, bs=24, gen=4, want_hidden=False):
    """Greedy free generation. Returns parsed labels, raw text, and (optionally) a dict with the final-norm hidden
    state at the last prompt position (the vector whose logits pick the first answer token) and the top-1 logit."""
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out_lab, out_raw, out_h, out_top, out_t0 = [], [], [], [], []
    ps = prompts(tok, texts)
    order = sorted(range(len(ps)), key=lambda i: len(ps[i]))
    res = {}
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        enc = tok([ps[j] for j in idx], return_tensors="pt", padding=True).to(model.device)
        g = model.generate(**enc, max_new_tokens=gen, do_sample=False, output_hidden_states=want_hidden,
                           output_logits=want_hidden, return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
        txt = tok.batch_decode(g.sequences[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        if want_hidden:
            hs = g.hidden_states[0][-1][:, -1].float().cpu()
            top = g.logits[0].float().max(-1).values.cpu()
            t0 = g.sequences[:, enc.input_ids.shape[1]].cpu()
        for k, j in enumerate(idx):
            res[j] = (parse(txt[k]), txt[k], hs[k] if want_hidden else None, top[k] if want_hidden else None,
                      t0[k] if want_hidden else None)
    for j in range(len(ps)):
        out_lab.append(res[j][0]); out_raw.append(res[j][1]); out_h.append(res[j][2]); out_top.append(res[j][3]); out_t0.append(res[j][4])
    extra = {"h": torch.stack(out_h), "top": torch.stack(out_top), "tok0": torch.stack(out_t0)} if want_hidden else None
    return out_lab, out_raw, extra


def moe_blocks(model):
    return [l.mlp for l in model.model.layers]


@torch.no_grad()
def routing_mass(model, tok, texts, bs=16):
    """Total top-k routing weight each expert receives over all real (non-pad) prompt tokens."""
    blocks = moe_blocks(model); E = blocks[0].num_experts
    mass = torch.zeros(len(blocks), E, dtype=torch.float64); cur = {}
    hooks = []
    for li, b in enumerate(blocks):
        def h(mod, inp, out, li=li):
            w = F.softmax(out[1].float(), dim=-1); v, s = torch.topk(w, mod.top_k, dim=-1)
            v = v / v.sum(-1, keepdim=True); v = v * cur["mask"][:, None]
            mass[li].index_add_(0, s.flatten().cpu(), v.flatten().double().cpu())
        hooks.append(b.register_forward_hook(h))
    tok.padding_side = "left"
    ps = prompts(tok, texts)
    for i in range(0, len(ps), bs):
        enc = tok(ps[i:i + bs], return_tensors="pt", padding=True).to(model.device)
        cur["mask"] = enc.attention_mask.flatten().float()
        model(**enc)
    for h in hooks:
        h.remove()
    return mass


# ------------------------------- compression: prune experts + round weights -------------------------------

def rtn(W, bits, group=128, grid=(1.0, 0.95, 0.9, 0.85, 0.8, 0.75)):
    """Symmetric round-to-nearest, one scale per `group` input columns, scale picked from a fixed clip grid by MSE.
    Returns integer codes (int8) and fp16 scales. Nothing is optimised; every weight only moves by rounding."""
    o, i = W.shape
    Wg = W.float().reshape(o, i // group, group)
    qmax = 2 ** (bits - 1) - 1
    amax = Wg.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    best_e, best_s = None, None
    for r in grid:
        s = (amax * r / qmax).half().float()
        e = ((torch.clamp(torch.round(Wg / s), -qmax, qmax) * s - Wg) ** 2).sum(-1, keepdim=True)
        if best_e is None:
            best_e, best_s = e, s
        else:
            m = e < best_e; best_e = torch.where(m, e, best_e); best_s = torch.where(m, s, best_s)
    q = torch.clamp(torch.round(Wg / best_s), -qmax, qmax).to(torch.int8).reshape(o, i)
    return q, best_s.squeeze(-1).half()


def rtn_asym(W, bits, group=128, grid=(1.0, 0.95, 0.9, 0.85)):
    """Asymmetric round-to-nearest (scale + zero point per group), range shrink picked from a fixed grid by MSE.
    Returns the dequantized weight directly (diagnostics)."""
    o, i = W.shape
    Wg = W.float().reshape(o, i // group, group)
    lo, hi = Wg.amin(-1, keepdim=True), Wg.amax(-1, keepdim=True)
    qm = 2 ** bits - 1; best_e, best_r = None, None
    for r in grid:
        mid = (hi + lo) / 2; half = (hi - lo) / 2 * r
        s = (2 * half / qm).clamp_min(1e-8).half().float(); z = torch.round((mid - half) / s)
        R = (torch.clamp(torch.round(Wg / s) - z, 0, qm) + z) * s
        e = ((R - Wg) ** 2).sum(-1, keepdim=True)
        if best_e is None:
            best_e, best_r = e, R
        else:
            m = e < best_e; best_e = torch.where(m, e, best_e); best_r = torch.where(m, R, best_r)
    return best_r.reshape(o, i)


def dequant(q, s, group=128, dtype=torch.bfloat16):
    o, i = q.shape
    return (q.float().reshape(o, i // group, group) * s.float()[..., None]).reshape(o, i).to(dtype)


def quantize(W, bits, mode="clip", group=128):
    """Round-to-nearest with one fp16 scale per `group` inputs. Returns (codes uint8, scales fp16, zeros int8|None)
    with W ~= (codes - zero) * scale; symmetric modes use the fixed zero 2^(bits-1)-1.
    mode: clip = symmetric, clip ratio from a fixed grid by MSE; absmax = symmetric, no clipping;
          asym = scale + zero point, range shrink from a fixed grid by MSE."""
    if mode == "asym":
        o, i = W.shape
        Wg = W.float().reshape(o, i // group, group)
        lo, hi = Wg.amin(-1, keepdim=True), Wg.amax(-1, keepdim=True)
        qm = 2 ** bits - 1; best = None
        for r in (1.0, 0.95, 0.9, 0.85):
            mid = (hi + lo) / 2; half = (hi - lo) / 2 * r
            s = (2 * half / qm).clamp_min(1e-8).half().float(); z = torch.round((mid - half) / s)
            cde = torch.clamp(torch.round(Wg / s) - z, 0, qm)
            e = (((cde + z) * s - Wg) ** 2).sum(-1, keepdim=True)
            if best is None:
                best = [e, cde, s, z]
            else:
                m = e < best[0]
                best = [torch.where(m, e, best[0]), torch.where(m, cde, best[1]), torch.where(m, s, best[2]), torch.where(m, z, best[3])]
        zero = -best[3].squeeze(-1)
        assert zero.min() >= -128 and zero.max() <= 127
        return best[1].to(torch.uint8).reshape(o, i), best[2].squeeze(-1).half(), zero.to(torch.int8)
    q, s = rtn(W, bits, group, grid=(1.0,) if mode == "absmax" else (1.0, 0.95, 0.9, 0.85, 0.8, 0.75))
    return (q.to(torch.int16) + 2 ** (bits - 1) - 1).to(torch.uint8), s, None


def dequantize(codes, s, zero, bits, group=128, dtype=torch.bfloat16):
    o, i = codes.shape
    z = zero.float()[..., None] if zero is not None else float(2 ** (bits - 1) - 1)
    return ((codes.float().reshape(o, i // group, group) - z) * s.float()[..., None]).reshape(o, i).to(dtype)


def keep_from_mass(mass, K):
    """Uniform: the K experts with most routing mass in every layer."""
    return [sorted(torch.topk(mass[l], K).indices.tolist()) for l in range(mass.shape[0])]


def keep_variable(mass, total, min_per_layer=16):
    """Variable per layer: rank all experts by their share of their own layer's routing mass and keep the top `total`
    overall (at least `min_per_layer` per layer), so layers with flatter routing keep more experts."""
    L, E = mass.shape
    frac = mass / mass.sum(1, keepdim=True)
    keep = [set(torch.topk(frac[l], min_per_layer).indices.tolist()) for l in range(L)]
    order = torch.argsort(frac.flatten(), descending=True).tolist()
    n = sum(len(k) for k in keep)
    for i in order:
        if n >= total:
            break
        l, e = divmod(i, E)
        if e not in keep[l]:
            keep[l].add(e); n += 1
    return [sorted(k) for k in keep]


def part(name):
    if ".experts." in name: return "experts"
    if "self_attn" in name: return "attn"
    if "embed_tokens" in name: return "embed"
    if name == "lm_head": return "head"
    raise ValueError(name)


@torch.no_grad()
def compress_inplace(model, keep, bits, modes=None, group=128, store=None):
    """bits: {"experts", "attn", "embed", "head"} -> 2..8, or 16 = keep bf16; modes: same keys -> clip/absmax/asym.
    Slices every MoE layer to its kept experts (router rows sliced = the base router restricted to them), then rounds
    weights. If `store` is a dict, keeps (codes, scales, zeros, bits) or (bf16 tensor, None, None, 16) for packaging."""
    modes = modes or {}
    for li, b in enumerate(moe_blocks(model)):
        k = keep[li]
        b.gate.weight = torch.nn.Parameter(b.gate.weight.data[k].clone(), requires_grad=False)
        b.gate.out_features = len(k)
        b.experts = torch.nn.ModuleList([b.experts[e] for e in k])
        b.num_experts = len(k)
    model.config.num_experts = max(len(k) for k in keep)
    model.config.tpc_experts_per_layer = [len(k) for k in keep]
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.Linear, torch.nn.Embedding)) and not name.endswith("mlp.gate"):
            p = part(name); nb = bits[p]
            if nb >= 16:
                if store is not None:
                    store[name] = (mod.weight.data.to(torch.bfloat16).cpu(), None, None, 16)
                continue
            cd, s, z = quantize(mod.weight.data, nb, modes.get(p, "clip"), group)
            if store is not None:
                store[name] = (cd.cpu(), s.cpu(), None if z is None else z.cpu(), nb)
            mod.weight.data.copy_(dequantize(cd, s, z, nb, group, mod.weight.dtype))
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def size_bytes(cfg, keep, bits, group=128):
    """On-disk bytes of the packed tensors (codes bit-packed + fp16 scales; bf16 for unrounded, norms, router)."""
    H, L, V, I = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["vocab_size"], cfg["moe_intermediate_size"]
    qd, kvd = cfg["num_attention_heads"] * cfg["head_dim"], cfg["num_key_value_heads"] * cfg["head_dim"]
    def qb(n, b): return 2 * n if b >= 16 else math.ceil(n * b / 8) + (n // group) * 2
    nk = sum(len(k) for k in keep)
    return (L * qb(H * qd * 2 + H * kvd * 2, bits["attn"]) + qb(V * H, bits["embed"]) + qb(V * H, bits["head"])
            + nk * qb(3 * H * I, bits["experts"]) + (nk * H + L * (2 * H + 2 * cfg["head_dim"]) + H) * 2)


def fp16_bytes(cfg):
    H, L, V, I, E = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["vocab_size"], cfg["moe_intermediate_size"], cfg["num_experts"]
    qd, kvd = cfg["num_attention_heads"] * cfg["head_dim"], cfg["num_key_value_heads"] * cfg["head_dim"]
    per = H * qd * 2 + H * kvd * 2 + E * H + E * 3 * H * I + 2 * H + 2 * cfg["head_dim"]
    return 2 * (L * per + 2 * V * H + H)


# ------------------------------- feedback: closed-form edit of existing output rows -------------------------------

def label_token(tok, label, stats=None):
    """The token the model actually emits first for `label` (recorded from calibration answers when available)."""
    if stats is not None and f"_tok_{label}" in stats:
        return int(stats[f"_tok_{label}"].item())
    ids = tok.encode(label, add_special_tokens=False)
    return ids[0]


def stats_from(H, preds, lam=0.1, rank=256):
    """Calibration statistics used only to compute feedback edits (never at answer time): mean final hidden state per
    predicted label and overall, plus a compact form of the inverse shrunk covariance (top `rank` eigenpairs)."""
    H = H.float(); mu = H.mean(0)
    st = {"_all": mu}
    for q in BASE_LABELS:
        idx = [i for i, p in enumerate(preds) if p == q]
        if idx:
            st[q] = H[idx].mean(0)
    C = torch.cov((H - mu).T); d = C.shape[0]; tau = torch.trace(C) / d
    ev, V = torch.linalg.eigh(C); ev, V = ev.flip(0)[:rank], V.flip(1)[:, :rank]
    rho = (torch.trace(C) - ev.sum()) / (d - rank)
    st["_V"] = V; st["_inv_top"] = 1 / ((1 - lam) * ev + lam * tau); st["_inv_rest"] = (1 / ((1 - lam) * rho + lam * tau)).reshape(1)
    return st


def whiten(st, u):
    V = st["_V"].float(); a = V.T @ u
    return V @ (st["_inv_top"].float() * a) + st["_inv_rest"].float() * (u - V @ a)


def plan_edit(H, top, preds, labels, rows, stats, delta, qt, contrast=0.0):
    """Closed form, no optimisation. For each taught label y (only items the model got wrong):
       u = unit( Cinv @ (mean h of those items - mean stats of what the model predicted for them) )   (whitened)
       alpha = the qt-quantile over those items of the smallest step along u that puts y's logit `delta` above the
               current winner.  New row: W[y] + alpha * u.
    H [n,d] hidden states, top [n] current max logit, rows {label: current W[y] (d,)}. Returns {label: new row}."""
    H = H.float(); top = top.float().clone(); new = {}
    Ci = "_V" in stats
    for y in sorted(set(labels)):
        I = [i for i in range(len(labels)) if labels[i] == y and preds[i] != y]
        if not I:
            continue
        ref = torch.stack([stats.get(preds[i], stats["_all"]).float() for i in I]).mean(0)
        O = [i for i in range(len(labels)) if labels[i] != y]
        if contrast > 0 and O:  # also point away from the other taught labels' examples
            ref = (1 - contrast) * ref + contrast * H[O].mean(0)
        u = H[I].mean(0) - ref
        u = whiten(stats, u) if Ci else u
        u = u / u.norm()
        w = rows[y].float(); proj = H[I] @ u
        need = (top[I] - H[I] @ w + delta) / proj
        need = need[proj > 0]
        if not len(need):
            continue
        alpha = torch.quantile(need.clamp_min(0), qt).item()
        new[y] = w + alpha * u
        top = torch.maximum(top, H @ new[y])
    return new


def requant_row(w, bits=4, mode="clip", group=128):
    cd, s, z = quantize(w[None].float(), bits, mode, group)
    return cd[0], s[0], (None if z is None else z[0])
