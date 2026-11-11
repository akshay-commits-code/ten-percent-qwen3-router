"""The shipped artifact: bit-packed integer codes + fp16 scales (+ int8 zero points where asymmetric) + bf16
router/norms, the tokenizer, the task definition and this code. load() rebuilds a standard HF Qwen3-MoE whose
layers keep a varying number of the base's experts, plus any experts that feedback restored (grow.py)."""
import json, os, shutil
import torch
from safetensors.torch import save_file, load_file

GROUP = 128
PROJ = ("gate_proj", "up_proj", "down_proj")


def pack_bits(codes, bits):
    """uint8 codes in [0, 2^bits) -> uint8 bitstream (little-endian bit order)."""
    v = codes.to(torch.uint8).flatten()
    if bits == 8:
        return v.contiguous()
    if bits == 4:
        v = v.view(-1, 2)
        return (v[:, 0] | (v[:, 1] << 4)).contiguous()
    b = ((v[:, None] >> torch.arange(bits, device=v.device, dtype=torch.uint8)) & 1).flatten()
    b = torch.cat([b, b.new_zeros((-b.numel()) % 8)]).view(-1, 8)
    return (b << torch.arange(8, device=v.device, dtype=torch.uint8)).sum(1, dtype=torch.uint8).contiguous()


def unpack_bits(p, bits, n):
    if bits == 8:
        return p[:n]
    if bits == 4:
        return torch.stack([p & 15, p >> 4], 1).flatten()[:n]
    b = ((p[:, None] >> torch.arange(8, device=p.device, dtype=torch.uint8)) & 1).flatten()[: n * bits].view(n, bits)
    return (b << torch.arange(bits, device=p.device, dtype=torch.uint8)).sum(1, dtype=torch.uint8)


def _put(dst, meta, name, entry):
    codes, s, z, bits = entry
    meta["bits"][name] = bits; meta["shapes"][name] = list(codes.shape)
    if bits >= 16:
        dst[name] = codes.to(torch.bfloat16).contiguous(); return
    dst[name + "::q"] = pack_bits(codes, bits); dst[name + "::s"] = s.contiguous()
    if z is not None:
        dst[name + "::z"] = z.contiguous()


def write(store, model, keep, src_dir, out, code_files):
    """store: module name -> (codes, scales, zeros, bits) from core.compress_inplace(..., store=...)."""
    os.makedirs(out, exist_ok=True)
    main, head, meta = {}, {}, {"group": GROUP, "bits": {}, "shapes": {}}
    for name, e in store.items():
        if ".experts." not in name:
            _put(head if name == "lm_head" else main, meta, name + ".weight", e)
    for l, k in enumerate(keep):
        for p in PROJ:
            es = [store[f"model.layers.{l}.mlp.experts.{e}.{p}"] for e in range(len(k))]
            st = lambda j: None if es[0][j] is None else torch.stack([x[j] for x in es])
            _put(main, meta, f"model.layers.{l}.mlp.experts.{p}", (st(0), st(1), st(2), es[0][3]))
    for name, t in model.state_dict().items():  # not rounded: norms and the sliced router, bf16
        if name.endswith("norm.weight") or name.endswith("mlp.gate.weight"):
            main[name] = t.detach().to(torch.bfloat16).cpu().contiguous()
    save_file({k: v.cpu() for k, v in main.items()}, f"{out}/model.safetensors")
    save_file({k: v.cpu() for k, v in head.items()}, f"{out}/lm_head.safetensors")
    cfg = json.load(open(f"{src_dir}/config.json"))
    cfg["num_experts"] = max(len(k) for k in keep); cfg["tpc_experts_per_layer"] = [len(k) for k in keep]
    json.dump(cfg, open(f"{out}/config.json", "w"), indent=1)
    json.dump(meta, open(f"{out}/quant.json", "w"))
    for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        shutil.copy(f"{src_dir}/{f}", f"{out}/{f}")
    for f in code_files:
        shutil.copy(f, f"{out}/{os.path.basename(f)}")


def _deq(t, meta, name, device):
    from core import dequantize
    bits, shape = meta["bits"][name], meta["shapes"][name]
    if bits >= 16:
        return t[name].to(device)
    n = 1
    for d in shape:
        n *= d
    cd = unpack_bits(t[name + "::q"].to(device), bits, n).view(*shape)
    s = t[name + "::s"].to(device); z = t[name + "::z"].to(device) if name + "::z" in t else None
    if len(shape) == 3:
        return torch.stack([dequantize(cd[e], s[e], None if z is None else z[e], bits, meta["group"]) for e in range(shape[0])])
    return dequantize(cd, s, z, bits, meta["group"])


def load(d, device="cuda"):
    from accelerate import init_empty_weights
    from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig.from_pretrained(d); meta = json.load(open(f"{d}/quant.json"))
    t = load_file(f"{d}/model.safetensors"); t.update(load_file(f"{d}/lm_head.safetensors"))
    sd = {}
    for name in meta["bits"]:
        w = _deq(t, meta, name, device)
        if ".mlp.experts." in name:
            base, p = name.rsplit(".", 1)
            for e in range(w.shape[0]):
                sd[f"{base}.{e}.{p}.weight"] = w[e]
        else:
            sd[name] = w
    for k, v in t.items():
        if "::" not in k and k not in meta["bits"]:
            sd[k] = v.to(device)
    with init_empty_weights():
        m = Qwen3MoeForCausalLM(cfg)
        for l, b in enumerate(m.model.layers):  # layers keep different numbers of experts
            K = cfg.tpc_experts_per_layer[l]
            b.mlp.gate = torch.nn.Linear(cfg.hidden_size, K, bias=False)
            b.mlp.experts = torch.nn.ModuleList(b.mlp.experts[:K]); b.mlp.num_experts = K
    m.load_state_dict(sd, strict=True, assign=True)
    m = m.to(device=device, dtype=torch.bfloat16).eval()
    import grow
    m.tpc_grown = grow.load_growth(d, m)  # experts restored by feedback (growth.safetensors), if any
    return m, AutoTokenizer.from_pretrained(d)


def disk_bytes(d):
    """Bytes of everything shipped in the artifact folder (hidden folders and Python caches are not part of it)."""
    n = 0
    for r, ds, fs in os.walk(d):
        ds[:] = [x for x in ds if not x.startswith(".") and x != "__pycache__"]
        n += sum(os.path.getsize(os.path.join(r, f)) for f in fs if not f.startswith("."))
    return n
