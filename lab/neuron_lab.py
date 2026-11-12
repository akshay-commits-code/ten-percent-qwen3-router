"""Self-contained idea: feedback only prunes single neurons inside the shipped experts (never fetches or changes a
value). Neurons are ranked by a first-order estimate of how much removing them moves the annotated items toward their
labels (one backward pass), minus how much it disturbs the guard answers. pool / dev files only."""
import json, random, time
import torch
import core


class NeuronLab:
    def __init__(self, model, tok, labels):
        self.m, self.tok, self.LAB = model, tok, labels
        model.requires_grad_(False)
        self.dev = model.lm_head.weight.device
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.label_ids = torch.tensor([core.label_token(tok, y) for y in labels], device=self.dev)
        self.exps = [(l, e, ex) for l, b in enumerate(core.moe_blocks(model)) for e, ex in enumerate(b.experts)]
        self.I = self.exps[0][2].down_proj.weight.shape[1]
        self.acc = torch.zeros(len(self.exps), self.I, device=self.dev)
        self.on = False
        self.hooks = [ex.down_proj.register_forward_pre_hook(self._mk(k)) for k, (_, _, ex) in enumerate(self.exps)]
        self.backup = {}

    def _mk(self, k):
        def pre(mod, inp):
            h = inp[0]
            if self.on and h.requires_grad:
                def grab(g, h=h, k=k):
                    self.acc[k].add_((g.float() * h.detach().float()).sum(0))  # returns None: the gradient is not altered
                h.register_hook(grab)
        return pre

    def encode(self, texts):
        return [self.tok(p).input_ids for p in core.prompts(self.tok, texts)]

    def _batch(self, ids, chunk):
        T = max(len(ids[i]) for i in chunk)
        inp = torch.full((len(chunk), T), self.tok.pad_token_id, dtype=torch.long); att = torch.zeros_like(inp)
        for k, i in enumerate(chunk):
            inp[k, T - len(ids[i]):] = torch.tensor(ids[i]); att[k, T - len(ids[i]):] = 1
        pos = (att.cumsum(1) - 1).clamp(min=0)
        return inp.to(self.dev), att.to(self.dev), pos.to(self.dev)

    def _logits(self, ids, chunk, grad=False):
        inp, att, pos = self._batch(ids, chunk)
        emb = self.m.model.embed_tokens(inp)
        if grad:
            emb = emb.detach().requires_grad_(True)
        h = self.m.model(inputs_embeds=emb, attention_mask=att, position_ids=pos, use_cache=False).last_hidden_state[:, -1]
        lg = self.m.lm_head(h).float()
        lab = lg[:, self.label_ids]
        mask = torch.zeros_like(lg); mask[:, self.label_ids] = float("-inf")
        return torch.cat([lab, (lg + mask).max(-1, keepdim=True).values], 1)

    @torch.no_grad()
    def score(self, ids, bs=64):
        order = sorted(range(len(ids)), key=lambda i: len(ids[i])); out = torch.empty(len(ids), len(self.LAB) + 1)
        for a in range(0, len(order), bs):
            chunk = order[a:a + bs]; out[torch.tensor(chunk)] = self._logits(ids, chunk).cpu()
        return out

    def preds(self, S):
        return [self.LAB[i] if i < len(self.LAB) else "?" for i in S.argmax(1).tolist()]

    def grad_effect(self, ids, y, bs=12, y_neg=None, c=0.0):
        """[n_experts, neurons]: first-order change of sum_i log p(label_i) if a neuron is removed (positive = helps).
        With y_neg and c > 0 the objective also pushes away from another label: log p(y) - c * log p(y_neg)."""
        self.acc.zero_(); self.on = True
        try:
            for a in range(0, len(ids), bs):
                chunk = list(range(a, min(a + bs, len(ids))))
                with torch.enable_grad():
                    S = self._logits(ids, chunk, grad=True); lp = torch.log_softmax(S, -1)
                    J = lp.gather(1, y[chunk][:, None].to(S.device)).sum()
                    if y_neg is not None and c > 0:
                        J = J - c * lp.gather(1, y_neg[chunk][:, None].to(S.device)).sum()
                J.backward()
        finally:
            self.on = False
        return -self.acc.clone()

    def prune(self, flat_idx):
        """Zero the output weights of the chosen neurons (flat index = expert * neurons + neuron)."""
        by = {}
        for f in flat_idx.tolist():
            by.setdefault(f // self.I, []).append(f % self.I)
        for k, js in by.items():
            W = self.exps[k][2].down_proj.weight.data; j = torch.tensor(js, device=W.device)
            self.backup.setdefault(k, []).append((j, W[:, j].clone())); W[:, j] = 0

    def restore(self):
        for k, lst in self.backup.items():
            W = self.exps[k][2].down_proj.weight.data
            for j, cols in reversed(lst):
                W[:, j] = cols
        self.backup = {}


def margins(S, y):
    own = S.gather(1, y[:, None])[:, 0]
    oth = S.clone(); oth.scatter_(1, y[:, None], float("-inf"))
    return own - oth.max(1).values


def experiment(nl, name, pool, dev, en, guard_texts, pick, Ks, lam, log=print):
    """pick(pool, preds) -> indices of the feedback items (mistakes) and of a second held-out batch of mistakes."""
    LAB = nl.LAB; Y = lambda xs: torch.tensor([LAB.index(x["label"]) for x in xs]); T = lambda xs: [x["text"] for x in xs]
    acc = lambda S, xs: round(100 * sum(p == x["label"] for p, x in zip(nl.preds(S), xs)) / len(xs), 2)
    ids_pool, ids_dev, ids_en, ids_g = nl.encode(T(pool)), nl.encode(T(dev)), nl.encode(T(en)), nl.encode(guard_texts)
    S_pool, S_dev0, S_en0, S_g0 = nl.score(ids_pool), nl.score(ids_dev), nl.score(ids_en), nl.score(ids_g)
    F1, F2 = pick(pool, nl.preds(S_pool))
    ids_f1, ids_f2 = [ids_pool[i] for i in F1], [ids_pool[i] for i in F2]
    y1, y2 = Y([pool[i] for i in F1]), Y([pool[i] for i in F2])
    en0, g0 = nl.preds(S_en0), nl.preds(S_g0); yg = S_g0.argmax(1)
    R = {"name": name, "dev_acc0": acc(S_dev0, dev), "en_acc0": acc(S_en0, en), "n_f1": len(F1), "n_f2": len(F2),
         "f1_labels": [pool[i]["label"] for i in F1], "runs": {}}
    log(json.dumps({k: v for k, v in R.items() if k != "runs"}))
    t = time.time(); Gg = nl.grad_effect(ids_g, yg); R["secs_guard_grad"] = round(time.time() - t)
    sh = y1.tolist(); random.Random(0).shuffle(sh)
    for vn, y in (("real", y1), ("shuf0", torch.tensor(sh))):
        t = time.time(); G = nl.grad_effect(ids_f1, y)
        score = G / len(F1) - lam * Gg.abs() / len(guard_texts)
        order = score.flatten().argsort(descending=True)
        m0 = margins(nl.score(ids_f1), y)
        for K in Ks:
            nl.prune(order[:K])
            Sf, S2, Sg, Sd, Se = nl.score(ids_f1), nl.score(ids_f2), nl.score(ids_g), nl.score(ids_dev), nl.score(ids_en)
            mm = margins(Sf, y)
            r = {"predicted_gain_per_item": round(G.flatten()[order[:K]].sum().item() / len(F1), 2),
                 "margin_gain_per_item": round((mm.clamp(-8, 3) - m0.clamp(-8, 3)).mean().item(), 3),
                 "moved_share": round(((mm - m0) > 0.5).float().mean().item(), 3),
                 "f1_fixed": sum(p == pool[i]["label"] for p, i in zip(nl.preds(Sf), F1)),
                 "f2_fixed": sum(p == pool[i]["label"] for p, i in zip(nl.preds(S2), F2)),
                 "guard_changed": sum(a != b for a, b in zip(nl.preds(Sg), g0)),
                 "dev_acc": acc(Sd, dev), "en_acc": acc(Se, en), "en_same": sum(a == b for a, b in zip(nl.preds(Se), en0)), "en_n": len(en0)}
            R["runs"][f"{vn}/K{K}"] = r; log(f"{name} {vn} K={K} " + json.dumps(r))
            nl.restore()
    return R


def experiment2(nl, pool, dev, en, guard_texts, pick, Ks, configs, log=print):
    """New-skill teaching with a larger guard and a locality filter. configs: [{"lam", "c", "q"}]:
    lam = weight of the guard disturbance, c = push away from the other taught label, q = only neurons whose guard
    disturbance is below this quantile are eligible."""
    LAB = nl.LAB; Y = lambda xs: torch.tensor([LAB.index(x["label"]) for x in xs]); T = lambda xs: [x["text"] for x in xs]
    acc = lambda S, xs: round(100 * sum(p == x["label"] for p, x in zip(nl.preds(S), xs)) / len(xs), 2)
    ids_pool, ids_dev, ids_en, ids_g = nl.encode(T(pool)), nl.encode(T(dev)), nl.encode(T(en)), nl.encode(guard_texts)
    S_pool, S_dev0, S_en0, S_g0 = nl.score(ids_pool), nl.score(ids_dev), nl.score(ids_en), nl.score(ids_g)
    F1, F2 = pick(pool, nl.preds(S_pool)); ids_f1 = [ids_pool[i] for i in F1]; y1 = Y([pool[i] for i in F1])
    taught = sorted(set(y1.tolist())); other = lambda y: torch.tensor([taught[1] if v == taught[0] else taught[0] for v in y.tolist()])
    en0, g0 = nl.preds(S_en0), nl.preds(S_g0)
    R = {"dev_acc0": acc(S_dev0, dev), "en_acc0": acc(S_en0, en), "n_guard": len(guard_texts), "runs": {}}
    by0 = {q: acc(S_dev0[[i for i, x in enumerate(dev) if x["label"] == q]], [x for x in dev if x["label"] == q]) for q in ("recipe", "music")}
    log(json.dumps({k: v for k, v in R.items() if k != "runs"} | {"by_label0": by0}))
    Gg = nl.grad_effect(ids_g, S_g0.argmax(1)).abs() / len(guard_texts)
    sh = y1.tolist(); random.Random(0).shuffle(sh); ysh = torch.tensor(sh)
    for cf in configs:
        for vn, y in (("real", y1), ("shuf0", ysh)):
            G = nl.grad_effect(ids_f1, y, y_neg=other(y), c=cf["c"]) / len(F1)
            score = G - cf["lam"] * Gg
            if cf.get("q", 1.0) < 1.0:
                thr = torch.quantile(Gg.flatten()[torch.randperm(Gg.numel(), device=Gg.device)[:1000000]], cf["q"])
                score = torch.where(Gg <= thr, score, torch.full_like(score, float("-inf")))
            order = score.flatten().argsort(descending=True)
            for K in Ks:
                nl.prune(order[:K])
                Sg, Sd, Se = nl.score(ids_g), nl.score(ids_dev), nl.score(ids_en); pd = nl.preds(Sd)
                r = {"guard_changed": sum(a != b for a, b in zip(nl.preds(Sg), g0)), "dev_acc": acc(Sd, dev),
                     "recipe": round(100 * sum(p == x["label"] for p, x in zip(pd, dev) if x["label"] == "recipe") / sum(x["label"] == "recipe" for x in dev), 1),
                     "music": round(100 * sum(p == x["label"] for p, x in zip(pd, dev) if x["label"] == "music") / sum(x["label"] == "music" for x in dev), 1),
                     "en_acc": acc(Se, en), "en_same": sum(a == b for a, b in zip(nl.preds(Se), en0)), "en_n": len(en0)}
                key = f"lam{cf['lam']}_c{cf['c']}_q{cf.get('q', 1.0)}/{vn}/K{K}"; R["runs"][key] = r; log(key + " " + json.dumps(r))
                nl.restore()
    return R
