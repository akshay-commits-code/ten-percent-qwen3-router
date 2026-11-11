"""Feedback by pruning.

Feedback never changes a weight value and never adds anything. It only removes more of the model's experts, the same
operation the compression itself used (drop an expert and its router row). After feedback the model is a strict subset
of the model before feedback; the list of removed experts is the only thing stored.

Everything here is forward passes: no gradients, no optimiser.
"""
import torch, torch.nn.functional as F
import core

LAB = core.BASE_LABELS


class Pruner:
    """Wraps a loaded model. `removed[l]` = set of expert indices of layer l that are switched off (router logit = -inf,
    identical to slicing the expert and its router row away). Extra per-row removals let many candidates be tried in
    one batch."""

    def __init__(self, model, tok, removed=None):
        self.m, self.tok = model, tok
        self.blocks = core.moe_blocks(model)
        self.E = [b.num_experts for b in self.blocks]
        self.removed = [set(r) for r in removed] if removed else [set() for _ in self.blocks]
        self.dev = model.lm_head.weight.device
        self.cur, self.T, self.mass, self.pos_w = {}, 1, None, None
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.label_ids = torch.tensor([core.label_token(tok, y) for y in LAB], device=self.dev)
        self.P = len(tok(core.prompts(tok, [""])[0].split("<|im_start|>user")[0]).input_ids)  # system-prompt tokens
        self._hooks = [b.gate.register_forward_hook(self._gate_hook(l)) for l, b in enumerate(self.blocks)]
        self._hooks += [b.register_forward_hook(self._mass_hook(l)) for l, b in enumerate(self.blocks)]

    def close(self):
        for h in self._hooks:
            h.remove()

    def _gate_hook(self, l):
        def h(mod, inp, out):
            b = self.cur.get(l)
            if b is None:
                return None
            b = b.repeat_interleave(self.T, 0) if out.dim() == 2 else b[:, None, :]
            return out + b.to(out.dtype)
        return h

    def _mass_hook(self, l):
        def h(mod, inp, out):
            if self.mass is None:
                return
            w = F.softmax(out[1].float(), -1); w = w.reshape(-1, w.shape[-1])
            v, s = torch.topk(w, mod.top_k, -1); v = v / v.sum(-1, keepdim=True)
            self.mass[l].index_add_(0, s.flatten(), (v * self.pos_w[:, None]).flatten().double())
        return h

    def _set_bias(self, extra):
        R, cur = len(extra), {}
        layers = {l for l in range(len(self.blocks)) if self.removed[l]} | {l for ex in extra for l, _ in ex}
        for l in layers:
            b = torch.zeros(R, self.E[l], device=self.dev)
            if self.removed[l]:
                b[:, sorted(self.removed[l])] = float("-inf")
            for r, ex in enumerate(extra):
                for ll, e in ex:
                    if ll == l:
                        b[r, e] = float("-inf")
            cur[l] = b
        self.cur = cur

    def encode(self, texts):
        return [self.tok(p).input_ids for p in core.prompts(self.tok, texts)]

    @torch.no_grad()
    def score(self, ids, rows=None, bs=96):
        """ids: token-id lists (one per item). rows: [(item_index, ((layer, expert), ...)), ...], one output row each,
        with those experts removed on top of self.removed; default = every item once.
        Returns [n_rows, 8]: the first-answer-token logits of the 7 skill names, then the best logit of any other token."""
        if rows is None:
            rows = [(i, ()) for i in range(len(ids))]
        order = sorted(range(len(rows)), key=lambda r: len(ids[rows[r][0]]))
        out = torch.empty(len(rows), len(LAB) + 1)
        for a in range(0, len(order), bs):
            chunk = order[a:a + bs]
            T = max(len(ids[rows[r][0]]) for r in chunk)
            inp = torch.full((len(chunk), T), self.tok.pad_token_id, dtype=torch.long); att = torch.zeros_like(inp)
            for k, r in enumerate(chunk):
                x = ids[rows[r][0]]; inp[k, T - len(x):] = torch.tensor(x); att[k, T - len(x):] = 1
            self._set_bias([rows[r][1] for r in chunk]); self.T = T
            if self.mass is not None:  # count routing only on real tokens after the system prompt
                pos = torch.arange(T)[None] - (T - att.sum(1, keepdim=True))
                self.pos_w = (pos >= self.P).flatten().float().to(self.dev)
            pos_ids = (att.cumsum(1) - 1).clamp(min=0)  # real tokens always sit at positions 0..n-1, whatever the padding
            h = self.m.model(input_ids=inp.to(self.dev), attention_mask=att.to(self.dev), position_ids=pos_ids.to(self.dev),
                             use_cache=False).last_hidden_state[:, -1]
            lg = self.m.lm_head(h).float()
            lab = lg[:, self.label_ids].clone(); lg[:, self.label_ids] = float("-inf")
            out[torch.tensor(chunk)] = torch.cat([lab, lg.max(-1, keepdim=True).values], 1).cpu()
        self.cur = {}
        return out

    def routing(self, ids, bs=64):
        """Per layer: total top-k routing weight each expert gets on the message part of these prompts."""
        self.mass = [torch.zeros(E, dtype=torch.float64, device=self.dev) for E in self.E]
        self.score(ids, bs=bs)
        m, self.mass = self.mass, None
        return [x.cpu() for x in m]


def margins(S, y):
    """S [n, 8], y [n] label indices -> logit of the annotated label minus the best competing token."""
    own = S.gather(1, y[:, None])[:, 0]
    oth = S.clone(); oth.scatter_(1, y[:, None], float("-inf"))
    return own - oth.max(1).values


def preds(S):
    return [LAB[i] if i < len(LAB) else "?" for i in S.argmax(1).tolist()]
