"""The first step-5 mechanism (rejected by the maintainer on 10 Nov 2026): closed-form edit of two output rows.\nKept for the record only; nothing here is used or shipped any more. Removed from core.py / artifact.py."""\nimport torch\n\n# ------------------------------- feedback: closed-form edit of existing output rows -------------------------------

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
