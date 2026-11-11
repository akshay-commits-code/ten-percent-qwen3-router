"""Design lab for prune-only feedback. Uses pool / dev files only; locked files are never read here."""
import json, random, time
import torch
from prune import Pruner, margins, preds, LAB


def greedy(pr, ids_f, y, cands, order, ids_g, g0, tau, cap, floor, max_accept, min_keep, spec=4, checkpoints=(), on_cp=None):
    """Try candidates in `order`; an expert stays removed only if the annotated items move toward their labels by at
    least `tau` (sum of clipped margins) and no guard answer changes."""
    nf = len(ids_f)
    obj = lambda S: margins(S, y).clamp(floor, cap).sum().item()
    cur = obj(pr.score(ids_f)); log, acc, queue = [], [], list(order)
    while queue and len(acc) < max_accept:
        chunk = [c for c in queue[:spec] if pr.E[cands[c][0]] - len(pr.removed[cands[c][0]]) > min_keep]
        used = min(spec, len(queue))
        if chunk:
            S = pr.score(ids_f, [(i, (cands[c],)) for c in chunk for i in range(nf)]).view(len(chunk), nf, -1)
            for k, c in enumerate(chunk):
                l, e = cands[c]; o = obj(S[k])
                if o - cur < tau:
                    log.append([l, e, round(o - cur, 3), "no"]); continue
                Sg = pr.score(ids_g, [(i, (cands[c],)) for i in range(len(ids_g))])
                ch = sum(a != b for a, b in zip(preds(Sg), g0))
                if ch:
                    log.append([l, e, round(o - cur, 3), f"guard{ch}"]); continue
                pr.removed[l].add(e); acc.append([l, e]); log.append([l, e, round(o - cur, 3), "yes"]); cur = o
                if on_cp and len(acc) in checkpoints:
                    on_cp(len(acc))
                used = queue.index(c) + 1  # the rest of this chunk is re-tested against the new state
                break
        queue = queue[used:]
    return acc, log, cur


def run_lab(pr, pool, dev, en_dev, guard_texts, cfg, save):
    t0 = time.time(); R = {"cfg": cfg}; Tn = {}
    Y = lambda xs: torch.tensor([LAB.index(x["label"]) for x in xs])
    acc = lambda S, xs: round(100 * sum(p == x["label"] for p, x in zip(preds(S), xs)) / len(xs), 2)
    T = lambda xs: [x["text"] for x in xs]
    ids_pool, ids_dev, ids_en, ids_g = pr.encode(T(pool)), pr.encode(T(dev)), pr.encode(T(en_dev)), pr.encode(guard_texts)
    S_pool, S_dev0, S_en0, S_g0 = pr.score(ids_pool), pr.score(ids_dev), pr.score(ids_en), pr.score(ids_g)
    p_pool, g0, en0 = preds(S_pool), preds(S_g0), preds(S_en0)
    wrong = [i for i, (p, x) in enumerate(zip(p_pool, pool)) if p != x["label"]]
    nf = cfg["n_feedback"]; F1, F2 = wrong[:nf], wrong[nf:2 * nf]
    R.update(pool_acc=acc(S_pool, pool), dev_acc0=acc(S_dev0, dev), en_acc0=acc(S_en0, en_dev), n_wrong=len(wrong), F1=F1, F2=F2,
             F1_pred=[p_pool[i] for i in F1], F1_label=[pool[i]["label"] for i in F1], secs_base=round(time.time() - t0))
    print(json.dumps({k: R[k] for k in ("pool_acc", "dev_acc0", "en_acc0", "n_wrong", "F1_pred", "F1_label", "secs_base")}), flush=True)
    ids_f1 = [ids_pool[i] for i in F1]; items = ids_f1 + [ids_pool[i] for i in F2]
    mass, mass_g = pr.routing(ids_f1), pr.routing(ids_g)
    flat = sorted(((mass[l][e].item(), l, e) for l in range(len(mass)) for e in range(len(mass[l]))), reverse=True)[:cfg["N"]]
    cands = [(l, e) for _, l, e in flat]
    t1 = time.time()
    S_f0 = pr.score(items)
    S_tab = pr.score(items, [(i, (c,)) for c in cands for i in range(len(items))]).view(len(cands), len(items), -1)
    R.update(cands=cands, cand_mass=[round(m, 3) for m, _, _ in flat], cand_mass_guard=[round(mass_g[l][e].item(), 3) for l, e in cands],
             secs_table=round(time.time() - t1))
    Tn.update(S_pool=S_pool, S_dev0=S_dev0, S_en0=S_en0, S_g0=S_g0, S_f0=S_f0, S_tab=S_tab,
              mass=torch.nn.utils.rnn.pad_sequence(mass, batch_first=True), mass_g=torch.nn.utils.rnn.pad_sequence(mass_g, batch_first=True))
    print("table", R["secs_table"], "s", flush=True); save(R, Tn)
    y_real = Y([pool[i] for i in F1]); sh = y_real.tolist(); random.Random(cfg["seed"]).shuffle(sh); y_shuf = torch.tensor(sh)
    y2 = Y([pool[i] for i in F2]) if F2 else None
    for name, y in (("real", y_real), ("shuf", y_shuf)):
        t2 = time.time()
        obj = lambda S: margins(S, y).clamp(cfg["floor"], cfg["cap"]).sum().item()
        base = obj(S_f0[:nf]); gain = torch.tensor([obj(S_tab[c, :nf]) - base for c in range(len(cands))])
        order = [c for c in gain.argsort(descending=True).tolist() if gain[c] > 0][:cfg["M"]]
        pr.removed = [set() for _ in pr.blocks]; cps = {}
        def cp(k):
            Sd, Se = pr.score(ids_dev), pr.score(ids_en)
            cps[k] = {"dev": acc(Sd, dev), "en_acc": acc(Se, en_dev), "en_same": sum(a == b for a, b in zip(preds(Se), en0))}
            print(name, "cp", k, cps[k], flush=True)
        accd, log, fin = greedy(pr, ids_f1, y, cands, order, ids_g, g0, cfg["tau"], cfg["cap"], cfg["floor"], cfg["max_accept"],
                                cfg["min_keep"], checkpoints=cfg["checkpoints"], on_cp=cp)
        Sd, Se, Sf = pr.score(ids_dev), pr.score(ids_en), pr.score(items)
        R[name] = {"accepted": accd, "log": log, "n_order": len(order), "obj0": round(base, 3), "obj1": round(fin, 3), "cps": cps,
                   "dev_acc": acc(Sd, dev), "en_acc": acc(Se, en_dev), "en_same": sum(a == b for a, b in zip(preds(Se), en0)), "en_n": len(en0),
                   "f1_fixed": sum(p == pool[i]["label"] for p, i in zip(preds(Sf[:nf]), F1)),
                   "f2_fixed": sum(p == pool[i]["label"] for p, i in zip(preds(Sf[nf:]), F2)), "f2_n": len(F2),
                   "dev_pred": preds(Sd), "secs": round(time.time() - t2)}
        Tn[f"S_dev_{name}"] = Sd; Tn[f"S_en_{name}"] = Se
        print(name, json.dumps({k: v for k, v in R[name].items() if k not in ("log", "dev_pred")}), flush=True); save(R, Tn)
    pr.removed = [set() for _ in pr.blocks]
    R["secs_total"] = round(time.time() - t0); save(R, Tn)
    return R
