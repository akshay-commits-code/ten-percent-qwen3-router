"""Choose the feedback edit's constants (shrinkage lam, delta, quantile) on teach_dev + dev only, from saved hidden states.
Rule (amendment 2): among settings changing <= 0.1% of dev answers, maximise teach_dev accuracy minus the
shuffled-label teach_dev accuracy (what the labels themselves contribute)."""
import json, sys, random
import torch
import core

tag = sys.argv[1]
W = f"work/{tag}"
P = json.load(open(f"{W}_preds.json"))
rd = lambda n: [json.loads(l) for l in open(f"data/{n}.jsonl")]
S = {n: torch.load(f"{W}_{n}_h.pt") for n in ["dev", "teach_dev", "teach_pool", "calib"]}
R = torch.load(f"{W}_label_rows.pt"); rows0 = {w: R["rows"][i] for i, w in enumerate(R["ids"])}
STATS = {lam: core.stats_from(S["calib"]["h"], P["calib"]["pred"], lam) for lam in (0.02, 0.1, 0.3, 0.6)}


def feedback_set(labels_from):
    xs, pr = rd("teach_pool"), P["teach_pool"]["pred"]
    out = []
    for y in ("recipe", "music"):
        out += [i for i, x in enumerate(xs) if x["label"] == y and pr[i] != y][:12]
    return out


def simulate(new_rows, name):
    H, top, pred = S[name]["h"].float(), S[name]["top"].float(), list(P[name]["pred"])
    best, arg = top.clone(), [None] * len(pred)
    for y, w in new_rows.items():
        z = H @ w
        m = z > best; best = torch.where(m, z, best)
        for i in torch.nonzero(m).flatten().tolist():
            arg[i] = y
    return [a if a is not None else p for a, p in zip(arg, pred)]


def edit(idx, labels, delta, qt, lam):
    stats = STATS[lam]
    Hp, tp, pp = S["teach_pool"]["h"][idx], S["teach_pool"]["top"][idx], [P["teach_pool"]["pred"][i] for i in idx]
    new = core.plan_edit(Hp, tp, pp, labels, {y: rows0[y] for y in set(labels)}, stats, delta, qt)
    out = {}
    for y, w in new.items():
        cd, s, z = core.requant_row(w)
        out[y] = core.dequantize(cd[None], s[None], None, 4, dtype=torch.float32)[0]
    return out


idx = feedback_set(None)
xs_fb = rd("teach_pool"); labels = [xs_fb[i]["label"] for i in idx]
td, dv = rd("teach_dev"), rd("dev")
base_dev = json.load(open("work/base_preds.json"))["dev"]["pred"]
small_dev = P["dev"]["pred"]
res = []
rnd_labels = labels[:]; random.Random(0).shuffle(rnd_labels)
acc = lambda pr: sum(p == x["label"] for p, x in zip(pr, td)) / len(td)
for lam in STATS:
    for delta in [1.0, 4.0]:
        for qt in [0.5, 0.75, 1.0]:
            nr = edit(idx, labels, delta, qt, lam)
            t_pred = simulate(nr, "teach_dev"); d_pred = simulate(nr, "dev")
            s_pred = simulate(edit(idx, rnd_labels, delta, qt, lam), "teach_dev")
            res.append({"lam": lam, "delta": delta, "quantile": qt, "teach_dev_acc": acc(t_pred), "shuffled_teach_dev_acc": acc(s_pred),
                        "dev_changed": sum(a != b for a, b in zip(d_pred, small_dev)),
                        "dev_base_correct_lost": sum(b == x["label"] and a != x["label"] and s == x["label"]
                                                     for a, s, b, x in zip(d_pred, small_dev, base_dev, dv))})
            print(res[-1])
ok = [r for r in res if r["dev_changed"] <= 0.001 * len(dv)]
pick = max(ok, key=lambda r: (r["teach_dev_acc"] - r["shuffled_teach_dev_acc"], -r["dev_changed"]))
print("PICK", pick, "feedback items", len(idx))
json.dump({"lam": pick["lam"], "delta": pick["delta"], "quantile": pick["quantile"], "grid": res}, open(f"{W}_feedback_design.json", "w"), indent=1)
