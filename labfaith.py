import json, sys
rd = lambda n: [json.loads(l) for l in open(f"data/{n}.jsonl")]
xs = rd("dev"); R = json.load(open(sys.argv[1])); bases = {k for k in R if k.endswith("none")}
for name, r in R.items():
    bk = [b for b in bases if name.startswith(b[:-4])] if not name.endswith("none") else [name]
    bk = max(bk, key=len) if bk else "none"
    B = R[bk]["pred"]; S = r["pred"]
    bc = [i for i, x in enumerate(xs) if B[i] == x["label"]]; kept = sum(S[i] == xs[i]["label"] for i in bc)
    per = {v: round(100 * sum(S[i] == v for i in bc if xs[i]["label"] == v) / max(1, sum(1 for i in bc if xs[i]["label"] == v)), 2) for v in sorted({x["label"] for x in xs})}
    print(f"{name:12s} (base {bk}) base-acc {100*len(bc)/len(xs):.2f} faithful {kept/len(bc)*100:6.2f}% lost {len(bc)-kept:3d} min {min(per.values())} {per}")
