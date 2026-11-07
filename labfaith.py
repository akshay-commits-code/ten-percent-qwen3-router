import json, sys
rd = lambda n: [json.loads(l) for l in open(f"data/{n}.jsonl")]
xs = rd("dev"); R = json.load(open(sys.argv[1])); B = R["none"]["pred"]
bc = [i for i, x in enumerate(xs) if B[i] == x["label"]]
print("base dev acc", round(100 * len(bc) / len(xs), 2), {s: round(100 * sum(B[i] == x["label"] for i, x in enumerate(xs) if x["source"] == s) / sum(1 for x in xs if x["source"] == s), 1) for s in sorted({x["source"] for x in xs})})
for name, r in R.items():
    S = r["pred"]; kept = sum(S[i] == xs[i]["label"] for i in bc)
    kinds = {}
    for key in ("label", "source"):
        for v in sorted({x[key] for x in xs}):
            I = [i for i in bc if xs[i][key] == v]; kinds[v] = round(100 * sum(S[i] == xs[i]["label"] for i in I) / len(I), 2)
    print(f"{name:8s} faithful {kept/len(bc)*100:6.2f}% lost {len(bc)-kept:3d} kinds {kinds} invalid {sum(p not in ('ham','spam') for p in S)}")
