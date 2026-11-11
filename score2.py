"""Turn the revised step-5 run's raw answers (final2/raw.json) into results2.json.
Usage: python score2.py [raw.json] [exam.jsonl] [teach.jsonl] [out.json]"""
import collections, json, math, sys

a = sys.argv[1:]
RAW = a[0] if len(a) > 0 else "final2/raw.json"
EXAM = a[1] if len(a) > 1 else "data/exam.jsonl"
TEACH = a[2] if len(a) > 2 else "data/teach2/hi-IN/locked.jsonl"
OUT = a[3] if len(a) > 3 else "results2.json"
rd = lambda p: [json.loads(l) for l in open(p)]
exam, teach, R = rd(EXAM), rd(TEACH), json.load(open(RAW))
FP16 = 61064245248


def acc(pred, xs, keep=lambda x: True):
    I = [i for i, x in enumerate(xs) if keep(x)]
    return [round(100 * sum(pred[i] == xs[i]["label"] for i in I) / len(I), 2), len(I)]


def faith(base, small, xs, keep=lambda x: True):
    bc = [i for i, x in enumerate(xs) if keep(x) and base[i] == x["label"]]
    k = sum(small[i] == xs[i]["label"] for i in bc)
    return {"kept": k, "base_correct": len(bc), "pct": round(100 * k / len(bc), 2)}


def kinds(xs, fn, *args):
    out = {"overall": fn(*args, xs)}
    for q in sorted({x["label"] for x in xs}):
        out[f"label={q}"] = fn(*args, xs, lambda x, q=q: x["label"] == q)
    for s in sorted({x["source"] for x in xs}):
        out[f"source={s}"] = fn(*args, xs, lambda x, s=s: x["source"] == s)
    return out


def paired(before, after, xs):
    """Items fixed / broken by feedback, with an exact two-sided sign test."""
    f = sum(b != x["label"] and c == x["label"] for b, c, x in zip(before, after, xs))
    k = sum(b == x["label"] and c != x["label"] for b, c, x in zip(before, after, xs))
    n = f + k
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(0, min(f, k) + 1)) / 2 ** n) if n else 1.0
    return {"fixed": f, "broken": k, "sign_test_p": float(f"{p:.3g}")}


S, F = R["small_exam"], R["fb_real_exam"]; B = R.get("base_exam")
SEEDS = [s for s in range(1, 20) if f"fb_shuffled{s}_teach" in R]; CTRL = ["empty"] + [f"shuffled{s}" for s in SEEDS]
variants = [("small_before_feedback", "small_teach"), ("real_feedback", "fb_real_teach"), ("empty_feedback", "fb_empty_teach")] + \
           [(f"shuffled_labels_seed{s}", f"fb_shuffled{s}_teach") for s in SEEDS]
if R.get("base_teach"):
    variants = [("base", "base_teach")] + variants
res = {
    "run": {"started": R["started"], "finished": R.get("finished"), "locale": R["locale"], "rehearsal_on_dev_files": R.get("rehearsal", False), "gpu": R.get("gpu")},
    "size": {"base_fp16_bytes": FP16, "limit_bytes": FP16 // 10, "artifact_bytes": R["artifact_bytes"], "pct": round(100 * R["artifact_bytes"] / FP16, 4),
             "after_real_feedback_bytes": R["fb_real_bytes"], "after_real_feedback_pct": round(100 * R["fb_real_bytes"] / FP16, 4),
             "after_controls_bytes": {v: R[f"fb_{v}_bytes"] for v in CTRL}},
    "teach_items": len(teach), "teach_per_label": dict(collections.Counter(x["label"] for x in teach)),
    "teach": {v: kinds(teach, acc, R[k]) for v, k in variants},
    "teach_real_vs_before": paired(R["small_teach"], R["fb_real_teach"], teach),
    "teach_real_vs_controls": {v: paired(R[f"fb_{v}_teach"], R["fb_real_teach"], teach) for v in CTRL},
    "experts_restored": {v: len(R[f"fb_{v}_log"]["restored"]) for v in ["real"] + CTRL},
    "feedback_items": R["feedback_items"],
    "feedback_logs": {v: {k: x for k, x in R[f"fb_{v}_log"].items() if k != "restored"} | {"restored": R[f"fb_{v}_log"]["restored"]}
                      for v in ["real"] + CTRL},
    "exam_items": len(exam),
    "exam_answers_changed_by_real_feedback": sum(x != y for x, y in zip(S, F)),
    "exam_accuracy": {"small_before_feedback": kinds(exam, acc, S), "small_after_real_feedback": kinds(exam, acc, F)},
    "answer_command_check": R.get("answer_cli"), "feedback_command_secs": R.get("fb_real_cli", {}).get("secs"),
    "growth_sha256": R.get("growth_sha256"),
}
if B:
    res["exam_accuracy"]["base"] = kinds(exam, acc, B)
    res["exam_faithfulness_before_feedback"] = kinds(exam, faith, B, S)
    res["exam_faithfulness_after_real_feedback"] = kinds(exam, faith, B, F)
    for v in CTRL[1:]:
        if R.get(f"fb_{v}_exam"):
            res[f"exam_faithfulness_after_{v}"] = kinds(exam, faith, B, R[f"fb_{v}_exam"])
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
show = lambda d: {k: (v[0] if isinstance(v, list) else v.get("pct")) for k, v in d.items()}
print("size", res["size"])
for v, _ in variants:
    print(f"teach {v:28s}", show(res["teach"][v]))
print("real vs before", res["teach_real_vs_before"], "| restored", res["experts_restored"])
print("exam changed by real feedback:", res["exam_answers_changed_by_real_feedback"], "of", len(exam))
if B:
    print("faithfulness before", show(res["exam_faithfulness_before_feedback"]))
    print("faithfulness after ", show(res["exam_faithfulness_after_real_feedback"]))
print("answer command check:", res["answer_command_check"])
