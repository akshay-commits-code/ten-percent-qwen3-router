"""Turn the single exam run's raw answers (final/raw.json) into results.json."""
import json, collections

rd = lambda n: [json.loads(l) for l in open(f"data/{n}.jsonl")]
exam, teach = rd("exam"), rd("teach_locked")
task = json.load(open("data/task.json")); R = json.load(open("final/raw.json"))
FP16 = 61064245248


def acc(pred, xs, keep=lambda x: True):
    I = [i for i, x in enumerate(xs) if keep(x)]
    return round(100 * sum(pred[i] == xs[i]["label"] for i in I) / len(I), 2), len(I)


def faith(base, small, xs, keep=lambda x: True):
    bc = [i for i, x in enumerate(xs) if keep(x) and base[i] == x["label"]]
    k = sum(small[i] == xs[i]["label"] for i in bc)
    return {"kept": k, "base_correct": len(bc), "pct": round(100 * k / len(bc), 2)}


def kinds(fn, *a):
    out = {"overall": fn(*a)}
    for q in sorted({x["label"] for x in exam}):
        out[f"label={q}"] = fn(*a, lambda x, q=q: x["label"] == q)
    for s in sorted({x["source"] for x in exam}):
        out[f"source={s}"] = fn(*a, lambda x, s=s: x["source"] == s)
    return out


B, S, F = R["base_exam"], R["small_exam"], R["fb_real_exam"]
near = set(task["nearby_intents"])
res = {
    "size": {"base_fp16_bytes": FP16, "artifact_bytes": R["artifact_bytes"], "pct": round(100 * R["artifact_bytes"] / FP16, 3),
             "after_feedback_bytes": R["fb_real_bytes"]},
    "exam_items": len(exam), "exam_per_kind": dict(collections.Counter(x["label"] for x in exam)) | dict(collections.Counter(x["source"] for x in exam)),
    "accuracy": {"base": kinds(acc, B, exam), "small": kinds(acc, S, exam), "small_after_feedback": kinds(acc, F, exam)},
    "faithfulness_small": kinds(faith, B, S, exam),
    "faithfulness_after_feedback": kinds(faith, B, F, exam),
    "nearby_items_after_feedback": faith(B, F, exam, lambda x: x["intent"] in near),
    "nearby_changed_by_feedback": sum(S[i] != F[i] for i, x in enumerate(exam) if x["intent"] in near),
    "exam_answers_changed_by_feedback": sum(a != b for a, b in zip(S, F)),
    "teach_locked": {v: {"all": acc(R[k], teach)[0],
                         "banking77": acc(R[k], teach, lambda x: x["source"] == "banking77")[0],
                         "clinc150_report_fraud": acc(R[k], teach, lambda x: x["source"] == "clinc150")[0]}
                     for v, k in [("base", "base_teach"), ("small_before_feedback", "small_teach"), ("real_feedback", "fb_real_teach"),
                                  ("shuffled_labels", "fb_shuffled_teach"), ("empty_feedback", "fb_empty_teach")]},
    "feedback_items": len(R["feedback_items"]), "feedback_logs": {v: R[f"fb_{v}_log"] for v in ("real", "shuffled", "empty")},
    "run": {"started": R["started"], "finished": R.get("finished")},
}
json.dump(res, open("results.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ("size", "faithfulness_small", "faithfulness_after_feedback", "teach_locked", "nearby_items_after_feedback",
                                      "exam_answers_changed_by_feedback")}, indent=1))
print("accuracy overall: base", res["accuracy"]["base"]["overall"], "small", res["accuracy"]["small"]["overall"],
      "after fb", res["accuracy"]["small_after_feedback"]["overall"])
