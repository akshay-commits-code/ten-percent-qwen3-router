"""Step-5 revision: build and LOCK fresh teach sets before any model run on them.

What the feedback will teach: requests in a language the small model was not built for. MASSIVE is a parallel corpus,
so only utterance ids whose English text was never used in any earlier split (exam, dev, calib, teach*) are used here.
The same ids are split into pool / dev / locked for 12 candidate locales; the locale is chosen later on dev only, and
only that locale's locked file is ever opened (once, by the final run).
"""
import hashlib, json, os, random, re
import pandas as pd

R = os.path.dirname(os.path.abspath(__file__)); RAW, OUT = f"{R}/raw", f"{R}/data/teach2"
LOCALES = ["es-ES", "de-DE", "fr-FR", "zh-CN", "ja-JP", "hi-IN", "ar-SA", "ru-RU", "ko-KR", "th-TH", "tr-TR", "vi-VN"]
BASE = {"weather": "weather", "email": "email", "transport": "transport", "lists": "lists", "calendar": "calendar", "iot": "home", "news": "news"}
CAP = {"news": 9, "lists": 36, "transport": 40, "weather": 54, "home": 80, "email": 80, "calendar": 80}  # per split
norm = lambda t: re.sub(r"\s+", " ", t).strip().lower()
rng = random.Random(20261111)

used = set()
for n in ("exam", "dev", "calib", "teach_locked", "teach_dev", "teach_pool"):
    used |= {norm(json.loads(l)["text"]) for l in open(f"{R}/data/{n}.jsonl")}
en = pd.concat([pd.read_parquet(f"{RAW}/massive_{s}.parquet") for s in ("train", "validation", "test")])
left = en[en.label_text.isin(BASE) & ~en.text.map(norm).isin(used)].drop_duplicates(subset="text")
split = {"pool": [], "dev": [], "locked": []}
for sc, lab in BASE.items():
    ids = sorted(left[left.label_text == sc].id.tolist(), key=int); rng.shuffle(ids)
    for k, part in enumerate(("pool", "dev", "locked")):
        split[part] += [(i, lab) for i in ids[k * CAP[lab]:(k + 1) * CAP[lab]]]
lock = {"locked_files": {}, "other_files": {}, "counts": {k: len(v) for k, v in split.items()}, "per_label": CAP, "locales": LOCALES}
for loc in LOCALES:
    df = pd.concat([pd.read_parquet(f"{RAW}/loc/{loc}_{s}.parquet") for s in ("train", "validation", "test")]).set_index("id")
    os.makedirs(f"{OUT}/{loc}", exist_ok=True)
    for part, rows in split.items():
        rows = rows[:]; random.Random(f"{loc}-{part}").shuffle(rows)
        p = f"{OUT}/{loc}/{part}.jsonl"
        with open(p, "w") as f:
            for j, (i, lab) in enumerate(rows):
                assert BASE[df.loc[i, "label_text"]] == lab
                f.write(json.dumps({"text": df.loc[i, "text"], "label": lab, "source": f"massive-{loc}", "massive_id": i,
                                    "id": f"{loc}-{part}-{j}"}, ensure_ascii=False) + "\n")
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        (lock["locked_files"] if part == "locked" else lock["other_files"])[f"teach2/{loc}/{part}.jsonl"] = h
json.dump(lock, open(f"{R}/LOCK2.json", "w"), indent=1)
print(json.dumps({k: lock[k] for k in ("counts", "per_label")}), len(lock["locked_files"]), "locked files")
