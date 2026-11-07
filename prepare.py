"""Attempt 5: build every split and LOCK the exam + teach set before any model run on this task.

Task: route a voice-assistant request to the skill that should handle it. Answer = the skill name.
Main source: Amazon MASSIVE (en-US). Different source (exam only, never used to build/tune/calibrate): CLINC150
utility/travel/auto/home/work intents (none of these items were used or seen in earlier attempts).
Feedback teaches two skills the base task does not have: recipe (cooking) and music.
"""
import hashlib, json, os, random, re
import pandas as pd, pyarrow.parquet as pq

R = os.path.dirname(os.path.abspath(__file__)); RAW, DATA = f"{R}/raw", f"{R}/data"
rng = random.Random(20261107)
SKILLS = {
    "weather": "weather conditions and forecasts",
    "email": "reading, writing, sending or managing email and email contacts",
    "transport": "taxis, trains, flights, tickets, traffic and directions",
    "lists": "creating, reading or editing lists (shopping lists, to-do lists)",
    "calendar": "calendar events, meetings and reminders",
    "home": "smart-home devices: lights, plugs, vacuum, coffee machine, speakers' wemo/hue style controls",
    "news": "news headlines and updates",
}
BASE = {"weather": ["weather"], "email": ["email"], "transport": ["transport"], "lists": ["lists"], "calendar": ["calendar"],
        "home": ["iot"], "news": ["news"]}
TEACH = {"recipe": ["cooking"], "music": ["music"]}
CLINC = {"weather": ["weather"], "transport": ["uber", "traffic", "directions", "distance", "book_flight", "flight_status", "car_rental"],
         "lists": ["shopping_list", "shopping_list_update", "todo_list", "todo_list_update"],
         "calendar": ["calendar", "calendar_update", "meeting_schedule", "schedule_meeting", "reminder", "reminder_update"],
         "home": ["smart_home"]}
CLINC_TEACH = {"recipe": ["recipe", "cook_time", "ingredients_list", "ingredient_substitution", "meal_suggestion"]}
SYSTEM = ("You route requests for a voice assistant to the skill that should handle them.\nSkills:\n" +
          "\n".join(f"- {k}: {v}" for k, v in SKILLS.items()) + "\nReply with the skill name only.")
SYSTEM = SYSTEM.replace(", speakers' wemo/hue style controls", "")

norm = lambda t: re.sub(r"\s+", " ", t).strip().lower()
ms = pd.concat([pd.read_parquet(f"{RAW}/massive_{s}.parquet") for s in ("train", "validation", "test")])
names = json.loads(pq.read_schema(f"{RAW}/clinc_train.parquet").metadata[b"huggingface"])["info"]["features"]["intent"]["names"]
cl = pd.concat([pd.read_parquet(f"{RAW}/clinc_{s}.parquet") for s in ("train", "validation", "test")])
cl["intent_name"] = [names[i] for i in cl.intent]


def items(pairs, source):
    seen, out = set(), []
    for t, l, sc in pairs:
        k = norm(t)
        if k in seen:
            continue
        seen.add(k); out.append({"text": t, "label": l, "scenario": sc, "source": source})
    rng.shuffle(out)
    return out


inv = lambda m: {v: k for k, vs in m.items() for v in vs}
mb, mt = inv(BASE), inv(TEACH)
m_all = items([(t, mb.get(s, mt.get(s)), s) for t, s in zip(ms.text, ms.label_text)], "massive")
c_all = items([(t, inv(CLINC).get(s, inv(CLINC_TEACH).get(s)), s) for t, s in zip(cl.text, cl.intent_name)
               if s in inv(CLINC) or s in inv(CLINC_TEACH)], "clinc150")


def take(xs, n, pred):
    a = [x for x in xs if pred(x)][:n]; ids = {id(x) for x in a}
    return a, [x for x in xs if id(x) not in ids]


sp = {k: [] for k in ("exam", "dev", "calib", "teach_locked", "teach_dev", "teach_pool")}
rest = m_all
for lab in BASE:
    for part, n in (("exam", 330), ("dev", 200), ("calib", 150)):
        a, rest = take(rest, n, lambda x: x["label"] == lab); sp[part] += a
for lab in TEACH:
    for part, n in (("teach_locked", 120), ("teach_dev", 80)):
        a, rest = take(rest, n, lambda x: x["label"] == lab); sp[part] += a
    sp["teach_pool"] += [x for x in rest if x["label"] == lab]
a, _ = take(rest, 300, lambda x: x["label"] is None); sp["calib"] += [dict(x, label=None) for x in a]  # other scenarios, unlabeled calibration text
cb = [x for x in c_all if x["label"] in BASE]
per = {}
for x in cb:
    per.setdefault(x["scenario"], []).append(x)
k_per = -(-600 // len(per))
sp["exam"] += [x for s in sorted(per) for x in per[s][:k_per]][:max(600, 0)]
sp["teach_locked"] += [x for x in c_all if x["label"] == "recipe"][:150]


def dump(name, xs):
    xs = xs[:]; rng.shuffle(xs)
    with open(f"{DATA}/{name}.jsonl", "w") as f:
        for i, x in enumerate(xs):
            f.write(json.dumps(dict(x, id=f"{name}-{i}")) + "\n")


os.makedirs(DATA, exist_ok=True)
for k, v in sp.items():
    dump(k, v)
json.dump({"labels": list(SKILLS), "system": SYSTEM, "teach_labels": list(TEACH)}, open(f"{DATA}/task.json", "w"), indent=1)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
cnt = lambda xs, key: {str(k): int(v) for k, v in pd.Series([x[key] for x in xs]).value_counts().items()}
lock = {"locked_files": {n: sha(f"{DATA}/{n}.jsonl") for n in ("exam", "teach_locked")},
        "other_files": {n: sha(f"{DATA}/{n}.jsonl") for n in ("dev", "calib", "teach_dev", "teach_pool")} | {"task.json": sha(f"{DATA}/task.json")},
        "counts": {k: len(v) for k, v in sp.items()}, "exam_per_label": cnt(sp["exam"], "label"), "exam_per_source": cnt(sp["exam"], "source"),
        "teach_locked_per_label": cnt(sp["teach_locked"], "label"), "teach_pool_per_label": cnt(sp["teach_pool"], "label")}
json.dump(lock, open(f"{R}/LOCK.json", "w"), indent=1)
print(json.dumps(lock, indent=1)); print(SYSTEM)
