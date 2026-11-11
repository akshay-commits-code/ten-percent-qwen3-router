"""Feedback: python feedback.py <artifact_dir> <annotations.jsonl> [--base <folder or Hugging Face repo of the original model>]
Each line: {"text": ..., "label": <correct skill>}, at most 25 lines, marking mistakes. Items the model already gets
right are ignored. Restores pruned experts of the original model into the artifact (growth.safetensors); prints a log.
By default the needed experts are fetched from the original model's public repo (only their bytes are downloaded)."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import artifact, grow

args = sys.argv[1:]
base = args.pop(args.index("--base") + 1) if "--base" in args else None
args = [a for a in args if a != "--base"]
d, f = args[0], args[1]
items = [(x["text"], x["label"].strip().lower()) for x in map(json.loads, (l for l in open(f) if l.strip())) if x.get("text")]
assert len(items) <= 25, "at most 25 annotated items per round"
m, tok = artifact.load(d)
cfg = json.load(open(f"{d}/growth_cfg.json"))
src = grow.Base(base, None) if base else grow.Base(cfg["base"], cfg.get("revision"))
print(json.dumps(grow.apply_feedback(d, m, tok, items, base=src, cfg=cfg, grown=m.tpc_grown), indent=1))
