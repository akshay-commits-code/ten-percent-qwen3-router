"""Feedback: python feedback.py <artifact_dir> <annotations.jsonl>
Each line: {"text": ..., "label": <correct queue name>}. Items the model already gets right are ignored (feedback is
for marking mistakes). Edits the artifact in place (same size on disk); prints what changed."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import artifact

d, f = sys.argv[1], sys.argv[2]
items = [(x["text"], x["label"].strip().lower()) for x in map(json.loads, open(f)) if x.get("text")]
assert len(items) <= 25, "at most 25 annotated items per round"
m, tok = artifact.load(d)
print(json.dumps(artifact.apply_feedback(d, m, tok, items)))
