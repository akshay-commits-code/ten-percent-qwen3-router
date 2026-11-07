"""Answer: python answer.py <artifact_dir> <messages.jsonl|.txt>  -> one queue name per line on stdout.
Input: JSONL with a "text" field per line, or plain text with one message per line."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import artifact, core

d, f = sys.argv[1], sys.argv[2]
lines = [l.rstrip("\n") for l in open(f) if l.strip()]
texts = [json.loads(l)["text"] for l in lines] if f.endswith(".jsonl") else lines
m, tok = artifact.load(d)
for lab in core.run(m, tok, texts)[0]:
    print(lab)
