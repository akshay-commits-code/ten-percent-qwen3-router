"""Assemble the shipped artifact folder: the unchanged weight files + tokenizer + this repo's code and small files."""
import os, shutil

CODE = ("core.py", "artifact.py", "grow.py", "answer.py", "feedback.py", "keep.json", "growth_cfg.json")
KEEP = ("config.json", "quant.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")
BIG = ("model.safetensors", "lm_head.safetensors")


def stage(src, dst, app="/root/app", link=True):
    """src: the frozen first-submission artifact (weights byte-identical). link=True symlinks the two weight files."""
    shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
    for f in BIG:
        os.symlink(f"{src}/{f}", f"{dst}/{f}") if link else shutil.copy(f"{src}/{f}", f"{dst}/{f}")
    for f in KEEP:
        shutil.copy(f"{src}/{f}", f"{dst}/{f}")
    for f in CODE:
        shutil.copy(f"{app}/{f}", f"{dst}/{f}")
    shutil.copy(f"{app}/data/task.json", f"{dst}/task.json"); shutil.copy(f"{app}/data/guard.jsonl", f"{dst}/guard.jsonl")
    return dst
