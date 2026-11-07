# The 10% challenge: Qwen3-30B-A3B at 9.89% of its fp16 size, faithful on a locked exam, learning from 24 labels

**Task: voice-assistant request routing.** A request is sent to the skill that should handle it: weather, email, transport, lists, calendar, home (smart-home devices) or news. The model's real output is the skill name, produced by free greedy generation.

**Base model:** `Qwen/Qwen3-30B-A3B-Instruct-2507` (Apache-2.0). It has 30.5B parameters and 61,064,245,248 bytes at bf16/fp16.

## Results on the locked exam (opened once, `results.json`)

| Rule | Requirement | Result |
|---|---|---|
| Size, everything shipped | ≤ 10% of fp16 | **6,040,715,980 B = 9.892%**. This counts weights, tokenizer, code, task file and feedback statistics. |
| Faithful | ≥ 98% of base-correct items kept | **99.36%** (2,775 / 2,793) |
| Every kind | ≥ 95% | Lowest is **98.56%** (weather). The other labels are 98.73–100%. MASSIVE is 99.42%; CLINC150 (different source) is 99.12%. |
| Exam | ≥ 2,000 items, ≥ 300 per kind, ≥ 500 from an unused source | 2,910 items: every label has 330 or more, MASSIVE 2,310, CLINC150 600 |
| Base has something to keep | Well above chance, below ceiling | Base 95.98% (chance is 14.3%); small model 96.12% |
| Feedback teaches (24 annotated mistakes) | Clear gain on locked items | Locked teach set (390 items): **18.97% → 64.87%**. Recipe 0% → 61.1%; CLINC recipe intents (another source) 0% → 54.7% |
| Controls | Shuffled and empty clearly worse | Shuffled labels 29.23%; empty 18.97% |
| Nothing breaks | Exam still passes after feedback | 99.36% after feedback. 1 exam answer changed out of 2,910. Nearby labels (lists, home) keep 99.62%. |
| Size after feedback | Unchanged | 6,040,715,980 B |

The small model gets more exam items right than the base (96.12% vs 95.98%). That is not used as the match criterion; the match is the 99.36% of base-correct items kept.

## How it is small (no weight training anywhere)

- **Task-specific expert pruning.** Keep 2,640 of the 6,144 routed experts. The count varies per layer, from about 37 to 81. Experts are ranked by their share of their layer's top-8 routing mass on unlabeled calibration text (task requests plus generic English). The router is sliced to the kept experts, which is exactly the base router restricted to them. Every kept expert is a byte-exact slice of the base.
- **Round-to-nearest only.** Experts use 3-bit asymmetric codes, attention 6-bit, embeddings 5-bit, and the output layer 4-bit. Every group of 128 weights shares one fp16 scale, chosen from a fixed grid by weight MSE. Norms and router stay bf16. There is no GPTQ and no optimisation of any weight.
- **Shipping.** Codes are bit-packed into safetensors. `artifact.load()` rebuilds a standard Hugging Face `Qwen3MoeForCausalLM` with fewer experts per layer.

## How it learns from feedback (no weight training, no side memory, nothing in the prompt)

`python feedback.py <artifact> annotations.jsonl` (each line: `{"text": ..., "label": ...}`, at most 25 items) does the following:

1. Runs the small model on the annotated items. Items it already gets right are ignored, because feedback marks mistakes.
2. For each taught label, takes the final hidden states of the mistaken items and computes one direction in closed form: their mean, minus the mean of the other taught label's items, whitened by a stored low-rank inverse covariance of calibration states.
3. Adds a closed-form multiple of that direction to the **existing output-layer row of the label's token**. The multiple is the median of the smallest per-item steps that make the label win by a margin.
4. Rewrites only those rows, re-rounded to 4 bits, in place. File size is unchanged.

At answer time the computation is the original model's, with two existing output rows changed. Nothing is looked up and nothing is added to the prompt. `feedback_stats.safetensors` (1 MB) is used only to compute edits.

## Exam hygiene

**Locked before any model run on this task.** `exam.jsonl` and `teach_locked.jsonl` were hash-locked (`LOCK.json`) and committed first. The git history shows this.

**Never used to build or choose.** All choices were made on `dev`, `calib`, `teach_dev` and `teach_pool`:

- the prompt wording
- the calibration text
- the expert-selection weighting
- the bit allocation
- the feedback constants

Each decision and its dev numbers is in `PROTOCOL.md`: amendments 1–3, all committed before the exam was opened.

**Opened once.** `modal_final.py` opened the exam once. It ran the base, the small model, and real / shuffled / empty feedback.

**Gate change, disclosed.** The pre-registered dev gate was 99% overall and 98% per label. Dev reached 99.10%, with the news label at 97.85%. The gate was lowered to the challenge's own thresholds before opening the exam, and that is disclosed in amendment 2.

**Earlier attempts.** Attempts 1–4 on other tasks and models were stopped or failed, and their sealed exams were never reused here. CLINC150 banking intents were an exam source in attempt 2. The CLINC150 items used here (utility, travel, auto, home and work intents) were never used or seen.

## Reproduce

```bash
python prepare.py   # builds the splits and the lock from raw MASSIVE + CLINC150
modal run modal_app.py::lab_stage --configs-json '<configs>'   # dev-only lab (configs in PROTOCOL.md)
modal run modal_app.py::build_final                              # writes the artifact
python design_feedback.py C1                                     # feedback constants from teach_dev + dev
modal run modal_final.py::final && python score.py               # the single exam run -> results.json
```

## Using the artifact

```bash
python answer.py   <artifact_dir> messages.txt        # one skill name per line
python feedback.py <artifact_dir> annotations.jsonl   # teaches; edits the artifact in place
```
