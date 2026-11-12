# The 10% challenge: Qwen3-30B-A3B at 9.89% of its fp16 size, faithful on a locked exam, learning from 24 corrections with no weight training

**Artifact (6.04 GB):** https://huggingface.co/akshay-commits-code/qwen3-30b-a3b-10pct-voice-router

**Task: voice-assistant request routing.** A request is sent to the skill that should handle it: weather, email, transport, lists, calendar, home (smart-home devices) or news. The model's real output is the skill name, produced by free greedy generation.

**Base model:** `Qwen/Qwen3-30B-A3B-Instruct-2507` (Apache-2.0). It has 30.5B parameters and 61,064,245,248 bytes at bf16/fp16.

## Status

- **Steps 1–4 (compression and locked exam)** are unchanged since the first submission on 7 Nov. The weight files are byte-identical (`ARTIFACT.json`).
- **Step 5 (feedback) was replaced.** The first version edited two output-layer rows; the maintainer ruled on 10 Nov that this fits weight values to the feedback. The new step never changes a weight value. It was pre-registered in `PROTOCOL.md` amendments 4–5 (commit `75b380c`) before any locked file was opened.
- **Timing.** The one locked run of the new step started on 12 Nov at 16:38 UTC and finished at 17:21 UTC, after the 17:05 UTC deadline. The pre-registration and the dev results were public before the deadline.

## Results

### Compression (locked exam opened once on 7 Nov, `results.json`)

| Rule | Requirement | Result |
|---|---|---|
| Size, everything shipped | ≤ 10% of fp16 | **6,039,657,231 B = 9.891%** before feedback |
| Faithful | ≥ 98% of base-correct items kept | **99.36%** (2,775 / 2,793) |
| Every kind | ≥ 95% | Lowest is **98.56%** (weather). MASSIVE 99.42%; CLINC150 (different source) 99.12% |
| Exam | ≥ 2,000 items, ≥ 300 per kind, ≥ 500 from an unused source | 2,910 items: every label has 330 or more, MASSIVE 2,310, CLINC150 600 |
| Base has something to keep | Well above chance, below ceiling | Base 95.98% (chance is 14.3%); small model 96.12% |

### Feedback (locked Hindi file opened once on 12 Nov, `results2.json`)

What the feedback teaches: requests in a language the small model handles badly. The labels are the same 7 skills.

| Measure | Result |
|---|---|
| Locked teach set | 379 Hindi requests (MASSIVE hi-IN), from utterance ids never used in any earlier split |
| Feedback given | The first 24 real mistakes of the small model in the Hindi `pool` file |
| Base model on the locked set | 94.99% |
| Small model, no feedback | 54.35% |
| **Small model after feedback** | **90.77%** (142 items fixed, 4 broken) |
| Empty feedback | 54.35% |
| Shuffled labels, seeds 1–6 | 54.35%, 54.35%, 54.35%, 54.35%, 54.35%, 54.35%. Experts restored across all controls: 0 |
| Experts restored by real feedback | 35 |
| Size after feedback | 6,105,631,828 B = **9.9987%** |
| Exam after feedback (second opening, same run) | Base-correct items kept: 99.28% before feedback → **99.32%** after. 10 of 2,910 answers changed. Lowest kind after feedback: 98.56% (weather) |
| Shipped commands | `feedback.py` ran as a separate process (169 s, about 360 MB fetched from the base repo). `answer.py`, run as a separate process on the first 20 locked items, gave 19 of 20 answers identical to the full-set run; the one that differs is a low-margin item in a different batch (the batch sensitivity is measured in amendment 5) |

Per label on the locked Hindi set, before → after: calendar 50.0% · email 78.8% · home 26.2% · lists 44.4% · news 0.0% · transport 42.5% · weather 90.7% → calendar 86.2% · email 90.0% · home 91.2% · lists 97.2% · news 66.7% · transport 92.5% · weather 96.3%.

This run used an NVIDIA H100 80GB HBM3; the 7 Nov run used an A100. Batch numerics differ slightly between the two, so this run's own before-feedback faithfulness (99.28%) is reported next to the after-feedback one.

Dev-only checks with the same frozen code (`PROTOCOL.md` amendment 5): Turkish 68.07% → 87.86%, Arabic 74.14% → 90.50%, shuffled labels unchanged in both.

## How it is small (no weight training anywhere)

- **Task-specific expert pruning.** Keep 2,640 of the 6,144 routed experts. The count varies per layer, from about 37 to 81. Experts are ranked by their share of their layer's top-8 routing mass on unlabeled calibration text (task requests plus generic English). The router is sliced to the kept experts, which is exactly the base router restricted to them. Every kept expert is a byte-exact slice of the base.
- **Round-to-nearest only.** Experts use 3-bit asymmetric codes, attention 6-bit, embeddings 5-bit, and the output layer 4-bit. Every group of 128 weights shares one fp16 scale, chosen from a fixed grid by weight MSE. Norms and router stay bf16. There is no GPTQ and no optimisation of any weight.
- **Shipping.** Codes are bit-packed into safetensors. `artifact.load()` rebuilds a standard Hugging Face `Qwen3MoeForCausalLM` with fewer experts per layer.

## How it learns from feedback (no weight training, no head, no lookup, nothing in the prompt)

The small model is the base with most of its experts given zero bits. Feedback changes which experts have bits. It never changes a weight value and never creates one.

`python feedback.py <artifact> annotations.jsonl` (each line: `{"text": ..., "label": ...}`, at most 25 items) does the following:

1. **Run and judge.** The small model answers the annotated items. Items it already gets right are ignored, because feedback marks mistakes.
2. **Trace.** For the mistaken items, read which pruned experts the base router would send those requests to (one forward pass, the base router applied to the small model's own hidden states). That ranks the pruned experts.
3. **Move bits.** Put the top-ranked group back: each expert is the base's own expert, rounded by the same fixed 3-bit rule as every kept expert, with the base's own router row. The group fills the room left under 10% (35 experts).
4. **Check that the mistakes actually recover.** The group stays only if the average margin of the annotated items toward their annotated labels rises by a set amount, at least 60% of those items move toward their label, and no more than 3 of 64 unlabeled English guard requests change their answer. Otherwise the group is removed again and nothing changes.

Restored experts are stored in `growth.safetensors`, which counts toward size. At answer time the model is the original model's own computation with more of its original experts present. Answering needs only the artifact.

How the annotated items are used: the texts pick the candidate experts, and the labels decide whether they stay. With shuffled labels the mistakes do not recover, so nothing is kept.

## What it cannot do

- **It restores what the base can do.** It cannot teach a skill the base never outputs (for example a new skill name such as "recipe").
- **Feedback needs the base model.** By default `feedback.py` fetches only the needed tensors (about 360 MB) from the base's public repo; `--base <folder>` uses a local copy.
- **One round fills the room.** The first round uses the free space under 10%. Later rounds would need to retire rarely used experts to make room; that is not implemented.

## Exam hygiene

**Locked before any model run.** `exam.jsonl` and the first teach set were hash-locked (`LOCK.json`) and committed first. The Hindi teach set and 11 other locales were hash-locked (`LOCK2.json`) and committed before any model run on them. The git history shows this.

**Never used to build or choose.** All choices were made on `dev`, `calib`, and the `pool` / `dev` files of the teach sets. Each decision and its dev numbers is in `PROTOCOL.md`, committed before the matching locked file was opened.

**Opened once each.** The exam was opened on 7 Nov for the first submission (`old_step5/modal_final_first_submission.py`). `modal_final2.py` opened the Hindi locked file once on 12 Nov and re-ran the exam in the same run, only to check the model after feedback. Only the Hindi locked file was opened; the other 11 locales' locked files remain unopened.

**Disclosures.**

- The exam was opened a second time, as announced in amendments 4 and 5. No exam output was used to design the new step.
- A rehearsal of the final script on dev files, run after the artifact and constants were frozen, already applied shuffle seeds 1–3 to the same 24 `pool` items. Seeds 4–6 were added after that and had never been run.
- The first submission's shuffled control used one fixed shuffle, the same one used while tuning.
- When the exam was built, its sentences were read once to exclude them from the calibration text.
- The pre-registered dev gate for compression was 99% overall and 98% per label. Dev reached 99.10%, with the news label at 97.85%. The gate was lowered to the challenge's own thresholds before opening the exam (amendment 2).
- Self-contained alternatives (feedback that only removes experts or neurons, or changes how many experts each token uses) were tried on dev and did not teach; see amendment 5 and `lab/`.
- The "locked first" order rests on this repo's git history and the Hugging Face upload times.
- Earlier attempts on other tasks or models failed or stopped at dev, and their sealed exams were never reused here.

## Commands

```
python answer.py <artifact_dir> messages.txt            # one skill name per line
python feedback.py <artifact_dir> annotations.jsonl     # at most 25 annotated mistakes
```

## Data and licences

- Base model: Qwen3-30B-A3B-Instruct-2507, Apache-2.0.
- Task data: Amazon MASSIVE (en-US, and the hi-IN and other locale files in `data/teach2/`; CC BY 4.0) and CLINC150 (CC BY 3.0). The subsets used are in `data/`.
