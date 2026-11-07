**Title:** Proof: Qwen3-30B-A3B at 9.892% of fp16, keeps 99.36% of base-correct items on a 2,910-item locked exam, learns two new skills from 24 labels

**Artifact + full proof:** https://huggingface.co/akshay-commits-code/qwen3-30b-a3b-10pct-voice-router. The repo root is the shipped artifact; [`proof/`](https://huggingface.co/akshay-commits-code/qwen3-30b-a3b-10pct-voice-router/tree/main/proof) holds the code, locks, protocol, git history and results.

**Base:** `Qwen/Qwen3-30B-A3B-Instruct-2507`, 61,064,245,248 fp16 bytes.

**Task:** voice-assistant skill routing (weather / email / transport / lists / calendar / home / news). The answer is the skill name from free generation.

| Rule | Result |
|---|---|
| ≤10% of fp16, everything shipped | **9.892%** (6,040,715,980 B: weights, tokenizer, code, task file, feedback stats) |
| Faithful (≥98% of base-correct kept) | **99.36%** (2,775 / 2,793) |
| ≥95% per kind | Lowest **98.56%** (weather). CLINC150 99.12%, MASSIVE 99.42% |
| Exam | 2,910 locked items. Each label has ≥330. 600 come from CLINC150, which was never used to build, tune or calibrate. |
| Base has something to keep | Base 95.98% (chance 14.3%); small model 96.12% |
| Feedback (24 annotated mistakes) | Locked teach set (390 items): 18.97% → **64.87%**. New skill `recipe`: 0% → 61.1%; its CLINC items (another source) 0% → 54.7%. |
| Controls | Shuffled labels 29.23%; empty 18.97% |
| Doesn't break the exam | Still 99.36% after feedback. 1 of 2,910 answers changed; nearby labels 99.62% |

**Compression (no weight training).**

- Task-specific expert pruning: 2,640 of 6,144 routed experts kept, chosen by routing mass on unlabeled calibration text. Experts are exact slices of the base, and the router is sliced to them.
- Round-to-nearest: experts 3-bit asymmetric, attention 6-bit, embeddings 5-bit, output layer 4-bit, with fp16 group scales.

**Feedback (no training, no side memory, nothing in the prompt).** A closed-form edit of the existing output-layer rows of the taught label tokens:

- The direction is the mean hidden state of the mistaken items, contrasted with the other taught label and whitened by stored calibration statistics (1 MB, used only to compute edits).
- The step size is the median per-item step needed to flip each item.
- Rows are re-rounded in place, so the size is unchanged.

**Commands.** `python answer.py <artifact> messages.txt` and `python feedback.py <artifact> annotations.jsonl` (≤25 items).

**Hygiene.**

- The exam and teach set were hash-locked and committed before any model run on this task.
- Every choice was made on dev, with each one recorded in `PROTOCOL.md` amendments committed before the exam was opened.
- The exam was opened once (`modal_final.py`).

**Disclosures.**

- The pre-registered dev gate was 99% overall and 98% per label. Dev reached 99.10%, with news at 97.85%, so I lowered the gate to the challenge's thresholds before opening the exam.
- On the `music` subset alone, shuffled labels scored higher (88.3%) than real labels (73.3%). The base already partly knew "music"; `recipe` is where the labels clearly teach.
- Earlier attempts on other tasks or models failed or stopped at dev, and their exams were never reused.
