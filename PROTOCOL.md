# Attempt 5: pre-registered protocol (written before any model run on this task)

## What earlier attempts taught

- **Ambiguity sets a floor.** Faithfulness is limited by how decisive the base is. Attempt 2's bank-queue routing had about 3% ambiguous items that flip under any compression. On attempt 3's spam task, Qwen3 shifted one way, towards "spam", under every compression step.
- **BitNet doesn't solve it.** Attempt 4 used BitNet b1.58 2B, where the ternary body is exactly lossless. But the base scored only 62% on spam, close to chance, and cutting the vocabulary to fit 10% was destructive. The exam was not opened.

## Task

Route a voice-assistant request to the skill that should handle it: weather, email, transport, lists, calendar, home or news. These skills are clearly distinct. The answer is the skill name, produced by free greedy generation.

**Base.** Qwen3-30B-A3B-Instruct-2507, bf16.

## Locked files (`LOCK.json`)

**`data/exam.jsonl`.** 2,910 items:

- 2,310 from MASSIVE en-US.
- 600 from CLINC150's utility, travel, auto, home and work intents. This is the different source; none of these CLINC items was used or seen before.

**`data/teach_locked.jsonl`.** 390 items covering the two new skills: `recipe` (MASSIVE cooking + CLINC recipe intents) and `music` (MASSIVE music).

## Compression

**Recipe.** The attempt-2 recipe, re-derived on this task's `calib`. Candidates:

| Candidate | Experts kept | Attention | Size |
|---|---|---|---|
| C1 | 2,650 | 6-bit | 9.89% |
| C1_a8 | 2,520 | 8-bit | 9.86% |

Pick the one with the highest dev faithfulness.

**Go/no-go on dev (1,400 MASSIVE items).** Faithfulness must be ≥ 99.0% overall and ≥ 98.0% for every label. Otherwise the exam stays locked.

## Feedback

**Feedback set.** The first 12 mistakes per new skill in `teach_pool`, 24 items in all.

**Mechanism.** The same whitened closed-form edit of existing output rows. Constants are picked by the attempt-2 rule on `teach_dev` + dev.

**Controls.** Shuffled-label and empty feedback.
