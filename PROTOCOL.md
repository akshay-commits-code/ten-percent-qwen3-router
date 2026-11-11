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

## Amendment 1 (dev only; exam still locked)

**Prompt.** The skill descriptions are now clearer (prompt P2): news covers current events, sports results, markets and prices. Base dev accuracy rose from 94.8% to 95.6%.

**What dev showed.** C1 keeps 98.6–98.7% of base-correct items. The `news` label keeps only 93.6–95.7%: broad current-events queries drift to weather or calendar.

**Change.** Calibration for expert selection is broadened with 1,200 unlabeled MASSIVE requests from out-of-task scenarios and 400 generic English lines (wikitext). Neither source is in the exam.

**Next step.** Re-run on dev. The go/no-go gate is unchanged.

## Amendment 2 (dev only; exam still locked)

**Dev results** (base dev accuracy 95.64%). Every config below uses expert selection by plain routing mass on the broadened calibration text (system tokens ×1, message ×1, last position ×0).

| Config | Base-correct kept | Weakest label (news) |
|---|---|---|
| 2,700 experts, 4-bit embeddings | 99.18% | 97.31% |
| 2,640 experts, 5-bit embeddings | 99.10% | 97.85% |
| 2,600 experts, 6-bit embeddings | 99.10% | 97.85% |

**Floor.** About 11 fragile items are lost in every config.

**Choice.** The selection rule is: highest minimum per-label faithfulness, then highest overall, then more experts (pruning is the main risk under source shift). It picks **2,640 experts, experts 3-bit asymmetric, attention 6-bit, embeddings 5-bit, output layer 4-bit**, about 9.83% of fp16 with everything counted.

**Gate.** The overall gate (≥99%) is met. The per-label gate (≥98%) misses by one news item. Following the user's instruction to go for it, the gate is set to the challenge's own thresholds: 98% overall and 95% per kind. Dev meets these with margins of +1.1 and +2.85 points.

## Amendment 3 (feedback design on teach_dev + dev only; exam still locked)

**Why the edit changed.** Before feedback, the small model already answers "music" for many music requests: teach_dev accuracy is 30% with no feedback. With the plain whitened edit, shuffled labels did as well as real ones. The edit direction now also points away from the other taught label's examples (`contrast`), so it carries information specific to each label.

**Picked by the attempt-2 rule:** shrinkage 0.3, contrast 1.0, delta 4, quantile 0.5.

| Measure (on dev) | Value |
|---|---|
| teach_dev, no feedback | 30.0% |
| teach_dev, real labels | 67.5% |
| teach_dev, shuffled labels | 38.1% |
| teach_dev, empty feedback | 30.0% |
| Dev answers changed | 1 |

**Artifact.** 6,040,715,643 bytes, which is 9.892% of 61,064,245,248 fp16 bytes. On a 300-item round trip, 299 answers were identical (one differs by batch numerics).

**Nearby exam items, pre-registered:** labels `lists` and `home`.

## Amendment 4: step 5 replaced (written before any model run on the new teach sets)

**Why.** The maintainer ruled on 10 Nov that the first feedback step does not qualify: it wrote new values into two output-layer rows, and those values were fitted to the feedback. The compression side was not questioned. The shipped weights (`model.safetensors`, `lm_head.safetensors`) stay byte-identical; only the feedback step is replaced.

**New mechanism: regrow.** Feedback never changes or creates a weight value. It only decides which of the base model's own pruned experts are put back.

1. Run the small model on the annotated items (at most 25, each a real mistake).
2. Candidates: pruned experts the base router would use on those items, ranked by the routing mass they would receive on the message tokens.
3. Try the candidates one at a time, in that order. An expert stays only if the annotated items move toward their annotated labels (the sum of capped label margins rises by a set amount). Otherwise it is removed again.
4. Restored experts are copied from the base and rounded by the same fixed rule as every kept expert (3-bit asymmetric, fp16 group scales). They are stored in `growth.safetensors`, which counts toward size. Growth stops before everything shipped exceeds 10% of the base's fp16 bytes.

At answer time the model is the original model's own computation with more of its original experts present. No head, no adapter, no lookup, nothing in the prompt.

**What the feedback teaches.** Requests in a language the small model was not built for (it was selected and calibrated on English). Twelve MASSIVE locales are candidates. MASSIVE is a parallel corpus, so the new sets use only utterance ids whose English text was never in any earlier split. For each locale the same ids are split into `pool`, `dev` and `locked` (379 items each); all 12 `locked` files are hash-locked in `LOCK2.json` before any model run.

**Choices, all on `pool` / `dev` and the English `dev` only.**

- Locale: the one with the largest gap between base and small accuracy on its `dev`, with base accuracy at least 85%.
- Growth constants (number of candidates tried, margin cap, acceptance threshold, growth cap).
- Feedback set: the first 24 small-model mistakes in the locale's `pool`, in file order.

**Final run, once.** On the chosen locale's `locked` file and the exam:

- base, small before feedback, real feedback, empty feedback, and shuffled-label feedback with three different shuffles (seeds 1–3; design uses seed 0)
- the exam after real feedback

**Disclosed in advance.** The exam was already opened once for the first submission. It is opened a second time here, only to check the model after the new feedback. No exam output was used to design the new step. Only the chosen locale's `locked` file is opened.

**Pass conditions.** Locked accuracy rises clearly over no feedback; shuffled and empty do clearly worse than real; exam faithfulness after feedback stays at or above 98% overall and 95% per kind; total size stays at or under 10%.
