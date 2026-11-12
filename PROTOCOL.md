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

## Amendment 5: step 5 finalised on pool / dev only (the hi-IN locked file is still unopened; the exam has not been re-opened)

**Self-contained alternatives were tried first and dropped.** All on hi-IN `pool` / `dev` and the English `dev` (`lab/`):

| Idea (nothing fetched from the base) | Unseen Hindi `dev` |
|---|---|
| No feedback | 55.7% |
| Feedback removes more experts (prune-only) | No usable gain. Removing one expert moves the 24 annotated items by about as much as run-to-run numeric noise. |
| Feedback changes how many experts each token uses (4 / 6 / 12 / 16 instead of 8) | 42.0% / 53.6% / 57.0% / 53.8% |
| Upper limit of only removing the "answer weather when unsure" habit | 63.9% |

The small model fails on Hindi because the experts that handle Hindi were pruned. Only putting them back fixes that, so the regrow mechanism of amendment 4 stays.

**Changes to amendment 4, with reasons.**

- **One group instead of one expert at a time.** Candidates are put back as a group that fills the room left under 10% (35 experts). A single expert's effect on 24 items is about as large as numeric noise, so one-at-a-time decisions were unreliable. On hi-IN `dev`: groups of 4 kept 4 experts (57.8%), groups of 8 kept 32 (81.8%), groups of 12 kept 12 (66.2%), one group of 35 kept 35 (86.3%).
- **Stricter label test.** A group stays only if (a) the average clipped margin of the annotated mistakes toward their annotated labels rises by at least 0.5, and (b) at least 60% of those items each move toward their label by more than 0.5. Test (b) was added because test (a) alone accepted some shuffled-label feedback: 3% of 2,000 simulated shuffles of the Hindi items, and the seed-0 shuffle of the Turkish items, where 19 of the 24 mistakes carry one of two labels so a shuffle leaves many labels correct. With (b), 0 of 2,000 simulated Hindi shuffles pass; the real labels pass with 88% of items moving.
- **Guard.** `guard.jsonl` holds 64 unlabeled English requests from the calibration text. A group is rejected if more than 3 of the model's own answers on them change.
- **Where restored experts come from.** By default `feedback.py` fetches only the needed tensors (the base router and the chosen experts, about 360 MB) from the public base repo, pinned to revision `0d7cf23`. `--base <folder>` uses a local copy. Answering needs nothing but the artifact.
- **Scoring inside feedback.** The margin is read at the first answer token: the logit of the annotated skill name minus the best other token.

**Frozen constants (`growth_cfg.json`).** Group 35, at most 70 candidates, stop after 2 rejected groups, margin clip −8 to +3, average gain ≥ 0.5, share of items moved ≥ 60% (each by more than 0.5), guard ≤ 3 of 64 changed, limit 6,106,424,524 bytes.

**Locale.** By the amendment-4 rule (largest base-minus-small gap on `dev`, base at least 85%): **hi-IN**, base 94.5%, small 54.1% (probe saved before any design work).

**Dress rehearsal on dev with the frozen code and constants** (feedback = first 24 small-model mistakes in the locale's `pool`; design shuffle seed 0):

| Locale | `dev` before | Real feedback | Shuffled labels | Experts restored (real / shuffled) | English `dev` faithfulness before → after |
|---|---|---|---|---|---|
| hi-IN (chosen) | 55.67% | **86.28%** | 55.67% | 35 / 0 | 99.10% → 99.18% (1,400 items, 7 answers changed) |
| tr-TR (extra check) | 68.07% | 87.86% | 68.07% | 35 / 0 | 98.69% → 98.69% (first 400 items) |
| ar-SA (extra check) | 74.14% | 90.50% | 74.14% | 35 / 0 | 98.69% → 98.69% (first 400 items) |

After real feedback the hi-IN artifact is 6,105,631,828 bytes (9.9987%). A fresh load from disk gave the same 379 `dev` answers as the in-memory model.

**Rehearsal of the final script on dev files.** After the constants and the artifact were frozen, `modal_final2.py --rehearsal` ran the whole final procedure with hi-IN `dev` in place of `locked` and the English `dev` in place of the exam (no base model, no locked file in the image). It exists to make sure the one real run cannot fail on a technicality. Result: real feedback through the shipped `feedback.py` command restored 35 experts and a fresh load scored 86.28% (120 items fixed, 4 broken); empty and shuffled seeds 1–3 restored nothing (55.67%); `answer.py` gave the same answers as the in-process model; 7 of 1,400 English `dev` answers changed. Nothing was changed after the rehearsal except adding shuffle seeds 4–6. Saved in `work/step5/rehearsal_raw.json` and `work/step5/rehearsal_results.json`.

**Frozen artifact (`ARTIFACT.json`).** 6,039,657,231 bytes before feedback (9.891%). `model.safetensors` and `lm_head.safetensors` are byte-identical to the first submission. The old `feedback.json` and `feedback_stats.safetensors` are gone; `grow.py`, `keep.json`, `growth_cfg.json` and `guard.jsonl` are added.

**Final run, once (`modal_final2.py`).** As in amendment 4, on hi-IN `locked` and the exam:

- base; small before feedback; empty feedback; shuffled-label feedback with seeds 1 to 6
- real feedback, run through the shipped `feedback.py` command, then a fresh load of the artifact answers hi-IN `locked` and the whole exam
- the exam is re-run after any control that restores an expert

**Pass conditions (unchanged).** Locked accuracy rises clearly over no feedback; shuffled and empty do clearly worse than real; exam faithfulness after feedback stays at or above 98% overall and 95% per kind; everything shipped stays at or under 10%.

**Disclosed in advance.**

- Feedback needs the original model's files (public repo or a local copy). It adds only slices of the original model.
- The annotated texts pick the candidate experts; the labels decide whether they stay. When the labels are rejected nothing changes, so a rejected shuffled run equals the empty run.
- The exam was opened once for the first submission and is opened a second time here, only to check the model after the new feedback. No exam output was used to design this step.
- The first submission's shuffled control used one fixed shuffle (seed 0), the same one used while tuning. This run uses six shuffles. Seeds 1–3 were pre-registered in amendment 4 and no choice was ever made on them, but the rehearsal below already applied them to the same 24 `pool` items (all three were rejected). Seeds 4–6 were added after that and have never been run.
- When the exam was built, its sentences were read once to exclude them from the calibration text.
- The "locked first" order rests on this repo's git history and the Hugging Face upload times.

## Result of the final run (12 Nov, `final2/raw.json`, `results2.json`)

The run started at 16:38 UTC and finished at 17:21 UTC on an H100 (the script allows an A100-80GB or an H100). It ran once, with the code and artifact frozen in commit `75b380c`.

| Measure on the locked hi-IN file (379 items) | Result |
|---|---|
| Base | 94.99% |
| Small, no feedback | 54.35% |
| Small after real feedback (24 mistakes, shipped `feedback.py`, fresh load) | **90.77%** (142 fixed, 4 broken; 35 experts restored) |
| Empty feedback | 54.35% |
| Shuffled labels, seeds 1–6 | 54.35% each; no group passed the label test, nothing restored |

| Other pass conditions | Result |
|---|---|
| Exam faithfulness in this run, before → after real feedback | 99.28% → 99.32%; 10 of 2,910 answers changed; lowest kind after feedback 98.56% (weather) |
| Size before → after feedback | 9.8907% → 9.9987% (6,105,631,828 bytes) |

**Notes.**

- The run finished 16 minutes after the challenge deadline (17:05 UTC). Amendment 5 and the dev results were public before it.
- The 7 Nov run measured 99.36% exam faithfulness on an A100; this run measures 99.28% before feedback on an H100. Base and small answers both shift slightly with hardware batch numerics.
- `answer.py`, run as a separate process on the first 20 locked items, matched the full-set run on 19 of 20. The item that differs has a low margin and sat in a different batch.
- After amendment 5 was committed, one more self-contained idea was tried on pool / dev only (feedback prunes single neurons, `lab/neuron_lab.py`). It did not teach: on Hindi it only memorised the annotated items (unseen dev 55.2% → 51.7%), and on new skill names it either changed English answers or learned nothing label-specific. Nothing in the frozen code or constants was changed.
