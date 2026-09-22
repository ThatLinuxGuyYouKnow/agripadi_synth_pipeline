# Distractor-vs-Correct Length Bias — Discovery & Fix

## Discovery (measured on `smoke.jsonl`: 424 rows, 216 MC)

- **79% of items** had the correct answer as the *uniquely longest* choice
  (char-based; 137/216 strictly longest by word count).
- Correct answer averaged **10.0 words** vs **6.5 words** per distractor
  (1.54× ratio).
- Answer length rank distribution: `0 (longest) = 178, 1 = 17, 2 = 11, 3 = 10`.

## Why it is a problem

- A model trained on this distribution learns the **heuristic
  "pick the longest, most detailed option"** instead of agronomy. Judge-written
  hidden items will not have this systematic bias, so the heuristic misfires.
- The profiler's `_extract_score` **prefers `acc_norm,none`**, and acc_norm
  normalizes by continuation byte-length. Teaching the model to dump probability
  mass on the longest option is the worst case under exactly that metric.

## Fix applied to `question_gen.py`

1. **Generation prompt** (`_FORMAT_RULES`, shared by grid and excerpt tracks):
   added an explicit *mechanical* instruction — the correct answer must NOT be
   the longest option; count words; if the answer has the most, lengthen a
   distractor or shorten the answer; ban hedging padding such as
   "It is recommended to always...".

2. **Deterministic gate** (`validate_mc`): reject any item where the correct
   answer is *both* uniquely the longest *and* ≥1.25× the distractor mean word
   count. Rejects flow into the existing `rejected` file with a `length_bias:`
   note so the per-run rate is visible.

3. **LLM verifier** (`VERIFIER_SYSTEM`): added a rejection bullet for the
   stylistic version of the same tell — "the correct answer stands out
   stylistically as noticeably longer, more detailed, or more carefully hedged
   than every distractor."

## Verification

- Unit checks pass (balanced item passes; length-biased item rejected;
  longest-but-close item passes; pre-existing checks unchanged).
- Retro-applied to `smoke.jsonl`: **129/216 items would now be rejected** for
  length bias, confirming the generator badly needed the constraint and that
  the current smoke set should **not** be used for training as-is.

## Concrete bad examples (from `smoke.jsonl`)

**Example 1 — `cassava_pest_a1dcb281`**
```json
"question": "A cassava farmer has noticed the local cassava green mite causing serious damage in the dry season. What is the best first step to control it?",
"choices": [
  "Uproot all affected plants and burn them right away",
  "Apply a strong chemical pesticide immediately to kill the mites",
  "Flood the field with water to drown the mites",
  "Plant a cassava variety known to be resistant to the cassava green mite"
]
"answer_index": 3,
"answer": "Plant a cassava variety known to be resistant to the cassava green mite",
```
- Answer: **10 words**. Distractors: **7, 8, 7** words. Answer uniquely longest
  and ~1.4× the distractor mean. A test-taker can eliminate B (explicitly warned
  against) and C (physically nonsensical) and guess A vs D by length/detail.

**Example 2 — `cassava_pest_6815c4fe`**
```json
"question": "I see this white cotton-like thing full my cassava top and the leaves dey bunch together. What I go do make am stop?",
"choices": [
  "Cut off all the leaves and leave the stems",
  "Remove and avoid using cuttings from the infested plants to stop the spread",
  "Spray the whole farm with a strong insecticide immediately",
  "Bend the affected stems down and bury them"
]
"answer_index": 1,
"answer": "Remove and avoid using cuttings from the infested plants to stop the spread",
```
- Answer: **11 words**. Distractors: **6, 7, 6** words. Same tell: the correct
  option is the longest and most hedged ("avoid using cuttings from the infested
  plants to stop the spread"), so it reads as "the careful one."

## Action

- Regenerate the training set under the new prompt + gate rules before the next
  serious training run.
- Watch the `length_bias:` notes in the rejected file on the first post-fix run;
  if reject rate stays above ~25%, escalate from "instruction" to
  "generate-N-and-drop-one" length balancing.

---

# Excerpt Fact-Conflict Sweep — Fixes & Dataset Impact (2026-08-09)

Full read of all 132 excerpts in `excerpts.json`. One cross-source conflict
(pepper rotation: Nigeria 3-4 yr vs Ghana 2 yr — left as-is, both are source
claims), plus the following errors. All fixes applied to `excerpts.json`:

1. **`goat_markets_001` — arithmetic error, FIXED.**
   Source said: "selling a doe for 900 meticais while spending 60 on it,
   40 on treatment and 20 to go to market, gives a profit of **840**."
   900 − (60+40+20) = **780**. Corrected to 780.

2. **`cassava_storage_002` — conflicting trench dimensions, one removed.**
   First sentence gave "pits or trenches, usually 1 meter long and
   30-40 cm wide"; later text (kept) gives "trenches measuring 1 meter wide
   and 30 to 40 cm deep". Removed the dimension clause from the first
   sentence; the detailed version (with drainage-ditch spec) is authoritative.

3. **`policy_extension_002` — duplicate committee name, one removed.**
   Same body called "Technical Sub-Committee on Crops" and "Seed Registration
   and Release Subcommittee". Unified to "Technical Sub-Committee on Crops".

4. **`pepper_disease_002` — wrong claim removed.**
   "sombo and atawere last about four days" is implausible (harvesting is
   weekly; the 2-yr claim for tatase/atarodo is already contested by the
   verifier, see below). Replaced with "are shorter-lived".

5. **`pepper_disease_003` — Deltamethrin rate, fixed after web check.**
   Source (Ghana MOFA Pepper_Production.pdf) itself prints "Deltamethrin
   product at 75-100 mls/L" — excerpt copied it faithfully, but real
   deltamethrin labels run ~0.3-1 ml/L (Decis 2.5EC: 10-15 ml/20 L for
   thrips; Decis Protech: 33 ml/100 L; Decis 100: ~0.5 ml/L). The "per
   litre" is a source typo, ~100x too high. Corrected to "75-100 ml per
   15 litres of water" (matching the adjacent Lambda-Cyhalothrin rate in
   the same guide).

## Dataset impact (checked agripadi_v3, sft, train, advisory_sft,
advisory_questions, additions, main_updates, retry, smoke, rejected/flagged)

- **Goat 840 error: did NOT propagate.** The generated item
  (`goat_markets_agribusiness_70789adb`, in agripadi_v3/sft/train) asks
  "sells a goat for 900 meticais but spent 120 ... What is the profit?" and
  the answer/answer_index is **780** — the generator silently recomputed the
  correct value even though the excerpt said 840 at generation time.
- **Cassava trench dims: only the kept version in data.** train.jsonl /
  advisory_sft.jsonl ("make the trench 1 meter wide and 30-40 cm deep")
  matches the surviving text; no "1 meter long / 30-40 cm wide" anywhere.
- **Pepper "four days" / longevity claim: rejected, not in training data.**
  The generated "Which pepper variety is more productive for a longer period,
  tatase or sombo?" landed in `agripadi_v3_rejected.jsonl` — the verifier
  flagged it: "sombo typically yields over a longer period. Answer is wrong."
  So the source's 2-year claim is itself contested; nothing from this excerpt
  reached accepted/training sets.
- **Deltamethrin 75-100 ml and the duplicate committee name: zero trace**
  in any dataset file. Neither influenced generated content.

Verdict: the five excerpts.json errors did not contaminate the training data
(the verifier + generator math caught or avoided them); `excerpts.json` is now
clean.

---

# Hausa Supplement Heist (2026-08-09)

Origin: `~/Downloads/adoption_tuning/data.jsonl` (Adaption remaster of our own
pipeline output, 2,562 unique IDs — a subset of ours). Harvested its only
genuinely-new content: **216 Hausa translations** (mc=123, short_answer=93,
verified, schema-identical to our rows).

Applied:
- Extracted to `hausa_supplement.jsonl` (ids suffixed `-ha`, language="Hausa").
- Appended to `agripadi_v3.jsonl` (4507 rows) — English parallels remain, so
  each item trains in BOTH languages (double coverage).
- Re-rendered via `render_sft.py` → `sft.jsonl` 5,983 rows; `train.jsonl`
  6,720 rows (+339: mc_letter=123, mc_arc=123, short_answer=93; raw/chat ~50/50).
- No advisory rows in the Hausa set → zero API spend, `advisory_sft.jsonl`
  untouched.
- Backups: `train.jsonl.pre_hausa`, `agripadi_v3.jsonl.pre_hausa`.
- Tokenizer check: tiny-aya-earth encodes ɗ/ƙ/ɓ as real SentencePiece units,
  no byte-fallback garbage. Safe to train.

Update (same day): dropped 14 Hausa rows where `answer` was still English
(hausa q -> english a). Kept 202 clean hausa->hausa pairs (320 train
examples). Train now 6,701 rows; duplication ~4.5%, bilingual augmentation
not verbatim duplication so low overfit risk.
