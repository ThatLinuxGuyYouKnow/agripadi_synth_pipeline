# AgriPadi — Full Pipeline Runbook (Aug 2026)

Local data-generation run. Everything uses the `DEEPSEEK_API_KEY` in `.env` — loaded automatically via `python-dotenv` at import time in `question_gen.py` (a shell `export` still overrides it). If the key is missing, the scripts exit immediately with a clear message.

**Provider/model:** `--provider deepseek --model deepseek-v4-flash`
(If `deepseek-v4-flash` 404s, the old `deepseek-chat` alias still works — just drop the `--model` flag.)

Note: `question_gen.py` disables DeepSeek "thinking" mode on every call (`extra_body={"thinking": {"type": "disabled"}}`). v4-flash is a reasoning model and its chain-of-thought counts against `max_tokens` — with thinking on, batches returned 6k–16k tokens of reasoning and *empty* content, failing JSON parsing. Smoke test with thinking off: 36/36 items per batch, ~5.4k output tokens per 3-excerpt call.

**Total runtime estimate:** ~40–60 min (gen ~20–25, verifier ~10, answers ~15).

**Total cost estimate:** well under $2 (most of the system-prompt input hits the cache-hit rate).

---

## 0. Smoke test (1 min, optional but recommended tonight)

Verifies the model name + batched output parsing before the full spend.

```bash
python3 -c "import json; json.dump(json.load(open('excerpts.json'))[:3], open('/tmp/smoke_excerpts.json','w'))"
python3 question_gen.py --excerpts /tmp/smoke_excerpts.json --grid-items 0 \
    --out /tmp/smoke_test.jsonl --provider deepseek --model deepseek-v4-flash
wc -l /tmp/smoke_test.jsonl /tmp/smoke_test.jsonl* 2>/dev/null
```

Expect `~36 items` (3 excerpts × 12), a mix of `mc` / `short_answer` / `advisory`, and no `FAILED` lines.

---

## 1. Generate questions (stage 1)

Excerpt track (all 132 excerpts) + grid track (breadth curriculum, 3000 target).

```bash
python3 question_gen.py \
    --excerpts excerpts.json \
    --grid-items 3000 \
    --out agripadi_v3.jsonl \
    --flagged-out agripadi_v3_flagged.jsonl \
    --rejected-out agripadi_v3_rejected.jsonl \
    --provider deepseek \
    --model deepseek-v4-flash
```

**Planned:** `4,584` items = `2,292` MC + `1,528` short_answer + `764` advisory
- Excerpt track: 132 × 12 = `1,584` (6 MC / 4 short / 2 advisory each)
- Grid track: 500 cells × 6 = `3,000` (3 MC / 2 short / 1 advisory each)

**Expected accepted** (after dedup ~1–2%, source-voice flagging ~2–3%, verifier rejections ~2–5%):
**~4,300–4,500 rows**. Watch the printed summary:
- `length_bias:` notes in `agripadi_v3_rejected.jsonl` — the gate is doing its job, but with v4-flash the smoke test rejected **9/18 MC items (~50%)** on this, so expect the final MC count to land well under the planned 2,292 (total rows maybe ~3,400–3,800, not 4,300+). If MC yield matters (it renders 2 variants each and is the profiler's main surface), consider a retry pass for rejected MCs before training.
- Flagged rate over ~5%: skim `agripadi_v3_flagged.jsonl` before training.

## 2. Render SFT (stage 1b)

```bash
python3 render_sft.py --in agripadi_v3.jsonl \
    --out sft.jsonl \
    --advisory-out advisory_questions.jsonl
```

**Expected output:** ~5,700–6,000 SFT examples (each MC row renders 2 variants — `mc_letter` + `mc_arc` — half as raw text, half as chat; each short_answer renders 1), plus ~750 advisory questions passed through to `advisory_questions.jsonl`.

## 3. Generate advisory answers (stage 2)

```bash
python3 answer_gen.py \
    --in advisory_questions.jsonl \
    --excerpts excerpts.json \
    --out advisory_sft.jsonl \
    --provider deepseek \
    --model deepseek-v4-flash
```

**Expected output:** ~700–750 advisory answer pairs (40–160 words each, with one feedback-driven retry pass).

## 4. Combine into the final training file

```bash
cat sft.jsonl advisory_sft.jsonl > train.jsonl
wc -l train.jsonl
```

**Expected:** ~6,400–6,750 total rows.

## 5. Sanity checks

```bash
python3 -c "
import json
from collections import Counter
rows = [json.loads(l) for l in open('train.jsonl')]
print(Counter(r.get('variant') for r in rows))
print(Counter(r.get('surface') for r in rows if 'surface' in r))
print('advisory:', sum(1 for r in rows if r.get('variant')=='advisory'))
letters = Counter(r.get('answer_index') for r in rows if 'answer_index' in r)
print('MC letters:', dict(sorted(letters.items())))
"
```

- MC answer letters should be roughly uniform (no letter position bias).
- No `format: advisory` rows should remain in `sft.jsonl` — they all live in `advisory_questions.jsonl`.

---

## Kaggle training (separate, after tonight's run)

- **`train_qlora.py`** reads a `question_gen.py` output directly (`DATA_PATH = "agripadi_v3.jsonl"`) and uses ONLY `mc` + `short_answer` rows — advisory rows are skipped. Point it at `agripadi_v3.jsonl` (NOT `train.jsonl`, which it can't parse: no `format` field).
- To train on the advisory answers too, `train_qlora.py` needs a small edit to consume `train.jsonl` (rows keyed by `variant`/`surface`, not `format`) — decide after seeing how the judge path weighs in. Do not hold tonight's run for this.
- Then `eval_arc_easy.py` merges the LoRA → exports Q4_K_M GGUF → scores via lm-eval (same runtime as the audit).

---

## File map

| File | Contents |
|---|---|
| `agripadi_v3.jsonl` | Accepted stage-1 rows (mc / short_answer / advisory) |
| `agripadi_v3_flagged.jsonl` | Possible source-voice leakage (review, don't train) |
| `agripadi_v3_rejected.jsonl` | Shape/verifier/length-bias rejections |
| `sft.jsonl` | Rendered MC + short-answer SFT examples |
| `advisory_questions.jsonl` | Advisory questions (stage-2 input) |
| `advisory_sft.jsonl` | Advisory question → answer pairs |
| `train.jsonl` | `sft.jsonl` + `advisory_sft.jsonl` (final local set) |
