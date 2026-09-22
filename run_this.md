# AgriPadi — Pipeline Runbook

Local data-generation run for the AgriPadi synthetic SFT pipelines.
Everything uses the `DEEPSEEK_API_KEY` in `.env` — loaded automatically via
`python-dotenv` at import time in `question_gen.py` (a shell `export` still
overrides it). If the key is missing, the scripts exit immediately with a
clear message.

**Provider/model:** `--provider deepseek --model deepseek-v4-flash`
(If `deepseek-v4-flash` 404s, the old `deepseek-chat` alias still works —
just drop the `--model` flag.)

Note: `question_gen.py` disables DeepSeek "thinking" mode on every call
(`extra_body={"thinking": {"type": "disabled"}}`). v4-flash is a reasoning
model and its chain-of-thought counts against `max_tokens` — with thinking
on, batches returned 6k–16k tokens of reasoning and *empty* content,
failing JSON parsing. Smoke test with thinking off: 36/36 items per batch,
~5.4k output tokens per 3-excerpt call.

**Total runtime estimate:** ~40–60 min (gen ~20–25, verifier ~10,
answers ~15).

**Total cost estimate:** well under $2 (most of the system-prompt input
hits the cache-hit rate).

---

## 0. Smoke test (~1 min, optional but recommended)

Verifies the model name + batched output parsing before the full spend.

```bash
python3 -c "import json; json.dump(json.load(open('excerpts.json'))[:3], open('/tmp/smoke_excerpts.json','w'))"
python3 question_gen.py --excerpts /tmp/smoke_excerpts.json --grid-items 0 \
    --out /tmp/smoke_test.jsonl --provider deepseek --model deepseek-v4-flash
wc -l /tmp/smoke_test.jsonl /tmp/smoke_test.jsonl* 2>/dev/null
```

Expect `~36 items` (3 excerpts × 12), a mix of `mc` / `short_answer` /
`advisory`, and no `FAILED` lines.

---

## 1. Generate questions (stage 1)

Excerpt track (all 132 excerpts) + grid track (breadth curriculum, 3000
target).

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

**Planned:** `4,584` items = `2,292` MC + `1,528` short_answer + `764`
advisory
- Excerpt track: 132 × 12 = `1,584` (6 MC / 4 short / 2 advisory each)
- Grid track: 500 cells × 6 = `3,000` (3 MC / 2 short / 1 advisory each)

**Expected accepted** (after dedup ~1–2%, source-voice flagging ~2–3%,
verifier rejections ~2–5%): **~4,300–4,500 rows**. Watch the printed
summary:

- `length_bias:` notes in `agripadi_v3_rejected.jsonl` — the gate is doing
  its job, but with v4-flash the smoke test rejected **9/18 MC items
  (~50%)** on this, so expect the final MC count to land well under the
  planned 2,292 (total rows maybe ~3,400–3,800, not 4,300+). If MC yield
  matters (it renders 2 variants each), run the repair passes below before
  rendering.
- Flagged rate over ~5%: skim `agripadi_v3_flagged.jsonl` before
  publishing.

### 1c. Repair passes (as needed)

Run these between stage 1 and rendering when the summary shows failed
batches or a large rejected pile.

**Replay transiently-failed API batches** (stage-1 batches that died on
connection errors — batch indices are known and deterministic):

```bash
python3 retry_failed_batches.py \
    --excerpts excerpts.json \
    --existing agripadi_v3.jsonl \
    --out agripadi_v3_retry.jsonl \
    --flagged-out agripadi_v3_retry_flagged.jsonl \
    --rejected-out agripadi_v3_retry_rejected.jsonl \
    --provider deepseek --model deepseek-v4-flash
```

**Regenerate gate/verifier rejects with feedback**, then re-verify
anything still lacking a verdict. Writes additions/updates — never
overwrites the main file:

```bash
python3 regenerate_rejected.py \
    --rejected agripadi_v3_rejected.jsonl \
    --retry agripadi_v3_retry.jsonl \
    --main agripadi_v3.jsonl \
    --excerpts excerpts.json \
    --out agripadi_v3_additions.jsonl \
    --failed-out agripadi_v3_regen_failed.jsonl \
    --main-updates-out agripadi_v3_main_updates.jsonl \
    --provider deepseek --model deepseek-v4-flash
```

After a repair pass, append additions / apply main updates into
`agripadi_v3.jsonl`, then continue to stage 1b.

---

## 2. Render SFT (stage 1b)

```bash
python3 render_sft.py --in agripadi_v3.jsonl \
    --out sft.jsonl \
    --advisory-out advisory_questions.jsonl
```

**Expected output:** ~5,700–6,000 SFT examples (each MC row renders 2
variants — `mc_letter` + `mc_arc` — and each short_answer renders 1), plus
~750 advisory questions passed through to `advisory_questions.jsonl`.

**Surfaces:** by default each rendered example is coin-flipped to either
a chat `messages` row (`surface: "chat"`) or a raw completion `text` row
(`surface: "raw"`, matching lm-eval's template-free profiling surface),
via `--raw-frac` (default `0.5`).

For a **chat-only** output (the form published to Hugging Face — raw
completions are excluded), render with:

```bash
python3 render_sft.py --in agripadi_v3.jsonl \
    --out sft.jsonl \
    --advisory-out advisory_questions.jsonl \
    --raw-frac 0
```

---

## 3. Generate advisory answers (stage 2)

```bash
python3 answer_gen.py \
    --in advisory_questions.jsonl \
    --excerpts excerpts.json \
    --out advisory_sft.jsonl \
    --provider deepseek \
    --model deepseek-v4-flash
```

**Expected output:** ~700–750 advisory answer pairs (40–160 words each,
with one feedback-driven retry pass).

---

## 4. Combine into the final set

```bash
cat sft.jsonl advisory_sft.jsonl > train.jsonl
wc -l train.jsonl
```

**Expected:** ~6,400–6,750 total rows (production run: **6,701**).

---

## 5. Sanity checks

```bash
python3 -c "
import json
from collections import Counter
rows = [json.loads(l) for l in open('train.jsonl')]
print('variants:', Counter(r.get('variant') for r in rows))
print('surfaces:', Counter(r.get('surface') for r in rows if 'surface' in r))
print('advisory:', sum(1 for r in rows if r.get('variant')=='advisory'))
print('missing messages:', sum(1 for r in rows if not r.get('messages')))
print('dup ids:', len(rows) - len({r['id'] for r in rows if r.get('id')}))
"

# MC letter balance lives on the stage-1 rows (train.jsonl has no answer_index)
python3 -c "
import json
from collections import Counter
mc = [r for r in map(json.loads, open('agripadi_v3.jsonl')) if r.get('format')=='mc']
print('MC letters:', dict(sorted(Counter(r['answer_index'] for r in mc).items())))
"
```

- MC answer letters should be roughly uniform (no letter position bias).
- No `format: advisory` rows should remain in `sft.jsonl` — they all live
  in `advisory_questions.jsonl`.
- Every `train.jsonl` row should have `messages` (chat form) when rendered
  with `--raw-frac 0`.

---

## File map

| File | Contents |
|---|---|
| `agripadi_v3.jsonl` | Accepted stage-1 rows (`mc` / `short_answer` / `advisory`) |
| `agripadi_v3_flagged.jsonl` | Possible source-voice leakage (review; do not publish) |
| `agripadi_v3_rejected.jsonl` | Shape / verifier / length-bias rejections |
| `agripadi_v3_retry.jsonl` | Rows from replayed failed batches |
| `agripadi_v3_retry_flagged.jsonl` | Flagged rows from the retry pass |
| `agripadi_v3_retry_rejected.jsonl` | Rejected rows from the retry pass |
| `agripadi_v3_additions.jsonl` | Regen additions (append to main after review) |
| `agripadi_v3_main_updates.jsonl` | Re-verification patches for the main file |
| `agripadi_v3_regen_failed.jsonl` | Regens that still failed after feedback |
| `hausa_supplement.jsonl` | Hausa-translated supplement (ids suffixed `-ha`) |
| `sft.jsonl` | Rendered MC + short-answer SFT examples |
| `advisory_questions.jsonl` | Advisory questions (stage-2 input) |
| `advisory_sft.jsonl` | Advisory question → answer pairs |
| `advisory_rejected.jsonl` | Stage-2 rejects |
| `train.jsonl` | `sft.jsonl` + `advisory_sft.jsonl` (final local set) |

Generated data artifacts are gitignored — this repository is the
pipeline source. The published chat-format dataset lives on Hugging Face
(see the dataset card, which links back here).
