# AgriPadi Synth Pipelines

**Repo:** https://github.com/ThatLinuxGuyYouKnow/agripadi_synth_pipeline

Synthetic data generation pipelines for **AgriPadi**, an agriculture advisory
dataset for West Africa (English, Nigerian Pidgin, Hausa). These scripts
produce the structured rows, quality-gate them, and render them into
chat-SFT examples.

This repository contains **only the data pipelines** — no training, no eval,
no model export. The published dataset lives on Hugging Face; this repo is
the reproducible source that produced it.

## Pipeline overview

```
excerpts.json  (seed corpus: 132 sourced crop/pest/policy excerpts)
      │
      ▼
┌─────────────────────────────  stage 1  ─────────────────────────────┐
│ question_gen.py                                                     │
│  • excerpt track (grounded in excerpts.json) + grid track (breadth) │
│  • emits rows: mc / short_answer / advisory                         │
│  • gates: dedup → banned actives → length-bias → source-voice       │
│           → LLM verifier                                            │
│  → agripadi_v3.jsonl (+ _flagged, + _rejected)                      │
└──────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────  stage 1b  ─────────────────────────────┐
│ render_sft.py                                                       │
│  • mc → 2 variants (mc_letter + mc_arc), raw-text and chat surfaces │
│  • short_answer → 1 variant                                         │
│  • advisory rows routed out untouched                               │
│  → sft.jsonl  +  advisory_questions.jsonl                           │
└──────────────────────────────────────────────────────────────────────┘
      │                                │
      │                                ▼
      │                 ┌─────────────  stage 2  ─────────────┐
      │                 │ answer_gen.py                        │
      │                 │  • behavior-conditioned answers      │
      │                 │    (answer_directly/clarify/reassure)│
      │                 │  • gates: 40–160-word window →       │
      │                 │           LLM verifier → 1 retry     │
      │                 │  → advisory_sft.jsonl (+ _rejected)  │
      │                 └──────────────────────────────────────┘
      ▼                                ▼
            cat sft.jsonl advisory_sft.jsonl > train.jsonl
```

Repair passes (run as needed between stages):

| Script | Purpose |
|---|---|
| `retry_failed_batches.py` | Replays stage-1 batches that failed on transient API errors (batch indices are deterministic and known). |
| `regenerate_rejected.py` | Regenerates gate-rejected rows with the rejection reason fed back as feedback, then re-verifies anything lacking a verdict. Writes additions/updates — never overwrites the main file. |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then paste your DEEPSEEK_API_KEY
```

`question_gen.py` loads `.env` via `python-dotenv` at import time (a shell
`export` still overrides it). Scripts exit immediately with a clear message
if the key is missing.

Generation uses DeepSeek's OpenAI-compatible API:

```bash
--provider deepseek --model deepseek-v4-flash
```

(`deepseek-chat` still works if `deepseek-v4-flash` 404s — drop the `--model`
flag.) Thinking mode is disabled on every call; on v4-flash, reasoning
tokens otherwise consume `max_tokens` and return empty content.

**Cost/time for a full run:** ~40–60 min, well under $2 (most system-prompt
input hits the cache-hit rate).

## Quickstart

Smoke test first (~1 min, validates model name + batch parsing):

```bash
python3 -c "import json; json.dump(json.load(open('excerpts.json'))[:3], open('/tmp/smoke_excerpts.json','w'))"
python3 question_gen.py --excerpts /tmp/smoke_excerpts.json --grid-items 0 \
    --out /tmp/smoke_test.jsonl --provider deepseek --model deepseek-v4-flash
```

Full run — exact commands, expected counts, and sanity checks are in the
runbook: **[run_this.md](run_this.md)**.

```bash
# 1. generate questions
python3 question_gen.py --excerpts excerpts.json --grid-items 3000 \
    --out agripadi_v3.jsonl \
    --flagged-out agripadi_v3_flagged.jsonl \
    --rejected-out agripadi_v3_rejected.jsonl \
    --provider deepseek --model deepseek-v4-flash

# 2. render SFT
python3 render_sft.py --in agripadi_v3.jsonl \
    --out sft.jsonl --advisory-out advisory_questions.jsonl

# 3. answer advisory questions
python3 answer_gen.py --in advisory_questions.jsonl --excerpts excerpts.json \
    --out advisory_sft.jsonl \
    --provider deepseek --model deepseek-v4-flash

# 4. combine
cat sft.jsonl advisory_sft.jsonl > train.jsonl
```

## Quality gates

Rows must clear every gate to reach `train.jsonl`. Design rationale and the
length-bias investigation are in [notes.md](notes.md).

**Deterministic (no API cost):**

- **Length-bias gate** (`validate_mc`) — rejects MC items where the correct
  answer is *both* uniquely the longest *and* ≥1.25× the distractor mean
  word count. Without this, models learn "pick the longest option" instead
  of agronomy — worst case under `acc_norm`.
- **Banned actives** (`contains_banned_active`) — blocks inappropriate
  agrochemical recommendations.
- **Dedup** (`dedup_rows`) — near-duplicate question collapse across tracks.
- **Source-voice flag** (`looks_like_document_voice`) — routes rows that
  read like pasted source text to `*_flagged.jsonl` for review.
- **Answer-letter balance** — MC choices are shuffled deterministically
  per-item (seeded by id), so letter position is unbiased and reproducible.
- **Word-count window** (stage 2) — advisory answers must land in
  40–160 words so they survive greedy decoding with a 256-token budget.

**LLM verifier** — every row gets an independent verification pass
(correct answer, single correct choice, no fabrication, stylistic
length-bias tell). Failures go to `*_rejected.jsonl` with a note;
`regenerate_rejected.py` feeds that note back as regeneration feedback.

### Watch thresholds (from the runbook)

- `length_bias:` rate in the rejected file — gate is working; if >25%,
  escalate to generate-N-and-drop-one balancing.
- Flagged rate >5% — skim `*_flagged.jsonl` before publishing.
- MC answer letters roughly uniform in the final set.

## Output artifacts

Generated files are **gitignored** — this repo is the pipeline, the dataset
is published separately.

| File | Contents |
|---|---|
| `agripadi_v3.jsonl` | Accepted stage-1 rows (`mc` / `short_answer` / `advisory`) |
| `agripadi_v3_flagged.jsonl` | Possible source-voice leakage (review; not published) |
| `agripadi_v3_rejected.jsonl` | Shape / verifier / length-bias rejections |
| `sft.jsonl` | Rendered MC + short-answer SFT examples |
| `advisory_questions.jsonl` | Advisory questions (stage-2 input) |
| `advisory_sft.jsonl` | Advisory question → answer pairs |
| `train.jsonl` | `sft.jsonl` + `advisory_sft.jsonl` — the production set |

## Repository layout

```
question_gen.py            stage 1 — question generation + gates + verifier
render_sft.py              stage 1b — render structured rows to SFT examples
answer_gen.py              stage 2 — advisory answers + gates + verifier
retry_failed_batches.py    repair — replay transiently-failed API batches
regenerate_rejected.py     repair — regen rejects with feedback, re-verify
main.py                    legacy v1 generator (standalone, pre-v3)
excerpts.json              seed corpus (132 sourced excerpts)
run_this.md                full runbook: commands, counts, sanity checks
notes.md                   quality-gate design notes
test.md                    smoke-test instructions
```

`main.py` is the earlier crop×topic×language generator kept for reference;
the v3 pipeline (`question_gen.py` onward) is what produced the published
dataset.

## Not included here

Training, evaluation, model export (`train_*.py`, `eval_arc_easy.py`,
`export_gguf.py`), dataset upload, and model upload live outside this
repository by design — see the dataset card on Hugging Face for the
trained model and full provenance.
