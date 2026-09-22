# Smoke Test

## Command

Run from `/home/alabi-ayobami/agripadi-synth`:

```bash
set -a && . ./.env && set +a
python3 -u question_gen.py --excerpts excerpts.json --grid-items 0 \
  --skip-verify --out smoke.jsonl --provider deepseek
```

## What it does

- Generates questions from the **excerpt track only** (51 excerpts in `excerpts.json`)
- No grid track (`--grid-items 0`)
- Skips the verifier pass (`--skip-verify`) to save API credits — this is the smoke run
- Uses DeepSeek (`--provider deepseek`), model `deepseek-chat` (default)
- `--skip-verify` means the output will NOT have `verified` flags
- Expects `DEEPSEEK_API_KEY` in the environment (loaded from `.env`)

## Expected output

- ~51 excerpts x ~12 items per excerpt ≈ **~600 items** written to `smoke.jsonl`
- Batching: 3 excerpts per LLM call → **17 batches**
- ~37s per batch → **~11 min total**

## Check after completion

Run this summary to inspect distribution:

```bash
python3 - <<'EOF'
import json
from collections import Counter
rows = [json.loads(l) for l in open("smoke.jsonl")]
print("total:", len(rows))
print("formats:", dict(Counter(r["format"] for r in rows)))
print("subdomains:", dict(Counter(r["subdomain"] for r in rows)))
print("mc letters:", dict(Counter(chr(65+r["answer_index"]) for r in rows if r["format"]=="mc")))
EOF
```
