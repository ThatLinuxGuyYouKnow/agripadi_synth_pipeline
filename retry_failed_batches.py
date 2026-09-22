#!/usr/bin/env python3
"""Retry the stage-1 generation batches that failed with transient API errors.

Which batches failed is known exactly (reconstructed and cross-checked against
the run's outputs -- see regenerate_rejected.py's header for the same story):

- Excerpt batches are POSITIONAL chunks of excerpts.json (3 per call), so
  batch N = excerpts[(N-1)*3 : N*3]. Failed: batches 9 and 28
  -> excerpts 24-26 and 81-83 (0-indexed).
- Grid cells come from build_grid_cells(3000, 6), which is deterministic when
  the RNG state is replayed: seed(42), then build all 132 excerpt plans in
  file order (exactly what the real run did before the grid track), then call
  build_grid_cells. Verified: every accepted/flagged/rejected grid row maps
  to a reconstructed (subdomain, subject, theme) cell, and no row maps into
  the failed batches. Failed: batches 15, 19, 28, 99, 106 (4 cells each).

Plans/cell_ids are regenerated fresh (new uuids) -- only the excerpt contents
and the cells' (subdomain, subject, theme) triples need to match the original
run, and those are what the reconstruction guarantees.

Rows written here are NOT verifier-checked; regenerate_rejected.py runs one
verifier pass over everything outstanding (these rows + repaired rejects +
the main file's "verifier unavailable" rows). Doc-voice flagging and dedup
against the existing dataset ARE applied, same as the main pipeline.

Usage:
    python3 retry_failed_batches.py --provider deepseek --model deepseek-v4-flash
"""

import argparse
import json
import random
import time
import uuid
from collections import Counter

import question_gen as qg

# 1-indexed batch numbers from the stage-1 run log (transient empty-content
# failures). Excerpt batches: 3 excerpts each. Grid batches: 4 cells each.
FAILED_EXCERPT_BATCHES = (9, 28)
FAILED_GRID_BATCHES = (15, 19, 28, 99, 106)


def make_client(provider):
    """Provider client with a bounded request timeout: the SDK's default 600s
    turns a network blip into a 10-minute hang per call, and the stage-1 run
    (and the verifier streak) died that way."""
    if provider == "anthropic":
        return qg.get_anthropic_client()
    import openai
    return openai.OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        timeout=90,
        max_retries=3,
    )


def reconstruct_failed_batches(args, excerpts):
    """Replay the run's RNG consumption so build_grid_cells reproduces the
    original cells, then pick out the failed batches by position."""
    random.seed(args.seed)
    for ex in excerpts:  # same draws as the real excerpt track, in order
        qg.build_excerpt_plan(ex, args.items_per_excerpt, args.mc_frac, args.short_frac)
    cells = qg.build_grid_cells(args.grid_items, args.items_per_cell)

    ex_batches = list(qg.chunked(excerpts, args.excerpts_per_call))
    grid_batches = list(qg.chunked(cells, args.cells_per_call))
    return ([ex_batches[n - 1] for n in FAILED_EXCERPT_BATCHES],
            [grid_batches[n - 1] for n in FAILED_GRID_BATCHES])


def gen_excerpt_batches(args, provider, client, model, ex_batches, rejected):
    rows = []
    for ex_batch in ex_batches:
        situations, plans_by_group = [], {}
        for ex in ex_batch:
            plan = qg.build_excerpt_plan(ex, args.items_per_excerpt,
                                         args.mc_frac, args.short_frac)
            plans_by_group[ex["id"]] = {p["id"]: p for p in plan}
            situations.append(qg.EXCERPT_SITUATION_TEMPLATE.format(
                excerpt_id=ex["id"], crop=ex["crop"], topic=ex["topic"],
                excerpt=ex["excerpt_text"], specs=qg.format_specs(plan),
            ))
        try:
            raw = qg.call_with_retry(provider, client, model, qg.EXCERPT_SYSTEM_PROMPT,
                                     "\n\n".join(situations), tries=args.tries)
            results = qg.parse_json_array(raw)
        except Exception as e:
            print(f"  [excerpt retry {[e['id'] for e in ex_batch]}] FAILED AGAIN: {e}")
            continue

        def base(excerpt_id, _ex=ex_batch):
            ex = next(e for e in _ex if e["id"] == excerpt_id)
            return {"track": "excerpt",
                    "subdomain": qg.EXCERPT_TOPIC_TO_SUBDOMAIN.get(ex["topic"], "crops_agronomy"),
                    "subject": ex["crop"], "crop": ex["crop"], "topic": ex["topic"],
                    "source_excerpt_id": ex["id"]}

        got = qg.ingest_items(results, plans_by_group, base, args.seed, rejected)
        rows.extend(got)
        print(f"  [excerpt retry {[e['id'] for e in ex_batch]}] -> {len(got)} items")
        time.sleep(0.2)
    return rows


def gen_grid_batches(args, provider, client, model, grid_batches, rejected):
    rows = []
    for cell_batch in grid_batches:
        for cell in cell_batch:  # fresh ids; only (subdomain, subject, theme) must match the run
            cell["cell_id"] = f"{cell['subdomain']}_{uuid.uuid4().hex[:8]}"
        blocks, plans_by_group = [], {}
        for cell in cell_batch:
            plan = qg.build_cell_plan(cell, args.items_per_cell, args.mc_frac, args.short_frac)
            plans_by_group[cell["cell_id"]] = {p["id"]: p for p in plan}
            blocks.append(qg.GRID_CELL_TEMPLATE.format(
                cell_id=cell["cell_id"], subdomain=cell["subdomain"],
                subject=cell["subject"], theme=cell["theme"], specs=qg.format_specs(plan),
            ))
        label = [(c["subdomain"], c["subject"], c["theme"]) for c in cell_batch]
        try:
            raw = qg.call_with_retry(provider, client, model, qg.GRID_SYSTEM_PROMPT,
                                     "\n\n".join(blocks), tries=args.tries)
            results = qg.parse_json_array(raw)
        except Exception as e:
            print(f"  [grid retry {label}] FAILED AGAIN: {e}")
            continue

        def base(cell_id, _batch=cell_batch):
            cell = next(c for c in _batch if c["cell_id"] == cell_id)
            crop = cell["subject"] if cell["subdomain"] in (
                "crops_agronomy", "pest_disease_weeds") else None
            return {"track": "grid", "subdomain": cell["subdomain"],
                    "subject": cell["subject"], "crop": crop, "topic": cell["theme"],
                    "source_excerpt_id": None}

        got = qg.ingest_items(results, plans_by_group, base, args.seed, rejected)
        rows.extend(got)
        print(f"  [grid retry {cell_batch[0]['subdomain']} x{len(cell_batch)} cells] "
              f"-> {len(got)} items")
        time.sleep(0.2)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excerpts", type=str, default="excerpts.json")
    ap.add_argument("--existing", type=str, nargs="+",
                    default=["agripadi_v3.jsonl", "agripadi_v3_flagged.jsonl"],
                    help="rows already in the dataset (dedup reference)")
    ap.add_argument("--out", type=str, default="agripadi_v3_retry.jsonl")
    ap.add_argument("--flagged-out", type=str, default="agripadi_v3_retry_flagged.jsonl")
    ap.add_argument("--rejected-out", type=str, default="agripadi_v3_retry_rejected.jsonl")
    ap.add_argument("--items-per-excerpt", type=int, default=12)
    ap.add_argument("--excerpts-per-call", type=int, default=3)
    ap.add_argument("--grid-items", type=int, default=3000)
    ap.add_argument("--items-per-cell", type=int, default=6)
    ap.add_argument("--cells-per-call", type=int, default=4)
    ap.add_argument("--mc-frac", type=float, default=qg.DEFAULT_MC_FRAC)
    ap.add_argument("--short-frac", type=float, default=qg.DEFAULT_SHORT_FRAC)
    ap.add_argument("--tries", type=int, default=4,
                    help="API attempts per batch (the original run used 2)")
    ap.add_argument("--provider", choices=["anthropic", "deepseek"], default="deepseek")
    ap.add_argument("--model", type=str, default="deepseek-v4-flash")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model = args.model
    client = make_client(args.provider)

    with open(args.excerpts) as f:
        excerpts = json.load(f)
    excerpt_lookup = {ex["id"]: ex for ex in excerpts}

    ex_batches, grid_batches = reconstruct_failed_batches(args, excerpts)
    print(f"Retrying {len(ex_batches)} excerpt batches "
          f"({sum(len(b) for b in ex_batches)} excerpts) and "
          f"{len(grid_batches)} grid batches ({sum(len(b) for b in grid_batches)} cells)")

    rejected = []
    rows = gen_excerpt_batches(args, args.provider, client, model, ex_batches, rejected)
    rows += gen_grid_batches(args, args.provider, client, model, grid_batches, rejected)

    # Dedup against the existing dataset (existing rows win ties, so any
    # near-dup of an already-accepted question is dropped from the retry set).
    existing = []
    for path in args.existing:
        try:
            with open(path) as f:
                existing.extend(json.loads(l) for l in f)
        except FileNotFoundError:
            print(f"(dedup reference {path} not found, skipping)")
    new_ids = {r["id"] for r in rows}
    kept = qg.dedup_rows(existing + rows)
    rows = [r for r in kept if r["id"] in new_ids]
    print(f"Dedup vs existing: {len(new_ids)} -> {len(rows)} new rows")

    # Document-voice flag (excerpt track only), same gate as the main pipeline.
    accepted, flagged = [], []
    for r in rows:
        ex_text = (excerpt_lookup.get(r["source_excerpt_id"] or "") or {}).get("excerpt_text")
        if ex_text and qg.looks_like_document_voice(r["question"], ex_text):
            r["reason"] = "possible verbatim phrase overlap with source excerpt"
            flagged.append(r)
        else:
            accepted.append(r)

    with open(args.out, "w") as f:
        for r in accepted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.flagged_out, "w") as f:
        for r in flagged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.rejected_out, "w") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(accepted)} unverified rows to {args.out}")
    print(f"Flagged {len(flagged)} -> {args.flagged_out}")
    print(f"Shape-rejected {len(rejected)} -> {args.rejected_out} "
          f"(feed to regenerate_rejected.py)")
    print("Formats: " + ", ".join(f"{k}={v}" for k, v in
          Counter(r["format"] for r in accepted).most_common()))


if __name__ == "__main__":
    main()
