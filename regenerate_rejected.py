#!/usr/bin/env python3
"""Repair rejected stage-1 items, then run ONE verifier pass over everything
that still lacks a verdict.

Three leftover problems from the stage-1 run are handled together:

1. Shape-rejected rows (agripadi_v3_rejected.jsonl + the retry script's
   rejects): 92% are MC length_bias -- the deterministic gate rejected them,
   and the note says exactly why. Each item is REGENERATED with its rejection
   reason fed back (excerpt-track items stay grounded in their source
   excerpt), up to --rounds attempts, with the same gates re-applied locally.
2. retry_failed_batches.py rows (agripadi_v3_retry.jsonl): generated but not
   yet verifier-checked.
3. Main-file rows with verifier_note="verifier unavailable" (the 17-call
   connection-error streak left them kept-but-unverified): re-verified here;
   results go to --main-updates-out so the main file can be patched.

Nothing overwrites the main file: this script writes additions, failures and
main-row updates; merging is a separate explicit step.

Usage:
    python3 regenerate_rejected.py --provider deepseek --model deepseek-v4-flash
"""

import argparse
import json
import os
import time
from collections import Counter

import question_gen as qg

REGEN_SYSTEM = f"""You are repairing rejected items from an agricultural
training-data generation run for smallholder farming in Nigeria and West
Africa. Each spec below shows an item that an automated quality gate rejected
and the EXACT reason. Write one corrected replacement per spec, keeping the
same agricultural intent.

- If the rejection concerned only the answer/options (length bias, wrong
  number of options, a banned active in the correct answer), KEEP the
  previous question text (improve wording only if needed) and rewrite the
  answer and distractors.
- Otherwise write a fresh question on the same subject and theme.
- If a reference excerpt is provided, the corrected question and its correct
  answer must be derivable from it.

{qg._FORMAT_RULES}
{qg.GLOBAL_RULES}
Output shape: [{{"id": "...", "question": "...", "answer": "...",
"distractors": ["...", "...", "..."], "explanation": "one sentence"}}].
"distractors" is required for "mc" specs, omitted for "short_answer" specs.
Each item's "id" must exactly match its spec.
"""


def load_jsonl(path):
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


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


def build_user_block(key, chunk, excerpt_lookup):
    lines = []
    if key[0] == "excerpt":
        ex = excerpt_lookup.get(key[1]) or {}
        lines += ["Reference excerpt (ground corrections in it; do not mimic wording):",
                  "---", ex.get("excerpt_text", ""), "---", ""]
    else:
        lines += [f"Subdomain: {key[1]}", ""]
    lines.append("Item specs to repair (write ONE corrected item per spec):")
    for r in chunk:
        lines.append(f"- id: {r['id']} | format: {r['format']} | "
                     f"subject: {r.get('subject')} | theme: {r.get('topic')}")
        lines.append(f"  previous question: {r.get('question', '')}")
        if r.get("choices"):  # verifier-rejected rows are fully assembled
            marked = next((c for i, c in enumerate(r["choices"])
                           if i == r.get("answer_index")), r.get("answer", ""))
            lines.append(f"  previous marked answer: {marked}")
        lines.append(f"  rejection reason: {r.get('note', '')}")
    return "\n".join(lines)


def assemble_repair(spec, item, args):
    """Validate a repaired item and build a full dataset row from the original
    rejected row. Returns (row, "") or (None, reason)."""
    for k in ("note", "verified", "verifier_note", "choices", "answer_index",
              "answer", "explanation"):
        spec.pop(k, None)
    row = dict(spec)
    q = (item.get("question") or "").strip()
    if not q:
        return None, "empty question"
    row["question"] = q
    if row["format"] == "mc":
        answer = (item.get("answer") or "").strip()
        distractors = item.get("distractors") or []
        ok, reason = qg.validate_mc(answer, distractors)
        if not ok:
            return None, f"invalid_mc: {reason}"
        choices, idx = qg.assemble_choices(answer, distractors, args.seed, row["id"])
        row.update({"choices": choices, "answer_index": idx, "answer": choices[idx],
                    "explanation": (item.get("explanation") or "").strip()})
    else:  # short_answer
        answer = (item.get("answer") or "").strip()
        if not answer:
            return None, "invalid_short_answer: empty answer"
        row.update({"answer": answer,
                    "explanation": (item.get("explanation") or "").strip()})
    # Shape-rejected rows never got question_type/voice; backfill so repaired
    # rows carry the same fields as main-file rows.
    if "question_type" not in row:
        allowed = qg.SUBDOMAINS.get(row.get("subdomain"), {}).get("qtypes", qg.ALL_QTYPES)
        row["question_type"] = qg.sample_qtype(allowed)
    if "voice" not in row:
        row["voice"] = qg.weighted_choice(qg.VOICES_WEIGHTED)
    return row, ""


def repair_round(pending, args, provider, client, model, excerpt_lookup):
    """One repair attempt over all pending rows. Returns (ok_rows, still)."""
    groups = {}
    for r in pending:
        if r.get("track") == "excerpt" and r.get("source_excerpt_id"):
            key = ("excerpt", r["source_excerpt_id"])
        else:
            key = ("grid", r.get("subdomain") or "crops_agronomy")
        groups.setdefault(key, []).append(r)

    ok_rows, still = [], []
    n_calls = sum(1 for rs in groups.values() for _ in qg.chunked(rs, args.batch_size))
    call_no = 0
    for key, rs in groups.items():
        for chunk in qg.chunked(rs, args.batch_size):
            call_no += 1
            try:
                raw = qg.call_with_retry(provider, client, model, REGEN_SYSTEM,
                                         build_user_block(key, chunk, excerpt_lookup),
                                         tries=args.tries)
                results = {it.get("id"): it for it in qg.parse_json_array(raw)
                           if isinstance(it, dict)}
            except Exception as e:
                print(f"  [repair call {call_no}/{n_calls}] FAILED: {e} "
                      f"({len(chunk)} items carry over)")
                for r in chunk:
                    r["note"] = f"repair call failed: {e}"
                    still.append(r)
                continue
            for r in chunk:
                item = results.get(r["id"])
                if item is None:
                    r["note"] = "repair call did not return this id"
                    still.append(r)
                    continue
                row, reason = assemble_repair(r, item, args)
                if row is None:
                    r["note"] = reason
                    still.append(r)
                else:
                    ok_rows.append(row)
            time.sleep(0.2)
    return ok_rows, still


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rejected", type=str, nargs="+",
                    default=["agripadi_v3_rejected.jsonl", "agripadi_v3_retry_rejected.jsonl"])
    ap.add_argument("--retry", type=str, default="agripadi_v3_retry.jsonl",
                    help="unverified rows from retry_failed_batches.py")
    ap.add_argument("--main", type=str, default="agripadi_v3.jsonl")
    ap.add_argument("--excerpts", type=str, default="excerpts.json")
    ap.add_argument("--out", type=str, default="agripadi_v3_additions.jsonl")
    ap.add_argument("--failed-out", type=str, default="agripadi_v3_regen_failed.jsonl")
    ap.add_argument("--main-updates-out", type=str, default="agripadi_v3_main_updates.jsonl")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--verify-batch-size", type=int, default=40)
    ap.add_argument("--no-verify-main-unverified", dest="verify_main_unverified",
                    action="store_false")
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--provider", choices=["anthropic", "deepseek"], default="deepseek")
    ap.add_argument("--model", type=str, default="deepseek-v4-flash")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model = args.model
    client = make_client(args.provider)

    with open(args.excerpts) as f:
        excerpt_lookup = {ex["id"]: ex for ex in json.load(f)}
    main_rows = load_jsonl(args.main)
    retry_rows = load_jsonl(args.retry)
    rejected = []
    for path in args.rejected:
        rejected.extend(load_jsonl(path))

    pending = [r for r in rejected if r.get("format") in ("mc", "short_answer")]
    skipped = [r for r in rejected if r.get("format") not in ("mc", "short_answer")]
    if skipped:
        print(f"Skipping {len(skipped)} non-mc/short_answer rejected rows")
    print(f"Repairing {len(pending)} rejected rows "
          f"({Counter(r['format'] for r in pending)})")

    repaired = []
    for round_no in range(1, args.rounds + 1):
        if not pending:
            break
        print(f"-- repair round {round_no}: {len(pending)} items --")
        ok_rows, pending = repair_round(pending, args, args.provider, client,
                                        model, excerpt_lookup)
        repaired.extend(ok_rows)
        print(f"   round {round_no}: {len(ok_rows)} repaired, {len(pending)} still failing")
    failed = pending

    # Dedup repaired rows against everything already in the dataset
    # (existing rows win ties; only surviving repaired rows are kept).
    rep_ids = {r["id"] for r in repaired}
    survivors = [r for r in qg.dedup_rows(main_rows + retry_rows + repaired)
                 if r["id"] in rep_ids]
    print(f"Dedup: {len(repaired)} repaired -> {len(survivors)} after dedup vs dataset")

    # One verifier pass over everything outstanding.
    to_verify = survivors + retry_rows
    main_unverified = []
    if args.verify_main_unverified:
        main_unverified = [r for r in main_rows
                           if r.get("verified") is None
                           and r.get("verifier_note") == "verifier unavailable"]
        to_verify += main_unverified
        print(f"Including {len(main_unverified)} unverified main-file rows")
    qg.verify_rows(to_verify, argparse.Namespace(verify_batch_size=args.verify_batch_size),
                   args.provider, client, model)

    additions, verif_failed = [], []
    for r in survivors + retry_rows:
        if r.get("verified") is False:
            r["note"] = f"verifier: {r.get('verifier_note', '')}"
            verif_failed.append(r)
        else:
            additions.append(r)

    with open(args.out, "w") as f:
        for r in additions:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.failed_out, "w") as f:
        for r in failed + verif_failed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.main_updates_out, "w") as f:
        for r in main_unverified:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(additions)} verified/kept rows to {args.out}")
    print(f"Still failing -> {args.failed_out}: {len(failed)} unrepairable "
          f"+ {len(verif_failed)} verifier-rejected")
    print(f"Main-row updates -> {args.main_updates_out}: {len(main_unverified)} "
          f"(apply verified=True updates to {args.main}; drop verified=False ones)")
    print("Additions formats: " + ", ".join(f"{k}={v}" for k, v in
          Counter(r["format"] for r in additions).most_common()))


if __name__ == "__main__":
    main()
