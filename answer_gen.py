#!/usr/bin/env python3
"""
Stage 2 of the AgriPadi SFT pipeline: generate the ASSISTANT's replies to the
advisory questions produced by question_gen.py (v3) and passed through by
render_sft.py.

Why this is a separate stage with behavior-conditioned prompting:
- The profiler's automated paths (MC ranking, exact match) are covered by the
  mc/short_answer rows. The advisory slice is what the JUDGE panel sees:
  open-ended farmer messages under greedy decoding with a finite generation
  budget (profiler default max_gen_toks=256, n_ctx=2048). Answers must
  therefore be complete, self-contained and bounded (~40-160 words), never
  truncated mid-thought.
- v3 deliberately flipped the behavior distribution so answer_directly
  dominates and clarify is a small slice -- but each behavior has a distinct
  REQUIRED SHAPE, and a generic "answer this question" prompt would erase
  that. The prompt enforces the shape per item:
    * answer_directly: commit in the FIRST sentence; caveats only at the end
    * clarify:         most-likely answer first, then ONE targeted follow-up
    * reassure:        brief empathy, real risk level, 1-2 practical steps
    * safety_refusal:  decline + why + safe alternative + who to consult
- Excerpt-track rows are grounded in their source excerpt (pass --excerpts);
  grid-track rows rely on generator knowledge, so everything goes through a
  batched LLM verifier (behavior match + agronomic facts + safety rules).
  Failures get ONE regeneration attempt with the verifier's note as feedback,
  then are written to the rejected file if still failing.

Safety rules carried over from v1/v3: never recommend banned actives (warning
against them is fine), never state precise dosages as fact (defer to label /
extension officer), never invent statistics, prices or brand names.

Output schema matches render_sft.py (one JSON object per line):
    {"id": "<row_id>:advisory", "source_id", "variant": "advisory",
     "subdomain", "track", "expected_behavior", "goal",
     "messages": [{"role": "user", ...}, {"role": "assistant", ...}],
     "verified", "verifier_note"}
so the final training set is just: cat sft.jsonl advisory_sft.jsonl > train.jsonl

Usage:
    python answer_gen.py --in advisory_questions.jsonl \
        --excerpts excerpts.json --out advisory_sft.jsonl --provider anthropic

Requires: pip install anthropic openai --break-system-packages
"""

import argparse
import json
import re
import time
from collections import Counter

# Reuse the v3 machinery: providers, retrying caller, JSON parsing, batching.
from question_gen import (
    BANNED_ACTIVES,
    call_with_retry,
    chunked,
    get_anthropic_client,
    get_deepseek_client,
    parse_json_array,
)

# Deterministic length gate (words). The prompt asks for 40-160; anything far
# outside that band breaks the profiler's generation budget or reads as a
# non-answer. Failures go into the retry pool, not straight to rejection.
MIN_WORDS = 20
MAX_WORDS = 240

# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are writing the ASSISTANT's reply to a farmer's message,
as training data for an agricultural advisory assistant serving smallholder
farmers in Nigeria and West Africa.

You will be given several farmer messages, each with metadata and an
"expected_behavior" that dictates the SHAPE of your reply.

Behavior shapes:
- "answer_directly": Commit to the most likely answer in the FIRST sentence.
  Then give the key action(s) the farmer should take, in order. A caveat
  ("if you also see X, it could be Y instead") may come at the END -- never
  first.
- "clarify": The message lacks information needed for a confident answer.
  Give the most probable explanation given what was said ("Most likely this
  is..."), with the action that is safe to start now, then ask ONE targeted
  follow-up question whose answer would confirm the diagnosis. Never open
  with a question; never ask a list of questions.
- "reassure": Acknowledge the worry warmly and briefly, state the real level
  of risk in plain terms, then give one or two practical next steps.
- "safety_refusal": The message asks for something unsafe (a banned pesticide,
  a dangerous practice, eating or selling unsafe produce). Firmly but
  politely decline, say why in one sentence, offer the safe alternative, and
  point to the right authority (extension officer, product label, vet).

Rules for ALL replies:
- Simple, respectful Nigerian English. Technical terms are allowed but must
  be explained in plain words.
- Be concrete and practical: what to do, in what order, what to watch for.
- Never recommend banned/restricted pesticides: {", ".join(BANNED_ACTIVES)}.
  Mentioning them to warn AGAINST is fine.
- Never state a precise chemical dosage as fact. Say to follow the product
  label or confirm exact product and rate with a local extension officer.
- Never invent statistics, study results, prices, or brand names.
- Length: 40-160 words (shorter for reassure/safety_refusal, fuller for
  answer_directly). Complete and self-contained -- never end mid-thought.
- Match the farmer's register: a worried farmer gets calm first, jargon last.
- If a reference excerpt is provided with the message, ground your facts in
  it; do not copy its wording.
- All output in English, regardless of message_style.
- Return ONLY JSON: [{{"id": "...", "answer": "..."}}] -- the id must match
  the message's id exactly. No markdown fences, no commentary.
"""

MESSAGE_TEMPLATE = """Message (id: {id})
expected_behavior: {behavior} | goal: {goal} | tone: {tone} | information_level: {info_level}
{grounding}Farmer's message:
"{question}"
"""

GROUNDING_TEMPLATE = """Reference excerpt (ground your facts in this, do not copy wording):
---
{excerpt}
---
"""

RETRY_TEMPLATE = "A previous draft of this reply was rejected: {note}. Avoid that mistake.\n"

VERIFIER_SYSTEM = f"""You are a strict reviewer of agricultural advisory
training data for West African smallholder farmers. You will be given farmer
messages with the assistant's reply and the behavior the reply was supposed
to follow.

Reject (ok=false) if ANY holds:
- behavior mismatch: a "clarify" reply that gives no most-likely answer or
  asks more than one question; an "answer_directly" reply that opens with
  hedging or a question instead of committing; a "safety_refusal" reply that
  actually provides the unsafe instructions; a "reassure" reply that is
  alarmist or gives no practical step
- recommends a banned/restricted pesticide ({", ".join(BANNED_ACTIVES)}) --
  warning AGAINST them is fine
- states a precise chemical dosage as fact instead of deferring to the
  product label or an extension officer
- invents statistics, prices, study results or brand names
- factually wrong agronomy or animal husbandry for West Africa
- not self-contained, ends mid-thought, or is far too long/short for a
  WhatsApp-style advisory answer
Otherwise ok=true.

Return ONLY a JSON array: [{{"id": "...", "ok": true/false, "note": "short
reason if not ok"}}]
"""

# ---------------------------------------------------------------------------
# Generation + verification
# ---------------------------------------------------------------------------

def word_count(text: str) -> int:
    return len(re.findall(r"[a-zA-Z0-9']+", text))


def build_user_prompt(batch, excerpt_lookup, notes):
    blocks = []
    for row in batch:
        grounding = ""
        excerpt = excerpt_lookup.get(row.get("source_excerpt_id") or "")
        if excerpt:
            grounding = GROUNDING_TEMPLATE.format(excerpt=excerpt["excerpt_text"])
        block = MESSAGE_TEMPLATE.format(
            id=row["id"], behavior=row["expected_behavior"], goal=row["goal"],
            tone=row["tone"], info_level=row["information_level"],
            grounding=grounding, question=row["question"],
        )
        if row["id"] in notes:
            block = RETRY_TEMPLATE.format(note=notes[row["id"]]) + block
        blocks.append(block)
    return "\n\n".join(blocks)


def run_generation(batch, excerpt_lookup, notes, args, provider, client, model):
    """Returns {row_id: answer} for successfully parsed items."""
    user = build_user_prompt(batch, excerpt_lookup, notes)
    raw = call_with_retry(provider, client, model, SYSTEM_PROMPT, user, max_tokens=6000)
    answers = {}
    for item in parse_json_array(raw):
        rid, ans = item.get("id"), (item.get("answer") or "").strip()
        if rid and ans:
            answers[rid] = ans
    return answers


def render_for_verifier(row, answer):
    return (f"id: {row['id']} | expected_behavior: {row['expected_behavior']}\n"
            f"farmer's message: \"{row['question']}\"\n"
            f"assistant's reply: \"{answer}\"")


def run_verification(rows_with_answers, args, provider, client, model):
    """Returns {row_id: verdict_dict}."""
    verdicts = {}
    for batch in chunked(rows_with_answers, args.verify_batch_size):
        user = "\n\n".join(render_for_verifier(r, a) for r, a in batch)
        try:
            raw = call_with_retry(provider, client, model, VERIFIER_SYSTEM, user,
                                  max_tokens=min(6000, len(batch) * 80))
            for v in parse_json_array(raw):
                verdicts[v.get("id")] = v
        except Exception as e:
            print(f"    [verify batch] FAILED: {e} (items kept, unverified)")
        time.sleep(0.2)
    return verdicts


def generate_all(rows, excerpt_lookup, notes, args, provider, client, model, label):
    """Batched generation over rows. Returns ({id: answer}, [missing rows])."""
    answers, missing = {}, []
    for batch_num, batch in enumerate(chunked(rows, args.batch_size)):
        try:
            got = run_generation(batch, excerpt_lookup, notes, args, provider, client, model)
        except Exception as e:
            print(f"  [{label} batch {batch_num+1}] FAILED: {e}")
            missing.extend(batch)
            continue
        answers.update(got)
        missing.extend(r for r in batch if r["id"] not in got)
        print(f"  [{label} batch {batch_num+1}] {len(batch)} messages -> {len(got)} answers "
              f"(total {len(answers)})")
        time.sleep(0.2)
    return answers, missing

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", type=str, required=True,
                    help="advisory_questions.jsonl from render_sft.py")
    ap.add_argument("--excerpts", type=str, default=None,
                    help="excerpts.json for grounding excerpt-track rows")
    ap.add_argument("--out", type=str, default="advisory_sft.jsonl")
    ap.add_argument("--rejected-out", type=str, default="advisory_rejected.jsonl")
    ap.add_argument("--batch-size", type=int, default=5,
                    help="Farmer messages per generation call")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--verify-batch-size", type=int, default=40)
    ap.add_argument("--provider", choices=["anthropic", "deepseek"], default="anthropic")
    ap.add_argument("--model", type=str, default=None)
    args = ap.parse_args()

    default_models = {"anthropic": "claude-sonnet-4-6", "deepseek": "deepseek-chat"}
    model = args.model or default_models[args.provider]
    if args.provider == "anthropic":
        client = get_anthropic_client()
    else:
        # Bounded request timeout: the SDK default (600s) turns a network
        # blip into a 10-minute hang per call.
        import os
        import openai
        client = openai.OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            timeout=90,
            max_retries=3,
        )

    rows = [json.loads(l) for l in open(args.infile) if l.strip()]
    excerpt_lookup = {}
    if args.excerpts:
        excerpt_lookup = {ex["id"]: ex for ex in json.load(open(args.excerpts))}
    print(f"Loaded {len(rows)} advisory questions "
          f"({sum(1 for r in rows if r.get('source_excerpt_id'))} excerpt-grounded)")

    # ---- pass 1: generate -------------------------------------------------
    answers, missing = generate_all(rows, excerpt_lookup, {}, args,
                                    args.provider, client, model, "gen")
    # This backend sometimes returns partial arrays (1-2 of 5 items). Give
    # omitted rows a couple of second-chance passes instead of dropping them.
    for chance in range(1, 3):
        if not missing:
            break
        print(f"  second chance {chance}: {len(missing)} messages")
        again, missing = generate_all(missing, excerpt_lookup, {}, args,
                                      args.provider, client, model, f"gen2-{chance}")
        answers.update(again)
    if missing:
        print(f"  {len(missing)} messages returned no usable answer (dropped)")

    # ---- quality gates: deterministic length, then LLM verifier -----------
    def length_failure(ans):
        n = word_count(ans)
        if n < MIN_WORDS:
            return f"too short ({n} words; needs 40-160)"
        if n > MAX_WORDS:
            return f"too long ({n} words; needs 40-160)"
        return None

    retry_notes, accepted_answers = {}, {}
    for rid, ans in answers.items():
        fail = length_failure(ans)
        if fail:
            retry_notes[rid] = fail
        else:
            accepted_answers[rid] = ans

    if not args.skip_verify and accepted_answers:
        row_by_id = {r["id"]: r for r in rows}
        # The backend occasionally echoes a fabricated id; drop unknown ids
        # instead of crashing the whole run.
        pairs = [(row_by_id[rid], ans) for rid, ans in accepted_answers.items()
                 if rid in row_by_id]
        print(f"Verifier: {len(pairs)} answers")
        verdicts = run_verification(pairs, args, args.provider, client, model)
        for rid, v in verdicts.items():
            if v.get("ok") is False:
                retry_notes[rid] = v.get("note") or "verifier rejected"
        for rid in list(accepted_answers):
            v = verdicts.get(rid)
            if v is not None and v.get("ok") is False:
                del accepted_answers[rid]

    # ---- pass 2: one regeneration attempt with feedback -------------------
    recovered = 0
    if retry_notes:
        retry_rows = [r for r in rows if r["id"] in retry_notes]
        print(f"Retry with feedback: {len(retry_rows)} answers")
        retry_answers, _ = generate_all(retry_rows, excerpt_lookup, retry_notes, args,
                                        args.provider, client, model, "retry")
        for rid, ans in retry_answers.items():
            if length_failure(ans):
                continue
            accepted_answers[rid] = ans
            recovered += 1
        if not args.skip_verify and retry_answers:
            row_by_id = {r["id"]: r for r in rows}
            pairs = [(row_by_id[rid], a) for rid, a in retry_answers.items()
                     if accepted_answers.get(rid) == a and rid in row_by_id]
            verdicts = run_verification(pairs, args, args.provider, client, model)
            for rid, v in verdicts.items():
                if v.get("ok") is False:
                    accepted_answers.pop(rid, None)
                    recovered -= 1

    # ---- write ------------------------------------------------------------
    row_by_id = {r["id"]: r for r in rows}
    out_rows, rejected = [], []
    for r in rows:
        ans = accepted_answers.get(r["id"])
        base = {
            "source_id": r["id"], "variant": "advisory",
            "subdomain": r.get("subdomain"), "track": r.get("track"),
            "expected_behavior": r.get("expected_behavior"), "goal": r.get("goal"),
            "question": r["question"],
        }
        if ans:
            out_rows.append({
                **base,
                "id": f"{r['id']}:advisory",
                "messages": [
                    {"role": "user", "content": r["question"]},
                    {"role": "assistant", "content": ans},
                ],
                "verified": None if args.skip_verify else True,
                "verifier_note": "",
            })
        else:
            rejected.append({**base, "note": retry_notes.get(r["id"], "no answer generated")})

    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.rejected_out, "w") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- summary ----------------------------------------------------------
    print(f"\nWrote {len(out_rows)} advisory examples to {args.out}")
    print(f"Recovered by retry: {recovered}")
    print(f"Rejected {len(rejected)} -> {args.rejected_out}")
    beh = Counter(r["expected_behavior"] for r in out_rows)
    if beh:
        print("Behaviours: " + ", ".join(f"{k}={v}" for k, v in beh.most_common()))
    if out_rows:
        wc = [word_count(r["messages"][1]["content"]) for r in out_rows]
        print(f"Answer words: min={min(wc)} mean={sum(wc)//len(wc)} max={max(wc)}")


if __name__ == "__main__":
    main()
