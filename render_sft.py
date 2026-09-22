#!/usr/bin/env python3
"""
Stage 1b of the AgriPadi SFT pipeline: render the structured rows emitted by
question_gen.py (v3) into chat-SFT `messages` pairs.

Why this exists (profiler alignment):
- The ADTC profiler scores the quantized model with lm-evaluation-harness via
  llama-cpp-python. Multiple-choice tasks are scored by loglikelihood of a
  continuation, and we do NOT know how the judges' task YAML serializes
  choices. The two canonical serializations are:
    * MMLU-style: context lists lettered choices, continuation is the LETTER
      (" B").
    * ARC-style:   context is the bare question, continuation is the choice
      TEXT (" Striga").
  Training only one surface form risks dumping probability mass on the wrong
  one at eval time, so every MC row is rendered BOTH ways (format
  augmentation -- content dedup already happened in question_gen.py, do not
  re-dedup after rendering):
    * "mc_letter": question + lettered choices -> "B. Striga" (letter-first,
      so P(" B" | ctx) is trained directly; a fraction also appends the
      explanation so the model can justify itself to human judges).
    * "mc_arc":    bare question -> "Striga" (trains P(answer text | question)).
- Short-answer tasks are scored by exact match under greedy decoding, so
  short_answer rows render as question -> canonical answer ONLY. Never append
  explanations here -- trailing text breaks exact match.
- Advisory rows carry no answers yet (they are stage-2 input); they are
  written verbatim to --advisory-out, not rendered.

No system message is emitted by default: lm-eval presents eval contexts with
no system prompt, and training with one then evaluating without shifts the
distribution. Pass --system only if you know what you are doing (it applies
to chat-surface rows only).

Surface mix (--raw-frac, default 0.5): the profiler's lm-eval adapter
(adtc-profiler accuracy.py) tokenizes context+continuation with
add_bos=True, special=False -- no chat template is ever applied and chat
role tokens can never appear at eval time. A model trained ONLY on
chat-templated text is therefore scored on a surface it never saw in
training. Each rendered example is emitted as EITHER a chat `messages` row
OR a raw completion `text` row (seeded coin flip per row, so re-runs are
reproducible). Raw rows mirror the lm-eval serializations:
  mc_letter:    "{question}\n\nA. ..\nB. ..\n\nAnswer: B. Striga"
  mc_arc/sa:    "{question}\n\nAnswer: Striga"
Trainers must NOT apply a chat template to `text` rows (append EOS instead).

Usage:
    python render_sft.py --in agripadi_v3.jsonl --out sft.jsonl \
        --advisory-out advisory_questions.jsonl

Output schema (one JSON object per line):
    {"id", "source_id", "variant", "surface", "subdomain", "track",
     ... and EITHER "messages" (surface="chat": user/assistant pairs,
     system only if --system) OR "text" (surface="raw": plain completion)}
Any trainer (Unsloth / axolotl / TRL) can apply tiny-aya-earth's chat
template to `messages` rows directly; `text` rows train as-is.
"""

import argparse
import json
import random
from collections import Counter

LETTERS = "ABCD"

MC_LETTER_PROMPT = "{question}\n\n{choices}\n\nAnswer:"


def render_mc(row, explain: bool):
    """Render one MC row into its two variants. Returns list of rendered dicts."""
    choices = row["choices"]
    idx = row["answer_index"]
    letter = LETTERS[idx]
    answer = row["answer"]
    explanation = (row.get("explanation") or "").strip()

    lettered = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    letter_completion = f"{letter}. {answer}"
    if explain and explanation:
        letter_completion += f"\n{explanation}"

    return [
        {
            "variant": "mc_letter",
            "user": MC_LETTER_PROMPT.format(question=row["question"], choices=lettered),
            "assistant": letter_completion,
        },
        {
            "variant": "mc_arc",
            "user": row["question"],
            "assistant": answer,
        },
    ]


def render_short_answer(row):
    # Exact-match aligned: canonical answer only, never an explanation.
    return [{
        "variant": "short_answer",
        "user": row["question"],
        "assistant": row["answer"],
    }]


def render_raw(variant, user, assistant):
    """Render one example the way lm-eval presents it to the profiler: raw
    context + continuation text, no chat template, no system prompt."""
    if variant == "mc_letter":
        return f"{user} {assistant}"         # user prompt already ends "Answer:"
    return f"{user}\n\nAnswer: {assistant}"  # ARC-style: question -> "Answer: X"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", type=str, required=True,
                    help="agripadi_v3.jsonl from question_gen.py")
    ap.add_argument("--out", type=str, default="sft.jsonl")
    ap.add_argument("--advisory-out", type=str, default="advisory_questions.jsonl",
                    help="Advisory rows pass through here for the stage-2 answerer")
    ap.add_argument("--mc-variants", choices=["both", "letter", "arc"], default="both")
    ap.add_argument("--explain-frac", type=float, default=0.2,
                    help="Fraction of mc_letter rows that append the explanation")
    ap.add_argument("--raw-frac", type=float, default=0.5,
                    help="Fraction of rendered examples emitted as raw completion "
                         "text (no chat messages) matching the profiler's "
                         "template-free lm-eval surface")
    ap.add_argument("--system", type=str, default=None,
                    help="Optional system message prepended to every example "
                         "(default none: lm-eval contexts have no system prompt)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_rows, advisory_rows, skipped = [], [], 0
    with open(args.infile) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            if row.get("format") == "advisory":
                advisory_rows.append(row)
                continue

            if row.get("verified") is False:
                skipped += 1  # question_gen already strips these; belt-and-braces
                continue

            if row["format"] == "mc":
                if (len(row.get("choices") or []) != 4
                        or not (0 <= row.get("answer_index", -1) < 4)
                        or not row.get("answer")):
                    skipped += 1
                    continue
                # Per-row seeded RNG: reproducible regardless of input order.
                rng = random.Random(f"{args.seed}:{row['id']}")
                rendered = render_mc(row, explain=rng.random() < args.explain_frac)
            elif row["format"] == "short_answer":
                if not row.get("answer"):
                    skipped += 1
                    continue
                # Per-row seeded RNG (same scheme as mc) for the surface flip.
                rng = random.Random(f"{args.seed}:{row['id']}")
                rendered = render_short_answer(row)
            else:
                skipped += 1
                continue

            if row["format"] == "mc" and args.mc_variants != "both":
                keep = f"mc_{args.mc_variants}"
                rendered = [r for r in rendered if r["variant"] == keep]

            for r in rendered:
                out = {
                    "id": f"{row['id']}:{r['variant']}",
                    "source_id": row["id"],
                    "variant": r["variant"],
                    "subdomain": row.get("subdomain"),
                    "track": row.get("track"),
                }
                # Surface flip (seeded per row): raw text matches the
                # profiler's template-free eval surface; chat keeps usability.
                if rng.random() < args.raw_frac:
                    out["surface"] = "raw"
                    out["text"] = render_raw(r["variant"], r["user"], r["assistant"])
                else:
                    out["surface"] = "chat"
                    messages = []
                    if args.system:
                        messages.append({"role": "system", "content": args.system})
                    messages.append({"role": "user", "content": r["user"]})
                    messages.append({"role": "assistant", "content": r["assistant"]})
                    out["messages"] = messages
                out_rows.append(out)

    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.advisory_out, "w") as f:
        for r in advisory_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = Counter(r["variant"] for r in out_rows)
    surfaces = Counter(r["surface"] for r in out_rows)
    print(f"Wrote {len(out_rows)} SFT examples to {args.out}")
    print("Variants: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("Surfaces: " + ", ".join(f"{k}={v}" for k, v in sorted(surfaces.items())))
    print(f"Advisory rows passed through to {args.advisory_out}: {len(advisory_rows)}")
    if skipped:
        print(f"Skipped {skipped} rows (failed verification or malformed)")


if __name__ == "__main__":
    main()
