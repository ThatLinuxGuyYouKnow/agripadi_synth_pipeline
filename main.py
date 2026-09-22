#!/usr/bin/env python3
"""
AgriPadi synthetic SFT dataset generator (DeepSeek Edition).

Generates farmer-style Q&A pairs for pest control, disease ID/mitigation,
and planting cycles across English, Pidgin, and Hausa.

Usage:
    export DEEPSEEK_API_KEY="your_api_key_here"
    python generate_dataset.py --rows 6000 --out dataset.jsonl

Requires: pip install openai
"""

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from openai import OpenAI

MODEL = "deepseek-chat"
VERIFIER_MODEL = "deepseek-chat"

CROPS = [
    "cassava", "maize", "rice", "yam", "sorghum", "millet",
    "cowpea", "groundnut", "tomato", "pepper", "okra",
]

LANGUAGES = ["Nigerian English", "Nigerian Pidgin", "Hausa"]

TOPICS = [
    ("pest", 0.28),
    ("disease", 0.28),
    ("planting_cycle", 0.20),
    ("soil_fertilizer_irrigation", 0.10),
    ("ambiguous_multiturn", 0.10),
    ("safety_edge_case", 0.05),
    ("rotation_storage", 0.05),
]

BANNED_ACTIVES = [
    "endosulfan", "monocrotophos", "DDT", "aldrin", "dieldrin", "paraquat",
    "methyl parathion", "phorate", "lindane", "chlordane", "toxaphene",
]

SYSTEM_PROMPT = f"""You are generating synthetic training data for an agricultural
advisory assistant for smallholder farmers in Nigeria and West Africa.

Rules:
- Questions must sound like a real farmer texting or speaking casually.
- Answers must follow this structure: likely cause -> immediate action -> non-chemical control -> chemical control (generic class only) -> prevention tip.
- NEVER recommend these banned/restricted actives: {", ".join(BANNED_ACTIVES)}.
- Do not state precise chemical dosages. Tell the farmer to confirm exact product and dosage with a local agricultural extension officer.
- Return ONLY valid JSON.
"""

GEN_PROMPT_TEMPLATE = """Generate {n} distinct question-answer pairs for:
Crop: {crop}
Topic: {topic}
Language: {language}
{grounding_block}

CRITICAL LANGUAGE INSTRUCTIONS:
If Language is "Nigerian Pidgin": The question must be in authentic, casual Nigerian Pidgin. The answer must be helpful, respectful Pidgin.
If Language is "Hausa": The question and answer must be in grammatically correct Hausa using proper agricultural terms (e.g., 'Takin zamani', 'Kwari').
If Language is "Nigerian English": Use standard English with slight Nigerian conversational nuances.

Return a JSON object with a single key "data" containing an array of objects. Each object must have keys "question" and "answer" only.
"""

VERIFIER_SYSTEM = """You are a strict fact-checker for agricultural advice aimed at
smallholder farmers in Nigeria. Given a question and answer, respond ONLY with a JSON object:
{"ok": true/false, "note": "short reason if not ok"}

Flag ok=false if the answer:
- states a specific numeric dosage/rate as fact instead of deferring to an extension officer
- recommends a banned/restricted pesticide
- makes up a specific yield/price/study statistic
- is factually implausible agronomy (e.g. wrong season, wrong pest-crop pairing)
Otherwise ok=true.
"""

@dataclass
class Row:
    crop: str
    topic: str
    language: str
    question: str
    answer: str
    verified: bool = False
    verifier_note: str = ""

def build_topic_plan(total_rows: int):
    plan = []
    for topic, frac in TOPICS:
        topic_rows = round(total_rows * frac)
        per_combo = max(1, topic_rows // (len(CROPS) * len(LANGUAGES)))
        for crop in CROPS:
            for lang in LANGUAGES:
                plan.append((crop, topic, lang, per_combo))
    return plan

def call_deepseek(client, system, user, max_tokens=4000):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        max_tokens=max_tokens,
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content.strip()

def parse_json_response(text: str):
    try:
        data = json.loads(text)
        # We asked for {"data": [...]}, so extract the array
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        # Fallback if it just returned the array wrapped in some other key
        for key in data:
            if isinstance(data[key], list):
                return data[key]
        return []
    except json.JSONDecodeError:
        return []

def generate_batch(client, crop, topic, lang, n, grounding_text=""):
    grounding_block = f"Ground your answers in this reference material:\n---\n{grounding_text[:3000]}\n---" if grounding_text else ""
    user_prompt = GEN_PROMPT_TEMPLATE.format(n=n, crop=crop, topic=topic, language=lang, grounding_block=grounding_block)
    
    raw = call_deepseek(client, SYSTEM_PROMPT, user_prompt)
    items = parse_json_response(raw)

    rows = []
    for item in items:
        q, a = item.get("question", "").strip(), item.get("answer", "").strip()
        if not q or not a: continue
        rows.append(Row(crop=crop, topic=topic, language=lang, question=q, answer=a))
    return rows

def contains_banned_active(text: str) -> bool:
    lower = text.lower()
    return any(b.lower() in lower for b in BANNED_ACTIVES)

def verify_row(client, row: Row) -> Row:
    if contains_banned_active(row.answer):
        row.verified = False
        row.verifier_note = "contains banned active ingredient"
        return row
    
    user = f"Question: {row.question}\nAnswer: {row.answer}"
    try:
        raw = call_deepseek(client, VERIFIER_SYSTEM, user, max_tokens=200)
        result = json.loads(raw)
        row.verified = bool(result.get("ok", False))
        row.verifier_note = result.get("note", "")
    except Exception as e:
        row.verified = False
        row.verifier_note = f"verifier error: {e}"
    return row

def dedup_rows(rows, threshold=0.85):
    def norm_tokens(s):
        return set(re.findall(r"[a-z]+", s.lower()))

    kept = []
    kept_token_sets = []
    for r in rows:
        toks = norm_tokens(r.question)
        is_dup = False
        for kt in kept_token_sets:
            if not toks or not kt: continue
            overlap = len(toks & kt) / max(1, len(toks | kt))
            if overlap >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(r)
            kept_token_sets.append(toks)
    return kept

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=6000, help="Target total rows before dedup/verification loss")
    ap.add_argument("--out", type=str, default="dataset.jsonl")
    ap.add_argument("--rejected-out", type=str, default="rejected.jsonl")
    ap.add_argument("--batch-size", type=int, default=10, help="QA pairs requested per API call")
    ap.add_argument("--skip-verify", action="store_true", help="Skip the verifier pass")
    ap.add_argument("--grounding-dir", type=str, default=None, help="Optional dir of .txt source docs")
    args = ap.parse_args()

    # Initialize DeepSeek Client
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY environment variable not set.")
        return

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    grounding = {}
    if args.grounding_dir:
        for crop in CROPS:
            path = os.path.join(args.grounding_dir, f"{crop}.txt")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    grounding[crop] = f.read()

    plan = build_topic_plan(args.rows)
    all_rows: list[Row] = []

    print(f"Plan: {len(plan)} crop/topic/lang cells, target ~{args.rows} rows")
    for i, (crop, topic, lang, target_n) in enumerate(plan):
        remaining = target_n
        while remaining > 0:
            n = min(args.batch_size, remaining)
            batch = generate_batch(client, crop, topic, lang, n, grounding.get(crop, ""))
            all_rows.extend(batch)
            remaining -= n
            time.sleep(0.5) # Rate limit protection for DeepSeek
        print(f"  [{i+1}/{len(plan)}] {crop} | {topic} | {lang}: generated (running total {len(all_rows)})")

    print(f"\nRaw generated: {len(all_rows)}")
    all_rows = dedup_rows(all_rows)
    print(f"After dedup: {len(all_rows)}")

    accepted, rejected = [], []
    if not args.skip_verify:
        print("\nStarting Verification Pass...")
        for i, row in enumerate(all_rows):
            row = verify_row(client, row)
            (accepted if row.verified else rejected).append(row)
            if (i + 1) % 100 == 0:
                print(f"  verified {i+1}/{len(all_rows)}")
            time.sleep(0.1) # Slight delay to prevent hammering the API
    else:
        accepted = all_rows

    print(f"\nAccepted: {len(accepted)}  Rejected: {len(rejected)}")

    # Write out the accepted dataset
    with open(args.out, "w", encoding="utf-8") as f:
        for r in accepted:
            f.write(json.dumps({
                "prompt": r.question,
                "completion": r.answer,
                "crop": r.crop,
                "topic": r.topic,
                "language": r.language # Added language to the output!
            }, ensure_ascii=False) + "\n")

    # Write out the rejected dataset (great for debugging)
    with open(args.rejected_out, "w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps({
                "prompt": r.question,
                "completion": r.answer,
                "crop": r.crop,
                "topic": r.topic,
                "language": r.language,
                "note": r.verifier_note,
            }, ensure_ascii=False) + "\n")

    print(f"Wrote {args.out} and {args.rejected_out}")

if __name__ == "__main__":
    main()
