#!/usr/bin/env python3
"""
Stage 1 of the AgriPadi SFT pipeline (v3): evaluation-aligned synthetic data
generation for the ADTC 2026 agriculture domain.

Why v3 exists (profiler-driven redesign):
- The ADTC profiler scores accuracy by running lm-evaluation-harness against
  the QUANTIZED model in-process via llama-cpp-python. Two automated scoring
  paths exist, plus a human judge path:
    1. loglikelihood(context, continuation) -> multiple-choice RANKING tasks
       (ARC/MMLU-style, acc / acc_norm). The model never generates; it must
       assign the highest probability to the correct continuation.
    2. generate_until (temperature=0, exact_match-family metrics) -> SHORT
       canonical answers. Rambling or hedged answers fail exact match.
    3. Judge panel reads open-ended responses -> advisory quality still
       matters, but the model must COMMIT to an answer, not only clarify.
- loglikelihood_rolling is NOT supported by the profiler -> no perplexity
  tasks, so the hidden set is MC and/or short-generation.
- v2's clarify-heavy distribution (70% of pest/disease items) actively hurts
  paths (1) and (2). Advisory items remain, but answer_directly dominates and
  clarify shrinks to a small slice (CLARIFY_PROB).

What v3 emits (row `format` field):
- "mc"           (~50%): question + 4 choices + answer_index. Generated as
                 answer + 3 distractors and shuffled DETERMINISTICALLY
                 per-item (seed+id), so letter position is unbiased and
                 reproducible. Robust to both ARC-style (choice text as
                 continuation) and MMLU-style (letter as continuation) tasks.
- "short_answer" (~30%): question + short canonical answer (+ 1-sentence
                 explanation usable for SFT variants).
- "advisory"     (~20%): farmer-voice QUESTIONS ONLY (answers still come from
                  the stage-2 pipeline), carrying the v2 metadata (tone,
                  information_level, message_style, goal, expected_behavior)
                  plus a conversation_ready flag marking clarify rows that a
                  later multi-turn stage can extend.

Two generation tracks:
- "excerpt" track: grounded in excerpts.json (depth; cassava/cowpea
  pest/disease today). MC/short answers must be derivable from the excerpt.
- "grid" track: a curriculum of (subdomain x subject x theme x question_type)
  cells spanning all of agriculture (breadth), followed by a batched LLM
  verifier pass that rejects wrong/multi-correct/fabricated items.

Languages: ENGLISH ONLY here (v2 decision kept). Hausa/Pidgin/Yoruba/etc. are
produced at translation time in a later stage with metadata passthrough, so
facts never drift between languages.

Usage:
    python question_gen.py --excerpts excerpts.json --grid-items 3000 \
        --out agripadi_v3.jsonl --provider anthropic

Requires: pip install anthropic openai --break-system-packages
"""

import argparse
import json
import math
import os
import random
import re
import time
import uuid
from collections import Counter

# Load .env at import time so both this script and importers (answer_gen.py)
# see DEEPSEEK_API_KEY etc. Real environment variables take precedence.
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Format mix (set from answers to design questions; overridable via CLI)
# ---------------------------------------------------------------------------

DEFAULT_MC_FRAC = 0.50
DEFAULT_SHORT_FRAC = 0.30
# advisory fraction = 1 - mc - short

# How often a minimal/partial-info diagnosis-flavoured advisory item becomes a
# "clarify" example (v2 effectively made this ~70%; that behaviour now fails
# the benchmark, so it is a small slice).
CLARIFY_PROB = 0.20

# Voice mix for mc/short_answer items. The hidden eval items (if any) will be
# written in neutral/exam voice; farmer voice aids generalisation and the
# judge-facing chat persona. Advisory items are always farmer voice.
VOICES_WEIGHTED = [("neutral", 0.60), ("farmer", 0.40)]

# ---------------------------------------------------------------------------
# Topic grid: subdomains, subjects, themes, question types
# ---------------------------------------------------------------------------
# Weights mirror the ADTC domain description ("crop, livestock, weather, and
# market advisory") plus post-harvest (high farmer value) and the existing
# excerpt strength (pest/disease). Must sum to 1.0.

QUESTION_TYPE_WEIGHTS = [
    ("factual_recall", 0.18),     # name/term/role/sign (kept a minority)
    ("diagnosis_id", 0.20),       # symptoms -> pest/disease/problem
    ("best_action", 0.28),        # situation -> single best agronomic action
    ("comparison", 0.14),         # which option is better for a stated goal
    ("timing_sequence", 0.12),    # when / in what order
    ("safety_regulation", 0.08),  # safe handling, banned inputs, who to consult
]

ALL_QTYPES = [q for q, _ in QUESTION_TYPE_WEIGHTS]

SUBDOMAINS = {
    "crops_agronomy": {
        "weight": 0.18,
        "qtypes": ALL_QTYPES,
        "subjects": [
            "maize", "cassava", "rice (lowland)", "rice (upland)", "yam",
            "sorghum", "millet", "cowpea", "groundnut", "soybean", "tomato",
            "pepper", "okra", "onion", "plantain", "sweet potato", "cocoyam",
            "sesame", "cocoa", "oil palm", "cashew", "cotton",
            "leafy vegetables (ugu/amaranth)",
        ],
        "themes": [
            "variety selection", "land preparation", "planting date and season",
            "spacing and plant population", "seed rate and seed treatment",
            "intercropping and crop rotation", "thinning and gap filling",
            "nutrient deficiency symptoms", "harvesting and maturity signs",
        ],
    },
    "pest_disease_weeds": {
        "weight": 0.20,
        "qtypes": ALL_QTYPES,
        "subjects": [
            "fall armyworm", "stem borers", "grasshoppers", "aphids",
            "cassava mealybug", "cassava green mite", "whiteflies as virus vectors",
            "termites", "nematodes", "storage weevils", "pod borers (Maruca)",
            "fruit flies", "quelea birds", "field rodents",
            "maize streak virus", "cassava mosaic disease",
            "cassava bacterial blight", "groundnut rosette disease",
            "rice blast", "bacterial leaf blight", "tomato early/late blight",
            "damping-off", "root rots", "smut diseases", "rust diseases",
            "striga (witchweed)", "spear grass", "broadleaf weeds",
            "IPM and natural enemies", "safe pesticide handling",
            "pesticide resistance management",
        ],
        "themes": [
            "identification by symptoms", "damage recognition",
            "scouting and action thresholds", "cultural control",
            "biological control", "botanical/organic control",
            "chemical control safety", "prevention", "spread and quarantine",
        ],
    },
    "soil_fertilizer_irrigation": {
        "weight": 0.13,
        "qtypes": ALL_QTYPES,
        "subjects": [
            "soil types and texture", "soil testing", "NPK grades and nutrient roles",
            "urea and nitrogen management", "compost and farmyard manure",
            "green manure and legumes", "liming and soil pH", "micronutrients",
            "fertilizer application methods", "fertilizer timing",
            "erosion control", "mulching and water conservation",
            "rainfed vs irrigated cropping", "small-scale and drip irrigation",
            "drainage and waterlogging",
        ],
        "themes": [
            "nutrient roles and deficiency signs", "application method",
            "application timing", "choosing a rate (soil test / extension)",
            "organic amendments", "soil conservation", "water management",
        ],
    },
    "livestock_poultry_fish": {
        "weight": 0.15,
        "qtypes": ALL_QTYPES,
        "subjects": [
            "broiler chickens", "layer chickens", "local/indigenous chickens",
            "goats", "sheep", "cattle", "pigs", "rabbits", "catfish",
            "tilapia", "ducks", "guinea fowl", "honey bees",
        ],
        "themes": [
            "housing and stocking density", "feeding and feed basics",
            "water provision", "vaccination schedules",
            "common diseases and their signs", "deworming and parasite control",
            "breeding and selection", "brooding and chick management",
            "biosecurity", "pond management and water quality",
        ],
    },
    "weather_climate": {
        "weight": 0.10,
        "qtypes": ["factual_recall", "best_action", "comparison", "timing_sequence"],
        "subjects": [
            "rainfall onset and cessation", "dry spells and drought",
            "harmattan season", "floods and waterlogging", "heat stress in crops",
            "heat stress in livestock", "windstorms", "agro-ecological zones of Nigeria",
            "seasonal forecasts and climate information", "changing rainfall patterns",
            "soil moisture conservation", "planting windows",
        ],
        "themes": [
            "reading seasonal signs", "planning around weather risk",
            "adapting planting decisions", "protecting crops and animals",
            "water harvesting and conservation",
        ],
    },
    "markets_agribusiness": {
        "weight": 0.08,
        "qtypes": ["factual_recall", "best_action", "comparison", "timing_sequence"],
        "subjects": [
            "farm-gate vs market prices", "seasonal price patterns",
            "deciding when to sell vs store", "aggregation and cooperatives",
            "bargaining and middlemen", "grading and quality standards",
            "transport and market access", "record keeping and farm budgets",
            "credit and input finance", "agricultural insurance",
            "contract farming", "value addition decisions",
        ],
        "themes": [
            "getting a better price", "reducing selling costs",
            "group marketing", "farm money management", "managing price risk",
        ],
    },
    "postharvest_storage": {
        "weight": 0.10,
        "qtypes": ALL_QTYPES,
        "subjects": [
            "sun and solar drying", "threshing and shelling",
            "moisture content and safe storage", "hermetic bags (PICS) and silos",
            "cribs and barns", "aflatoxin and storage moulds",
            "storage pests (weevils, larger grain borer)", "traditional storage methods",
            "processing (gari, flour, parboiling, oil extraction)",
            "handling perishables (tomato, leafy vegetables)", "packaging",
            "storing seed for planting",
        ],
        "themes": [
            "reducing post-harvest losses", "safe moisture levels",
            "pest-free storage", "mould and toxin prevention",
            "choosing a storage technology", "when processing adds value",
        ],
    },
    "policy_extension": {
        "weight": 0.06,
        "qtypes": ["factual_recall", "best_action", "comparison", "safety_regulation"],
        "subjects": [
            "agricultural extension services", "government input subsidy programmes",
            "land access and tenure", "farmer cooperatives and associations",
            "agricultural insurance schemes", "quarantine and movement regulations",
            "pesticide registration and banned actives", "veterinary services",
        ],
        "themes": [
            "where to get help", "rules farmers must follow",
            "accessing programmes and services", "organising with other farmers",
        ],
    },
}

# Advisory information-level distributions per subdomain (grid track).
# Diagnosis-flavoured subdomains skew partial (mirrors real extension calls);
# the rest skew complete.
GRID_ADVISORY_INFO_DIST = {
    "pest_disease_weeds": {"minimal": 0.10, "partial": 0.70, "complete": 0.20},
    "livestock_poultry_fish": {"minimal": 0.10, "partial": 0.60, "complete": 0.30},
    "markets_agribusiness": {"minimal": 0.05, "partial": 0.15, "complete": 0.80},
    "weather_climate": {"minimal": 0.05, "partial": 0.15, "complete": 0.80},
    "policy_extension": {"minimal": 0.05, "partial": 0.15, "complete": 0.80},
}
DEFAULT_GRID_INFO_DIST = {"minimal": 0.05, "partial": 0.25, "complete": 0.70}

# Subdomains where a vague advisory message plausibly needs a clarifying
# follow-up (diagnosis-like). Clarify is still gated by CLARIFY_PROB.
DIAGNOSIS_HEAVY = {"pest_disease_weeds", "livestock_poultry_fish", "pest", "disease"}

# Banned/restricted actives (from main.py). They may appear as WRONG MC
# distractors, but never inside a correct answer.
BANNED_ACTIVES = [
    "endosulfan", "monocrotophos", "DDT", "aldrin", "dieldrin", "paraquat",
    "methyl parathion", "phorate", "lindane", "chlordane", "toxaphene",
]

# ---------------------------------------------------------------------------
# Advisory-slice config (carried over from v2; tone/style distributions and
# the light-Pidgin surface ported from the questions-only generator design)
# ---------------------------------------------------------------------------

TONES_WEIGHTED = [
    ("worried", 0.18),
    ("confused", 0.15),
    ("curious", 0.12),
    ("urgent", 0.08),
    ("frustrated", 0.08),
    ("calm", 0.10),
    ("matter-of-fact", 0.18),
    ("hopeful", 0.05),
    ("skeptical", 0.06),
]

# Style strings are fed VERBATIM to the generator LLM via format_specs, so
# each must be self-describing. Pidgin stays a light influence at low rate:
# this batch is English-first; full localization happens in a later stage.
MESSAGE_STYLES_WEIGHTED = [
    ("plain farmer question", 0.15),
    ("WhatsApp message", 0.20),
    ("SMS", 0.08),
    ("voice-note transcription", 0.10),
    ("phone call transcript with an extension officer", 0.07),
    ("spoken aloud to a neighbour", 0.08),
    ("short blunt question", 0.10),
    ("rambling farmer message", 0.10),
    ("casual Nigerian English", 0.07),
    ("English with a light, natural Nigerian Pidgin influence", 0.05),
]

# Goals are scoped per subdomain (v3.1): pest-flavoured goals like "is it
# spreading" are agronomically incoherent for soil, weather or market topics.
# Every list keeps the exact string "general worry / reassurance check-in"
# (determine_expected_behavior keys on it for the "reassure" behaviour).
GOALS_PEST = [
    ("diagnosis", 0.25),
    ("will I lose the whole crop", 0.15),
    ("is it safe to eat / sell", 0.12),
    ("is it spreading to other plants", 0.12),
    ("should I remove or destroy the affected plants", 0.10),
    ("can I still plant something else here next season", 0.08),
    ("why is only part of the field/plant affected", 0.08),
    ("general worry / reassurance check-in", 0.10),
]

GOALS_LIVESTOCK = [
    ("diagnosis", 0.25),
    ("will I lose the animal / flock", 0.15),
    ("is it spreading to the other animals", 0.12),
    ("should I isolate or cull the affected animals", 0.12),
    ("is the meat / milk / egg still safe", 0.10),
    ("when should I call the vet or animal health worker", 0.10),
    ("general worry / reassurance check-in", 0.16),
]

GOALS_SOIL = [
    ("why are my plants yellow or stunted", 0.22),
    ("is it worth applying fertilizer / manure", 0.20),
    ("how do I fix this cheaply with what I have", 0.18),
    ("will this reduce my yield", 0.15),
    ("can I still plant here next season", 0.10),
    ("general worry / reassurance check-in", 0.15),
]

GOALS_WEATHER = [
    ("should I plant now or wait", 0.22),
    ("will the rains still come / will this dry spell end", 0.20),
    ("will my crop survive this weather", 0.18),
    ("how do I protect what I already planted", 0.18),
    ("general worry / reassurance check-in", 0.22),
]

GOALS_CROPS = [
    ("did I plant at the right time / in the right way", 0.22),
    ("why is germination or growth poor", 0.20),
    ("which variety / spacing is best for me", 0.18),
    ("will it mature before the season ends", 0.15),
    ("general worry / reassurance check-in", 0.25),
]

GOALS_POSTHARVEST = [
    ("is my stored produce going bad", 0.25),
    ("how do I stop the losses", 0.22),
    ("is it still safe to eat / sell", 0.18),
    ("should I sell now or store longer", 0.15),
    ("general worry / reassurance check-in", 0.20),
]

GOALS_MARKETS = [
    ("is this a fair price", 0.25),
    ("when / where should I sell", 0.25),
    ("is it worth the transport cost", 0.18),
    ("can I get a loan / input credit", 0.12),
    ("general worry / reassurance check-in", 0.20),
]

GOALS_POLICY = [
    ("where can I get help / inputs / training", 0.35),
    ("am I eligible for this programme", 0.25),
    ("what does this regulation mean for me", 0.20),
    ("general worry / reassurance check-in", 0.20),
]

DEFAULT_GOALS = [
    ("diagnosis", 0.20),
    ("what should I do about it", 0.30),
    ("will it get worse", 0.20),
    ("general worry / reassurance check-in", 0.30),
]

GOALS_BY_SUBDOMAIN = {
    "pest_disease_weeds": GOALS_PEST,
    "livestock_poultry_fish": GOALS_LIVESTOCK,
    "soil_fertilizer_irrigation": GOALS_SOIL,
    "weather_climate": GOALS_WEATHER,
    "crops_agronomy": GOALS_CROPS,
    "postharvest_storage": GOALS_POSTHARVEST,
    "markets_agribusiness": GOALS_MARKETS,
    "policy_extension": GOALS_POLICY,
}

# Per-topic information_level distributions for the EXCERPT track (v2 config).
TOPIC_INFO_LEVEL_DIST = {
    "pest": {"minimal": 0.10, "partial": 0.70, "complete": 0.20},
    "disease": {"minimal": 0.10, "partial": 0.70, "complete": 0.20},
    "planting_cycle": {"minimal": 0.05, "partial": 0.15, "complete": 0.80},
    "soil_fertilizer_irrigation": {"minimal": 0.05, "partial": 0.25, "complete": 0.70},
    "rotation_storage": {"minimal": 0.05, "partial": 0.15, "complete": 0.80},
    "ambiguous_multiturn": {"minimal": 0.60, "partial": 0.40, "complete": 0.0},
    "safety_edge_case": {"minimal": 0.50, "partial": 0.50, "complete": 0.0},
    "weed_management": {"minimal": 0.25, "partial": 0.25, "complete": 0.50},
}
DEFAULT_INFO_DIST = {"minimal": 0.33, "partial": 0.34, "complete": 0.33}

# Map excerpt topics to grid subdomains (for qtype/voice sampling).
EXCERPT_TOPIC_TO_SUBDOMAIN = {
    "pest": "pest_disease_weeds",
    "disease": "pest_disease_weeds",
    "weed_management": "pest_disease_weeds",
    "planting_cycle": "crops_agronomy",
    "soil_fertilizer_irrigation": "soil_fertilizer_irrigation",
    "rotation_storage": "postharvest_storage",
    "crops_agronomy": "crops_agronomy",
    "postharvest_storage": "postharvest_storage",
    "livestock_poultry_fish": "livestock_poultry_fish",
    "weather_climate": "weather_climate",
    "markets_agribusiness": "markets_agribusiness",
    "policy_extension": "policy_extension",
}

# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------

def weighted_choice(options_weighted):
    options, weights = zip(*options_weighted)
    return random.choices(options, weights=weights, k=1)[0]


def sample_from_dist(dist):
    options, weights = zip(*dist.items())
    return random.choices(options, weights=weights, k=1)[0]


def sample_qtype(allowed):
    pool = [(q, w) for q, w in QUESTION_TYPE_WEIGHTS if q in allowed]
    return weighted_choice(pool)


def determine_expected_behavior(topic_or_subdomain: str, info_level: str, goal: str) -> str:
    """Map a generation plan to an expected assistant behaviour.

    v3 change: clarify is a SMALL slice (CLARIFY_PROB), not the default for
    under-specified diagnosis questions. The benchmark rewards commitment;
    the stage-2 answerer handles 'most-likely-answer first, one targeted
    follow-up after' for clarify rows.
    """
    if topic_or_subdomain == "safety_edge_case":
        return "safety_refusal"
    if goal == "general worry / reassurance check-in":
        return "reassure"
    if (info_level in ("minimal", "partial")
            and topic_or_subdomain in DIAGNOSIS_HEAVY
            and random.random() < CLARIFY_PROB):
        return "clarify"
    return "answer_directly"


def split_formats(n: int, mc_frac: float, short_frac: float):
    """Deterministic per-batch format counts: (n_mc, n_short, n_advisory)."""
    n_mc = round(n * mc_frac)
    n_short = round(n * short_frac)
    return n_mc, n_short, n - n_mc - n_short


def goals_for(topic_or_subdomain: str):
    """Goal list for a grid subdomain or an excerpt topic (mapped via
    EXCERPT_TOPIC_TO_SUBDOMAIN; unknown keys get the generic default)."""
    if topic_or_subdomain in GOALS_BY_SUBDOMAIN:
        return GOALS_BY_SUBDOMAIN[topic_or_subdomain]
    mapped = EXCERPT_TOPIC_TO_SUBDOMAIN.get(topic_or_subdomain)
    return GOALS_BY_SUBDOMAIN.get(mapped, DEFAULT_GOALS)


def make_advisory_spec(row_id, info_dist, topic_or_subdomain):
    info_level = sample_from_dist(info_dist)
    goal = weighted_choice(goals_for(topic_or_subdomain))
    return {
        "id": row_id,
        "format": "advisory",
        "information_level": info_level,
        "tone": weighted_choice(TONES_WEIGHTED),
        "message_style": weighted_choice(MESSAGE_STYLES_WEIGHTED),
        "goal": goal,
        "expected_behavior": determine_expected_behavior(topic_or_subdomain, info_level, goal),
    }


def build_excerpt_plan(excerpt, n, mc_frac, short_frac):
    """Plan for one excerpt: same 50/30/20 mix as the grid, grounded."""
    topic = excerpt["topic"]
    subdomain = EXCERPT_TOPIC_TO_SUBDOMAIN.get(topic, "crops_agronomy")
    allowed = SUBDOMAINS[subdomain]["qtypes"]
    info_dist = TOPIC_INFO_LEVEL_DIST.get(topic, DEFAULT_INFO_DIST)
    n_mc, n_short, n_adv = split_formats(n, mc_frac, short_frac)

    plan = []
    for i in range(n_mc + n_short):
        fmt = "mc" if i < n_mc else "short_answer"
        plan.append({
            "id": f"{excerpt['crop']}_{topic}_{uuid.uuid4().hex[:8]}",
            "format": fmt,
            "question_type": sample_qtype(allowed),
            "voice": weighted_choice(VOICES_WEIGHTED),
        })
    for _ in range(n_adv):
        row_id = f"{excerpt['crop']}_{topic}_{uuid.uuid4().hex[:8]}"
        plan.append(make_advisory_spec(row_id, info_dist, topic))
    return plan


def build_grid_cells(grid_items: int, items_per_cell: int):
    """Allocate cells across subdomains by weight (largest remainder), then
    cycle through shuffled (subject, theme) combos for variety."""
    total_cells = max(1, math.ceil(grid_items / items_per_cell))
    raw = {sd: total_cells * cfg["weight"] for sd, cfg in SUBDOMAINS.items()}
    counts = {sd: int(v) for sd, v in raw.items()}
    remainder = total_cells - sum(counts.values())
    for sd in sorted(raw, key=lambda s: raw[s] - int(raw[s]), reverse=True)[:remainder]:
        counts[sd] += 1

    cells = []
    for sd, n_cells in counts.items():
        cfg = SUBDOMAINS[sd]
        combos = [(s, t) for s in cfg["subjects"] for t in cfg["themes"]]
        random.shuffle(combos)
        for i in range(n_cells):
            subject, theme = combos[i % len(combos)]
            cells.append({
                "cell_id": f"{sd}_{uuid.uuid4().hex[:8]}",
                "subdomain": sd,
                "subject": subject,
                "theme": theme,
            })
    return cells


def build_cell_plan(cell, n, mc_frac, short_frac):
    sd = cell["subdomain"]
    allowed = SUBDOMAINS[sd]["qtypes"]
    info_dist = GRID_ADVISORY_INFO_DIST.get(sd, DEFAULT_GRID_INFO_DIST)
    n_mc, n_short, n_adv = split_formats(n, mc_frac, short_frac)

    plan = []
    for i in range(n_mc + n_short):
        fmt = "mc" if i < n_mc else "short_answer"
        plan.append({
            "id": f"{sd}_{uuid.uuid4().hex[:8]}",
            "format": fmt,
            "question_type": sample_qtype(allowed),
            "voice": weighted_choice(VOICES_WEIGHTED),
        })
    for _ in range(n_adv):
        plan.append(make_advisory_spec(f"{sd}_{uuid.uuid4().hex[:8]}", info_dist, sd))
    return plan


def format_specs(plan):
    lines = []
    for item in plan:
        if item["format"] == "advisory":
            lines.append(
                f"- id: {item['id']} | format: advisory | "
                f"information_level: {item['information_level']} | "
                f"tone: {item['tone']} | message_style: {item['message_style']} | "
                f"goal: {item['goal']}"
            )
        else:
            lines.append(
                f"- id: {item['id']} | format: {item['format']} | "
                f"question_type: {item['question_type']} | voice: {item['voice']}"
            )
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

_FORMAT_RULES = """Three item formats exist; each spec line says which to write.

1. "mc" -- a multiple-choice question with exactly ONE unambiguously correct
   answer and three plausible distractors.
   - Distractors must be the same KIND of thing as the answer (same category,
     similar specificity and length) and must be genuinely wrong, not
     debatably correct.
   - LENGTH BALANCE (critical, mechanical rule): the correct answer must NOT
     be the longest option. Write all four options at roughly the same word
     count. After drafting, count the words of each option; if the correct
     answer has the most words, lengthen a distractor or shorten the answer
     until it does not. Never pad the correct answer with hedging like "It is
     recommended to always..." -- that phrasing is a tell.
   - DISTRACTOR DIFFICULTY (critical): every wrong option must be a real,
     plausible alternative practice or fact from the same domain. Never
     absurd, physically nonsensical, self-evidently reckless, or already
     explicitly ruled out by the question or the source material. Someone
     with NO farming knowledge must NOT be able to eliminate any option by
     common sense alone.
     GOOD (hard): answer "rough handling of the stems" alongside options
     "stems taken from a 12-month-old plant", "planting in the wet season",
     "cutting the stems into 25cm pieces" -- all four are real agronomic
     variables.
     GOOD (hard): "dry season" vs "wet season" vs "planting season" vs
     "harvest season".
     BAD (too easy, never write these): "flood the field to drown the mites",
     "apply the strongest chemical you can find immediately" (where the
     context already warns against chemicals), "do nothing and hope".
   - Never use "all of the above", "none of the above", or combined options.
   - No trick questions and no obscure statistics -- answerable from knowledge
     a good West African extension officer would have.
   - Return: {"id": "...", "question": "...", "answer": "...",
     "distractors": ["...", "...", "..."], "explanation": "one sentence: why
     the answer is correct"}

2. "short_answer" -- a question with a short canonical answer (a term, name,
   number with unit, or one short phrase).
   - The answer must be the standard, unambiguous response an examiner would
     accept.
   - Prefer advisory-relevant facts (what to do, when, which sign, how much)
     over taxonomic recall. A Latin/scientific binomial may be the expected
     answer in AT MOST about 1 in 10 short-answer items, and only when the
     common name also appears in the question.
   - Return: {"id": "...", "question": "...", "answer": "...",
     "explanation": "one sentence"}

3. "advisory" -- ONLY a farmer's message (never the answer), in the farmer's
   own casual Nigerian-English voice, following the information_level, tone,
   message_style and goal in the spec:
   - "minimal": one vague symptom or complaint, no diagnosis attempt, may not
     even name the crop/animal
   - "partial": a couple of symptoms or some context, but missing something a
     real diagnosis would need
   - "complete": a fuller description, still in the farmer's own words
   - Do NOT over-specify: real farmers rarely name the pest, the disease, the
     exact variety or the scientific term -- they describe what they observe
     in ordinary words. Even "complete" messages stay layperson. Uncertainty
     and incomplete descriptions are realistic, not a flaw.
   - If the message_style mentions Pidgin, use only a LIGHT influence (a word
     or two, e.g. "dey", "wetin", "make I") inside otherwise plain English --
     never full Pidgin.
   - Return: {"id": "...", "question": "..."}

question_type meanings (mc/short_answer only):
- factual_recall: a specific fact (name, term, nutrient role, sign)
- diagnosis_id: given symptoms/signs, identify the pest, disease or problem
- best_action: given a situation, the single best agronomic action
- comparison: which of the options is better for a stated goal
- timing_sequence: when, or in what order, to do something
- safety_regulation: safe handling, banned/restricted inputs, correct
  authority to consult

voice (mc/short_answer only):
- "neutral": written like a clear exam or reference question
- "farmer": sounds like a farmer asking, but still with a definite,
  answerable intent (never vague for mc/short_answer)
"""

GLOBAL_RULES = f"""Rules for ALL items:
- Do NOT invent statistics, study results, product brand names, or precise
  chemical dosages. Where dosage matters, the correct answer defers to the
  product label or a local extension officer.
- Never present a banned/restricted pesticide as a recommendation:
  {", ".join(BANNED_ACTIVES)}. (They MAY appear as WRONG distractors.)
- West African smallholder context: local crops, breeds, seasons and market
  realities.
- Vary phrasing across items; no two items should share the same question
  shape.
- All output in English. Return ONLY JSON: an array with one object per
  situation/cell you were given. No markdown fences, no commentary.
"""

GRID_SYSTEM_PROMPT = f"""You are generating training data for an agricultural
advisory AND examination benchmark for smallholder farmers and extension
officers in Nigeria and West Africa. You will be given several "cells", each
a (subdomain, subject, theme) triple plus item specifications to fill.

{_FORMAT_RULES}
{GLOBAL_RULES}
Output shape: [{{"cell_id": "...", "items": [...]}}]. The "id" of each item
must exactly match the id in its specification.
"""

GRID_CELL_TEMPLATE = """Cell (cell_id: {cell_id})
Subdomain: {subdomain} | Subject: {subject} | Theme: {theme}
Item specifications:
{specs}
"""

EXCERPT_SYSTEM_PROMPT = f"""You are generating training data for an
agricultural advisory AND examination benchmark for smallholder farmers in
Nigeria and West Africa. You will be given, for one or more crop situations, a
reference excerpt describing the real underlying facts, plus item
specifications to fill.

Grounding rules (excerpt track):
- For "mc" and "short_answer" items, the QUESTION and the CORRECT ANSWER must
  be derivable from the excerpt. Distractors must be plausible agronomic
  alternatives that are NOT presented as correct in the excerpt, and must
  satisfy the DISTRACTOR DIFFICULTY rules below -- in particular, the excerpt
  must not make a distractor eliminable without domain knowledge (e.g. if the
  excerpt warns against chemicals, "apply chemicals" is too easy).
- The farmer has NOT read the reference document. Never copy the excerpt's
  wording, terminology, or sentence structure into any question -- rephrase
  into plain question language (answers may use the correct technical terms).
- Do not invent facts, crops, or symptoms absent from the excerpt.

{_FORMAT_RULES}
{GLOBAL_RULES}
Output shape: [{{"excerpt_id": "...", "items": [...]}}]. The "id" of each
item must exactly match the id in its specification.
"""

EXCERPT_SITUATION_TEMPLATE = """Situation (excerpt_id: {excerpt_id})
Crop: {crop}
Topic: {topic}
Reference excerpt (context only, do not mimic wording):
---
{excerpt}
---
Item specifications:
{specs}
"""

VERIFIER_SYSTEM = f"""You are a strict reviewer of agricultural training data
for West African smallholder farming. You will be given a batch of items
(multiple-choice or short-answer). For each, decide if it is usable.

Reject (ok=false) if ANY holds:
- mc: the marked answer is wrong, OR more than one choice could be defended
  as correct, OR a distractor is actually correct
- mc: the correct answer stands out stylistically -- noticeably longer, more
  detailed, or more carefully hedged than every distractor (a tell that lets
  a test-taker pick it without knowing the content)
- short_answer: the answer is wrong, not the standard answer, or not short
  and canonical
- any item: factually wrong agronomy for West Africa, invented statistics or
  brand names, a precise chemical dosage stated as fact, or a
  banned/restricted pesticide ({", ".join(BANNED_ACTIVES)}) presented as a
  recommendation
Otherwise ok=true.

Return ONLY a JSON array: [{{"id": "...", "ok": true/false, "note": "short
reason if not ok"}}]
"""

# ---------------------------------------------------------------------------
# Model backends
# ---------------------------------------------------------------------------

def get_anthropic_client():
    import anthropic
    return anthropic.Anthropic()


def get_deepseek_client():
    import openai
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit(
            "DEEPSEEK_API_KEY is not set. Put it in .env "
            "(DEEPSEEK_API_KEY=sk-...) or export it in the shell."
        )
    return openai.OpenAI(
        api_key=key,
        base_url="https://api.deepseek.com",
    )


def call_model(provider, client, model, system, user, max_tokens=8192):
    if provider == "anthropic":
        resp = client.messages.create(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()
    elif provider == "deepseek":
        resp = client.chat.completions.create(
            model=model, max_tokens=max_tokens,
            # deepseek-v4-flash is a reasoning model whose chain-of-thought
            # counts against max_tokens and can starve the actual output
            # (observed: 16k tokens of reasoning, empty content). These JSON
            # generation/verification tasks need no deliberation, so turn
            # thinking off. Harmless on non-thinking models (deepseek-chat).
            extra_body={"thinking": {"type": "disabled"}},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        ch = resp.choices[0]
        if ch.finish_reason == "length":
            raise RuntimeError(
                f"response truncated at max_tokens={max_tokens} (no usable JSON)"
            )
        return (ch.message.content or "").strip()
    raise ValueError(f"unknown provider {provider}")


def call_with_retry(provider, client, model, system, user, tries=2, max_tokens=6000):
    for attempt in range(tries):
        try:
            return call_model(provider, client, model, system, user, max_tokens=max_tokens)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2)


UNQUOTED_KEY_RE = re.compile(r'([{,])\s*([a-zA-Z_][\w]*)\s*:')


def _repair_json(text: str) -> str:
    return UNQUOTED_KEY_RE.sub(r'\1"\2":', text)


def parse_json_array(text: str):
    text = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)

    def _try_load(s):
        return json.loads(s)

    def _try_extract_and_load(s):
        m = re.search(r"\[.*\]", s, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        return None

    for attempt, raw in [(0, text), (1, _repair_json(text))]:
        for loader in [_try_load, _try_extract_and_load]:
            try:
                result = loader(raw)
                if result is not None:
                    return result
            except json.JSONDecodeError:
                continue
    raise json.JSONDecodeError("all repair + extraction attempts failed", text, 0)

# ---------------------------------------------------------------------------
# Validation, assembly, post-processing
# ---------------------------------------------------------------------------

BAD_CHOICE_PAT = re.compile(r"\b(all|none|both|neither) of the above\b", re.I)


def contains_banned_active(text: str) -> bool:
    lower = text.lower()
    return any(b.lower() in lower for b in BANNED_ACTIVES)


def validate_mc(answer, distractors):
    """Deterministic MC shape checks. Returns (ok, reason)."""
    if not answer or not isinstance(distractors, list) or len(distractors) != 3:
        return False, "needs exactly 1 answer + 3 distractors"
    choices = [answer] + distractors
    if any(not isinstance(c, str) or not c.strip() for c in choices):
        return False, "empty choice"
    if len({c.strip().lower() for c in choices}) != 4:
        return False, "duplicate choices"
    if any(BAD_CHOICE_PAT.search(c) for c in choices):
        return False, "meta choice (all/none of the above)"
    if contains_banned_active(answer):
        return False, "correct answer contains a banned active"
    # Length-balance gate: generators systematically write the correct answer
    # as the longest, most detailed option (~60-80% of items unprompted).
    # A model trained on that learns "pick the longest choice", which fails on
    # judge-written items and is punished by acc_norm's byte-length
    # normalisation. Reject when the answer is BOTH uniquely the longest AND
    # at least 25% above the mean distractor length.
    aw = len(answer.split())
    dws = [len(d.split()) for d in distractors]
    if aw > max(dws) and aw >= 1.25 * (sum(dws) / len(dws)):
        return False, (
            f"length_bias: answer is longest ({aw} words vs distractor "
            f"mean {sum(dws) / len(dws):.1f}); lengthen a distractor or "
            f"shorten the answer"
        )
    return True, ""


def assemble_choices(answer, distractors, seed, item_id):
    """Shuffle deterministically per item (stable regardless of run order),
    so correct-answer letter position is unbiased and reproducible."""
    rng = random.Random(f"{seed}:{item_id}")
    choices = [answer] + list(distractors)
    rng.shuffle(choices)
    return choices, choices.index(answer)


def looks_like_document_voice(question: str, excerpt: str) -> bool:
    q_words = re.findall(r"[a-z]+", question.lower())
    excerpt_lower = excerpt.lower()
    for i in range(len(q_words) - 3):
        gram = " ".join(q_words[i:i + 4])
        if gram in excerpt_lower:
            return True
    return False


def dedup_rows(rows, threshold=0.8):
    """Remove near-duplicate questions, but never collapse advisory rows that
    differ in information_level: a vague and a fully-specified version of the
    same problem are deliberately different training examples."""
    def norm_tokens(s):
        return set(re.findall(r"[a-z]+", s.lower()))
    kept, kept_sets = [], []
    for r in rows:
        toks = norm_tokens(r["question"])
        dup = False
        for kr, kt in zip(kept, kept_sets):
            if (r["format"] == "advisory" and kr["format"] == "advisory"
                    and r.get("information_level") != kr.get("information_level")):
                continue
            if toks and kt and len(toks & kt) / len(toks | kt) >= threshold:
                dup = True
                break
        if not dup:
            kept.append(r)
            kept_sets.append(toks)
    return kept


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

# ---------------------------------------------------------------------------
# Track runners
# ---------------------------------------------------------------------------

def ingest_items(results, plans_by_group, row_base, seed, rejected):
    """Shared item ingestion for both tracks. Returns accepted rows.

    row_base(group_key) -> dict of constant row fields (track, subdomain,
    subject, topic, crop, source_excerpt_id).
    """
    rows = []
    for result in results:
        group_key = result.get("cell_id") or result.get("excerpt_id")
        plan_lookup = plans_by_group.get(group_key)
        if not group_key or not plan_lookup:
            continue
        base = row_base(group_key)
        for item in result.get("items", []):
            spec = plan_lookup.get(item.get("id"))
            q = (item.get("question") or "").strip()
            if not spec or not q:
                continue
            row = dict(base)
            row.update({"id": spec["id"], "format": spec["format"],
                        "question": q, "language": "English"})
            if spec["format"] == "mc":
                answer = (item.get("answer") or "").strip()
                distractors = item.get("distractors") or []
                ok, reason = validate_mc(answer, distractors)
                if not ok:
                    bad = dict(row); bad["note"] = f"invalid_mc: {reason}"
                    rejected.append(bad)
                    continue
                choices, idx = assemble_choices(answer, distractors, seed, spec["id"])
                row.update({
                    "choices": choices, "answer_index": idx, "answer": choices[idx],
                    "explanation": (item.get("explanation") or "").strip(),
                    "question_type": spec["question_type"], "voice": spec["voice"],
                })
            elif spec["format"] == "short_answer":
                answer = (item.get("answer") or "").strip()
                if not answer:
                    bad = dict(row); bad["note"] = "invalid_short_answer: empty answer"
                    rejected.append(bad)
                    continue
                row.update({
                    "answer": answer,
                    "explanation": (item.get("explanation") or "").strip(),
                    "question_type": spec["question_type"], "voice": spec["voice"],
                })
            else:  # advisory
                row.update({
                    "information_level": spec["information_level"],
                    "tone": spec["tone"], "message_style": spec["message_style"],
                    "goal": spec["goal"], "expected_behavior": spec["expected_behavior"],
                    # Marks rows a later multi-turn stage can extend with a
                    # clarifying follow-up exchange.
                    "conversation_ready": spec["expected_behavior"] == "clarify",
                })
            rows.append(row)
    return rows


def run_excerpt_track(args, provider, client, model, excerpts, rejected):
    rows = []
    for batch_num, ex_batch in enumerate(chunked(excerpts, args.excerpts_per_call)):
        situations, plans_by_group = [], {}
        for ex in ex_batch:
            plan = build_excerpt_plan(ex, args.items_per_excerpt, args.mc_frac, args.short_frac)
            plans_by_group[ex["id"]] = {p["id"]: p for p in plan}
            situations.append(EXCERPT_SITUATION_TEMPLATE.format(
                excerpt_id=ex["id"], crop=ex["crop"], topic=ex["topic"],
                excerpt=ex["excerpt_text"], specs=format_specs(plan),
            ))
        try:
            raw = call_with_retry(provider, client, model, EXCERPT_SYSTEM_PROMPT,
                                  "\n\n".join(situations))
            results = parse_json_array(raw)
        except Exception as e:
            print(f"  [excerpt batch {batch_num+1}] FAILED: {e}")
            continue

        def base(excerpt_id, _ex=ex_batch):
            ex = next(e for e in _ex if e["id"] == excerpt_id)
            return {"track": "excerpt",
                    "subdomain": EXCERPT_TOPIC_TO_SUBDOMAIN.get(ex["topic"], "crops_agronomy"),
                    "subject": ex["crop"], "crop": ex["crop"], "topic": ex["topic"],
                    "source_excerpt_id": ex["id"]}

        got = ingest_items(results, plans_by_group, base, args.seed, rejected)
        rows.extend(got)
        print(f"  [excerpt batch {batch_num+1}] {len(ex_batch)} excerpts -> {len(got)} items "
              f"(track total {len(rows)})")
        time.sleep(0.2)
    return rows


def run_grid_track(args, provider, client, model, rejected):
    cells = build_grid_cells(args.grid_items, args.items_per_cell)
    print(f"Grid: {len(cells)} cells across {len(SUBDOMAINS)} subdomains "
          f"(~{len(cells) * args.items_per_cell} items)")
    rows = []
    for batch_num, cell_batch in enumerate(chunked(cells, args.cells_per_call)):
        blocks, plans_by_group = [], {}
        for cell in cell_batch:
            plan = build_cell_plan(cell, args.items_per_cell, args.mc_frac, args.short_frac)
            plans_by_group[cell["cell_id"]] = {p["id"]: p for p in plan}
            blocks.append(GRID_CELL_TEMPLATE.format(
                cell_id=cell["cell_id"], subdomain=cell["subdomain"],
                subject=cell["subject"], theme=cell["theme"], specs=format_specs(plan),
            ))
        try:
            raw = call_with_retry(provider, client, model, GRID_SYSTEM_PROMPT,
                                  "\n\n".join(blocks))
            results = parse_json_array(raw)
        except Exception as e:
            print(f"  [grid batch {batch_num+1}] FAILED: {e}")
            continue

        def base(cell_id, _batch=cell_batch):
            cell = next(c for c in _batch if c["cell_id"] == cell_id)
            crop = cell["subject"] if cell["subdomain"] in (
                "crops_agronomy", "pest_disease_weeds") else None
            return {"track": "grid", "subdomain": cell["subdomain"],
                    "subject": cell["subject"], "crop": crop, "topic": cell["theme"],
                    "source_excerpt_id": None}

        got = ingest_items(results, plans_by_group, base, args.seed, rejected)
        rows.extend(got)
        print(f"  [grid batch {batch_num+1}] {len(cell_batch)} cells -> {len(got)} items "
              f"(track total {len(rows)})")
        time.sleep(0.2)
    return rows

# ---------------------------------------------------------------------------
# Verifier (batched; mc + short_answer only)
# ---------------------------------------------------------------------------

def render_for_verifier(row):
    if row["format"] == "mc":
        lines = [f"id: {row['id']} | format: mc", f"question: {row['question']}", "choices:"]
        for i, c in enumerate(row["choices"]):
            mark = "  <-- marked correct" if i == row["answer_index"] else ""
            lines.append(f"  {chr(65 + i)}. {c}{mark}")
        return "\n".join(lines)
    return (f"id: {row['id']} | format: short_answer\n"
            f"question: {row['question']}\nanswer: {row['answer']}")


def verify_rows(rows, args, provider, client, model):
    verifiable = [r for r in rows if r["format"] in ("mc", "short_answer")]
    print(f"Verifier: {len(verifiable)} answered items "
          f"({math.ceil(len(verifiable) / args.verify_batch_size)} calls)")
    verdicts = {}
    for batch_num, batch in enumerate(chunked(verifiable, args.verify_batch_size)):
        user = "\n\n".join(render_for_verifier(r) for r in batch)
        try:
            raw = call_with_retry(provider, client, model, VERIFIER_SYSTEM, user,
                                  max_tokens=min(6000, len(batch) * 80))
            for v in parse_json_array(raw):
                verdicts[v.get("id")] = v
        except Exception as e:
            print(f"  [verify batch {batch_num+1}] FAILED: {e} (items kept, unverified)")
        time.sleep(0.2)
    for r in verifiable:
        v = verdicts.get(r["id"])
        if v is None:
            r["verified"], r["verifier_note"] = None, "verifier unavailable"
        else:
            r["verified"] = bool(v.get("ok", False))
            r["verifier_note"] = v.get("note", "")
    return rows

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excerpts", type=str, default=None,
                    help="Optional excerpts.json for the grounded track")
    ap.add_argument("--out", type=str, default="agripadi_v3.jsonl")
    ap.add_argument("--flagged-out", type=str, default="agripadi_v3_flagged.jsonl")
    ap.add_argument("--rejected-out", type=str, default="agripadi_v3_rejected.jsonl")
    ap.add_argument("--items-per-excerpt", type=int, default=12)
    ap.add_argument("--excerpts-per-call", type=int, default=3)
    ap.add_argument("--grid-items", type=int, default=3000,
                    help="Target grid-track items (before losses)")
    ap.add_argument("--items-per-cell", type=int, default=6)
    ap.add_argument("--cells-per-call", type=int, default=4)
    ap.add_argument("--mc-frac", type=float, default=DEFAULT_MC_FRAC)
    ap.add_argument("--short-frac", type=float, default=DEFAULT_SHORT_FRAC)
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--verify-batch-size", type=int, default=40)
    ap.add_argument("--provider", choices=["anthropic", "deepseek"], default="anthropic")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.excerpts and args.grid_items <= 0:
        ap.error("nothing to do: pass --excerpts and/or --grid-items > 0")

    random.seed(args.seed)
    default_models = {"anthropic": "claude-sonnet-4-6", "deepseek": "deepseek-chat"}
    model = args.model or default_models[args.provider]
    client = get_anthropic_client() if args.provider == "anthropic" else get_deepseek_client()

    all_rows, rejected = [], []

    if args.excerpts:
        with open(args.excerpts) as f:
            excerpts = json.load(f)
        excerpt_lookup = {ex["id"]: ex for ex in excerpts}
        all_rows.extend(run_excerpt_track(args, args.provider, client, model, excerpts, rejected))
    else:
        excerpt_lookup = {}

    if args.grid_items > 0:
        all_rows.extend(run_grid_track(args, args.provider, client, model, rejected))

    before = len(all_rows)
    all_rows = dedup_rows(all_rows)
    print(f"Dedup: {before} -> {len(all_rows)}")

    # Document-voice flag: excerpt-track questions only (answers are SUPPOSED
    # to be derivable from the excerpt, so only questions are checked).
    accepted, flagged = [], []
    for r in all_rows:
        ex_text = (excerpt_lookup.get(r["source_excerpt_id"] or "") or {}).get("excerpt_text")
        if ex_text and looks_like_document_voice(r["question"], ex_text):
            r["reason"] = "possible verbatim phrase overlap with source excerpt"
            flagged.append(r)
        else:
            accepted.append(r)

    if not args.skip_verify:
        accepted = verify_rows(accepted, args, args.provider, client, model)
        still_ok = []
        for r in accepted:
            if r.get("verified") is False:
                r["note"] = f"verifier: {r.get('verifier_note', '')}"
                rejected.append(r)
            else:
                still_ok.append(r)
        accepted = still_ok

    with open(args.out, "w") as f:
        for r in accepted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.flagged_out, "w") as f:
        for r in flagged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.rejected_out, "w") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- summary ---
    print(f"\nWrote {len(accepted)} items to {args.out}")
    print(f"Flagged {len(flagged)} for review -> {args.flagged_out}")
    print(f"Rejected {len(rejected)} -> {args.rejected_out}")
    fmt_counts = Counter(r["format"] for r in accepted)
    total = max(1, len(accepted))
    print("Formats: " + ", ".join(f"{k}={v} ({v/total:.0%})" for k, v in sorted(fmt_counts.items())))
    print("Tracks: " + ", ".join(f"{k}={v}" for k, v in Counter(r["track"] for r in accepted).items()))
    print("Subdomains: " + ", ".join(f"{k}={v}" for k, v in
          Counter(r["subdomain"] for r in accepted).most_common()))
    letters = Counter(chr(65 + r["answer_index"]) for r in accepted if r["format"] == "mc")
    if letters:
        print("MC answer letters: " + ", ".join(f"{k}={letters.get(k, 0)}" for k in "ABCD"))
    behaviors = Counter(r.get("expected_behavior") for r in accepted if r["format"] == "advisory")
    if behaviors:
        print("Advisory behaviours: " + ", ".join(f"{k}={v}" for k, v in behaviors.most_common()))


if __name__ == "__main__":
    main()
