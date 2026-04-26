#!/usr/bin/env python3
"""
generate_golden_set.py
Generate grounded evaluation questions from the actual knowledge base.

Uses GPT-4o to produce paraphrased and varied questions directly from:
  - FAQ source CSV  (data/library_faq_clean.csv)
  - Database descriptions CSV (data/Databases description.csv)

Output: data/golden_set_generated.json
        A companion file to data/golden_set.json. These generated questions
        are grounded in real source content — not biased toward what you
        already know works. Merge them into golden_set.json after review.

Usage:
    python scripts/generate_golden_set.py
    python scripts/generate_golden_set.py --faq-count 30 --db-count 20
    python scripts/generate_golden_set.py --output data/my_generated.json
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from backend.llm_client import chat_completion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")


# ---------------------------------------------------------------------------
# Source document loaders
# ---------------------------------------------------------------------------

def load_faqs(path: str) -> list:
    faqs = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip()
            if q and a and len(a) > 20:
                faqs.append({"question": q, "answer": a[:800]})
    return faqs


def load_databases(path: str) -> list:
    dbs = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("name") or "").strip()
            desc = (row.get("description") or "").strip()
            if name and desc and len(desc) > 20:
                # Strip HTML tags
                desc = re.sub(r"<[^>]+>", " ", desc)
                desc = re.sub(r"\s+", " ", desc).strip()[:600]
                dbs.append({"name": name, "description": desc})
    return dbs


# ---------------------------------------------------------------------------
# Question generators
# ---------------------------------------------------------------------------

_FAQ_GENERATION_PROMPT = """\
You are generating evaluation questions for a university library chatbot.

Given a library FAQ entry (question + answer), generate 3 DIFFERENT question phrasings
that test the same information. The questions must be:
1. Clearly answerable from the provided FAQ answer
2. Varied in phrasing (not just synonym swaps)
3. Realistic — something a student would actually ask
4. At least one should be indirect (not literally repeating the FAQ question)

Also provide:
- The key terms a correct answer MUST contain (2-4 words from the answer)
- The difficulty: "easy" if question closely matches FAQ, "medium" if indirect

Return ONLY valid JSON (no markdown):
{
  "questions": [
    {
      "question": "...",
      "difficulty": "easy" | "medium",
      "answer_keywords": ["...", "..."]
    }
  ]
}

FAQ Entry:
Q: {question}
A: {answer}"""


_DB_GENERATION_PROMPT = """\
You are generating evaluation questions for a university library chatbot that recommends databases.

Given a database name and description, generate 2 questions a student might ask that
this database would answer. The questions must be:
1. Grounded in the database description — the answer must come from this database
2. Realistic — a student asking about a research topic, not asking about the database by name
3. Varied (one topic-based, one task-based)

Also provide:
- The key terms a correct answer MUST contain (include the database name)
- The difficulty: "easy" if obvious, "medium" if requires subject inference

Return ONLY valid JSON (no markdown):
{
  "questions": [
    {
      "question": "...",
      "difficulty": "easy" | "medium",
      "answer_keywords": ["...", "..."]
    }
  ]
}

Database:
Name: {name}
Description: {description}"""


_ARABIC_TRANSLATION_PROMPT = """\
Translate this English library question into natural Arabic as a student would ask it.
Return ONLY a JSON object: {"arabic": "...", "keywords_ar": ["term1", "term2", "term3"]}

Where keywords_ar are 2-3 key Arabic terms the answer should contain.

English question: {question}"""


def generate_faq_questions(faqs: list, count: int) -> list:
    import random
    random.seed(42)
    sample = random.sample(faqs, min(count, len(faqs)))
    generated = []

    for i, faq in enumerate(sample):
        logger.info(f"Generating FAQ questions {i+1}/{len(sample)}: {faq['question'][:60]}")
        try:
            raw = chat_completion(
                messages=[{
                    "role": "user",
                    "content": _FAQ_GENERATION_PROMPT.format(
                        question=faq["question"],
                        answer=faq["answer"],
                    ),
                }],
                max_tokens=600,
                temperature=0.3,
            )
            parsed = json.loads(raw)
            for j, q in enumerate(parsed.get("questions", [])[:3]):
                generated.append({
                    "id": f"gen_faq_{i:03d}_{j}",
                    "question": q["question"],
                    "category": "faq_direct",
                    "language": "en",
                    "difficulty": q.get("difficulty", "medium"),
                    "expected_behavior": "answer",
                    "expected_source_type": "faq",
                    "answer_keywords": q.get("answer_keywords", []),
                    "notes": f"Generated from FAQ: '{faq['question'][:60]}'",
                })
        except Exception as e:
            logger.warning(f"FAQ generation failed for entry {i}: {e}")

    return generated


def generate_db_questions(dbs: list, count: int) -> list:
    import random
    random.seed(42)
    sample = random.sample(dbs, min(count, len(dbs)))
    generated = []

    for i, db in enumerate(sample):
        logger.info(f"Generating DB questions {i+1}/{len(sample)}: {db['name'][:50]}")
        try:
            raw = chat_completion(
                messages=[{
                    "role": "user",
                    "content": _DB_GENERATION_PROMPT.format(
                        name=db["name"],
                        description=db["description"],
                    ),
                }],
                max_tokens=400,
                temperature=0.3,
            )
            parsed = json.loads(raw)
            for j, q in enumerate(parsed.get("questions", [])[:2]):
                generated.append({
                    "id": f"gen_db_{i:03d}_{j}",
                    "question": q["question"],
                    "category": "database_recommendation",
                    "language": "en",
                    "difficulty": q.get("difficulty", "medium"),
                    "expected_behavior": "answer",
                    "expected_source_type": "database",
                    "answer_keywords": q.get("answer_keywords", [db["name"].lower()[:20]]),
                    "notes": f"Generated from database: '{db['name']}'",
                })
        except Exception as e:
            logger.warning(f"DB generation failed for {db['name']}: {e}")

    return generated


def generate_arabic_translations(questions: list, count: int) -> list:
    """Translate a sample of English questions to Arabic for bilingual parity testing."""
    import random
    random.seed(7)
    sample = random.sample(questions, min(count, len(questions)))
    arabic_questions = []

    for i, q in enumerate(sample):
        logger.info(f"Translating to Arabic {i+1}/{len(sample)}: {q['question'][:60]}")
        try:
            raw = chat_completion(
                messages=[{
                    "role": "user",
                    "content": _ARABIC_TRANSLATION_PROMPT.format(question=q["question"]),
                }],
                max_tokens=200,
                temperature=0.1,
            )
            parsed = json.loads(raw)
            arabic_questions.append({
                "id": f"gen_ar_{i:03d}",
                "question": parsed["arabic"],
                "category": "arabic",
                "language": "ar",
                "difficulty": q["difficulty"],
                "expected_behavior": q["expected_behavior"],
                "expected_source_type": q["expected_source_type"],
                "answer_keywords": parsed.get("keywords_ar", []),
                "notes": f"Arabic translation of: '{q['question'][:60]}'",
                "en_pair_id": q["id"],
            })
        except Exception as e:
            logger.warning(f"Arabic translation failed for q {i}: {e}")

    return arabic_questions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate grounded golden-set questions")
    parser.add_argument("--faq-count", type=int, default=15,
                        help="Number of FAQ entries to generate questions from (default: 15)")
    parser.add_argument("--db-count", type=int, default=15,
                        help="Number of database entries to generate questions from (default: 15)")
    parser.add_argument("--arabic-count", type=int, default=10,
                        help="Number of English questions to translate to Arabic (default: 10)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: data/golden_set_generated.json)")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set.")
        sys.exit(1)

    faq_path = os.path.join(DATA_DIR, "library_faq_clean.csv")
    db_path = os.path.join(DATA_DIR, "Databases description.csv")
    output_path = args.output or os.path.join(DATA_DIR, "golden_set_generated.json")

    faqs = load_faqs(faq_path)
    dbs = load_databases(db_path)
    logger.info(f"Loaded {len(faqs)} FAQs and {len(dbs)} database descriptions")

    faq_questions = generate_faq_questions(faqs, args.faq_count)
    logger.info(f"Generated {len(faq_questions)} FAQ questions")

    db_questions = generate_db_questions(dbs, args.db_count)
    logger.info(f"Generated {len(db_questions)} database questions")

    all_english = faq_questions + db_questions
    arabic_questions = generate_arabic_translations(all_english, args.arabic_count)
    logger.info(f"Generated {len(arabic_questions)} Arabic questions")

    all_questions = all_english + arabic_questions

    output = {
        "version": "1.0",
        "created": datetime.utcnow().strftime("%Y-%m-%d"),
        "description": (
            "Auto-generated golden-set questions from actual source documents. "
            "Review and merge into data/golden_set.json after validation."
        ),
        "total": len(all_questions),
        "category_counts": {
            "faq_direct": len(faq_questions),
            "database_recommendation": len(db_questions),
            "arabic": len(arabic_questions),
        },
        "questions": all_questions,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(all_questions)} generated questions to {output_path}")
    print(f"\nGenerated {len(all_questions)} questions → {output_path}")
    print("Review the file, then merge approved questions into data/golden_set.json")


if __name__ == "__main__":
    main()
