#!/usr/bin/env python3
"""
run_evaluation.py
Standalone evaluation pipeline for the AUB Libraries Assistant chatbot.

Usage:
    # Evaluate with default test questions
    python scripts/run_evaluation.py

    # Evaluate with custom questions file (one question per line)
    python scripts/run_evaluation.py --questions my_questions.txt

    # Evaluate with a specific number of questions
    python scripts/run_evaluation.py --limit 10

    # Output to a specific file
    python scripts/run_evaluation.py --output results/eval_results.json

Output: JSON file with per-question metrics + aggregate scores.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from backend.database import init_db, close_db
from backend.chatbot import LibraryChatbot
from backend.evaluation import run_evaluation_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Default test questions covering all chatbot capabilities
DEFAULT_QUESTIONS = [
    # FAQ questions
    "What are the library opening hours?",
    "How can I reserve a book?",
    "Are there any fines on overdue items?",
    "Can I access online resources from off-campus?",
    "How do I cite my sources?",
    "Where can I find previous exams?",
    "How do I borrow a book, and for how long?",
    "Can I print, scan and photocopy documents in the Library?",
    "How can I renew borrowed books?",
    "Can AUB alumni use AUB libraries?",
    # Database recommendation questions
    "Which database should I use for engineering articles?",
    "Where can I find medical research papers?",
    "Best database for computer science research?",
    "What databases are available for business studies?",
    "I need to find IEEE papers on machine learning",
    # Arabic questions
    "ما هي ساعات عمل المكتبة؟",
    "أين أبحث عن مقالات علمية؟",
    "كيف أحجز كتاباً من المكتبة؟",
    # Edge cases
    "How do I get help with my research?",
    "What services does the library offer?",
]


def load_questions(filepath: str) -> list:
    """Load questions from a text file (one per line)."""
    questions = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                questions.append(line)
    return questions


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation pipeline")
    parser.add_argument(
        "--questions", type=str, default=None,
        help="Path to a text file with one question per line",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of questions to evaluate",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path (default: eval_results_<timestamp>.json)",
    )
    parser.add_argument(
        "--language", type=str, default=None,
        help="Force language (en/ar). Default: auto-detect",
    )
    args = parser.parse_args()

    # Load questions
    if args.questions:
        questions = load_questions(args.questions)
        logger.info(f"Loaded {len(questions)} questions from {args.questions}")
    else:
        questions = DEFAULT_QUESTIONS
        logger.info(f"Using {len(questions)} default test questions")

    if args.limit:
        questions = questions[:args.limit]
        logger.info(f"Limited to {len(questions)} questions")

    # Output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            f"eval_results_{timestamp}.json",
        )

    # Initialize
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.error("OPENAI_API_KEY not set. Export it or add to .env file.")
        sys.exit(1)

    logger.info("Initializing database connection...")
    init_db()

    logger.info("Initializing chatbot...")
    chatbot = LibraryChatbot(api_key)

    # Run evaluation
    logger.info(f"Starting evaluation of {len(questions)} questions...")
    results = run_evaluation_pipeline(
        questions=questions,
        chatbot=chatbot,
        language=args.language,
    )

    # Add metadata
    results["metadata"] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "num_questions": len(questions),
        "language_override": args.language,
        "questions_source": args.questions or "default",
    }

    # Save results
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")

    # Print summary
    agg = results["aggregate"]
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Questions evaluated:   {agg['total_questions']}")
    print(f"  Successful:            {agg['successful_evaluations']}")
    print(f"  ---")
    print(f"  Groundedness:          {agg['groundedness']:.4f}")
    print(f"  Faithfulness:          {agg['faithfulness']:.4f}")
    print(f"  Context Relevance:     {agg['context_relevance']:.4f}")
    print(f"  Answer Relevance:      {agg['answer_relevance']:.4f}")
    print(f"  Citation Accuracy:     {agg['citation_accuracy']:.4f}")
    print(f"  Hallucination Rate:    {agg['hallucination_rate']:.4f}")
    print(f"  ---")
    print(f"  GROUNDING SCORE:       {agg['grounding_score']:.4f}")
    print(f"  (weights: {agg['grounding_weights']})")
    print("=" * 60)

    close_db()


if __name__ == "__main__":
    main()
