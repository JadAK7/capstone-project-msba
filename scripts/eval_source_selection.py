#!/usr/bin/env python3
"""
eval_source_selection.py
Evaluation suite for multi-source RAG retrieval quality.

Tests whether the chatbot correctly selects the highest-priority source
when multiple sources are relevant, and correctly falls back to lower-priority
sources when they have better evidence.

Usage:
    python scripts/eval_source_selection.py                   # Run all tests
    python scripts/eval_source_selection.py --category hours  # Run specific category
    python scripts/eval_source_selection.py --verbose         # Show full chunk details
    python scripts/eval_source_selection.py --dry-run         # Show test cases only

Requires: PostgreSQL running with indexed data, OPENAI_API_KEY set.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.source_config import FACULTY_TEXT, SCRAPED_WEBSITE, FACULTY_FAQ, DATABASES


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    """A diagnostic test case for source selection evaluation."""
    query: str
    expected_source: str              # Expected winning source_type
    category: str                     # Test category for filtering
    description: str                  # Why this source should win
    accept_alternatives: List[str] = field(default_factory=list)  # Also acceptable sources


# Diagnostic test questions organized by expected source
TEST_CASES = [
    # --- Questions that should prefer faculty_text ---
    TestCase(
        query="What is the library's policy on borrowing books?",
        expected_source=FACULTY_TEXT,
        category="policy",
        description="Policy questions should prefer faculty-curated text over scraped pages",
        accept_alternatives=[SCRAPED_WEBSITE],
    ),
    TestCase(
        query="What services does the library offer for graduate students?",
        expected_source=FACULTY_TEXT,
        category="services",
        description="Service descriptions curated by faculty should take priority",
        accept_alternatives=[SCRAPED_WEBSITE],
    ),
    TestCase(
        query="How does interlibrary loan work at AUB?",
        expected_source=FACULTY_TEXT,
        category="services",
        description="Detailed process explanations should prefer faculty text",
        accept_alternatives=[SCRAPED_WEBSITE, FACULTY_FAQ],
    ),
    TestCase(
        query="What are the rules for using study rooms?",
        expected_source=FACULTY_TEXT,
        category="policy",
        description="Rules/policies should come from faculty-authored content",
        accept_alternatives=[SCRAPED_WEBSITE],
    ),

    # --- Questions that should prefer scraped_website ---
    TestCase(
        query="What are the current library opening hours?",
        expected_source=SCRAPED_WEBSITE,
        category="hours",
        description="Current hours should come from the live website (freshest source)",
        accept_alternatives=[FACULTY_FAQ],
    ),
    TestCase(
        query="When does Jafet Library close today?",
        expected_source=SCRAPED_WEBSITE,
        category="hours",
        description="Time-sensitive info should favor scraped website",
        accept_alternatives=[FACULTY_FAQ],
    ),
    TestCase(
        query="Where is the library located on campus?",
        expected_source=SCRAPED_WEBSITE,
        category="location",
        description="Location/directions info is typically on the website",
        accept_alternatives=[FACULTY_FAQ],
    ),
    TestCase(
        query="What events are happening at the library?",
        expected_source=SCRAPED_WEBSITE,
        category="events",
        description="Events/news are dynamic content from the website",
        accept_alternatives=[FACULTY_TEXT],
    ),

    # --- Questions that should prefer faculty_faq ---
    TestCase(
        query="How can I reserve a book?",
        expected_source=FACULTY_FAQ,
        category="faq",
        description="Direct FAQ-style question that may have an exact FAQ match",
        accept_alternatives=[FACULTY_TEXT, SCRAPED_WEBSITE],
    ),
    TestCase(
        query="Can I borrow DVDs from the library?",
        expected_source=FACULTY_FAQ,
        category="faq",
        description="Simple yes/no FAQ question",
        accept_alternatives=[FACULTY_TEXT, SCRAPED_WEBSITE],
    ),
    TestCase(
        query="How do I connect to the library WiFi?",
        expected_source=FACULTY_FAQ,
        category="faq",
        description="Common FAQ that likely has a direct match",
        accept_alternatives=[FACULTY_TEXT, SCRAPED_WEBSITE],
    ),

    # --- Questions where faculty_text should beat FAQ ---
    TestCase(
        query="Explain the library's lending policy in detail",
        expected_source=FACULTY_TEXT,
        category="priority_test",
        description="Detailed explanation should prefer faculty text over short FAQ answer",
        accept_alternatives=[SCRAPED_WEBSITE],
    ),
    TestCase(
        query="What are all the different types of library memberships and their benefits?",
        expected_source=FACULTY_TEXT,
        category="priority_test",
        description="Comprehensive question should prefer detailed faculty text",
        accept_alternatives=[SCRAPED_WEBSITE],
    ),

    # --- Questions where scraped should beat faculty_text (freshness) ---
    TestCase(
        query="What time does the library open during Ramadan?",
        expected_source=SCRAPED_WEBSITE,
        category="freshness_test",
        description="Seasonal/temporary hours should come from website",
        accept_alternatives=[FACULTY_FAQ],
    ),
    TestCase(
        query="Is the library open during the holiday break?",
        expected_source=SCRAPED_WEBSITE,
        category="freshness_test",
        description="Holiday hours are dynamic info from website",
        accept_alternatives=[FACULTY_FAQ],
    ),

    # --- Database questions (should route to databases) ---
    TestCase(
        query="Which database should I use for engineering research?",
        expected_source=DATABASES,
        category="database",
        description="Database recommendation should use databases source",
        accept_alternatives=[],
    ),
    TestCase(
        query="I need to find articles about machine learning",
        expected_source=DATABASES,
        category="database",
        description="Research article queries should route to databases",
        accept_alternatives=[],
    ),
]


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    """Result of running a single test case."""
    test: TestCase
    actual_source: str
    chosen_source_label: str
    passed: bool
    score_by_source: dict
    top_chunks: List[dict]
    selection_reason: str
    response_time_ms: float


def run_test(test: TestCase, bot, verbose: bool = False) -> TestResult:
    """Run a single test case and evaluate source selection."""
    start = time.time()
    answer, debug = bot.answer(test.query)
    elapsed_ms = (time.time() - start) * 1000

    chosen_source = debug.get("chosen_source", "unknown")
    retrieved_chunks = debug.get("retrieved_chunks", [])
    best_by_source = debug.get("best_by_source", {})
    selection_reason = debug.get("source_selection_reason", "")

    # Determine actual source type from chosen_source label
    # Map back from label to source type
    label_to_source = {
        "faculty_text (admin)": FACULTY_TEXT,
        "custom notes (admin)": FACULTY_TEXT,
        "library pages (scraped)": SCRAPED_WEBSITE,
        "FAQ": FACULTY_FAQ,
        "database (keyword intent)": DATABASES,
        "database (semantic)": DATABASES,
    }
    actual_source = label_to_source.get(chosen_source, "unknown")

    # Also check from the top chunk's source_type
    if actual_source == "unknown" and retrieved_chunks:
        actual_source = retrieved_chunks[0].get("source_type", "unknown")

    # Check if the result matches expected source
    passed = (actual_source == test.expected_source) or (actual_source in test.accept_alternatives)

    result = TestResult(
        test=test,
        actual_source=actual_source,
        chosen_source_label=chosen_source,
        passed=passed,
        score_by_source=best_by_source,
        top_chunks=retrieved_chunks[:5],
        selection_reason=selection_reason,
        response_time_ms=elapsed_ms,
    )

    if verbose:
        _print_test_detail(result)

    return result


def _print_test_detail(result: TestResult):
    """Print detailed info for a single test result."""
    status = "PASS" if result.passed else "FAIL"
    print(f"\n{'='*80}")
    print(f"[{status}] {result.test.query}")
    print(f"  Category: {result.test.category}")
    print(f"  Expected: {result.test.expected_source} | Actual: {result.actual_source}")
    print(f"  Chosen label: {result.chosen_source_label}")
    print(f"  Reason: {result.test.description}")
    print(f"  Selection reason: {result.selection_reason}")
    print(f"  Time: {result.response_time_ms:.0f}ms")
    print(f"  Scores by source: {json.dumps(result.score_by_source, indent=2)}")
    if result.top_chunks:
        print(f"  Top chunks:")
        for i, chunk in enumerate(result.top_chunks[:3]):
            print(
                f"    [{i}] source={chunk.get('source_type', '?')} "
                f"score={chunk.get('score', 0):.3f} "
                f"(raw={chunk.get('raw_rerank_score', 0):.3f} "
                f"+boost={chunk.get('source_boost', 0):.3f}) "
                f"| {chunk.get('page_title', '?')[:50]}"
            )


def run_evaluation(
    categories: Optional[List[str]] = None,
    verbose: bool = False,
    dry_run: bool = False,
):
    """Run the full evaluation suite."""
    tests = TEST_CASES
    if categories:
        tests = [t for t in tests if t.category in categories]

    if not tests:
        print("No test cases match the specified categories.")
        return

    print(f"\n{'='*80}")
    print(f"SOURCE SELECTION EVALUATION SUITE")
    print(f"{'='*80}")
    print(f"Test cases: {len(tests)}")
    if categories:
        print(f"Categories: {', '.join(categories)}")
    print()

    if dry_run:
        for t in tests:
            print(f"  [{t.category}] {t.query}")
            print(f"    Expected: {t.expected_source} (also OK: {t.accept_alternatives})")
            print(f"    Reason: {t.description}")
            print()
        return

    # Initialize chatbot
    from dotenv import load_dotenv
    load_dotenv()
    from backend.database import init_db
    from backend.chatbot import LibraryChatbot

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    init_db()
    bot = LibraryChatbot(api_key=api_key)

    results = []
    for i, test in enumerate(tests):
        print(f"[{i+1}/{len(tests)}] Testing: {test.query[:60]}...")
        result = run_test(test, bot, verbose=verbose)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"  -> {status} (expected={test.expected_source}, got={result.actual_source})")

    # Print summary
    _print_summary(results)


def _print_summary(results: List[TestResult]):
    """Print evaluation summary with metrics."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    print(f"\n{'='*80}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total: {total} | Passed: {passed} | Failed: {failed} | Accuracy: {passed/total*100:.1f}%")

    # By category
    categories = set(r.test.category for r in results)
    print(f"\nBy category:")
    for cat in sorted(categories):
        cat_results = [r for r in results if r.test.category == cat]
        cat_passed = sum(1 for r in cat_results if r.passed)
        print(f"  {cat}: {cat_passed}/{len(cat_results)} passed")

    # By expected source
    sources = set(r.test.expected_source for r in results)
    print(f"\nBy expected source:")
    for src in sorted(sources):
        src_results = [r for r in results if r.test.expected_source == src]
        src_passed = sum(1 for r in src_results if r.passed)
        print(f"  {src}: {src_passed}/{len(src_results)} passed")

    # Source distribution in actual results
    actual_dist = {}
    for r in results:
        actual_dist[r.actual_source] = actual_dist.get(r.actual_source, 0) + 1
    print(f"\nActual source distribution:")
    for src, count in sorted(actual_dist.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count} ({count/total*100:.1f}%)")

    # List failures
    failures = [r for r in results if not r.passed]
    if failures:
        print(f"\nFailed tests:")
        for r in failures:
            print(f"  [{r.test.category}] {r.test.query}")
            print(f"    Expected: {r.test.expected_source} | Got: {r.actual_source}")
            print(f"    Scores: {json.dumps(r.score_by_source)}")
            if r.selection_reason:
                print(f"    Reason: {r.selection_reason}")

    # Average response time
    avg_time = sum(r.response_time_ms for r in results) / len(results) if results else 0
    print(f"\nAverage response time: {avg_time:.0f}ms")

    # Override rate: how often did a lower-priority source override a higher-priority one?
    override_count = 0
    priority = {FACULTY_TEXT: 3, SCRAPED_WEBSITE: 2, FACULTY_FAQ: 1, DATABASES: 0}
    for r in results:
        expected_p = priority.get(r.test.expected_source, 0)
        actual_p = priority.get(r.actual_source, 0)
        if actual_p < expected_p and not r.passed:
            override_count += 1
    print(f"Incorrect low-priority overrides: {override_count}/{total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate source selection quality")
    parser.add_argument("--category", "-c", nargs="+", help="Filter by category")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--dry-run", "-d", action="store_true", help="Show test cases only")
    args = parser.parse_args()

    run_evaluation(
        categories=args.category,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )
