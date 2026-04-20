"""
freshness_check.py
Automated freshness checker for the AUB Libraries Assistant.

Compares currently scraped content against live website content to detect
stale data. Can be run as a standalone cron job or triggered via API.

Usage:
  # Standalone (e.g., weekly cron job)
  python scripts/freshness_check.py

  # As a cron entry (every Sunday at 3 AM):
  0 3 * * 0 cd /path/to/project && python scripts/freshness_check.py

  # Dry-run mode (check only, don't rescrape)
  python scripts/freshness_check.py --dry-run

What it does:
  1. Samples a set of key pages from the stored document_chunks table
  2. Fetches the current live content for those pages
  3. Compares the live content against stored content (Jaccard similarity)
  4. If significant drift is detected (>20% of sampled pages changed),
     triggers a full rescrape + reindex
  5. Logs results for monitoring

This closes the "stale data" gap: if AUB updates their hours or policies,
the chatbot reflects the changes within the cron interval (e.g., weekly).
"""

import os
import sys
import logging
import hashlib
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Pages that matter most for freshness (hours, policies, services)
# These are checked first and weighted more heavily.
PRIORITY_PAGE_TYPES = ["hours_contact", "service", "policy", "faq"]

# If this fraction of sampled pages have changed, trigger a rescrape.
# "20% drift" = more than 20% of the sampled pages have word-level Jaccard
# similarity below CONTENT_CHANGE_THRESHOLD when comparing stored content
# against the current live website content.
DRIFT_THRESHOLD = 0.20  # 20% of pages changed

# Max pages to sample for freshness check (keeps it fast)
MAX_SAMPLE_PAGES = 20

# Word-level Jaccard similarity below this means the page content has changed.
# Jaccard = |words_stored ∩ words_live| / |words_stored ∪ words_live|
# 0.85 means at least 15% of the word vocabulary differs between stored and live.
CONTENT_CHANGE_THRESHOLD = 0.85

# Very short pages (< 200 chars) use a stricter threshold because small edits
# represent a large percentage change in Jaccard terms.
SHORT_PAGE_CONTENT_THRESHOLD = 200  # characters
SHORT_PAGE_CHANGE_THRESHOLD = 0.90  # stricter for short pages


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Word-level Jaccard similarity between two texts.

    J(A,B) = |A ∩ B| / |A ∪ B| where A and B are sets of lowercase words.
    Returns 1.0 for identical word sets, 0.0 for completely disjoint.
    This is the metric used for "drift detection" — a score below
    CONTENT_CHANGE_THRESHOLD means the page has changed significantly.
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return intersection / union if union > 0 else 0.0


def _content_hash(text: str) -> str:
    """Quick hash for content comparison."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def get_sample_pages() -> list:
    """Get a sample of stored pages to check for freshness.

    Prioritizes pages with time-sensitive page_types (hours, services, policies).
    """
    from backend.database import init_db, get_connection

    init_db()

    pages = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Get distinct page URLs, prioritizing important page types
            cur.execute("""
                SELECT DISTINCT ON (page_url) page_url, page_title, page_type,
                       chunk_text, created_at
                FROM document_chunks
                ORDER BY page_url,
                         CASE page_type
                           WHEN 'hours_contact' THEN 1
                           WHEN 'service' THEN 2
                           WHEN 'policy' THEN 3
                           WHEN 'faq' THEN 4
                           ELSE 5
                         END,
                         chunk_index
                LIMIT %s
            """, (MAX_SAMPLE_PAGES,))

            for row in cur.fetchall():
                pages.append({
                    "url": row[0],
                    "title": row[1],
                    "page_type": row[2],
                    "stored_text": row[3],
                    "indexed_at": row[4],
                })

    logger.info(f"Sampled {len(pages)} pages for freshness check")
    return pages


def fetch_live_content(url: str) -> str:
    """Fetch the current live content of a page.

    Uses requests (not Playwright) for speed. Falls back gracefully
    if the page requires JS rendering.
    """
    import requests

    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (compatible; AUBLibraryBot/1.0)"
        })
        resp.raise_for_status()

        # Extract text content (simple approach — good enough for diff detection)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise elements
        for tag in soup.find_all(["nav", "footer", "header", "script", "style"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return text

    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return ""


def check_freshness(dry_run: bool = False) -> dict:
    """Check if stored content is stale compared to live website.

    Returns a report dict with:
      - pages_checked: number of pages sampled
      - pages_changed: number that have significantly different content
      - drift_ratio: fraction of pages that changed
      - changed_pages: list of (url, title, similarity) for changed pages
      - action_taken: "rescrape_triggered" or "no_action" or "dry_run"
    """
    sample_pages = get_sample_pages()

    if not sample_pages:
        logger.warning("No pages in document_chunks to check. Skipping freshness check.")
        return {
            "pages_checked": 0,
            "pages_changed": 0,
            "drift_ratio": 0.0,
            "changed_pages": [],
            "action_taken": "no_pages",
            "checked_at": datetime.utcnow().isoformat(),
        }

    changed_pages = []
    checked = 0

    for page in sample_pages:
        url = page["url"]
        live_text = fetch_live_content(url)

        if not live_text:
            # Couldn't fetch — skip, don't count as changed
            continue

        checked += 1
        similarity = _jaccard_similarity(page["stored_text"], live_text)

        # Short pages use a stricter threshold: a small edit on a 100-word page
        # causes a bigger Jaccard drop than on a 1000-word page.
        is_short = len(page["stored_text"]) < SHORT_PAGE_CONTENT_THRESHOLD
        threshold = SHORT_PAGE_CHANGE_THRESHOLD if is_short else CONTENT_CHANGE_THRESHOLD

        if similarity < threshold:
            changed_pages.append({
                "url": url,
                "title": page["title"],
                "page_type": page["page_type"],
                "similarity": round(similarity, 3),
                "indexed_at": str(page["indexed_at"]),
            })
            logger.info(
                f"Content changed: {page['title'][:60]} "
                f"(similarity={similarity:.3f}, type={page['page_type']})"
            )

    drift_ratio = len(changed_pages) / checked if checked > 0 else 0.0

    report = {
        "pages_checked": checked,
        "pages_changed": len(changed_pages),
        "drift_ratio": round(drift_ratio, 3),
        "changed_pages": changed_pages,
        "checked_at": datetime.utcnow().isoformat(),
    }

    logger.info(
        f"Freshness check complete: {len(changed_pages)}/{checked} pages changed "
        f"(drift={drift_ratio:.1%}, threshold={DRIFT_THRESHOLD:.0%})"
    )

    # Decide whether to trigger rescrape
    if drift_ratio >= DRIFT_THRESHOLD and not dry_run:
        logger.info("Drift threshold exceeded — triggering automatic rescrape")
        report["action_taken"] = "rescrape_triggered"
        _trigger_rescrape()
    elif dry_run:
        report["action_taken"] = "dry_run"
        if drift_ratio >= DRIFT_THRESHOLD:
            logger.info("Drift threshold exceeded (dry run — no action taken)")
        else:
            logger.info("Content is fresh enough (dry run)")
    else:
        report["action_taken"] = "no_action"
        logger.info("Content is fresh enough — no rescrape needed")

    return report


def _trigger_rescrape():
    """Trigger a rescrape via the same pipeline used by the admin endpoint."""
    try:
        import requests
        # Try to hit the local API if running
        resp = requests.post("http://localhost:8000/api/admin/rescrape", timeout=10)
        if resp.status_code == 200:
            logger.info("Rescrape triggered via API successfully")
            return
        elif resp.status_code == 409:
            logger.info("Rescrape already in progress")
            return
    except requests.ConnectionError:
        logger.info("API not running — triggering rescrape directly")

    # Fallback: run scraper directly
    try:
        from scripts.scrape_aub_library import AUBLibraryScraper, \
            START_URLS, ALLOWED_DOMAIN, ALLOWED_PATHS, MAX_PAGES
        from backend.index_builder import IndexBuilder

        scraper = AUBLibraryScraper(
            start_urls=START_URLS,
            allowed_domain=ALLOWED_DOMAIN,
            allowed_paths=ALLOWED_PATHS,
        )
        scraped_data = scraper.crawl(max_pages=MAX_PAGES)
        if scraped_data:
            count = IndexBuilder.build_chunks_from_scraped(scraped_data)
            logger.info(f"Direct rescrape complete: {count} chunks indexed")
        else:
            logger.warning("Scraper returned no data")
    except Exception as e:
        logger.error(f"Direct rescrape failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check content freshness for AUB Library chatbot")
    parser.add_argument("--dry-run", action="store_true", help="Check only, don't trigger rescrape")
    args = parser.parse_args()

    report = check_freshness(dry_run=args.dry_run)

    print("\n=== Freshness Check Report ===")
    print(f"Pages checked:  {report['pages_checked']}")
    print(f"Pages changed:  {report['pages_changed']}")
    print(f"Drift ratio:    {report['drift_ratio']:.1%}")
    print(f"Action taken:   {report['action_taken']}")
    if report["changed_pages"]:
        print("\nChanged pages:")
        for p in report["changed_pages"]:
            print(f"  - [{p['page_type']}] {p['title'][:60]} (sim={p['similarity']:.3f})")
    print()
