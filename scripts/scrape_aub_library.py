
"""
AUB Library Website Scraper
Scrapes all pages from the AUB library website using Playwright (headless browser)
so that JavaScript-rendered content (e.g. opening hours) is captured.
Stores results in PostgreSQL with pgvector embeddings.
"""

import os
import sys
import re
import time
from urllib.parse import urlparse
from collections import deque
from tqdm import tqdm
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.database import init_db, get_connection

# Configuration
START_URLS = [
    "https://www.aub.edu.lb/libraries",
    "https://www.aub.edu.lb/Libraries",
]
ALLOWED_DOMAIN = "aub.edu.lb"
ALLOWED_PATHS = ["/libraries", "/Libraries"]
MAX_PAGES = 500
CRAWL_DELAY = 1
JS_WAIT_MS = 3000
PAGE_TIMEOUT = 60000  # 60 seconds for page navigation
MAX_RETRIES = 3  # Retry failed page loads
RETRY_DELAY = 2  # Seconds between retries
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"


class AUBLibraryScraper:
    def __init__(self, start_urls, allowed_domain, allowed_paths):
        self.start_urls = start_urls
        self.allowed_domain = allowed_domain
        self.allowed_paths = allowed_paths
        self.visited = set()
        self.visited_base = set()  # normalized URLs (no query params)
        self.to_visit = deque(start_urls)
        self.scraped_data = []
        self.failed_pages = []

    # Paths that produce massive HTML (newsletters, embedded PDFs, news articles)
    # but don't contain useful FAQ/hours/services content for the chatbot.
    SKIP_PATH_PATTERNS = [
        "/newsletter", "/news/pages/",
    ]

    @staticmethod
    def normalize_url(url):
        """Strip query params and fragments so the same page isn't scraped twice.

        SharePoint pages like ?Expand=0, ?Expand=1 are the same content.
        """
        return url.split("?")[0].split("#")[0].rstrip("/")

    def is_valid_url(self, url):
        """Check if URL should be crawled."""
        parsed = urlparse(url)

        if self.allowed_domain not in parsed.netloc:
            return False

        if not any(parsed.path.startswith(path) for path in self.allowed_paths):
            return False

        path_lower = parsed.path.lower()

        # Skip binary files
        skip_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
                          '.zip', '.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mp3',
                          '.css', '.js']
        if any(path_lower.endswith(ext) for ext in skip_extensions):
            return False

        # Skip paths that produce bloated noise (newsletters, news articles)
        if any(pat in path_lower for pat in self.SKIP_PATH_PATTERNS):
            return False

        # Deduplicate: if we've already visited the base URL (without query params),
        # don't visit it again with different params.
        if self.normalize_url(url) in self.visited_base:
            return False

        return True

    @staticmethod
    def _wait_for_stable_content(page, max_wait_ms=8000, interval_ms=500):
        """Wait until the page's visible text stops growing.

        Instead of a fixed sleep, this polls document.body.innerText.length
        and returns once the value is stable for two consecutive checks.
        Falls back to max_wait_ms if content never stabilises.
        """
        prev_len = 0
        stable_checks = 0
        elapsed = 0
        while elapsed < max_wait_ms:
            curr_len = page.evaluate("document.body.innerText.length")
            if curr_len == prev_len and curr_len > 0:
                stable_checks += 1
                if stable_checks >= 2:
                    return
            else:
                stable_checks = 0
            prev_len = curr_len
            page.wait_for_timeout(interval_ms)
            elapsed += interval_ms

    def extract_text_and_html(self, page):
        """Extract text and raw HTML from the rendered page.

        Returns (text, html).
        • text  = innerText with line breaks preserved (for fallback extraction).
        • html  = full innerHTML (for proper HTML-based extraction in
                  content_extractor.py — noise removal happens there, not here).
        """
        # Capture the FULL innerHTML *before* removing anything, so the
        # downstream extractor can apply its own generic noise-removal.
        html = page.evaluate("document.body.innerHTML")

        # For the text fallback, remove the noisiest elements in-browser first
        # so innerText is cleaner.  Keep this minimal — the extractor handles
        # the rest.
        page.evaluate("""
            for (const el of document.querySelectorAll(
                'script, style, noscript, iframe, svg'
            )) { el.remove(); }
        """)

        text = page.evaluate("document.body.innerText")
        # Preserve line breaks — only collapse horizontal whitespace
        text = re.sub(r'[^\S\n]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        return text, html

    def extract_links(self, page, current_url):
        """Extract all valid links from rendered page."""
        links = []
        hrefs = page.evaluate("""
            Array.from(document.querySelectorAll('a[href]'))
                 .map(a => a.href)
        """)
        for href in hrefs:
            url = href.split('#')[0]
            if self.is_valid_url(url) and url not in self.visited:
                links.append(url)
        return links

    def scrape_page(self, page, url):
        """Scrape a single page using Playwright with retry logic."""
        for attempt in range(MAX_RETRIES):
            try:
                page.goto(url, timeout=PAGE_TIMEOUT, wait_until="domcontentloaded")
                # Smart wait: poll until content stops growing instead of
                # a fixed sleep.  Handles both fast-loading static pages and
                # slow JS-rendered SharePoint content.
                self._wait_for_stable_content(page)

                title = page.title() or url
                links = self.extract_links(page, url)
                text, html = self.extract_text_and_html(page)

                # No truncation — let the extraction pipeline see everything.
                if len(text) > 100:
                    self.scraped_data.append({
                        'url': url,
                        'title': title,
                        'content': text,
                        'html': html,
                    })

                return links

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"Error scraping {url} (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
                    print(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"Error scraping {url} after {MAX_RETRIES} attempts: {str(e)}")
                    self.failed_pages.append({'url': url, 'error': str(e)})
                    return []

        return []

    def crawl(self, max_pages=MAX_PAGES):
        """Crawl the website using BFS with a headless browser."""
        print(f"Starting crawl from {self.start_urls}")
        print(f"Max pages: {max_pages}")

        pbar = tqdm(total=max_pages, desc="Crawling pages")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            # Set realistic user-agent and Beirut timezone so JS-rendered
            # times (e.g. opening hours) display as they appear on the site.
            context = browser.new_context(
                user_agent=USER_AGENT,
                timezone_id="Asia/Beirut",
            )
            page = context.new_page()

            while self.to_visit and len(self.visited) < max_pages:
                url = self.to_visit.popleft()

                if url in self.visited:
                    continue

                self.visited.add(url)
                self.visited_base.add(self.normalize_url(url))
                pbar.update(1)
                pbar.set_postfix({'current': url[:50] + '...' if len(url) > 50 else url})

                new_links = self.scrape_page(page, url)

                for link in new_links:
                    if link not in self.visited:
                        self.to_visit.append(link)

                time.sleep(CRAWL_DELAY)

            browser.close()

        pbar.close()
        print(f"\nCrawl complete!")
        print(f"Pages visited: {len(self.visited)}")
        print(f"Pages with content: {len(self.scraped_data)}")

        # Print failure summary
        if self.failed_pages:
            print(f"\nFailed pages: {len(self.failed_pages)}")
            print("\nFailed URLs:")
            for failure in self.failed_pages[:10]:  # Show first 10 failures
                print(f"  - {failure['url']}")
                print(f"    Error: {failure['error']}")
            if len(self.failed_pages) > 10:
                print(f"  ... and {len(self.failed_pages) - 10} more")
        else:
            print("\nNo failed pages!")

        return self.scraped_data


def build_library_index(scraped_data):
    """Build PostgreSQL tables from scraped data using the full chunk pipeline.

    Truncates and rebuilds both document_chunks and library_pages tables.
    """
    from backend.index_builder import IndexBuilder

    print("\n" + "="*60)
    print("Building Library Index (chunk pipeline)")
    print("="*60)

    print(f"\nTotal pages scraped: {len(scraped_data)}")
    count = IndexBuilder.build_chunks_from_scraped(scraped_data)

    print(f"\n{count} chunks indexed.")
    print("\n" + "="*60)
    print("Index building complete!")
    print("="*60)

    return count


def main():
    """Main execution function."""
    print("="*60)
    print("AUB Library Website Scraper (Playwright)")
    print("="*60)

    if not os.environ.get("OPENAI_API_KEY"):
        print("\nWarning: OPENAI_API_KEY environment variable not set!")
        print("Please set it before running this script:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return

    # Initialize database
    init_db()

    scraper = AUBLibraryScraper(
        start_urls=START_URLS,
        allowed_domain=ALLOWED_DOMAIN,
        allowed_paths=ALLOWED_PATHS
    )

    scraped_data = scraper.crawl(max_pages=MAX_PAGES)

    if not scraped_data:
        print("\nNo data scraped! Please check the website URL and configuration.")
        return

    count = build_library_index(scraped_data)

    print("\nAll done!")
    print(f"\nLibrary pages are stored in PostgreSQL")
    print(f"Table 'library_pages' has {count} entries")

    print("\n" + "="*60)
    print("Sample of scraped pages:")
    print("="*60)
    for i, item in enumerate(scraped_data[:5]):
        print(f"\n{i+1}. {item['title']}")
        print(f"   URL: {item['url']}")
        print(f"   Content preview: {item['content'][:100]}...")


if __name__ == "__main__":
    main()