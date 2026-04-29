"""
AUB Library Website Scraper
Scrapes pages from the AUB library website using requests + BeautifulSoup.
Stores results in PostgreSQL with pgvector embeddings.

Note: this is a static (non-JS-rendering) scraper. JavaScript-only content
(e.g. dynamically rendered opening hours) will not be captured here — those
should be sourced from the FAQ CSV or maintained as a custom_note.
"""

import os
import sys
import re
import time
from urllib.parse import urlparse, urljoin
from collections import deque

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.database import init_db, get_connection

START_URLS = [
    "https://www.aub.edu.lb/libraries",
    "https://www.aub.edu.lb/Libraries",
]
ALLOWED_DOMAIN = "aub.edu.lb"
ALLOWED_PATHS = ["/libraries", "/Libraries"]
MAX_PAGES = 500
CRAWL_DELAY = 1
PAGE_TIMEOUT = 30  # seconds for HTTP request
MAX_RETRIES = 3
RETRY_DELAY = 2
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"

NOISE_TAGS = ("script", "style", "noscript", "iframe", "svg")


class AUBLibraryScraper:
    def __init__(self, start_urls, allowed_domain, allowed_paths):
        self.start_urls = start_urls
        self.allowed_domain = allowed_domain
        self.allowed_paths = allowed_paths
        self.visited = set()
        self.visited_base = set()
        self.to_visit = deque(start_urls)
        self.scraped_data = []
        self.failed_pages = []
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    SKIP_PATH_PATTERNS = [
        "/newsletter", "/news/pages/",
    ]

    @staticmethod
    def normalize_url(url):
        """Strip query params and fragments so the same page isn't scraped twice."""
        return url.split("?")[0].split("#")[0].rstrip("/")

    def is_valid_url(self, url):
        parsed = urlparse(url)

        if self.allowed_domain not in parsed.netloc:
            return False

        if not any(parsed.path.startswith(path) for path in self.allowed_paths):
            return False

        path_lower = parsed.path.lower()

        skip_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
                          '.zip', '.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mp3',
                          '.css', '.js']
        if any(path_lower.endswith(ext) for ext in skip_extensions):
            return False

        if any(pat in path_lower for pat in self.SKIP_PATH_PATTERNS):
            return False

        if self.normalize_url(url) in self.visited_base:
            return False

        return True

    @staticmethod
    def extract_text_and_html(soup):
        """Return (text, html) from the parsed page.

        • text = innerText-equivalent with line breaks preserved.
        • html = body innerHTML for the downstream content extractor.
        """
        body = soup.body or soup

        for tag in body.find_all(NOISE_TAGS):
            tag.decompose()

        html = body.decode_contents()

        text = body.get_text(separator="\n")
        text = re.sub(r'[^\S\n]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        return text, html

    def extract_links(self, soup, current_url):
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            absolute = urljoin(current_url, href).split('#')[0]
            if self.is_valid_url(absolute) and absolute not in self.visited:
                links.append(absolute)
        return links

    def scrape_page(self, url):
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.get(url, timeout=PAGE_TIMEOUT)
                resp.raise_for_status()

                soup = BeautifulSoup(resp.text, "html.parser")

                title_tag = soup.find("title")
                title = (title_tag.string.strip() if title_tag and title_tag.string else url)

                links = self.extract_links(soup, url)
                text, html = self.extract_text_and_html(soup)

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
        print(f"Starting crawl from {self.start_urls}")
        print(f"Max pages: {max_pages}")

        pbar = tqdm(total=max_pages, desc="Crawling pages")

        while self.to_visit and len(self.visited) < max_pages:
            url = self.to_visit.popleft()

            if url in self.visited:
                continue

            self.visited.add(url)
            self.visited_base.add(self.normalize_url(url))
            pbar.update(1)
            pbar.set_postfix({'current': url[:50] + '...' if len(url) > 50 else url})

            new_links = self.scrape_page(url)

            for link in new_links:
                if link not in self.visited:
                    self.to_visit.append(link)

            time.sleep(CRAWL_DELAY)

        pbar.close()
        print(f"\nCrawl complete!")
        print(f"Pages visited: {len(self.visited)}")
        print(f"Pages with content: {len(self.scraped_data)}")

        if self.failed_pages:
            print(f"\nFailed pages: {len(self.failed_pages)}")
            print("\nFailed URLs:")
            for failure in self.failed_pages[:10]:
                print(f"  - {failure['url']}")
                print(f"    Error: {failure['error']}")
            if len(self.failed_pages) > 10:
                print(f"  ... and {len(self.failed_pages) - 10} more")
        else:
            print("\nNo failed pages!")

        return self.scraped_data


def build_library_index(scraped_data):
    """Build PostgreSQL tables from scraped data using the full chunk pipeline."""
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
    print("="*60)
    print("AUB Library Website Scraper (requests + BeautifulSoup)")
    print("="*60)

    if not os.environ.get("OPENAI_API_KEY"):
        print("\nWarning: OPENAI_API_KEY environment variable not set!")
        print("Please set it before running this script:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return

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
