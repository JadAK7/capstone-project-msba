"""
AUB Library Website Scraper
Scrapes all pages from the AUB library website using Playwright (headless browser)
so that JavaScript-rendered content (e.g. opening hours) is captured.
Stores results in ChromaDB vector database.
"""

import os
import re
import time
from urllib.parse import urlparse
from collections import deque
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from tqdm import tqdm
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
CHROMA_DIR = "./chroma_db"
LIBRARY_COLLECTION = "library_pages"
EMBEDDING_MODEL = "text-embedding-3-small"


class AUBLibraryScraper:
    def __init__(self, start_urls, allowed_domain, allowed_paths):
        self.start_urls = start_urls
        self.allowed_domain = allowed_domain
        self.allowed_paths = allowed_paths
        self.visited = set()
        self.to_visit = deque(start_urls)
        self.scraped_data = []
        self.failed_pages = []  # Track failed pages

    def is_valid_url(self, url):
        """Check if URL should be crawled."""
        parsed = urlparse(url)

        if self.allowed_domain not in parsed.netloc:
            return False

        if not any(parsed.path.startswith(path) for path in self.allowed_paths):
            return False

        skip_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
                          '.zip', '.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mp3']
        if any(parsed.path.lower().endswith(ext) for ext in skip_extensions):
            return False

        return True

    def extract_text(self, page):
        """Extract meaningful text from rendered page using Playwright."""
        page.evaluate("""
            const selectors = [
                'script', 'style', 'nav', 'footer', 'header',
                '#useful-tools', '.useful-tools',
                '#s4-breadcrumb', '.breadcrumb', '.ms-breadcrumb',
                '#sideNavBox', '.ms-quickLaunch', '.ms-nav',
                '#DeltaTopNavigation', '.ms-topNavContainer'
            ];
            for (const sel of selectors) {
                for (const el of document.querySelectorAll(sel)) {
                    el.remove();
                }
            }
        """)

        text = page.evaluate("document.body.innerText")
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

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
                # Use domcontentloaded instead of networkidle to avoid SharePoint analytics timeout
                page.goto(url, timeout=PAGE_TIMEOUT, wait_until="domcontentloaded")
                page.wait_for_timeout(JS_WAIT_MS)

                title = page.title() or url
                links = self.extract_links(page, url)
                text = self.extract_text(page)

                if len(text) > 100:
                    self.scraped_data.append({
                        'url': url,
                        'title': title,
                        'content': text[:5000]
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
            # Set realistic user-agent to avoid bot detection
            context = browser.new_context(user_agent=USER_AGENT)
            page = context.new_page()

            while self.to_visit and len(self.visited) < max_pages:
                url = self.to_visit.popleft()

                if url in self.visited:
                    continue

                self.visited.add(url)
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
            print(f"\n⚠️  Failed pages: {len(self.failed_pages)}")
            print("\nFailed URLs:")
            for failure in self.failed_pages[:10]:  # Show first 10 failures
                print(f"  - {failure['url']}")
                print(f"    Error: {failure['error']}")
            if len(self.failed_pages) > 10:
                print(f"  ... and {len(self.failed_pages) - 10} more")
        else:
            print("\n✅ No failed pages!")

        return self.scraped_data


def build_library_index(scraped_data):
    """Build ChromaDB collection from scraped library pages."""
    print("\n" + "="*60)
    print("Building Library Pages Collection in ChromaDB")
    print("="*60)

    print(f"\nTotal pages scraped: {len(scraped_data)}")

    embedding_fn = OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name=EMBEDDING_MODEL,
    )
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection and recreate
    try:
        chroma_client.delete_collection(LIBRARY_COLLECTION)
    except (ValueError, Exception):
        pass

    collection = chroma_client.create_collection(
        name=LIBRARY_COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Upsert in batches
    batch_size = 50
    for i in tqdm(range(0, len(scraped_data), batch_size), desc="Upserting to ChromaDB"):
        batch = scraped_data[i:i + batch_size]

        ids = [f"page_{j}" for j in range(i, i + len(batch))]
        documents = [f"{item['title']}\n\n{item['content']}" for item in batch]
        metadatas = [
            {"url": item["url"], "title": item["title"], "content": item["content"]}
            for item in batch
        ]

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    count = collection.count()
    print(f"\nLibrary pages collection saved: {count} entries")

    print("\n" + "="*60)
    print("Collection building complete!")
    print("="*60)

    return collection


def main():
    """Main execution function."""
    print("="*60)
    print("AUB Library Website Scraper (Playwright)")
    print("="*60)

    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  Warning: OPENAI_API_KEY environment variable not set!")
        print("Please set it before running this script:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return

    scraper = AUBLibraryScraper(
        start_urls=START_URLS,
        allowed_domain=ALLOWED_DOMAIN,
        allowed_paths=ALLOWED_PATHS
    )

    scraped_data = scraper.crawl(max_pages=MAX_PAGES)

    if not scraped_data:
        print("\n⚠️  No data scraped! Please check the website URL and configuration.")
        return

    collection = build_library_index(scraped_data)

    print("\n✅ All done!")
    print(f"\nLibrary pages are stored in ChromaDB at: {CHROMA_DIR}")
    print(f"Collection '{LIBRARY_COLLECTION}' has {collection.count()} entries")

    print("\n" + "="*60)
    print("Sample of scraped pages:")
    print("="*60)
    for i, item in enumerate(scraped_data[:5]):
        print(f"\n{i+1}. {item['title']}")
        print(f"   URL: {item['url']}")
        print(f"   Content preview: {item['content'][:100]}...")


if __name__ == "__main__":
    main()
