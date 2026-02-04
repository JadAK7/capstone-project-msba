"""
AUB Library Website Scraper
Scrapes all pages from the AUB library website using Playwright (headless browser)
so that JavaScript-rendered content (e.g. opening hours) is captured.
Stores results in the vector database.
"""

import os
import re
import time
from urllib.parse import urljoin, urlparse
from collections import deque
import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from playwright.sync_api import sync_playwright

# Configuration
START_URLS = [
    "https://www.aub.edu.lb/libraries",
    "https://www.aub.edu.lb/Libraries",
]
ALLOWED_DOMAIN = "aub.edu.lb"
ALLOWED_PATHS = ["/libraries", "/Libraries"]  # Both casings used on the site
MAX_PAGES = 500  # Safety limit
CRAWL_DELAY = 1  # Seconds between requests (be respectful)
JS_WAIT_MS = 3000  # Time to wait for JS to render after page load
BATCH_SIZE = 200  # For embedding generation
OUTPUT_EMB_FILE = "library_pages_emb.npy"
OUTPUT_TEXT_FILE = "library_pages_text.parquet"

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


class AUBLibraryScraper:
    def __init__(self, start_urls, allowed_domain, allowed_paths):
        self.start_urls = start_urls
        self.allowed_domain = allowed_domain
        self.allowed_paths = allowed_paths
        self.visited = set()
        self.to_visit = deque(start_urls)
        self.scraped_data = []

    def is_valid_url(self, url):
        """Check if URL should be crawled."""
        parsed = urlparse(url)

        # Must be same domain
        if self.allowed_domain not in parsed.netloc:
            return False

        # Must start with allowed path
        if not any(parsed.path.startswith(path) for path in self.allowed_paths):
            return False

        # Skip files
        skip_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
                          '.zip', '.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mp3']
        if any(parsed.path.lower().endswith(ext) for ext in skip_extensions):
            return False

        return True

    def extract_text(self, page):
        """Extract meaningful text from rendered page using Playwright."""
        # Remove non-content elements before extracting text
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
        # Clean up whitespace
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
        """Scrape a single page using Playwright."""
        try:
            page.goto(url, timeout=30000, wait_until="networkidle")
            # Extra wait for any late JS rendering (e.g. hours widgets)
            page.wait_for_timeout(JS_WAIT_MS)

            # Extract title
            title = page.title() or url

            # Extract links before modifying the DOM
            links = self.extract_links(page, url)

            # Extract main content (this modifies the DOM by removing elements)
            text = self.extract_text(page)

            # Only store if we got meaningful content
            if len(text) > 100:
                self.scraped_data.append({
                    'url': url,
                    'title': title,
                    'content': text[:5000]
                })

            return links

        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return []

    def crawl(self, max_pages=MAX_PAGES):
        """Crawl the website using BFS with a headless browser."""
        print(f"Starting crawl from {self.start_urls}")
        print(f"Max pages: {max_pages}")

        pbar = tqdm(total=max_pages, desc="Crawling pages")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            while self.to_visit and len(self.visited) < max_pages:
                url = self.to_visit.popleft()

                if url in self.visited:
                    continue

                self.visited.add(url)
                pbar.update(1)
                pbar.set_postfix({'current': url[:50] + '...' if len(url) > 50 else url})

                # Scrape the page
                new_links = self.scrape_page(page, url)

                # Add new links to queue
                for link in new_links:
                    if link not in self.visited:
                        self.to_visit.append(link)

                # Be respectful - delay between requests
                time.sleep(CRAWL_DELAY)

            browser.close()

        pbar.close()
        print(f"\nCrawl complete!")
        print(f"Pages visited: {len(self.visited)}")
        print(f"Pages with content: {len(self.scraped_data)}")

        return self.scraped_data


def sanitize_matrix(matrix):
    """Sanitize embedding matrix by handling NaN/Inf and normalizing."""
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.clip(matrix, -10, 10)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    matrix = matrix / norms

    return matrix


def embed_batch(texts, batch_size=BATCH_SIZE):
    """Generate embeddings for a list of texts using OpenAI API."""
    embeddings = []

    print(f"Generating embeddings for {len(texts)} texts...")

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i + batch_size]

        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )

            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)

        except Exception as e:
            print(f"Error generating embeddings for batch {i//batch_size}: {str(e)}")
            embeddings.extend([[0.0] * 1536] * len(batch))

    return np.array(embeddings, dtype=np.float32)


def build_library_index(scraped_data):
    """Build vector index from scraped library pages."""
    print("\n" + "="*60)
    print("Building Library Pages Index")
    print("="*60)

    df = pd.DataFrame(scraped_data)
    print(f"\nTotal pages scraped: {len(df)}")

    texts_to_embed = []
    for _, row in df.iterrows():
        combined = f"{row['title']}\n\n{row['content']}"
        texts_to_embed.append(combined)

    embeddings = embed_batch(texts_to_embed)
    embeddings = sanitize_matrix(embeddings)

    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Embeddings dtype: {embeddings.dtype}")
    print(f"Embeddings range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")

    np.save(OUTPUT_EMB_FILE, embeddings)
    print(f"\n✓ Saved embeddings to {OUTPUT_EMB_FILE}")

    df.to_parquet(OUTPUT_TEXT_FILE, index=False)
    print(f"✓ Saved text data to {OUTPUT_TEXT_FILE}")

    print("\n" + "="*60)
    print("Index building complete!")
    print("="*60)

    return embeddings, df


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

    embeddings, df = build_library_index(scraped_data)

    print("\n✅ All done!")
    print(f"\nTo use these embeddings in your RAG system:")
    print(f"1. The embeddings are saved in: {OUTPUT_EMB_FILE}")
    print(f"2. The text data is saved in: {OUTPUT_TEXT_FILE}")
    print(f"3. Update your chat.py or app.py to load these files alongside FAQ and Database indices")

    print("\n" + "="*60)
    print("Sample of scraped pages:")
    print("="*60)
    for i, row in df.head(5).iterrows():
        print(f"\n{i+1}. {row['title']}")
        print(f"   URL: {row['url']}")
        print(f"   Content preview: {row['content'][:100]}...")


if __name__ == "__main__":
    main()
