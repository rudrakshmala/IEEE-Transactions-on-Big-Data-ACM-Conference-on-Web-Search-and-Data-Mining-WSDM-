import random
import time
from pathlib import Path

from crawler.discovery.browser import BrowserManager
from crawler.discovery.config import CATEGORY_URLS, HEADLESS


def crawl():
    browser_manager = BrowserManager(headless=HEADLESS)
    browser = browser_manager.start()
    page = browser_manager.new_page()

    debug_dir = Path("/tmp")
    debug_dir.mkdir(parents=True, exist_ok=True)

    for url in CATEGORY_URLS:
        print(f"Crawling: {url}")

        try:
            page.goto(url, wait_until="load", timeout=120000)

            # Conservative wait
            time.sleep(random.uniform(4, 7))

            # Gentle scrolling
            for _ in range(5):
                page.mouse.wheel(0, random.randint(800, 1800))
                time.sleep(random.uniform(0.8, 1.8))

            title = page.title()
            html = page.content()

            debug_file = debug_dir / f"{int(time.time())}.html"
            debug_file.write_text(html, encoding="utf-8")

            print(f"Page title: {title}")
            print(f"Rendered HTML size: {len(html)} bytes")
            print(f"Saved debug HTML to {debug_file}")

            if "captcha" in title.lower():
                print("CAPTCHA detected - backing off and skipping this URL")
                time.sleep(random.uniform(30, 60))
                continue

            links = page.evaluate(
                """
                () => [...new Set(
                    Array.from(document.querySelectorAll('a'))
                        .map(a => a.href)
                        .filter(Boolean)
                )]
                """
            )

            product_links = [h for h in links if "/item/" in h]
            print(f"Found {len(product_links)} product links")

        except Exception as e:
            print(f"Crawler error: {repr(e)}")

        # Random delay between URLs
        time.sleep(random.uniform(10, 20))

    browser_manager.stop()
    print("Finished crawl")


if __name__ == "__main__":
    crawl()