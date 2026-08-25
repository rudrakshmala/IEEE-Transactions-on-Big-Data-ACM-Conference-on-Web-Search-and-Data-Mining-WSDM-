from playwright.sync_api import sync_playwright


class BrowserManager:
    def __init__(self, headless=True):
        self.headless = headless
        self.playwright = None
        self.browser = None

    def start(self):
        self.playwright = sync_playwright().start()

        self.browser = self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--window-size=1366,768",
            ],
        )

        return self.browser

    def new_page(self):
        context = self.browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/127.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1366, "height": 768},
            locale="en-US",
            timezone_id="Asia/Singapore",
        )

        page = context.new_page()

        page.set_extra_http_headers(
            {
                "Accept-Language": "en-US,en;q=0.9",
                "Upgrade-Insecure-Requests": "1",
            }
        )

        return page

    def stop(self):
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()