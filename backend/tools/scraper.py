import random
import re
import time
import requests
from bs4 import BeautifulSoup
from utils.logger import get_logger

logger = get_logger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
]

PLATFORM_KEYWORDS = {
    "amazon":   ["amazon"],
    "flipkart": ["flipkart"],
    "croma":    ["croma"],
    "reliance": ["reliance", "jiomart"],
    "meesho":   ["meesho"],
    "myntra":   ["myntra"],
}


def get_headers(referer: str = "") -> dict:
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-IN,en-US;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0",
    }
    if referer:
        headers["Referer"] = referer
    return headers


def fetch_page(url: str, timeout: int = 15, delay: float = 1.5, referer: str = "") -> BeautifulSoup:
    time.sleep(delay)
    logger.info(f"Fetching: {url}")
    session = requests.Session()
    session.headers.update(get_headers(referer))
    response = session.get(url, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def parse_price(text: str) -> float:
    cleaned = re.sub(r"[₹,\s\xa0]", "", str(text))
    match = re.search(r"\d+\.?\d*", cleaned)
    if match:
        val = float(match.group())
        if val > 10:   # sanity: skip single-digit noise
            return val
    return 0.0


def normalize_platform(store: str) -> str:
    store_lower = store.lower()
    for platform, keywords in PLATFORM_KEYWORDS.items():
        if any(kw in store_lower for kw in keywords):
            return platform
    return "other"
