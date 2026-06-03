import re
from urllib.parse import quote_plus
from typing import Optional
from bs4 import BeautifulSoup
from tools.scraper import fetch_page
from graph.state import AgentState
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ─── Google Search rich-snippet selectors (in priority order) ────────────────
RATING_SELECTORS = [
    "div.Aq14fc",
    "span.Aq14fc",
    "div.aRct8",
    "span.oqSTJd",
    "div.pVA7K",
    "span.rhsB",
    "g-review-stars [aria-label]",
]


def scrape_google_rating(product_name: str, timeout: int) -> Optional[float]:
    query = quote_plus(f"{product_name} reviews rating")
    url = f"https://www.google.com/search?q={query}&hl=en&gl=in"
    try:
        soup = fetch_page(url, timeout=timeout, delay=1.0, referer="https://www.google.com/")

        # Try structured selectors first
        for sel in RATING_SELECTORS:
            elem = soup.select_one(sel)
            if elem:
                aria = elem.get("aria-label", "")
                text = aria if aria else elem.get_text(strip=True)
                match = re.search(r"(\d+\.?\d*)", text)
                if match:
                    val = float(match.group(1))
                    if 1.0 <= val <= 5.0:
                        logger.info(f"Google rating for '{product_name}': {val}")
                        return val

        # Fallback: scan visible text for "X.X out of 5" or "X.X stars"
        page_text = soup.get_text(" ", strip=True)
        for pattern in [
            r"(\d\.\d)\s*out\s*of\s*5",
            r"(\d\.\d)\s*stars?",
            r"Rating[:\s]+(\d\.\d)",
        ]:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                if 1.0 <= val <= 5.0:
                    logger.info(f"Google rating (fallback) for '{product_name}': {val}")
                    return val

    except Exception as e:
        logger.warning(f"Google review scrape failed: {e}")

    return None


def scrape_amazon_rating(product_name: str, timeout: int) -> Optional[float]:
    query = quote_plus(product_name)
    url = f"https://www.amazon.in/s?k={query}&i=electronics"
    try:
        soup = fetch_page(url, timeout=timeout, delay=1.0, referer="https://www.amazon.in/")
        selectors = [
            "span.a-icon-alt",
            "i.a-icon-star span.a-icon-alt",
            ".a-star-medium span.a-icon-alt",
        ]
        for sel in selectors:
            elems = soup.select(sel)
            ratings = []
            for e in elems[:10]:
                text = e.get_text(strip=True)
                match = re.search(r"(\d\.\d)", text)
                if match:
                    val = float(match.group(1))
                    if 1.0 <= val <= 5.0:
                        ratings.append(val)
            if ratings:
                avg = round(sum(ratings) / len(ratings), 1)
                logger.info(f"Amazon rating for '{product_name}': {avg}")
                return avg
    except Exception as e:
        logger.warning(f"Amazon rating scrape failed: {e}")
    return None


def sentiment_from_rating(rating: float) -> float:
    return round(rating / 5.0, 2)


def review_node(state: AgentState) -> dict:
    platforms = list(state["prices"].keys())
    if not platforms:
        return {"reviews": {}}

    product_name = state["product_name"]

    # Try Google first, then Amazon as fallback
    rating = scrape_google_rating(product_name, settings.scrape_timeout)
    if rating is None:
        rating = scrape_amazon_rating(product_name, settings.scrape_timeout)
    if rating is None:
        logger.warning("No rating found from any source, defaulting to 4.0")
        rating = 4.0

    reviews = {}
    for platform in platforms:
        reviews[platform] = {
            "rating": rating,
            "sentiment_score": sentiment_from_rating(rating),
        }

    logger.info(f"Review node: rating={rating} applied to {platforms}")
    return {"reviews": reviews}
