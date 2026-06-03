from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from tools.scraper import fetch_page, parse_price
from graph.state import AgentState
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ─── Amazon India ────────────────────────────────────────────────────────────

def scrape_amazon(product_name: str, timeout: int) -> float:
    url = f"https://www.amazon.in/s?k={quote_plus(product_name)}&i=electronics"
    try:
        soup = fetch_page(url, timeout=timeout, delay=1.5, referer="https://www.amazon.in/")
        # Multiple selector fallbacks for Amazon search result prices
        selectors = [
            ("span.a-price[data-a-size='xl'] span.a-price-whole",),
            ("span.a-price[data-a-size='l'] span.a-price-whole",),
            ("span.a-price span.a-price-whole",),
            (".s-price-instructions-style .a-price-whole",),
        ]
        prices = []
        for (sel,) in selectors:
            for elem in soup.select(sel)[:5]:
                p = parse_price(elem.get_text(strip=True))
                if p > 100:
                    prices.append(p)
            if prices:
                break

        if prices:
            best = min(prices)
            logger.info(f"Amazon price for '{product_name}': ₹{best}")
            return best
    except Exception as e:
        logger.warning(f"Amazon scrape failed: {e}")
    return 0.0


# ─── Flipkart ────────────────────────────────────────────────────────────────

def scrape_flipkart(product_name: str, timeout: int) -> float:
    url = f"https://www.flipkart.com/search?q={quote_plus(product_name)}&otracker=search"
    try:
        soup = fetch_page(url, timeout=timeout, delay=1.5, referer="https://www.flipkart.com/")
        selectors = [
            "div._30jeq3",
            "div._1_WHN1",
            "div.Nx9bqj",
            "div._25b18c div",
        ]
        prices = []
        for sel in selectors:
            for elem in soup.select(sel)[:5]:
                p = parse_price(elem.get_text(strip=True))
                if p > 100:
                    prices.append(p)
            if prices:
                break

        if prices:
            best = min(prices)
            logger.info(f"Flipkart price for '{product_name}': ₹{best}")
            return best
    except Exception as e:
        logger.warning(f"Flipkart scrape failed: {e}")
    return 0.0


# ─── Croma ───────────────────────────────────────────────────────────────────

def scrape_croma(product_name: str, timeout: int) -> float:
    url = f"https://www.croma.com/searchB?q={quote_plus(product_name)}%3Arelevance&inStoreSearch=false"
    try:
        soup = fetch_page(url, timeout=timeout, delay=1.5, referer="https://www.croma.com/")
        selectors = [
            "span.amount",
            "div.pdpPrice span",
            ".cp-price",
            "h3.pdpPrice",
        ]
        prices = []
        for sel in selectors:
            for elem in soup.select(sel)[:5]:
                p = parse_price(elem.get_text(strip=True))
                if p > 100:
                    prices.append(p)
            if prices:
                break

        if prices:
            best = min(prices)
            logger.info(f"Croma price for '{product_name}': ₹{best}")
            return best
    except Exception as e:
        logger.warning(f"Croma scrape failed: {e}")
    return 0.0


# ─── Reliance Digital ────────────────────────────────────────────────────────

def scrape_reliance(product_name: str, timeout: int) -> float:
    url = f"https://www.reliancedigital.in/search?q={quote_plus(product_name)}:relevance"
    try:
        soup = fetch_page(url, timeout=timeout, delay=1.5, referer="https://www.reliancedigital.in/")
        selectors = [
            "span.pdp__offerPrice",
            "span.product-price",
            ".price",
            "p.product-price",
        ]
        prices = []
        for sel in selectors:
            for elem in soup.select(sel)[:5]:
                p = parse_price(elem.get_text(strip=True))
                if p > 100:
                    prices.append(p)
            if prices:
                break

        if prices:
            best = min(prices)
            logger.info(f"Reliance Digital price for '{product_name}': ₹{best}")
            return best
    except Exception as e:
        logger.warning(f"Reliance Digital scrape failed: {e}")
    return 0.0


# ─── Node ────────────────────────────────────────────────────────────────────

SCRAPERS = {
    "amazon":   scrape_amazon,
    "flipkart": scrape_flipkart,
    "croma":    scrape_croma,
    "reliance": scrape_reliance,
}


def price_node(state: AgentState) -> dict:
    product_name = state["product_name"]
    prices: dict[str, float] = {}

    for platform, scraper_fn in SCRAPERS.items():
        price = scraper_fn(product_name, settings.scrape_timeout)
        if price > 0:
            prices[platform] = price

    if not prices:
        logger.error(f"No prices found for '{product_name}'")
        return {"error": f"No prices found for '{product_name}'. Try a more specific product name."}

    logger.info(f"Price node results: { {k: f'₹{v}' for k, v in prices.items()} }")
    return {"prices": prices}
