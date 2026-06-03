# Product Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a full-stack AI product recommendation app that scrapes Google Shopping in real-time and uses LangGraph multi-agent orchestration + Mistral (Ollama) to recommend the best platform to buy a product.

**Architecture:** Linear LangGraph StateGraph pipeline — Orchestrator → Price → Review → Platform → Decision — each node enriches shared `AgentState`. FastAPI serves a single `POST /recommend` endpoint consumed by a React/Vite frontend.

**Tech Stack:** Python 3.11, FastAPI, LangGraph, LangChain, langchain-ollama (Mistral via Ollama), requests, BeautifulSoup4, React 18, Vite, Axios, plain CSS.

---

## File Map

```
product_assistant/
├── backend/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── orchestrator.py       # normalize query → product_name
│   │   ├── price_agent.py        # Google Shopping scraper
│   │   ├── review_agent.py       # Google review/rating scraper
│   │   ├── platform_agent.py     # static platform metadata lookup
│   │   └── decision_agent.py     # scoring formula + mistral LLM reason
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py              # AgentState TypedDict
│   │   └── workflow.py           # LangGraph StateGraph definition
│   ├── tools/
│   │   ├── __init__.py
│   │   └── scraper.py            # shared HTTP + BeautifulSoup helpers
│   ├── utils/
│   │   ├── __init__.py
│   │   └── logger.py             # Python logging setup
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_scraper.py
│   │   ├── test_orchestrator.py
│   │   ├── test_price_agent.py
│   │   ├── test_review_agent.py
│   │   ├── test_platform_agent.py
│   │   ├── test_decision_agent.py
│   │   ├── test_workflow.py
│   │   └── test_main.py
│   ├── main.py                   # FastAPI app, CORS, /recommend endpoint
│   ├── config.py                 # Pydantic settings from .env
│   ├── .env                      # OLLAMA_BASE_URL, OLLAMA_MODEL, SCRAPE_TIMEOUT
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── SearchBox.jsx     # input + submit button
    │   │   ├── ResultCard.jsx    # recommended platform display
    │   │   └── SearchHistory.jsx # last 5 searches (localStorage)
    │   ├── App.jsx               # root state: loading, result, error, history
    │   ├── App.css               # centered modern layout
    │   ├── main.jsx
    │   └── api.js                # axios POST /recommend wrapper
    ├── index.html
    ├── package.json
    └── vite.config.js
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/.env`
- Create: `backend/config.py`
- Create: `backend/utils/__init__.py`, `backend/agents/__init__.py`, `backend/graph/__init__.py`, `backend/tools/__init__.py`, `backend/tests/__init__.py`

- [ ] **Step 1: Create backend directory structure**

```bash
cd /home/siva/siva/myprojects/product_assistant
mkdir -p backend/{agents,graph,tools,utils,tests}
touch backend/{agents,graph,tools,utils,tests}/__init__.py
```

- [ ] **Step 2: Create requirements.txt**

```
# backend/requirements.txt
fastapi==0.111.0
uvicorn[standard]==0.30.1
langchain==0.2.6
langchain-core==0.2.10
langchain-ollama==0.1.1
langgraph==0.1.9
requests==2.32.3
beautifulsoup4==4.12.3
python-dotenv==1.0.1
pydantic-settings==2.3.4
pytest==8.2.2
pytest-asyncio==0.23.7
httpx==0.27.0
```

- [ ] **Step 3: Create .env**

```bash
# backend/.env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
SCRAPE_TIMEOUT=10
```

- [ ] **Step 4: Create config.py**

```python
# backend/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"
    scrape_timeout: int = 10

    class Config:
        env_file = ".env"

settings = Settings()
```

- [ ] **Step 5: Install dependencies**

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Expected: All packages install without error.

- [ ] **Step 6: Commit**

```bash
cd /home/siva/siva/myprojects/product_assistant
git init
git add backend/
git commit -m "feat: scaffold backend project structure"
```

---

## Task 2: Logger Utility

**Files:**
- Create: `backend/utils/logger.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_logger.py
import logging
from utils.logger import get_logger

def test_get_logger_returns_logger():
    logger = get_logger("test")
    assert isinstance(logger, logging.Logger)

def test_get_logger_name():
    logger = get_logger("mymodule")
    assert logger.name == "mymodule"

def test_get_logger_level():
    logger = get_logger("test")
    assert logger.level == logging.INFO
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && source venv/bin/activate
pytest tests/test_logger.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'utils.logger'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/utils/logger.py
import logging

def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_logger.py -v
```

Expected: 3 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add backend/utils/logger.py backend/tests/test_logger.py
git commit -m "feat: add logger utility"
```

---

## Task 3: Scraper Utility

**Files:**
- Create: `backend/tools/scraper.py`
- Create: `backend/tests/test_scraper.py`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_scraper.py
from unittest.mock import patch, MagicMock
from tools.scraper import get_headers, fetch_page, parse_price, normalize_platform

def test_get_headers_returns_dict():
    headers = get_headers()
    assert "User-Agent" in headers
    assert "Accept" in headers

def test_get_headers_rotates_user_agent():
    agents = {get_headers()["User-Agent"] for _ in range(20)}
    assert len(agents) > 1  # at least 2 different agents in 20 calls

def test_parse_price_rupee_symbol():
    assert parse_price("₹56,999") == 56999.0

def test_parse_price_with_commas():
    assert parse_price("58,000") == 58000.0

def test_parse_price_invalid_returns_zero():
    assert parse_price("N/A") == 0.0

def test_normalize_platform_amazon():
    assert normalize_platform("amazon.in") == "amazon"
    assert normalize_platform("sold by amazon") == "amazon"

def test_normalize_platform_flipkart():
    assert normalize_platform("flipkart") == "flipkart"
    assert normalize_platform("Flipkart India") == "flipkart"

def test_normalize_platform_unknown():
    assert normalize_platform("somerandombrand.com") == "other"

def test_fetch_page_returns_soup():
    from bs4 import BeautifulSoup
    mock_response = MagicMock()
    mock_response.text = "<html><body><p>test</p></body></html>"
    mock_response.raise_for_status = MagicMock()

    with patch("tools.scraper.requests.get", return_value=mock_response):
        with patch("tools.scraper.time.sleep"):
            soup = fetch_page("http://example.com", timeout=5)

    assert isinstance(soup, BeautifulSoup)
    assert soup.find("p").text == "test"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_scraper.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tools.scraper'`

- [ ] **Step 3: Write implementation**

```python
# backend/tools/scraper.py
import random
import re
import time
import requests
from bs4 import BeautifulSoup
from utils.logger import get_logger

logger = get_logger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
]

PLATFORM_KEYWORDS = {
    "amazon": ["amazon"],
    "flipkart": ["flipkart"],
    "croma": ["croma"],
    "reliance": ["reliance", "jiomart"],
    "meesho": ["meesho"],
    "myntra": ["myntra"],
}


def get_headers() -> dict:
    """Return HTTP headers with a random User-Agent to reduce bot detection."""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-IN,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }


def fetch_page(url: str, timeout: int = 10) -> BeautifulSoup:
    """Fetch a URL and return parsed BeautifulSoup. Sleeps 1s before request."""
    time.sleep(1)
    logger.info(f"Fetching: {url}")
    response = requests.get(url, headers=get_headers(), timeout=timeout)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def parse_price(text: str) -> float:
    """Extract numeric price from strings like '₹56,999' or '58,000'. Returns 0.0 on failure."""
    cleaned = re.sub(r"[₹,\s]", "", text)
    match = re.search(r"\d+\.?\d*", cleaned)
    if match:
        return float(match.group())
    return 0.0


def normalize_platform(store: str) -> str:
    """Map raw store name string to a canonical platform key."""
    store_lower = store.lower()
    for platform, keywords in PLATFORM_KEYWORDS.items():
        if any(kw in store_lower for kw in keywords):
            return platform
    return "other"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_scraper.py -v
```

Expected: 9 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add backend/tools/scraper.py backend/tests/test_scraper.py
git commit -m "feat: add scraper utility with price parsing and platform normalization"
```

---

## Task 4: AgentState Definition

**Files:**
- Create: `backend/graph/state.py`
- Create: `backend/tests/test_state.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_state.py
from graph.state import AgentState

def test_agent_state_is_typed_dict():
    # Verify AgentState can be instantiated with all required fields
    state: AgentState = {
        "query": "iPhone 15",
        "product_name": "iPhone 15",
        "prices": {"amazon": 58999.0},
        "reviews": {"amazon": {"rating": 4.5, "sentiment_score": 0.85}},
        "platform_meta": {"amazon": {"delivery_days": 2, "return_policy_score": 0.9, "seller_rating": 4.6}},
        "scores": {"amazon": 0.75},
        "recommended_platform": "amazon",
        "recommended_price": 58999.0,
        "recommended_rating": 4.5,
        "reason": "Best overall value.",
        "all_platforms": [{"name": "amazon", "price": 58999.0, "rating": 4.5, "score": 0.75}],
        "error": None,
    }
    assert state["query"] == "iPhone 15"
    assert state["prices"]["amazon"] == 58999.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_state.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'graph.state'`

- [ ] **Step 3: Write implementation**

```python
# backend/graph/state.py
from typing import TypedDict, Optional, List, Dict

class AgentState(TypedDict):
    query: str                          # raw user query
    product_name: str                   # normalized product name
    prices: Dict[str, float]            # platform -> lowest price found
    reviews: Dict[str, Dict]            # platform -> {rating, sentiment_score}
    platform_meta: Dict[str, Dict]      # platform -> {delivery_days, return_policy_score, seller_rating}
    scores: Dict[str, float]            # platform -> final weighted score
    recommended_platform: str           # winner platform name
    recommended_price: float            # winner price
    recommended_rating: float           # winner rating
    reason: str                         # LLM-generated explanation
    all_platforms: List[Dict]           # full comparison list for frontend
    error: Optional[str]                # error message if any agent fails
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_state.py -v
```

Expected: 1 test PASSED

- [ ] **Step 5: Commit**

```bash
git add backend/graph/state.py backend/tests/test_state.py
git commit -m "feat: define AgentState TypedDict for LangGraph pipeline"
```

---

## Task 5: Orchestrator Agent

**Files:**
- Create: `backend/agents/orchestrator.py`
- Create: `backend/tests/test_orchestrator.py`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_orchestrator.py
from agents.orchestrator import normalize_query, orchestrator_node
from graph.state import AgentState

def test_normalize_strips_filler_words():
    assert normalize_query("best place to buy iPhone 15") == "iphone 15"
    assert normalize_query("where to buy Samsung Galaxy S24") == "samsung galaxy s24"

def test_normalize_strips_price_constraint():
    assert normalize_query("iPhone 15 under 60000") == "iphone 15"
    assert normalize_query("laptop below 50000") == "laptop"

def test_normalize_preserves_product_name():
    assert normalize_query("iPhone 15 128GB") == "iphone 15 128gb"

def test_orchestrator_node_sets_product_name():
    state: AgentState = {
        "query": "best place to buy iPhone 15 under 60000",
        "product_name": "",
        "prices": {}, "reviews": {}, "platform_meta": {},
        "scores": {}, "recommended_platform": "", "recommended_price": 0.0,
        "recommended_rating": 0.0, "reason": "", "all_platforms": [], "error": None,
    }
    result = orchestrator_node(state)
    assert result["product_name"] == "iphone 15"

def test_orchestrator_node_preserves_query():
    state: AgentState = {
        "query": "Samsung Galaxy S24",
        "product_name": "",
        "prices": {}, "reviews": {}, "platform_meta": {},
        "scores": {}, "recommended_platform": "", "recommended_price": 0.0,
        "recommended_rating": 0.0, "reason": "", "all_platforms": [], "error": None,
    }
    result = orchestrator_node(state)
    assert result["query"] == "Samsung Galaxy S24"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_orchestrator.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'agents.orchestrator'`

- [ ] **Step 3: Write implementation**

```python
# backend/agents/orchestrator.py
import re
from graph.state import AgentState
from utils.logger import get_logger

logger = get_logger(__name__)

FILLER_PATTERNS = [
    r"\bbest place to buy\b",
    r"\bwhere to buy\b",
    r"\bbest place to get\b",
    r"\bunder\s+\d+\b",
    r"\bbelow\s+\d+\b",
    r"\bwithin\s+\d+\b",
    r"\bless than\s+\d+\b",
]


def normalize_query(query: str) -> str:
    """Strip filler phrases and price constraints from user query, return cleaned product name."""
    text = query.lower().strip()
    for pattern in FILLER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    # Remove extra whitespace
    return " ".join(text.split())


def orchestrator_node(state: AgentState) -> dict:
    """Parse raw user query and seed AgentState with normalized product_name."""
    logger.info(f"Orchestrator processing query: {state['query']}")
    product_name = normalize_query(state["query"])
    logger.info(f"Normalized product name: {product_name}")
    return {"product_name": product_name}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_orchestrator.py -v
```

Expected: 5 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add backend/agents/orchestrator.py backend/tests/test_orchestrator.py
git commit -m "feat: add orchestrator agent with query normalization"
```

---

## Task 6: Price Agent

**Files:**
- Create: `backend/agents/price_agent.py`
- Create: `backend/tests/test_price_agent.py`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_price_agent.py
from unittest.mock import patch, MagicMock
from bs4 import BeautifulSoup
from agents.price_agent import parse_shopping_results, price_node
from graph.state import AgentState

SAMPLE_HTML = """
<html><body>
  <div class="sh-dgr__grid-result">
    <h3 class="Xjkr3b">Apple iPhone 15 128GB</h3>
    <span class="a8Pemb OFFNJ">₹58,999</span>
    <div class="aULzUe IuHnof">Amazon</div>
  </div>
  <div class="sh-dgr__grid-result">
    <h3 class="Xjkr3b">Apple iPhone 15 128GB</h3>
    <span class="a8Pemb OFFNJ">₹56,500</span>
    <div class="aULzUe IuHnof">Flipkart</div>
  </div>
</body></html>
"""

def test_parse_shopping_results_extracts_prices():
    soup = BeautifulSoup(SAMPLE_HTML, "html.parser")
    results = parse_shopping_results(soup)
    assert len(results) == 2

def test_parse_shopping_results_correct_prices():
    soup = BeautifulSoup(SAMPLE_HTML, "html.parser")
    results = parse_shopping_results(soup)
    prices = {r["store"]: r["price"] for r in results}
    assert prices["amazon"] == 58999.0
    assert prices["flipkart"] == 56500.0

def test_parse_shopping_results_empty_html():
    soup = BeautifulSoup("<html><body></body></html>", "html.parser")
    results = parse_shopping_results(soup)
    assert results == []

def test_price_node_populates_prices():
    mock_soup = BeautifulSoup(SAMPLE_HTML, "html.parser")
    state: AgentState = {
        "query": "iPhone 15", "product_name": "iphone 15",
        "prices": {}, "reviews": {}, "platform_meta": {},
        "scores": {}, "recommended_platform": "", "recommended_price": 0.0,
        "recommended_rating": 0.0, "reason": "", "all_platforms": [], "error": None,
    }
    with patch("agents.price_agent.fetch_page", return_value=mock_soup):
        result = price_node(state)
    assert "amazon" in result["prices"]
    assert "flipkart" in result["prices"]
    assert result["prices"]["amazon"] == 58999.0

def test_price_node_handles_scrape_failure():
    state: AgentState = {
        "query": "iPhone 15", "product_name": "iphone 15",
        "prices": {}, "reviews": {}, "platform_meta": {},
        "scores": {}, "recommended_platform": "", "recommended_price": 0.0,
        "recommended_rating": 0.0, "reason": "", "all_platforms": [], "error": None,
    }
    with patch("agents.price_agent.fetch_page", side_effect=Exception("Network error")):
        result = price_node(state)
    assert result["error"] is not None
    assert "price" in result["error"].lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_price_agent.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'agents.price_agent'`

- [ ] **Step 3: Write implementation**

```python
# backend/agents/price_agent.py
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from tools.scraper import fetch_page, parse_price, normalize_platform
from graph.state import AgentState
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

SHOPPING_URL = "https://www.google.com/search?q={query}&tbm=shop&hl=en&gl=in"

# Selector sets to try in order — Google changes these periodically
CONTAINER_SELECTORS = [
    "div.sh-dgr__grid-result",
    "div.KZmu8e",
    "div.u30d4",
]
PRICE_SELECTORS = ["span.a8Pemb", "span.OFFNJ", "span[aria-label]"]
STORE_SELECTORS = ["div.aULzUe", "span.E5ocAb", "div.IuHnof", "span.LbUacb"]
TITLE_SELECTORS = ["h3.Xjkr3b", "h3", "h4"]


def _select_first(element, selectors: list):
    """Try each selector in order, return first match or None."""
    for sel in selectors:
        found = element.select_one(sel)
        if found:
            return found
    return None


def parse_shopping_results(soup: BeautifulSoup) -> list:
    """Extract list of {title, price, store} from Google Shopping HTML."""
    results = []
    containers = []
    for sel in CONTAINER_SELECTORS:
        containers = soup.select(sel)
        if containers:
            break

    for container in containers:
        price_elem = _select_first(container, PRICE_SELECTORS)
        store_elem = _select_first(container, STORE_SELECTORS)
        title_elem = _select_first(container, TITLE_SELECTORS)

        if not price_elem or not store_elem:
            continue

        price = parse_price(price_elem.get_text(strip=True))
        store = normalize_platform(store_elem.get_text(strip=True))
        title = title_elem.get_text(strip=True) if title_elem else ""

        if price > 0 and store != "other":
            results.append({"title": title, "price": price, "store": store})

    return results


def price_node(state: AgentState) -> dict:
    """Scrape Google Shopping and return lowest price per platform."""
    try:
        query = quote_plus(state["product_name"])
        url = SHOPPING_URL.format(query=query)
        soup = fetch_page(url, timeout=settings.scrape_timeout)
        raw_results = parse_shopping_results(soup)

        # Keep only the lowest price per platform
        prices: dict[str, float] = {}
        for item in raw_results:
            platform = item["store"]
            if platform not in prices or item["price"] < prices[platform]:
                prices[platform] = item["price"]

        logger.info(f"Price agent found platforms: {list(prices.keys())}")
        return {"prices": prices}

    except Exception as e:
        logger.error(f"Price agent failed: {e}")
        return {"error": f"price scraping failed: {e}"}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_price_agent.py -v
```

Expected: 5 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add backend/agents/price_agent.py backend/tests/test_price_agent.py
git commit -m "feat: add price agent with Google Shopping scraper"
```

---

## Task 7: Review Agent

**Files:**
- Create: `backend/agents/review_agent.py`
- Create: `backend/tests/test_review_agent.py`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_review_agent.py
from unittest.mock import patch
from bs4 import BeautifulSoup
from agents.review_agent import extract_rating_from_soup, sentiment_from_rating, review_node
from graph.state import AgentState

SAMPLE_REVIEW_HTML = """
<html><body>
  <div class="Aq14fc">4.5</div>
  <span class="HiHjCd">2,345 reviews</span>
</body></html>
"""

SAMPLE_NO_RATING_HTML = "<html><body><p>No rating here</p></body></html>"


def test_extract_rating_finds_rating():
    soup = BeautifulSoup(SAMPLE_REVIEW_HTML, "html.parser")
    rating = extract_rating_from_soup(soup)
    assert rating == 4.5

def test_extract_rating_returns_none_when_missing():
    soup = BeautifulSoup(SAMPLE_NO_RATING_HTML, "html.parser")
    rating = extract_rating_from_soup(soup)
    assert rating is None

def test_sentiment_from_rating_high():
    assert sentiment_from_rating(4.5) == 0.9

def test_sentiment_from_rating_low():
    assert sentiment_from_rating(2.0) == 0.4

def test_sentiment_from_rating_midrange():
    score = sentiment_from_rating(3.5)
    assert 0.6 < score < 0.8

def test_review_node_populates_reviews():
    mock_soup = BeautifulSoup(SAMPLE_REVIEW_HTML, "html.parser")
    state: AgentState = {
        "query": "iPhone 15", "product_name": "iphone 15",
        "prices": {"amazon": 58999.0, "flipkart": 56500.0},
        "reviews": {}, "platform_meta": {},
        "scores": {}, "recommended_platform": "", "recommended_price": 0.0,
        "recommended_rating": 0.0, "reason": "", "all_platforms": [], "error": None,
    }
    with patch("agents.review_agent.fetch_page", return_value=mock_soup):
        result = review_node(state)
    assert "reviews" in result
    # Both platforms should have review data
    for platform in ["amazon", "flipkart"]:
        assert platform in result["reviews"]
        assert "rating" in result["reviews"][platform]
        assert "sentiment_score" in result["reviews"][platform]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_review_agent.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'agents.review_agent'`

- [ ] **Step 3: Write implementation**

```python
# backend/agents/review_agent.py
import re
from urllib.parse import quote_plus
from typing import Optional
from bs4 import BeautifulSoup
from tools.scraper import fetch_page
from graph.state import AgentState
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

REVIEW_URL = "https://www.google.com/search?q={query}+reviews&hl=en&gl=in"

# Google rich snippet / knowledge panel rating selectors
RATING_SELECTORS = [
    "div.Aq14fc",      # knowledge panel aggregate rating
    "span.Aq14fc",
    "div.aRct8",
    "span.oqSTJd",
    "g-review-stars span",
]


def extract_rating_from_soup(soup: BeautifulSoup) -> Optional[float]:
    """Pull star rating (0-5) from Google search result rich snippets."""
    for sel in RATING_SELECTORS:
        elem = soup.select_one(sel)
        if elem:
            text = elem.get_text(strip=True)
            match = re.search(r"\d+\.?\d*", text)
            if match:
                value = float(match.group())
                if 0.0 <= value <= 5.0:
                    return value

    # Fallback: look for any "X.X stars" or "X.X out of 5" pattern
    text = soup.get_text()
    match = re.search(r"(\d\.\d)\s*(?:stars?|out of 5)", text, re.IGNORECASE)
    if match:
        return float(match.group(1))

    return None


def sentiment_from_rating(rating: float) -> float:
    """Convert a 0-5 star rating to a 0-1 sentiment score."""
    return round(rating / 5.0, 2)


def review_node(state: AgentState) -> dict:
    """Scrape Google for product ratings and compute sentiment per platform."""
    platforms = list(state["prices"].keys())
    if not platforms:
        logger.warning("Review agent: no platforms in prices, skipping")
        return {"reviews": {}}

    reviews = {}
    try:
        query = quote_plus(state["product_name"])
        url = REVIEW_URL.format(query=query)
        soup = fetch_page(url, timeout=settings.scrape_timeout)
        global_rating = extract_rating_from_soup(soup)

        if global_rating is None:
            logger.warning("Review agent: no rating found, using default 4.0")
            global_rating = 4.0

        for platform in platforms:
            # Use same global product rating per platform (product rating is product-level, not platform-level)
            # Small variance per platform based on platform metadata will be applied in decision agent
            reviews[platform] = {
                "rating": global_rating,
                "sentiment_score": sentiment_from_rating(global_rating),
            }

        logger.info(f"Review agent found rating: {global_rating} for {len(platforms)} platforms")

    except Exception as e:
        logger.error(f"Review agent failed: {e}, using default rating 4.0")
        for platform in platforms:
            reviews[platform] = {"rating": 4.0, "sentiment_score": 0.8}

    return {"reviews": reviews}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_review_agent.py -v
```

Expected: 7 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add backend/agents/review_agent.py backend/tests/test_review_agent.py
git commit -m "feat: add review agent with Google review scraper and sentiment scoring"
```

---

## Task 8: Platform Agent

**Files:**
- Create: `backend/agents/platform_agent.py`
- Create: `backend/tests/test_platform_agent.py`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_platform_agent.py
from agents.platform_agent import PLATFORM_META, get_platform_meta, platform_node
from graph.state import AgentState

def test_platform_meta_has_amazon():
    assert "amazon" in PLATFORM_META
    assert "delivery_days" in PLATFORM_META["amazon"]
    assert "return_policy_score" in PLATFORM_META["amazon"]
    assert "seller_rating" in PLATFORM_META["amazon"]

def test_get_platform_meta_known_platform():
    meta = get_platform_meta("amazon")
    assert meta["delivery_days"] == 2
    assert meta["return_policy_score"] == 0.90
    assert meta["seller_rating"] == 4.6

def test_get_platform_meta_unknown_returns_defaults():
    meta = get_platform_meta("unknownstore")
    assert meta["delivery_days"] > 0
    assert 0.0 <= meta["return_policy_score"] <= 1.0

def test_platform_node_populates_meta():
    state: AgentState = {
        "query": "iPhone 15", "product_name": "iphone 15",
        "prices": {"amazon": 58999.0, "flipkart": 56500.0},
        "reviews": {"amazon": {"rating": 4.5, "sentiment_score": 0.9},
                    "flipkart": {"rating": 4.5, "sentiment_score": 0.9}},
        "platform_meta": {},
        "scores": {}, "recommended_platform": "", "recommended_price": 0.0,
        "recommended_rating": 0.0, "reason": "", "all_platforms": [], "error": None,
    }
    result = platform_node(state)
    assert "amazon" in result["platform_meta"]
    assert "flipkart" in result["platform_meta"]
    assert result["platform_meta"]["amazon"]["delivery_days"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_platform_agent.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'agents.platform_agent'`

- [ ] **Step 3: Write implementation**

```python
# backend/agents/platform_agent.py
from graph.state import AgentState
from utils.logger import get_logger

logger = get_logger(__name__)

# Static metadata — delivery SLA and reliability scores don't change per-query
PLATFORM_META = {
    "amazon":   {"delivery_days": 2,  "return_policy_score": 0.90, "seller_rating": 4.6},
    "flipkart": {"delivery_days": 3,  "return_policy_score": 0.85, "seller_rating": 4.4},
    "croma":    {"delivery_days": 4,  "return_policy_score": 0.80, "seller_rating": 4.2},
    "reliance": {"delivery_days": 5,  "return_policy_score": 0.75, "seller_rating": 4.0},
    "meesho":   {"delivery_days": 6,  "return_policy_score": 0.65, "seller_rating": 3.8},
    "myntra":   {"delivery_days": 4,  "return_policy_score": 0.78, "seller_rating": 4.1},
}

DEFAULT_META = {"delivery_days": 7, "return_policy_score": 0.60, "seller_rating": 3.5}


def get_platform_meta(platform: str) -> dict:
    """Return metadata dict for a platform, falling back to defaults for unknown platforms."""
    return PLATFORM_META.get(platform, DEFAULT_META)


def platform_node(state: AgentState) -> dict:
    """Attach delivery/return/seller metadata to each platform found in prices."""
    platforms = list(state["prices"].keys())
    platform_meta = {p: get_platform_meta(p) for p in platforms}
    logger.info(f"Platform agent enriched {len(platform_meta)} platforms")
    return {"platform_meta": platform_meta}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_platform_agent.py -v
```

Expected: 5 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add backend/agents/platform_agent.py backend/tests/test_platform_agent.py
git commit -m "feat: add platform agent with static delivery/reliability metadata"
```

---

## Task 9: Decision Agent

**Files:**
- Create: `backend/agents/decision_agent.py`
- Create: `backend/tests/test_decision_agent.py`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_decision_agent.py
from unittest.mock import patch, MagicMock
from agents.decision_agent import compute_scores, pick_winner, build_reason_prompt, decision_node
from graph.state import AgentState

PRICES = {"amazon": 58999.0, "flipkart": 56500.0}
REVIEWS = {
    "amazon":   {"rating": 4.5, "sentiment_score": 0.90},
    "flipkart": {"rating": 4.4, "sentiment_score": 0.88},
}
PLATFORM_META = {
    "amazon":   {"delivery_days": 2,  "return_policy_score": 0.90, "seller_rating": 4.6},
    "flipkart": {"delivery_days": 3,  "return_policy_score": 0.85, "seller_rating": 4.4},
}


def test_compute_scores_returns_score_per_platform():
    scores = compute_scores(PRICES, REVIEWS, PLATFORM_META)
    assert "amazon" in scores
    assert "flipkart" in scores
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_compute_scores_lower_price_gets_higher_price_score():
    scores = compute_scores(PRICES, REVIEWS, PLATFORM_META)
    # flipkart is cheaper so should have higher final score
    assert scores["flipkart"] > scores["amazon"]


def test_pick_winner_returns_highest_scoring_platform():
    scores = {"amazon": 0.71, "flipkart": 0.78}
    assert pick_winner(scores) == "flipkart"


def test_build_reason_prompt_contains_platform_name():
    prompt = build_reason_prompt("flipkart", 56500.0, 4.4, 0.78, PRICES, REVIEWS, PLATFORM_META)
    assert "flipkart" in prompt.lower()
    assert "56500" in prompt or "56,500" in prompt


def test_decision_node_returns_recommendation():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "Flipkart has the lowest price and fast delivery."

    state: AgentState = {
        "query": "iPhone 15", "product_name": "iphone 15",
        "prices": PRICES, "reviews": REVIEWS, "platform_meta": PLATFORM_META,
        "scores": {}, "recommended_platform": "", "recommended_price": 0.0,
        "recommended_rating": 0.0, "reason": "", "all_platforms": [], "error": None,
    }
    with patch("agents.decision_agent.ChatOllama", return_value=mock_llm):
        result = decision_node(state)

    assert result["recommended_platform"] == "flipkart"
    assert result["recommended_price"] == 56500.0
    assert len(result["all_platforms"]) == 2
    assert "Flipkart" in result["reason"]


def test_decision_node_fallback_reason_when_llm_fails():
    state: AgentState = {
        "query": "iPhone 15", "product_name": "iphone 15",
        "prices": PRICES, "reviews": REVIEWS, "platform_meta": PLATFORM_META,
        "scores": {}, "recommended_platform": "", "recommended_price": 0.0,
        "recommended_rating": 0.0, "reason": "", "all_platforms": [], "error": None,
    }
    with patch("agents.decision_agent.ChatOllama", side_effect=Exception("Ollama not running")):
        result = decision_node(state)

    assert result["recommended_platform"] == "flipkart"
    assert result["reason"] != ""  # fallback reason generated
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_decision_agent.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'agents.decision_agent'`

- [ ] **Step 3: Write implementation**

```python
# backend/agents/decision_agent.py
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from graph.state import AgentState
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def compute_scores(prices: dict, reviews: dict, platform_meta: dict) -> dict:
    """
    Compute weighted final score per platform.
    score = 0.4 * price_score + 0.3 * review_score + 0.3 * platform_score
    """
    if not prices:
        return {}

    max_price = max(prices.values())
    scores = {}

    for platform in prices:
        price = prices[platform]
        review = reviews.get(platform, {"rating": 4.0, "sentiment_score": 0.8})
        meta = platform_meta.get(platform, {"delivery_days": 7, "return_policy_score": 0.6, "seller_rating": 3.5})

        # Lower price → higher price_score
        price_score = 1.0 - (price / max_price) if max_price > 0 else 0.0

        # Average of normalized rating and sentiment
        review_score = (review["rating"] / 5.0 + review["sentiment_score"]) / 2.0

        # Delivery speed (normalise: 1 day = 1.0, 7+ days → lower)
        delivery_score = max(0.0, 1.0 - (meta["delivery_days"] - 1) / 7.0)
        platform_score = (
            delivery_score * 0.4
            + meta["return_policy_score"] * 0.3
            + (meta["seller_rating"] / 5.0) * 0.3
        )

        scores[platform] = round(
            0.4 * price_score + 0.3 * review_score + 0.3 * platform_score, 4
        )

    return scores


def pick_winner(scores: dict) -> str:
    """Return platform key with highest score."""
    return max(scores, key=scores.get)


def build_reason_prompt(
    winner: str, price: float, rating: float, score: float,
    prices: dict, reviews: dict, platform_meta: dict
) -> str:
    """Build a prompt for the LLM to explain the recommendation in 1-2 sentences."""
    others = [f"{p}: ₹{prices[p]:,.0f}" for p in prices if p != winner]
    others_str = ", ".join(others) if others else "no competitors found"
    meta = platform_meta.get(winner, {})

    return (
        f"You are a product recommendation assistant. Explain in 1-2 concise sentences "
        f"why {winner.title()} is the best platform to buy this product.\n\n"
        f"Data:\n"
        f"- Recommended: {winner.title()} at ₹{price:,.0f}\n"
        f"- Rating: {rating}/5.0\n"
        f"- Delivery: {meta.get('delivery_days', 'N/A')} days\n"
        f"- Return policy score: {meta.get('return_policy_score', 'N/A')}\n"
        f"- Competing platforms: {others_str}\n\n"
        f"Keep the explanation factual and under 40 words."
    )


def decision_node(state: AgentState) -> dict:
    """Score all platforms, pick winner, call Mistral via Ollama for reason text."""
    prices = state["prices"]
    reviews = state["reviews"]
    platform_meta = state["platform_meta"]

    scores = compute_scores(prices, reviews, platform_meta)
    winner = pick_winner(scores)

    winner_price = prices[winner]
    winner_rating = reviews.get(winner, {}).get("rating", 4.0)
    winner_score = scores[winner]

    # Build comparison list for frontend
    all_platforms = [
        {
            "name": p.title(),
            "price": prices[p],
            "rating": reviews.get(p, {}).get("rating", 0.0),
            "score": scores[p],
        }
        for p in prices
    ]
    all_platforms.sort(key=lambda x: x["score"], reverse=True)

    # Generate reason via Mistral — fallback to rule-based string if Ollama is down
    try:
        llm = ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url)
        prompt = build_reason_prompt(winner, winner_price, winner_rating, winner_score, prices, reviews, platform_meta)
        response = llm.invoke([HumanMessage(content=prompt)])
        reason = response.content.strip()
        logger.info(f"Decision agent LLM reason generated for {winner}")
    except Exception as e:
        logger.warning(f"Ollama unavailable ({e}), using rule-based reason")
        meta = platform_meta.get(winner, {})
        reason = (
            f"{winner.title()} offers the best value at ₹{winner_price:,.0f} "
            f"with {meta.get('delivery_days', 'N/A')}-day delivery and a "
            f"strong return policy (score: {meta.get('return_policy_score', 'N/A')})."
        )

    logger.info(f"Decision agent recommends: {winner} (score={winner_score})")
    return {
        "scores": scores,
        "recommended_platform": winner,
        "recommended_price": winner_price,
        "recommended_rating": winner_rating,
        "reason": reason,
        "all_platforms": all_platforms,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_decision_agent.py -v
```

Expected: 6 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add backend/agents/decision_agent.py backend/tests/test_decision_agent.py
git commit -m "feat: add decision agent with weighted scoring and Mistral LLM reasoning"
```

---

## Task 10: LangGraph Workflow

**Files:**
- Create: `backend/graph/workflow.py`
- Create: `backend/tests/test_workflow.py`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_workflow.py
from unittest.mock import patch, MagicMock
from bs4 import BeautifulSoup
from graph.workflow import build_graph, run_workflow

SAMPLE_SHOPPING_HTML = """
<html><body>
  <div class="sh-dgr__grid-result">
    <h3 class="Xjkr3b">iPhone 15</h3>
    <span class="a8Pemb OFFNJ">₹58,999</span>
    <div class="aULzUe IuHnof">Amazon</div>
  </div>
  <div class="sh-dgr__grid-result">
    <h3 class="Xjkr3b">iPhone 15</h3>
    <span class="a8Pemb OFFNJ">₹56,500</span>
    <div class="aULzUe IuHnof">Flipkart</div>
  </div>
</body></html>
"""

SAMPLE_REVIEW_HTML = """
<html><body><div class="Aq14fc">4.4</div></body></html>
"""


def test_build_graph_returns_compiled_graph():
    graph = build_graph()
    assert graph is not None


def test_run_workflow_returns_recommendation():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "Flipkart is cheapest with fast delivery."

    shopping_soup = BeautifulSoup(SAMPLE_SHOPPING_HTML, "html.parser")
    review_soup = BeautifulSoup(SAMPLE_REVIEW_HTML, "html.parser")

    def mock_fetch(url, timeout=10):
        if "tbm=shop" in url:
            return shopping_soup
        return review_soup

    with patch("agents.price_agent.fetch_page", side_effect=mock_fetch):
        with patch("agents.review_agent.fetch_page", side_effect=mock_fetch):
            with patch("agents.decision_agent.ChatOllama", return_value=mock_llm):
                result = run_workflow("best place to buy iPhone 15")

    assert result["recommended_platform"] in ["amazon", "flipkart"]
    assert result["recommended_price"] > 0
    assert result["reason"] != ""
    assert len(result["all_platforms"]) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_workflow.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'graph.workflow'`

- [ ] **Step 3: Write implementation**

```python
# backend/graph/workflow.py
from langgraph.graph import StateGraph, END
from graph.state import AgentState
from agents.orchestrator import orchestrator_node
from agents.price_agent import price_node
from agents.review_agent import review_node
from agents.platform_agent import platform_node
from agents.decision_agent import decision_node
from utils.logger import get_logger

logger = get_logger(__name__)


def build_graph():
    """Construct and compile the LangGraph StateGraph pipeline."""
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("price", price_node)
    workflow.add_node("review", review_node)
    workflow.add_node("platform", platform_node)
    workflow.add_node("decision", decision_node)

    # Linear pipeline edges
    workflow.set_entry_point("orchestrator")
    workflow.add_edge("orchestrator", "price")
    workflow.add_edge("price", "review")
    workflow.add_edge("review", "platform")
    workflow.add_edge("platform", "decision")
    workflow.add_edge("decision", END)

    return workflow.compile()


# Module-level compiled graph (reused across requests)
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_workflow(query: str) -> dict:
    """Run the full agent pipeline for a user query. Returns final AgentState."""
    logger.info(f"Starting workflow for query: {query}")

    initial_state: AgentState = {
        "query": query,
        "product_name": "",
        "prices": {},
        "reviews": {},
        "platform_meta": {},
        "scores": {},
        "recommended_platform": "",
        "recommended_price": 0.0,
        "recommended_rating": 0.0,
        "reason": "",
        "all_platforms": [],
        "error": None,
    }

    graph = get_graph()
    final_state = graph.invoke(initial_state)
    logger.info(f"Workflow complete. Recommended: {final_state.get('recommended_platform')}")
    return final_state
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_workflow.py -v
```

Expected: 2 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add backend/graph/workflow.py backend/tests/test_workflow.py
git commit -m "feat: add LangGraph workflow connecting all 5 agents in linear pipeline"
```

---

## Task 11: FastAPI Application

**Files:**
- Create: `backend/main.py`
- Create: `backend/tests/test_main.py`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_main.py
import pytest
from unittest.mock import patch
from httpx import AsyncClient, ASGITransport
from main import app

MOCK_WORKFLOW_RESULT = {
    "query": "iPhone 15",
    "product_name": "iphone 15",
    "prices": {"flipkart": 56500.0, "amazon": 58999.0},
    "reviews": {
        "flipkart": {"rating": 4.4, "sentiment_score": 0.88},
        "amazon": {"rating": 4.5, "sentiment_score": 0.90},
    },
    "platform_meta": {
        "flipkart": {"delivery_days": 3, "return_policy_score": 0.85, "seller_rating": 4.4},
        "amazon":   {"delivery_days": 2, "return_policy_score": 0.90, "seller_rating": 4.6},
    },
    "scores": {"flipkart": 0.78, "amazon": 0.71},
    "recommended_platform": "flipkart",
    "recommended_price": 56500.0,
    "recommended_rating": 4.4,
    "reason": "Flipkart has the lowest price and decent delivery.",
    "all_platforms": [
        {"name": "Flipkart", "price": 56500.0, "rating": 4.4, "score": 0.78},
        {"name": "Amazon",   "price": 58999.0, "rating": 4.5, "score": 0.71},
    ],
    "error": None,
}


@pytest.mark.asyncio
async def test_recommend_returns_200():
    with patch("main.run_workflow", return_value=MOCK_WORKFLOW_RESULT):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/recommend", json={"query": "iPhone 15"})
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_recommend_response_shape():
    with patch("main.run_workflow", return_value=MOCK_WORKFLOW_RESULT):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/recommend", json={"query": "iPhone 15"})
    data = response.json()
    assert data["recommended_platform"] == "Flipkart"
    assert data["price"] == 56500.0
    assert data["rating"] == 4.4
    assert "reason" in data
    assert "all_platforms" in data


@pytest.mark.asyncio
async def test_recommend_missing_query_returns_422():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/recommend", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_recommend_workflow_error_returns_500():
    with patch("main.run_workflow", side_effect=Exception("Scraping failed")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/recommend", json={"query": "iPhone 15"})
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_main.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: Write implementation**

```python
# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph.workflow import run_workflow
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Product Assistant API", version="1.0.0")

# Allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    query: str


class PlatformResult(BaseModel):
    name: str
    price: float
    rating: float
    score: float


class RecommendResponse(BaseModel):
    recommended_platform: str
    price: float
    rating: float
    reason: str
    all_platforms: list[PlatformResult]


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """Run the multi-agent LangGraph pipeline and return the best platform recommendation."""
    logger.info(f"Received recommend request: {request.query}")

    try:
        result = run_workflow(request.query)
    except Exception as e:
        logger.error(f"Workflow error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if result.get("error"):
        raise HTTPException(status_code=503, detail=result["error"])

    if not result.get("recommended_platform"):
        raise HTTPException(status_code=404, detail="No products found for this query")

    return RecommendResponse(
        recommended_platform=result["recommended_platform"].title(),
        price=result["recommended_price"],
        rating=result["recommended_rating"],
        reason=result["reason"],
        all_platforms=result["all_platforms"],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_main.py -v
```

Expected: 5 tests PASSED

- [ ] **Step 5: Run all backend tests**

```bash
pytest tests/ -v
```

Expected: All tests PASSED

- [ ] **Step 6: Commit**

```bash
git add backend/main.py backend/tests/test_main.py
git commit -m "feat: add FastAPI app with /recommend endpoint and CORS"
```

---

## Task 12: Frontend Scaffold

**Files:**
- Create: `frontend/` (Vite project)

- [ ] **Step 1: Scaffold Vite React project**

```bash
cd /home/siva/siva/myprojects/product_assistant
npm create vite@latest frontend -- --template react
cd frontend
npm install
npm install axios
```

- [ ] **Step 2: Verify dev server starts**

```bash
npm run dev
```

Expected: Vite dev server running at `http://localhost:5173`

Stop with Ctrl+C.

- [ ] **Step 3: Update vite.config.js to proxy API calls**

```js
// frontend/vite.config.js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/recommend': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
```

- [ ] **Step 4: Commit**

```bash
cd /home/siva/siva/myprojects/product_assistant
git add frontend/
git commit -m "feat: scaffold React Vite frontend with axios and API proxy"
```

---

## Task 13: API Client

**Files:**
- Create: `frontend/src/api.js`

- [ ] **Step 1: Create API wrapper**

```js
// frontend/src/api.js
import axios from 'axios'

const BASE_URL = 'http://localhost:8000'

/**
 * POST /recommend
 * @param {string} query - product search query
 * @returns {Promise<{recommended_platform, price, rating, reason, all_platforms}>}
 */
export async function getRecommendation(query) {
  const response = await axios.post(`${BASE_URL}/recommend`, { query })
  return response.data
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/api.js
git commit -m "feat: add axios API client for /recommend endpoint"
```

---

## Task 14: SearchBox Component

**Files:**
- Create: `frontend/src/components/SearchBox.jsx`

- [ ] **Step 1: Create component**

```jsx
// frontend/src/components/SearchBox.jsx
import { useState } from 'react'

/**
 * SearchBox — text input + submit button.
 * Props:
 *   onSearch(query: string) — called when user submits
 *   loading: bool — disables button when true
 */
export default function SearchBox({ onSearch, loading }) {
  const [query, setQuery] = useState('')

  function handleSubmit(e) {
    e.preventDefault()
    if (query.trim()) {
      onSearch(query.trim())
    }
  }

  return (
    <form className="search-box" onSubmit={handleSubmit}>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="e.g. iPhone 15 128GB"
        disabled={loading}
        className="search-input"
        aria-label="Product search"
      />
      <button
        type="submit"
        disabled={loading || !query.trim()}
        className="search-button"
      >
        {loading ? (
          <span className="spinner" aria-label="Loading" />
        ) : (
          'Find Best Platform'
        )}
      </button>
    </form>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/SearchBox.jsx
git commit -m "feat: add SearchBox component with loading state"
```

---

## Task 15: ResultCard Component

**Files:**
- Create: `frontend/src/components/ResultCard.jsx`

- [ ] **Step 1: Create component**

```jsx
// frontend/src/components/ResultCard.jsx
import { useState } from 'react'

/**
 * ResultCard — displays the top recommendation + expandable comparison table.
 * Props:
 *   result: { recommended_platform, price, rating, reason, all_platforms }
 */
export default function ResultCard({ result }) {
  const [showAll, setShowAll] = useState(false)

  const { recommended_platform, price, rating, reason, all_platforms } = result

  function renderStars(rating) {
    const full = Math.floor(rating)
    const half = rating - full >= 0.5 ? 1 : 0
    const empty = 5 - full - half
    return (
      <span className="stars" aria-label={`${rating} out of 5 stars`}>
        {'★'.repeat(full)}
        {half ? '½' : ''}
        {'☆'.repeat(empty)}
      </span>
    )
  }

  return (
    <div className="result-card">
      <div className="result-badge">{recommended_platform}</div>

      <div className="result-details">
        <div className="result-price">₹{price.toLocaleString('en-IN')}</div>
        <div className="result-rating">
          {renderStars(rating)}
          <span className="rating-value">{rating} / 5.0</span>
        </div>
      </div>

      <p className="result-reason">{reason}</p>

      <button
        className="toggle-button"
        onClick={() => setShowAll((v) => !v)}
      >
        {showAll ? 'Hide comparison ▲' : 'Compare all platforms ▼'}
      </button>

      {showAll && (
        <table className="comparison-table">
          <thead>
            <tr>
              <th>Platform</th>
              <th>Price (₹)</th>
              <th>Rating</th>
              <th>Score</th>
            </tr>
          </thead>
          <tbody>
            {all_platforms.map((p) => (
              <tr
                key={p.name}
                className={p.name === recommended_platform ? 'row-winner' : ''}
              >
                <td>{p.name}</td>
                <td>{p.price.toLocaleString('en-IN')}</td>
                <td>{p.rating}</td>
                <td>{(p.score * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/ResultCard.jsx
git commit -m "feat: add ResultCard with star rating and expandable comparison table"
```

---

## Task 16: SearchHistory Component

**Files:**
- Create: `frontend/src/components/SearchHistory.jsx`

- [ ] **Step 1: Create component**

```jsx
// frontend/src/components/SearchHistory.jsx
/**
 * SearchHistory — shows last 5 searches as clickable chips.
 * Props:
 *   history: string[]
 *   onSelect(query: string) — called when chip is clicked
 */
export default function SearchHistory({ history, onSelect }) {
  if (!history || history.length === 0) return null

  return (
    <div className="history-container">
      <span className="history-label">Recent:</span>
      {history.map((item, i) => (
        <button
          key={i}
          className="history-chip"
          onClick={() => onSelect(item)}
        >
          {item}
        </button>
      ))}
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/SearchHistory.jsx
git commit -m "feat: add SearchHistory component with localStorage chips"
```

---

## Task 17: App.jsx + CSS

**Files:**
- Modify: `frontend/src/App.jsx`
- Modify: `frontend/src/App.css`

- [ ] **Step 1: Write App.jsx**

```jsx
// frontend/src/App.jsx
import { useState, useEffect } from 'react'
import SearchBox from './components/SearchBox'
import ResultCard from './components/ResultCard'
import SearchHistory from './components/SearchHistory'
import { getRecommendation } from './api'
import './App.css'

const HISTORY_KEY = 'product_assistant_history'
const MAX_HISTORY = 5

export default function App() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [history, setHistory] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []
    } catch {
      return []
    }
  })

  function addToHistory(query) {
    setHistory((prev) => {
      const updated = [query, ...prev.filter((q) => q !== query)].slice(0, MAX_HISTORY)
      localStorage.setItem(HISTORY_KEY, JSON.stringify(updated))
      return updated
    })
  }

  async function handleSearch(query) {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await getRecommendation(query)
      setResult(data)
      addToHistory(query)
    } catch (err) {
      const message =
        err.response?.data?.detail ||
        err.message ||
        'Something went wrong. Please try again.'
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">🛒 Product Assistant</h1>
        <p className="app-subtitle">Find the best platform to buy any product</p>
      </header>

      <main className="app-main">
        <SearchBox onSearch={handleSearch} loading={loading} />
        <SearchHistory history={history} onSelect={handleSearch} />

        {error && (
          <div className="error-banner" role="alert">
            ⚠️ {error}
          </div>
        )}

        {loading && (
          <div className="loading-container" aria-live="polite">
            <div className="loading-spinner" />
            <p className="loading-text">Analyzing platforms...</p>
          </div>
        )}

        {result && !loading && <ResultCard result={result} />}
      </main>
    </div>
  )
}
```

- [ ] **Step 2: Write App.css**

```css
/* frontend/src/App.css */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #f0f4f8;
  color: #1a202c;
  min-height: 100vh;
}

.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 2rem 1rem;
}

.app-header { text-align: center; margin-bottom: 2rem; }
.app-title { font-size: 2rem; font-weight: 700; color: #2d3748; }
.app-subtitle { color: #718096; margin-top: 0.5rem; font-size: 1rem; }

.app-main {
  width: 100%;
  max-width: 640px;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

/* Search Box */
.search-box { display: flex; gap: 0.5rem; }
.search-input {
  flex: 1;
  padding: 0.75rem 1rem;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  font-size: 1rem;
  outline: none;
  transition: border-color 0.2s;
}
.search-input:focus { border-color: #4299e1; }
.search-input:disabled { background: #edf2f7; }

.search-button {
  padding: 0.75rem 1.5rem;
  background: #4299e1;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s;
  min-width: 160px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.search-button:hover:not(:disabled) { background: #3182ce; }
.search-button:disabled { background: #a0aec0; cursor: not-allowed; }

/* Spinner inside button */
.spinner {
  width: 18px; height: 18px;
  border: 3px solid rgba(255,255,255,0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
  display: inline-block;
}

/* History */
.history-container { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; }
.history-label { font-size: 0.85rem; color: #718096; }
.history-chip {
  padding: 0.3rem 0.75rem;
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 20px;
  font-size: 0.85rem;
  cursor: pointer;
  transition: background 0.2s;
}
.history-chip:hover { background: #ebf8ff; border-color: #90cdf4; }

/* Error */
.error-banner {
  background: #fff5f5;
  border: 1px solid #fc8181;
  color: #c53030;
  padding: 0.75rem 1rem;
  border-radius: 8px;
  font-size: 0.95rem;
}

/* Loading */
.loading-container { display: flex; flex-direction: column; align-items: center; gap: 0.75rem; padding: 2rem 0; }
.loading-spinner {
  width: 40px; height: 40px;
  border: 4px solid #e2e8f0;
  border-top-color: #4299e1;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
.loading-text { color: #718096; font-size: 0.95rem; }

@keyframes spin { to { transform: rotate(360deg); } }

/* Result Card */
.result-card {
  background: white;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 4px 6px rgba(0,0,0,0.07);
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.result-badge {
  display: inline-block;
  background: #48bb78;
  color: white;
  padding: 0.4rem 1rem;
  border-radius: 20px;
  font-weight: 700;
  font-size: 1.1rem;
  align-self: flex-start;
}

.result-details { display: flex; gap: 2rem; align-items: center; }
.result-price { font-size: 1.6rem; font-weight: 700; color: #2d3748; }
.result-rating { display: flex; align-items: center; gap: 0.5rem; }
.stars { color: #f6ad55; font-size: 1.2rem; }
.rating-value { color: #718096; font-size: 0.9rem; }

.result-reason { color: #4a5568; font-size: 0.95rem; line-height: 1.6; }

.toggle-button {
  background: none;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 0.4rem 0.75rem;
  cursor: pointer;
  font-size: 0.85rem;
  color: #4299e1;
  align-self: flex-start;
  transition: background 0.2s;
}
.toggle-button:hover { background: #ebf8ff; }

/* Comparison Table */
.comparison-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.comparison-table th, .comparison-table td {
  padding: 0.5rem 0.75rem;
  text-align: left;
  border-bottom: 1px solid #e2e8f0;
}
.comparison-table th { background: #f7fafc; font-weight: 600; color: #4a5568; }
.row-winner { background: #f0fff4; font-weight: 600; }
```

- [ ] **Step 3: Update main.jsx**

```jsx
// frontend/src/main.jsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
```

- [ ] **Step 4: Update index.html title**

```html
<!-- frontend/index.html -->
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Product Assistant</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

- [ ] **Step 5: Verify frontend builds**

```bash
cd frontend
npm run build
```

Expected: Build succeeds, `dist/` folder created.

- [ ] **Step 6: Commit**

```bash
cd /home/siva/siva/myprojects/product_assistant
git add frontend/src/App.jsx frontend/src/App.css frontend/src/main.jsx frontend/index.html
git commit -m "feat: complete React frontend with search, results, and history"
```

---

## Task 18: End-to-End Smoke Test

- [ ] **Step 1: Start Ollama with mistral**

```bash
ollama pull mistral
ollama serve &
```

Expected: Ollama server running at `http://localhost:11434`

- [ ] **Step 2: Start backend**

```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload
```

Expected: `Uvicorn running on http://127.0.0.1:8000`

- [ ] **Step 3: Test the /health endpoint**

```bash
curl http://localhost:8000/health
```

Expected: `{"status":"ok"}`

- [ ] **Step 4: Test the /recommend endpoint**

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "Samsung Galaxy S24"}'
```

Expected: JSON response with `recommended_platform`, `price`, `rating`, `reason`, `all_platforms`

- [ ] **Step 5: Start frontend**

```bash
cd frontend
npm run dev
```

Expected: App running at `http://localhost:5173`

- [ ] **Step 6: Manual smoke test**
  - Open `http://localhost:5173`
  - Type "iPhone 15" in search box
  - Click "Find Best Platform"
  - Verify loading spinner appears
  - Verify ResultCard shows platform, price, rating, reason
  - Verify "Compare all platforms" expander works
  - Verify search history chip appears for repeated searches

- [ ] **Step 7: Run full backend test suite**

```bash
cd backend
pytest tests/ -v --tb=short
```

Expected: All tests PASSED

- [ ] **Step 8: Final commit**

```bash
cd /home/siva/siva/myprojects/product_assistant
git add .
git commit -m "feat: complete product assistant — backend + frontend + all tests passing"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** All 5 agents implemented (orchestrator, price, review, platform, decision) ✓
- [x] `POST /recommend` endpoint with correct request/response shape ✓
- [x] Google Shopping scraping (BeautifulSoup + rotated User-Agent) ✓
- [x] Mistral via Ollama in decision agent with fallback ✓
- [x] CORS enabled for React frontend ✓
- [x] `.env` config via pydantic-settings ✓
- [x] All error scenarios handled (scrape failure, Ollama down, no results) ✓
- [x] Frontend: SearchBox, ResultCard, SearchHistory, loading/error states ✓
- [x] `all_platforms` comparison table in ResultCard ✓
- [x] localStorage history (last 5 searches) ✓
- [x] No placeholders or TBDs ✓
- [x] Types consistent across all tasks (AgentState fields match throughout) ✓
