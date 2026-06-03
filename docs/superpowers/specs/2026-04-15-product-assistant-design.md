# Product Assistant — Design Spec
**Date:** 2026-04-15
**Status:** Approved

---

## Goal

An AI-powered product recommendation system that scrapes Google Shopping in real-time and suggests the best platform (Amazon, Flipkart, Croma, etc.) to buy a product based on price, reviews, and platform reliability.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI (Python, async) |
| AI Orchestration | LangGraph `StateGraph` + LangChain |
| LLM | `mistral` via Ollama (`http://localhost:11434`) |
| Scraping | `requests` + `BeautifulSoup4` (Google Shopping) |
| Frontend | React (Vite) + Axios |
| Styling | Plain CSS (centered, modern, minimal) |

---

## Architecture: Linear Pipeline (Approach A)

```
User Query
   │
   ▼
POST /recommend (FastAPI)
   │
   ▼
LangGraph StateGraph
   │
   ├─► orchestrator_node   — parse & normalize query, seed AgentState
   ├─► price_node          — scrape Google Shopping for real prices
   ├─► review_node         — scrape Google Search for ratings/sentiment
   ├─► platform_node       — static lookup for delivery/return/seller data
   └─► decision_node       — score all platforms + mistral → reason text
   │
   ▼
JSON Response → React Frontend
```

Each node reads from and writes to a shared `AgentState` (TypedDict). Nodes run sequentially.

---

## Backend Project Structure

```
backend/
├── agents/
│   ├── orchestrator.py     # parses query, seeds state
│   ├── price_agent.py      # Google Shopping scraper
│   ├── review_agent.py     # Google review/rating scraper
│   ├── platform_agent.py   # static platform metadata lookup
│   └── decision_agent.py   # scoring + mistral LLM reasoning
├── graph/
│   └── workflow.py         # LangGraph StateGraph definition
├── tools/
│   └── scraper.py          # shared requests + BeautifulSoup utilities
├── utils/
│   └── logger.py           # Python logging setup
├── main.py                 # FastAPI app, CORS, /recommend endpoint
├── config.py               # settings (Ollama URL, model name, timeouts)
└── requirements.txt
```

---

## Frontend Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── SearchBox.jsx       # input + submit button
│   │   ├── ResultCard.jsx      # recommended platform card
│   │   └── SearchHistory.jsx   # last 5 searches from localStorage
│   ├── App.jsx                 # state: loading, result, error, history
│   ├── main.jsx
│   └── api.js                  # axios POST /recommend wrapper
├── index.html
├── package.json
└── vite.config.js
```

---

## Agent Details

### Orchestrator Node
- Receives raw user query string
- Normalizes product name (strips filler words like "best place to buy", "under X")
- Seeds `AgentState` with `query` and `product_name`

### Price Node
- Scrapes: `https://www.google.com/search?q={product_name}&tbm=shop`
- Headers: rotated User-Agent strings to avoid blocks
- Parses: product title, price (₹), seller/platform name from shopping cards
- Groups results by platform, picks lowest price per platform
- Rate limit: `time.sleep(1)` between requests

### Review Node
- Scrapes: `https://www.google.com/search?q={product_name}+reviews`
- Extracts: star ratings from rich snippets / knowledge panel
- Falls back to mistral sentiment estimation if no structured rating found
- Returns: `{ platform: { rating: float, sentiment_score: float (0-1) } }`

### Platform Node
- Static lookup table (no live scraping — this data is stable):
```python
PLATFORM_META = {
    "amazon":   { "delivery_days": 2,  "return_policy_score": 0.90, "seller_rating": 4.6 },
    "flipkart": { "delivery_days": 3,  "return_policy_score": 0.85, "seller_rating": 4.4 },
    "croma":    { "delivery_days": 4,  "return_policy_score": 0.80, "seller_rating": 4.2 },
    "reliance": { "delivery_days": 5,  "return_policy_score": 0.75, "seller_rating": 4.0 },
}
```

### Decision Node
**Scoring formula:**
```
price_score    = 1 - (platform_price / max_price)   # lower price = higher score
review_score   = (rating / 5.0 + sentiment_score) / 2
platform_score = (1 / delivery_days * 0.4) + (return_policy_score * 0.3) + (seller_rating / 5 * 0.3)

final_score = (0.4 * price_score) + (0.3 * review_score) + (0.3 * platform_score)
```
- Picks platform with highest `final_score`
- Calls mistral via `ChatOllama` to generate 1-2 sentence `reason` explanation

---

## API Contract

### `POST /recommend`

**Request:**
```json
{ "query": "iPhone 15 128GB" }
```

**Response:**
```json
{
  "recommended_platform": "Flipkart",
  "price": 56999,
  "rating": 4.4,
  "reason": "Flipkart offers the lowest price at ₹56,999 with strong seller ratings and a solid return policy.",
  "all_platforms": [
    { "name": "Amazon",   "price": 58999, "rating": 4.5, "score": 0.71 },
    { "name": "Flipkart", "price": 56999, "rating": 4.4, "score": 0.78 }
  ]
}
```

**Error response (500):**
```json
{ "detail": "Scraping failed: no results found for query" }
```

---

## Frontend UI

- **Layout:** centered single-column, max-width 640px
- **States:**
  - Default: search box + "Find Best Platform" button
  - Loading: spinner animation, button disabled
  - Result: `ResultCard` showing platform badge, price, star rating, reason text, expandable `all_platforms` comparison table
  - Error: red banner with message
- **History:** last 5 searches shown as clickable chips below search box (stored in `localStorage`)

---

## CORS & Configuration

- FastAPI CORS: allow `http://localhost:5173`
- `config.py` reads from `.env`:
  - `OLLAMA_BASE_URL=http://localhost:11434`
  - `OLLAMA_MODEL=mistral`
  - `SCRAPE_TIMEOUT=10`

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Google blocks scrape | Return 503 with `"Scraping temporarily blocked, try again"` |
| No results found | Return 404 with `"No products found for query"` |
| Ollama not running | Decision agent returns rule-based reason string (no LLM fallback crash) |
| Partial platform data | Score only platforms with complete data, skip others |

---

## Run Instructions

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

**Prerequisites:**
- Ollama running: `ollama serve` + `ollama pull mistral`
- Python 3.11+
- Node 18+
