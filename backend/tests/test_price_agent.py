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
