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
    for platform in ["amazon", "flipkart"]:
        assert platform in result["reviews"]
        assert "rating" in result["reviews"][platform]
        assert "sentiment_score" in result["reviews"][platform]
