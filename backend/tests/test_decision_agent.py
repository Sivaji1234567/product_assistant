from unittest.mock import patch, MagicMock
from agents.decision_agent import compute_scores, pick_winner, build_reason_prompt, decision_node
from graph.state import AgentState

PRICES = {"amazon": 75000.0, "flipkart": 54000.0}
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
    assert scores["flipkart"] > scores["amazon"]


def test_pick_winner_returns_highest_scoring_platform():
    scores = {"amazon": 0.71, "flipkart": 0.78}
    assert pick_winner(scores) == "flipkart"


def test_build_reason_prompt_contains_platform_name():
    prompt = build_reason_prompt("flipkart", 54000.0, 4.4, 0.78, PRICES, REVIEWS, PLATFORM_META)
    assert "flipkart" in prompt.lower()
    assert "54000" in prompt or "54,000" in prompt


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
    assert result["reason"] != ""
