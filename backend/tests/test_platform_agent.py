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
