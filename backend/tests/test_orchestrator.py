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
