from graph.state import AgentState


def test_agent_state_is_typed_dict():
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
