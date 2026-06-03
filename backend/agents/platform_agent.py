from graph.state import AgentState
from utils.logger import get_logger

logger = get_logger(__name__)

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
    return PLATFORM_META.get(platform, DEFAULT_META)


def platform_node(state: AgentState) -> dict:
    platforms = list(state["prices"].keys())
    platform_meta = {p: get_platform_meta(p) for p in platforms}
    logger.info(f"Platform agent enriched {len(platform_meta)} platforms")
    return {"platform_meta": platform_meta}
