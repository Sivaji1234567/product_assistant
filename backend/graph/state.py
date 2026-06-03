from typing import TypedDict, Optional, List, Dict


class AgentState(TypedDict):
    query: str
    product_name: str
    prices: Dict[str, float]
    reviews: Dict[str, Dict]
    platform_meta: Dict[str, Dict]
    scores: Dict[str, float]
    recommended_platform: str
    recommended_price: float
    recommended_rating: float
    reason: str
    all_platforms: List[Dict]
    error: Optional[str]
