from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from graph.state import AgentState
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def compute_scores(prices: dict, reviews: dict, platform_meta: dict) -> dict:
    if not prices:
        return {}

    max_price = max(prices.values())
    scores = {}

    for platform in prices:
        price = prices[platform]
        review = reviews.get(platform, {"rating": 4.0, "sentiment_score": 0.8})
        meta = platform_meta.get(platform, {"delivery_days": 7, "return_policy_score": 0.6, "seller_rating": 3.5})

        price_score = 1.0 - (price / max_price) if max_price > 0 else 0.0
        review_score = (review["rating"] / 5.0 + review["sentiment_score"]) / 2.0

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
    return max(scores, key=scores.get)


def build_reason_prompt(
    winner: str, price: float, rating: float, score: float,
    prices: dict, reviews: dict, platform_meta: dict
) -> str:
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
    prices = state["prices"]
    reviews = state["reviews"]
    platform_meta = state["platform_meta"]

    scores = compute_scores(prices, reviews, platform_meta)
    winner = pick_winner(scores)

    winner_price = prices[winner]
    winner_rating = reviews.get(winner, {}).get("rating", 4.0)
    winner_score = scores[winner]

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
