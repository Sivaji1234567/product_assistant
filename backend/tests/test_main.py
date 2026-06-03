import pytest
from unittest.mock import patch
from httpx import AsyncClient, ASGITransport
from main import app

MOCK_WORKFLOW_RESULT = {
    "query": "iPhone 15",
    "product_name": "iphone 15",
    "prices": {"flipkart": 56500.0, "amazon": 58999.0},
    "reviews": {
        "flipkart": {"rating": 4.4, "sentiment_score": 0.88},
        "amazon": {"rating": 4.5, "sentiment_score": 0.90},
    },
    "platform_meta": {
        "flipkart": {"delivery_days": 3, "return_policy_score": 0.85, "seller_rating": 4.4},
        "amazon":   {"delivery_days": 2, "return_policy_score": 0.90, "seller_rating": 4.6},
    },
    "scores": {"flipkart": 0.78, "amazon": 0.71},
    "recommended_platform": "flipkart",
    "recommended_price": 56500.0,
    "recommended_rating": 4.4,
    "reason": "Flipkart has the lowest price and decent delivery.",
    "all_platforms": [
        {"name": "Flipkart", "price": 56500.0, "rating": 4.4, "score": 0.78},
        {"name": "Amazon",   "price": 58999.0, "rating": 4.5, "score": 0.71},
    ],
    "error": None,
}


@pytest.mark.asyncio
async def test_recommend_returns_200():
    with patch("main.run_workflow", return_value=MOCK_WORKFLOW_RESULT):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/recommend", json={"query": "iPhone 15"})
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_recommend_response_shape():
    with patch("main.run_workflow", return_value=MOCK_WORKFLOW_RESULT):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/recommend", json={"query": "iPhone 15"})
    data = response.json()
    assert data["recommended_platform"] == "Flipkart"
    assert data["price"] == 56500.0
    assert data["rating"] == 4.4
    assert "reason" in data
    assert "all_platforms" in data


@pytest.mark.asyncio
async def test_recommend_missing_query_returns_422():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/recommend", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_recommend_workflow_error_returns_500():
    with patch("main.run_workflow", side_effect=Exception("Scraping failed")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/recommend", json={"query": "iPhone 15"})
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
