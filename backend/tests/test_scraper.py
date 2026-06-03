from unittest.mock import patch, MagicMock
from tools.scraper import get_headers, fetch_page, parse_price, normalize_platform


def test_get_headers_returns_dict():
    headers = get_headers()
    assert "User-Agent" in headers
    assert "Accept" in headers


def test_get_headers_rotates_user_agent():
    agents = {get_headers()["User-Agent"] for _ in range(20)}
    assert len(agents) > 1


def test_parse_price_rupee_symbol():
    assert parse_price("₹56,999") == 56999.0


def test_parse_price_with_commas():
    assert parse_price("58,000") == 58000.0


def test_parse_price_invalid_returns_zero():
    assert parse_price("N/A") == 0.0


def test_normalize_platform_amazon():
    assert normalize_platform("amazon.in") == "amazon"
    assert normalize_platform("sold by amazon") == "amazon"


def test_normalize_platform_flipkart():
    assert normalize_platform("flipkart") == "flipkart"
    assert normalize_platform("Flipkart India") == "flipkart"


def test_normalize_platform_unknown():
    assert normalize_platform("somerandombrand.com") == "other"


def test_fetch_page_returns_soup():
    from bs4 import BeautifulSoup
    mock_response = MagicMock()
    mock_response.text = "<html><body><p>test</p></body></html>"
    mock_response.raise_for_status = MagicMock()

    with patch("tools.scraper.requests.get", return_value=mock_response):
        with patch("tools.scraper.time.sleep"):
            soup = fetch_page("http://example.com", timeout=5)

    assert isinstance(soup, BeautifulSoup)
    assert soup.find("p").text == "test"
