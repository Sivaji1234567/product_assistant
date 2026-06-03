from unittest.mock import patch, MagicMock
from bs4 import BeautifulSoup
from graph.workflow import build_graph, run_workflow

SAMPLE_SHOPPING_HTML = """
<html><body>
  <div class="sh-dgr__grid-result">
    <h3 class="Xjkr3b">iPhone 15</h3>
    <span class="a8Pemb OFFNJ">₹58,999</span>
    <div class="aULzUe IuHnof">Amazon</div>
  </div>
  <div class="sh-dgr__grid-result">
    <h3 class="Xjkr3b">iPhone 15</h3>
    <span class="a8Pemb OFFNJ">₹56,500</span>
    <div class="aULzUe IuHnof">Flipkart</div>
  </div>
</body></html>
"""

SAMPLE_REVIEW_HTML = """
<html><body><div class="Aq14fc">4.4</div></body></html>
"""


def test_build_graph_returns_compiled_graph():
    graph = build_graph()
    assert graph is not None


def test_run_workflow_returns_recommendation():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "Flipkart is cheapest with fast delivery."

    shopping_soup = BeautifulSoup(SAMPLE_SHOPPING_HTML, "html.parser")
    review_soup = BeautifulSoup(SAMPLE_REVIEW_HTML, "html.parser")

    def mock_fetch(url, timeout=10):
        if "tbm=shop" in url:
            return shopping_soup
        return review_soup

    with patch("agents.price_agent.fetch_page", side_effect=mock_fetch):
        with patch("agents.review_agent.fetch_page", side_effect=mock_fetch):
            with patch("agents.decision_agent.ChatOllama", return_value=mock_llm):
                result = run_workflow("best place to buy iPhone 15")

    assert result["recommended_platform"] in ["amazon", "flipkart"]
    assert result["recommended_price"] > 0
    assert result["reason"] != ""
    assert len(result["all_platforms"]) > 0
