from langgraph.graph import StateGraph, END
from graph.state import AgentState
from agents.orchestrator import orchestrator_node
from agents.price_agent import price_node
from agents.review_agent import review_node
from agents.platform_agent import platform_node
from agents.decision_agent import decision_node
from utils.logger import get_logger

logger = get_logger(__name__)


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("price", price_node)
    workflow.add_node("review", review_node)
    workflow.add_node("platform", platform_node)
    workflow.add_node("decision", decision_node)

    workflow.set_entry_point("orchestrator")
    workflow.add_edge("orchestrator", "price")
    workflow.add_edge("price", "review")
    workflow.add_edge("review", "platform")
    workflow.add_edge("platform", "decision")
    workflow.add_edge("decision", END)

    return workflow.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_workflow(query: str) -> dict:
    logger.info(f"Starting workflow for query: {query}")

    initial_state: AgentState = {
        "query": query,
        "product_name": "",
        "prices": {},
        "reviews": {},
        "platform_meta": {},
        "scores": {},
        "recommended_platform": "",
        "recommended_price": 0.0,
        "recommended_rating": 0.0,
        "reason": "",
        "all_platforms": [],
        "error": None,
    }

    graph = get_graph()
    final_state = graph.invoke(initial_state)
    logger.info(f"Workflow complete. Recommended: {final_state.get('recommended_platform')}")
    return final_state
