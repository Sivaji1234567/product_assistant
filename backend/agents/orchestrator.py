import re
from graph.state import AgentState
from utils.logger import get_logger

logger = get_logger(__name__)

FILLER_PATTERNS = [
    r"\bbest place to buy\b",
    r"\bwhere (can i |should i )?buy\b",
    r"\bbest place to get\b",
    r"\bwhere to get\b",
    r"\bshould i buy\b",
    r"\bunder\s+\d+\b",
    r"\bbelow\s+\d+\b",
    r"\bwithin\s+\d+\b",
    r"\bless than\s+\d+\b",
    r"\bprice of\b",
    r"\bcost of\b",
    r"\bbuy\b",
    r"\bcheapest\b",
]


def normalize_query(query: str) -> str:
    text = query.lower().strip()
    for pattern in FILLER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def orchestrator_node(state: AgentState) -> dict:
    logger.info(f"Orchestrator processing query: {state['query']}")
    product_name = normalize_query(state["query"])
    logger.info(f"Normalized product name: {product_name}")
    return {"query": state["query"], "product_name": product_name}
