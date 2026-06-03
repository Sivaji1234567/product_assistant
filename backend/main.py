from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph.workflow import run_workflow
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Product Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    query: str


class PlatformResult(BaseModel):
    name: str
    price: float
    rating: float
    score: float


class RecommendResponse(BaseModel):
    recommended_platform: str
    price: float
    rating: float
    reason: str
    all_platforms: list[PlatformResult]


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    logger.info(f"Received recommend request: {request.query}")

    try:
        result = run_workflow(request.query)
    except Exception as e:
        logger.error(f"Workflow error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if result.get("error"):
        raise HTTPException(status_code=503, detail=result["error"])

    if not result.get("recommended_platform"):
        raise HTTPException(status_code=404, detail="No products found for this query")

    return RecommendResponse(
        recommended_platform=result["recommended_platform"].title(),
        price=result["recommended_price"],
        rating=result["recommended_rating"],
        reason=result["reason"],
        all_platforms=result["all_platforms"],
    )
