import logging
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.azure_client import llm_client
from app.schemas import (
    HealthResponse,
    JudgeRequest,
    JudgeResponse,
    RagEvalRequest,
    RagEvalResponse,
    AttackRunRequest,
    AttackRunResponse
)
from app.judge import evaluate_with_judge
from app.rag_metrics import evaluate_rag_pipeline
from app.attacks import run_attack_suite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("pravah_eval_worker")

app = FastAPI(
    title="Pravah Eval Worker",
    version="0.1.0",
    description="FastAPI LLM-as-a-judge & RAG Groundedness Evaluation Service for Pravah"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security schemes: X-API-Key, X-Pravah-Secret, and Bearer token
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
pravah_secret_header = APIKeyHeader(name="X-Pravah-Secret", auto_error=False)
bearer_auth = HTTPBearer(auto_error=False)


async def verify_auth_secret(
    api_key: Optional[str] = Security(api_key_header),
    pravah_secret: Optional[str] = Security(pravah_secret_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_auth),
) -> bool:
    """Verifies that the request provides a valid shared secret token."""
    expected_secret = settings.WORKER_AUTH_SECRET

    provided_secret = api_key or pravah_secret
    if not provided_secret and bearer:
        provided_secret = bearer.credentials

    if not provided_secret or provided_secret != expected_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication secret header (X-API-Key, X-Pravah-Secret, or Bearer token)",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Returns worker health and integration statuses."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        azure_configured=llm_client.is_configured,
        langfuse_configured=bool(settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY),
        mock_mode=settings.MOCK_LLM or not llm_client.is_configured
    )


@app.post("/judge", response_model=JudgeResponse, tags=["LLM Judge"])
async def judge_endpoint(
    request: JudgeRequest,
    _auth: bool = Depends(verify_auth_secret)
):
    """
    Evaluates candidate text using the requested versioned rubric with an LLM judge.
    Returns 1-5 scores and detailed reasoning string.
    """
    try:
        response = evaluate_with_judge(request)
        return response
    except Exception as e:
        logger.error(f"Error executing judge evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/rag-eval", response_model=RagEvalResponse, tags=["RAG Metrics"])
async def rag_eval_endpoint(
    request: RagEvalRequest,
    _auth: bool = Depends(verify_auth_secret)
):
    """
    Computes all 4 core RAG metrics:
    - Faithfulness (claim breakdown & groundedness)
    - Answer Relevance (rating 1-5)
    - Context Precision (rank-weighted MAP@K)
    - Context Recall (reference answer claim recovery)
    Returns combined summary stored in Pravah's judgeReasoning field.
    """
    try:
        response = evaluate_rag_pipeline(request)
        return response
    except Exception as e:
        logger.error(f"Error executing RAG evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/attack-run", response_model=AttackRunResponse, tags=["Security Attacks"])
async def attack_run_endpoint(
    request: AttackRunRequest,
    _auth: bool = Depends(verify_auth_secret)
):
    """
    Executes prompt injection and jailbreak attack suite against a pipeline.
    Calculates block rate and logs failed cases.
    """
    try:
        response = await run_attack_suite(request)
        return response
    except Exception as e:
        logger.error(f"Error executing attack suite: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
