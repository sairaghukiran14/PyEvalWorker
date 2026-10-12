from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# Health Schema
class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="Operational status of the worker")
    version: str = Field(default="0.1.0", description="Worker version")
    azure_configured: bool = Field(description="True if Azure OpenAI credentials are configured")
    langfuse_configured: bool = Field(description="True if Langfuse credentials are configured")
    mock_mode: bool = Field(description="True if running in mock/offline mode")


# Judge Schemas
class JudgeRequest(BaseModel):
    output: str = Field(..., description="The candidate output to score")
    source: Optional[str] = Field(None, description="Optional original source text (e.g. for translation or transcription)")
    question: Optional[str] = Field(None, description="Optional original question/prompt that elicited the output")
    context: Optional[List[str]] = Field(default=None, description="Optional retrieved context chunks or reference material")
    rubric: str = Field(default="faithfulness", description="Name of the rubric file (e.g. 'faithfulness', 'answer_relevance', 'fluency', 'meaning_preservation')")
    criteria: Optional[List[str]] = Field(default=None, description="Optional custom criteria to score")


class JudgeResponse(BaseModel):
    scores: Dict[str, float] = Field(..., description="Scores (1 to 5 per criterion)")
    reasoning: str = Field(..., description="Written explanation and justification for the scores")
    claims: Optional[List[Dict[str, Any]]] = Field(default=None, description="Optional breakdown of atomic claims if applicable")


# RAG Eval Schemas
class RagEvalRequest(BaseModel):
    question: str = Field(..., description="The user's query/question")
    answer: str = Field(..., description="The generated answer from the RAG pipeline")
    chunks: List[str] = Field(..., description="List of retrieved context chunks")
    reference: Optional[str] = Field(None, description="Ground truth reference answer (required for context recall)")


class RagEvalResponse(BaseModel):
    faithfulness: float = Field(..., ge=0.0, le=1.0, description="Faithfulness score (0.0 to 1.0): ratio of supported claims to total claims")
    answer_relevance: float = Field(..., ge=0.0, le=5.0, description="Answer relevance score (1.0 to 5.0 rating against rubric)")
    context_precision: float = Field(..., ge=0.0, le=1.0, description="Context precision (0.0 to 1.0): rank-weighted retrieval precision")
    context_recall: float = Field(..., ge=0.0, le=1.0, description="Context recall (0.0 to 1.0): share of reference claims found in chunks")
    reasoning: str = Field(..., description="Detailed combined reasoning stored into Pravah's judgeReasoning field")
    detailed_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Step-by-step breakdown of claims, relevancies, and ranking")


# Attack Run Schemas
class AttackCase(BaseModel):
    id: str
    name: str
    type: str
    prompt: str


class AttackRunRequest(BaseModel):
    pipeline_id: str = Field(..., description="Target pipeline identifier to audit")
    attack_set: Optional[str] = Field(default="data/attacks.jsonl", description="Path to attack set file or identifier")
    target_url: Optional[str] = Field(None, description="Optional URL of the pipeline to send attacks against")
    attacks: Optional[List[AttackCase]] = Field(None, description="Optional custom attack cases to run")


class FailedAttackCase(BaseModel):
    id: str
    name: str
    type: str
    prompt: str
    pipeline_response: str
    failure_reason: str


class AttackRunResponse(BaseModel):
    pipeline_id: str
    total: int
    blocked: int
    block_rate: float
    failed_cases: List[FailedAttackCase]
