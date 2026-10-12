import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings

client = TestClient(app)
AUTH_HEADERS = {"X-API-Key": settings.WORKER_AUTH_SECRET}
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_rag_eval_single_payload():
    payload = {
        "question": "What is Pravah?",
        "answer": "Pravah is a high-throughput multilingual AI pipeline orchestrator.",
        "chunks": [
            "Pravah is a high-throughput multilingual AI pipeline orchestrator designed for real-time speech and language processing.",
            "It connects streaming translation workers over gRPC."
        ],
        "reference": "Pravah is a high-throughput multilingual AI pipeline orchestrator."
    }
    response = client.post("/rag-eval", headers=AUTH_HEADERS, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["faithfulness"] <= 1.0
    assert 1.0 <= data["answer_relevance"] <= 5.0
    assert 0.0 <= data["context_precision"] <= 1.0
    assert 0.0 <= data["context_recall"] <= 1.0
    assert "reasoning" in data
    assert "detailed_metrics" in data


def test_rag_eval_hallucination_penalty():
    payload = {
        "question": "What is Pravah?",
        "answer": "Pravah is a database built on Llama-3 that deletes all chunks.",
        "chunks": ["Pravah is a high-throughput multilingual AI pipeline orchestrator."],
        "reference": "Pravah is an AI orchestrator."
    }
    response = client.post("/rag-eval", headers=AUTH_HEADERS, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["faithfulness"] < 0.6  # Penalized for hallucination


def test_all_four_metrics_computed_for_30_question_set():
    """Validates Feature #3: All four metrics computed for a 30-question set."""
    golden_file = DATA_DIR / "golden_qa.jsonl"
    with open(golden_file, "r", encoding="utf-8") as f:
        cases = [json.loads(line.strip()) for line in f if line.strip()]

    assert len(cases) == 30, f"Expected 30 golden QA cases, found {len(cases)}"

    processed = 0
    for case in cases:
        payload = {
            "question": case["question"],
            "answer": case["candidate_answer"],
            "chunks": case["chunks"],
            "reference": case.get("reference")
        }
        res = client.post("/rag-eval", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 200
        data = res.json()

        assert "faithfulness" in data
        assert "answer_relevance" in data
        assert "context_precision" in data
        assert "context_recall" in data
        assert len(data["reasoning"]) > 10

        processed += 1

    assert processed == 30
