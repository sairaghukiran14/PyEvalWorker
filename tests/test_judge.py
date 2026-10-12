import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings

client = TestClient(app)
AUTH_HEADERS = {"X-API-Key": settings.WORKER_AUTH_SECRET}
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_judge_faithfulness():
    payload = {
        "output": "Pravah is a high-throughput multilingual AI pipeline orchestrator.",
        "context": ["Pravah is a high-throughput multilingual AI pipeline orchestrator."],
        "rubric": "faithfulness"
    }
    response = client.post("/judge", headers=AUTH_HEADERS, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "faithfulness" in data["scores"]
    assert 1.0 <= data["scores"]["faithfulness"] <= 5.0
    assert len(data["reasoning"]) > 10


def test_judge_meaning_preservation():
    payload = {
        "source": "The build fails if candidate scores are lower than the baseline.",
        "output": "If the candidate model scores below baseline, CI terminates with an error.",
        "rubric": "meaning_preservation"
    }
    response = client.post("/judge", headers=AUTH_HEADERS, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "meaning_preservation" in data["scores"]
    assert data["scores"]["meaning_preservation"] >= 4.0
    assert "reasoning" in data


def test_judge_fluency():
    payload = {
        "output": "Azure OpenAI provides hosted model endpoints with enterprise reliability guarantees.",
        "rubric": "fluency"
    }
    response = client.post("/judge", headers=AUTH_HEADERS, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fluency" in data["scores"]
    assert data["scores"]["fluency"] >= 4.0
    assert "reasoning" in data


def test_judge_returns_scores_and_reasoning_for_30_cases():
    """Validates Feature #2: Judge returns scores and reasoning for 30 cases."""
    labels_file = DATA_DIR / "human_labels.jsonl"
    with open(labels_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 30, f"Expected 30 validation cases, found {len(lines)}"

    scored_count = 0
    for line in lines:
        case = json.loads(line)
        payload = {
            "output": case["output"],
            "source": case.get("source"),
            "question": case.get("question"),
            "context": case.get("context", []),
            "rubric": case["rubric"]
        }
        res = client.post("/judge", headers=AUTH_HEADERS, json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "scores" in data
        assert len(data["reasoning"]) > 5
        scored_count += 1

    assert scored_count == 30
