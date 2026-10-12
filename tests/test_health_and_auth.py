import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
    assert "azure_configured" in data
    assert "langfuse_configured" in data
    assert "mock_mode" in data


def test_auth_rejection_without_header():
    # Endpoints must reject without secret
    res_judge = client.post("/judge", json={"output": "test", "rubric": "faithfulness"})
    assert res_judge.status_code == 401

    res_rag = client.post("/rag-eval", json={
        "question": "q", "answer": "a", "chunks": ["c"]
    })
    assert res_rag.status_code == 401

    res_attack = client.post("/attack-run", json={"pipeline_id": "test-pipe"})
    assert res_attack.status_code == 401


def test_auth_rejection_with_invalid_secret():
    headers = {"X-API-Key": "invalid-wrong-secret"}
    response = client.post(
        "/judge",
        headers=headers,
        json={"output": "test", "rubric": "faithfulness"}
    )
    assert response.status_code == 401


def test_auth_success_with_api_key():
    headers = {"X-API-Key": settings.WORKER_AUTH_SECRET}
    response = client.post(
        "/judge",
        headers=headers,
        json={"output": "Pravah is an AI orchestrator.", "rubric": "faithfulness"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "scores" in data
    assert "reasoning" in data


def test_auth_success_with_pravah_secret_header():
    headers = {"X-Pravah-Secret": settings.WORKER_AUTH_SECRET}
    response = client.post(
        "/judge",
        headers=headers,
        json={"output": "Pravah is an AI orchestrator.", "rubric": "faithfulness"}
    )
    assert response.status_code == 200


def test_auth_success_with_bearer_token():
    headers = {"Authorization": f"Bearer {settings.WORKER_AUTH_SECRET}"}
    response = client.post(
        "/judge",
        headers=headers,
        json={"output": "Pravah is an AI orchestrator.", "rubric": "faithfulness"}
    )
    assert response.status_code == 200
