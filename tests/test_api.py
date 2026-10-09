"""
Context Surgeon - API Integration Tests
Tests all FastAPI endpoints with TestClient:
- GET /api/health
- POST /api/compress
- Input validation (invalid budgets, missing fields)
- GET /api/benchmarks
- POST /api/evaluate
- GET /api/benchmarks/results
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify health endpoint returns healthy status and tokenizer info."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "tokenizer" in data
    assert data["service"] == "Context Surgeon"


def test_compress_endpoint_valid_request():
    """Verify POST /api/compress returns compressed context with explainable sections."""
    payload = {
        "system_prompt": "You are a customer assistant.",
        "context": "Customer ORD-990 status is Shipped. Blue ribbon requested. Thank you for your business.",
        "query": "What is the status of order ORD-990?",
        "target_tokens": 30
    }
    response = client.post("/api/compress", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["system_prompt"] == payload["system_prompt"]
    assert data["query"] == payload["query"]
    assert "ORD-990" in data["compressed_context"]
    assert data["original_tokens"] > 0
    assert data["compressed_tokens"] > 0
    assert data["reduction_percentage"] >= 0.0
    assert len(data["sections"]) > 0
    assert "retained_sections" in data["summary"]


def test_compress_endpoint_empty_context():
    """Verify empty context returns 200 with 0 tokens and clean response."""
    payload = {
        "system_prompt": "Assistant",
        "context": "",
        "query": "Hello?",
        "target_tokens": 50
    }
    response = client.post("/api/compress", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["original_tokens"] == 0
    assert data["compressed_tokens"] == 0
    assert data["reduction_percentage"] == 0.0


def test_compress_endpoint_invalid_budget():
    """Verify target_tokens <= 0 returns validation error (422 Unprocessable Entity)."""
    payload = {
        "system_prompt": "Assistant",
        "context": "Some context",
        "query": "Some query",
        "target_tokens": -5
    }
    response = client.post("/api/compress", json=payload)
    assert response.status_code == 422


def test_list_benchmarks_endpoint():
    """Verify GET /api/benchmarks returns at least 30 cases."""
    response = client.get("/api/benchmarks")
    assert response.status_code == 200
    cases = response.json()
    assert isinstance(cases, list)
    assert len(cases) >= 30
    assert cases[0]["case_id"] == "case_01"


def test_evaluate_and_results_endpoints():
    """Verify POST /api/evaluate runs evaluation and GET /api/benchmarks/results retrieves it."""
    eval_resp = client.post("/api/evaluate")
    assert eval_resp.status_code == 200
    eval_data = eval_resp.json()
    assert eval_data["status"] == "success"
    assert "aggregate_metrics" in eval_data

    results_resp = client.get("/api/benchmarks/results")
    assert results_resp.status_code == 200
    results_data = results_resp.json()
    assert results_data["status"] == "available"
    assert "aggregate_metrics" in results_data
    assert len(results_data["cases"]) >= 30
