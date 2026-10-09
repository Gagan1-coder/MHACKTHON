"""
Context Surgeon - Benchmark Validation Tests
Validates dataset integrity, category coverage, and evaluation engine execution.
"""

import os
import json
import pytest
from backend.evaluator import BenchmarkEvaluator
from backend.schemas import BenchmarkCase


def test_benchmark_dataset_integrity():
    """Verify that cases.json contains at least 30 distinct cases with valid schemas."""
    evaluator = BenchmarkEvaluator()
    cases = evaluator.load_cases()

    assert len(cases) >= 30, f"Expected at least 30 cases, found {len(cases)}"

    case_ids = set()
    categories = set()

    for c in cases:
        assert c.case_id not in case_ids, f"Duplicate case ID: {c.case_id}"
        case_ids.add(c.case_id)
        categories.add(c.category)
        assert len(c.query.strip()) > 0
        assert c.target_tokens > 0

    # Verify at least 25 distinct categories are represented
    assert len(categories) >= 25, f"Expected rich category diversity, found {len(categories)}"


def test_evaluator_run_and_metrics():
    """Verify that evaluator runs across benchmark cases and generates valid stats."""
    evaluator = BenchmarkEvaluator()
    cases = evaluator.load_cases()

    aggregate, results = evaluator.run_all(cases)

    assert len(results) == len(cases)
    assert aggregate.total_cases == len(cases)
    # Check that mean token reduction is computed and positive
    assert aggregate.mean_token_reduction > 20.0
    # Check that critical fact retention rate is strong (> 85%)
    assert aggregate.critical_fact_retention_rate >= 85.0
    # Check that p95 latency is sub-100ms
    assert aggregate.p95_compression_latency_ms < 100.0

    # Verify result files were created
    json_path = os.path.join(evaluator.results_dir, "results.json")
    csv_path = os.path.join(evaluator.results_dir, "results.csv")
    assert os.path.exists(json_path)
    assert os.path.exists(csv_path)

    # Verify JSON content
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "aggregate_metrics" in data
    assert len(data["cases"]) == len(cases)
