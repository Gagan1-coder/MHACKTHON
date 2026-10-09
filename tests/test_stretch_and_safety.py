"""
Unit tests for 95% Stretch Token Reduction & Safety Protections.
Verifies:
1. Highly repetitive conversations (95%+ reduction target)
2. Duplicate message storms
3. Long irrelevant paragraphs with needle evidence
4. Structured record filtering (relevant vs irrelevant)
5. Important dates and numerical fact preservation
6. Negation and later correction handling
7. Code dependencies symbol traversal
8. Short dense contexts with little redundancy
9. Contexts where 95% reduction is unsafe (transparent safety limits)
10. Configurable preferred reduction target behavior
"""

import json
import pytest
from pathlib import Path
from backend.compressor import ContextSurgeon, compress_context


@pytest.fixture
def benchmark_cases():
    cases_path = Path(__file__).resolve().parent.parent / "benchmarks" / "cases.json"
    with open(cases_path, "r", encoding="utf-8") as f:
        return {c["case_id"]: c for c in json.load(f)}


def test_highly_repetitive_conversations(benchmark_cases):
    """Case 31: 18-turn repetitive chat reaches >= 95% reduction and preserves order ORD-5510, date, carrier."""
    c = benchmark_cases["case_31"]
    resp = compress_context(
        system_prompt=c.get("system_prompt", ""),
        context=c["context"],
        query=c["query"],
        target_tokens=c["target_tokens"],
        preferred_reduction_target=c.get("preferred_reduction_target", 95.0)
    )

    for fact in c["expected_critical_facts"]:
        assert fact.lower() in resp.compressed_context.lower(), f"Fact '{fact}' missing from compressed context"

    assert resp.reduction_percentage >= 95.0, f"Expected >= 95% reduction, got {resp.reduction_percentage}%"
    assert resp.target_achievement_status == "Target achieved"


def test_duplicate_message_storms(benchmark_cases):
    """Case 32: 45 repeated log alerts compressed by >= 95% while keeping root cause and deployment version."""
    c = benchmark_cases["case_32"]
    resp = compress_context(
        system_prompt=c.get("system_prompt", ""),
        context=c["context"],
        query=c["query"],
        target_tokens=c["target_tokens"],
        preferred_reduction_target=c.get("preferred_reduction_target", 95.0)
    )

    for fact in c["expected_critical_facts"]:
        assert fact.lower() in resp.compressed_context.lower(), f"Fact '{fact}' missing from compressed context"

    assert resp.reduction_percentage >= 95.0, f"Expected >= 95% reduction, got {resp.reduction_percentage}%"
    assert resp.target_achievement_status == "Target achieved"


def test_long_irrelevant_paragraphs_needle(benchmark_cases):
    """Case 33: Multi-paragraph corporate memo extracts needle with safety limit preventing unsafe fact loss."""
    c = benchmark_cases["case_33"]
    resp = compress_context(
        system_prompt=c.get("system_prompt", ""),
        context=c["context"],
        query=c["query"],
        target_tokens=c["target_tokens"],
        preferred_reduction_target=c.get("preferred_reduction_target", 95.0)
    )

    for fact in c["expected_critical_facts"]:
        assert fact.lower() in resp.compressed_context.lower(), f"Fact '{fact}' missing from compressed context"

    assert resp.reduction_percentage >= 80.0, f"Expected >= 80% reduction, got {resp.reduction_percentage}%"
    assert resp.target_achievement_status == "Safety limit reached"


def test_structured_record_filtering(benchmark_cases):
    """Case 34: 25 customer JSON records pruned by >= 95% keeping only target CUST-9901 details."""
    c = benchmark_cases["case_34"]
    resp = compress_context(
        system_prompt=c.get("system_prompt", ""),
        context=c["context"],
        query=c["query"],
        target_tokens=c["target_tokens"],
        preferred_reduction_target=c.get("preferred_reduction_target", 95.0)
    )

    for fact in c["expected_critical_facts"]:
        assert fact.lower() in resp.compressed_context.lower(), f"Fact '{fact}' missing from compressed context"

    assert "CUST-1001" not in resp.compressed_context
    assert "CUST-1002" not in resp.compressed_context
    assert resp.reduction_percentage >= 95.0
    assert resp.target_achievement_status == "Target achieved"


def test_important_dates_and_numerical_facts(benchmark_cases):
    """Case 35: Executive minutes preserve baseline date, revised date, and delay variance."""
    c = benchmark_cases["case_35"]
    resp = compress_context(
        system_prompt=c.get("system_prompt", ""),
        context=c["context"],
        query=c["query"],
        target_tokens=c["target_tokens"],
        preferred_reduction_target=c.get("preferred_reduction_target", 95.0)
    )

    for fact in c["expected_critical_facts"]:
        assert fact.lower() in resp.compressed_context.lower(), f"Fact '{fact}' missing from compressed context"


def test_negations_and_corrections(benchmark_cases):
    """Case 36: Policy handbook preserves prohibition, corrected budget, and effective date."""
    c = benchmark_cases["case_36"]
    resp = compress_context(
        system_prompt=c.get("system_prompt", ""),
        context=c["context"],
        query=c["query"],
        target_tokens=c["target_tokens"],
        preferred_reduction_target=c.get("preferred_reduction_target", 95.0)
    )

    for fact in c["expected_critical_facts"]:
        assert fact.lower() in resp.compressed_context.lower(), f"Fact '{fact}' missing from compressed context"


def test_code_dependencies_preservation(benchmark_cases):
    """Case 37: Multi-function Python script retains target function and its helper dependency."""
    c = benchmark_cases["case_37"]
    resp = compress_context(
        system_prompt=c.get("system_prompt", ""),
        context=c["context"],
        query=c["query"],
        target_tokens=c["target_tokens"],
        preferred_reduction_target=c.get("preferred_reduction_target", 95.0)
    )

    for fact in c["expected_critical_facts"]:
        assert fact.lower() in resp.compressed_context.lower(), f"Fact '{fact}' missing from compressed context"

    # Unrelated functions should be omitted
    assert "sync_s3_bucket" not in resp.compressed_context
    assert "generate_invoice_pdf" not in resp.compressed_context


def test_short_dense_contexts_safety_limit(benchmark_cases):
    """Case 38: Short dense context does not destroy facts to reach 95%; reports Safety limit reached."""
    c = benchmark_cases["case_38"]
    resp = compress_context(
        system_prompt=c.get("system_prompt", ""),
        context=c["context"],
        query=c["query"],
        target_tokens=c["target_tokens"],
        preferred_reduction_target=c.get("preferred_reduction_target", 95.0)
    )

    for fact in c["expected_critical_facts"]:
        assert fact.lower() in resp.compressed_context.lower(), f"Fact '{fact}' missing from compressed context"

    assert resp.target_achievement_status == "Safety limit reached"
    assert "Safety limit" in resp.evidence_preservation_status


def test_unsafe_95_percent_reduction_hazmat(benchmark_cases):
    """Case 39: Life-safety / HAZMAT incident preserves all 4 facts and avoids unsafe compression."""
    c = benchmark_cases["case_39"]
    resp = compress_context(
        system_prompt=c.get("system_prompt", ""),
        context=c["context"],
        query=c["query"],
        target_tokens=c["target_tokens"],
        preferred_reduction_target=c.get("preferred_reduction_target", 95.0)
    )

    for fact in c["expected_critical_facts"]:
        assert fact.lower() in resp.compressed_context.lower(), f"Fact '{fact}' missing from compressed context"

    # Must preserve life safety facts and honestly report Safety limit reached
    assert resp.target_achievement_status == "Safety limit reached"
    assert resp.reduction_percentage < 95.0


def test_configurable_preferred_reduction_target():
    """Verify preferred_reduction_target influences target status accurately."""
    context = (
        "Server SRV-101 is located in Dallas datacenter rack 4.\n\n"
        "Routine check at 08:00 OK.\n\n"
        "Routine check at 09:00 OK.\n\n"
        "Routine check at 10:00 OK.\n\n"
        "Routine check at 11:00 OK.\n\n"
        "Routine check at 12:00 OK."
    )
    query = "Where is server SRV-101 located?"

    # Target 70%
    r70 = compress_context("", context, query, target_tokens=100, preferred_reduction_target=70.0)
    assert r70.preferred_reduction_target == 70.0
    if r70.reduction_percentage >= 70.0:
        assert r70.target_achievement_status == "Target achieved"

    # Target 99% (unrealistically high for this small context)
    r99 = compress_context("", context, query, target_tokens=100, preferred_reduction_target=99.0)
    assert r99.preferred_reduction_target == 99.0
    assert r99.target_achievement_status in ("Safety limit reached", "Below target")
