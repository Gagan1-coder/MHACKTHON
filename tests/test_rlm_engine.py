"""
Unit and Integration Tests for Recursive Language Model (RLM) Engine.
Inspired by grishahq/recursive-llm (Grigori Gvadzabia / Zhang et al.).

Verifies:
1. ContextStore externalization, chunk partitioning, entity indexing, and inverted search.
2. SafeExplorationREPL query execution, sandboxing, and timeout protection.
3. RLMRecursiveEngine trajectory generation, step audit, and statistics.
4. Mode A ("fast_local") vs Mode B ("recursive") behavior in ContextSurgeon.
5. Auto mode selection based on context token threshold.
6. Important fact, date, number, negation, correction, and code dependency preservation in Mode B.
7. Long document needle retrieval (case_40).
8. API /api/compress data flow with mode parameter.
"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from backend.main import app
from backend.tokenizer import default_tokenizer
from backend.rlm_engine import (
    ContextStore,
    SafeExplorationREPL,
    RLMRecursiveEngine,
    TrajectoryEvent,
    RLMStats
)
from backend.compressor import ContextSurgeon, compress_context
from backend.schemas import CompressRequest

client = TestClient(app)


def test_context_store_externalization_and_indexing():
    """Verify that ContextStore partitions text and indexes entities outside prompt."""
    text = (
        "Server SRV-901 operates in Frankfurt Data Center. "
        "CPU temperature peaked at 85 degrees Celsius on 2026-03-15.\n\n"
        "```python\ndef get_status():\n    return 'OK'\n```\n\n"
        "{\"service\": \"auth-gateway\", \"status\": \"healthy\"}"
    )
    store = ContextStore(text)
    assert len(store.chunks) >= 3
    assert store.total_tokens > 0

    # Search for server ID
    hits = store.search("SRV-901", max_results=3)
    assert len(hits) > 0
    assert hits[0]["chunk_id"] == "chunk_000"
    assert "SRV-901" in hits[0]["identifiers"]

    # Filter by entity
    entity_hits = store.filter_by_entities("SRV-901")
    assert "chunk_000" in entity_hits


def test_safe_exploration_repl_query_execution():
    """Verify safe sandboxed REPL step execution."""
    text = (
        "Project DELTA milestone deadline is 18 November 2026.\n\n"
        "Unrelated cafeteria announcement: pizza on Friday."
    )
    store = ContextStore(text)
    repl = SafeExplorationREPL(store, timeout_seconds=2.0)
    res = repl.execute_query("What is the milestone deadline for Project DELTA?")

    assert "candidate_ids" in res
    assert len(res["candidate_ids"]) > 0
    assert "search_hits" in res


def test_rlm_engine_recursive_exploration_and_trajectory():
    """Verify RLM recursive engine produces observable trajectory and preserves facts."""
    context = (
        "Transaction TXN-5544 was processed on 22 April 2026 at 10:15 UTC.\n\n"
        "Customer transferred 4,200 USD via Swift wire to account ACC-9912.\n\n"
        "Office supplies inventory update: order 20 pens and 5 notebooks.\n\n"
        "Standard weather report for Geneva: 14 degrees Celsius with rain."
    )
    query = "What amount was transferred in transaction TXN-5544?"
    target_tokens = 50

    engine = RLMRecursiveEngine(max_iterations=6)
    compressed, trajectory, stats, retained = engine.process(
        system_prompt="Assistant",
        context=context,
        query=query,
        target_tokens=target_tokens
    )

    assert "TXN-5544" in compressed
    assert "4,200 USD" in compressed or "4,200" in compressed
    assert "pens" not in compressed
    assert "weather report" not in compressed
    assert len(trajectory) >= 3
    assert stats.iterations >= 2
    assert stats.exploration_tokens > 0
    assert stats.mode == "recursive"

    # Verify trajectory events structure
    for event in trajectory:
        assert isinstance(event, TrajectoryEvent)
        d = event.to_dict()
        assert "step" in d
        assert "action" in d
        assert "tokens_inspected" in d
        assert "elapsed_ms" in d


def test_context_surgeon_mode_a_vs_mode_b():
    """Verify ContextSurgeon behaves distinctly in Mode A (fast_local) and Mode B (recursive)."""
    context = (
        "Deployment release REL-880 deployed on 2026-05-10.\n\n"
        "Database migration executed successfully.\n\n"
        "Random note: kitchen water cooler fixed."
    )
    query = "When was release REL-880 deployed?"

    surgeon = ContextSurgeon()

    # Mode A: Fast Local
    res_a = surgeon.compress("Assistant", context, query, target_tokens=30, mode="fast_local")
    assert res_a.mode == "fast_local"
    assert res_a.recursive_trace is None
    assert res_a.recursive_stats is None
    assert "REL-880" in res_a.compressed_context
    assert res_a.original_tokens > res_a.compressed_tokens

    # Mode B: Recursive RLM
    res_b = surgeon.compress("Assistant", context, query, target_tokens=30, mode="recursive")
    assert res_b.mode == "recursive"
    assert res_b.recursive_trace is not None
    assert len(res_b.recursive_trace) >= 3
    assert res_b.recursive_stats is not None
    assert res_b.recursive_stats["iterations"] >= 2
    assert "REL-880" in res_b.compressed_context


def test_context_surgeon_auto_mode():
    """Verify 'auto' mode selects fast_local for normal contexts and recursive for large ones."""
    short_context = "Flight FL-100 departs at 14:00."
    surgeon = ContextSurgeon()
    res_short = surgeon.compress("Assistant", short_context, "Departure time?", target_tokens=30, mode="auto")
    assert res_short.mode == "fast_local"

    # Large context > 1200 tokens
    long_context = "Section header.\n\n" + "\n\n".join([
        f"Paragraph {i}: Detailed background description with lots of information about system {i}."
        for i in range(80)
    ]) + "\n\nCritical Key: KEY-99921."
    res_long = surgeon.compress("Assistant", long_context, "What is Critical Key?", target_tokens=50, mode="auto")
    assert res_long.mode == "recursive"
    assert "KEY-99921" in res_long.compressed_context


def test_recursive_mode_preserves_code_dependencies():
    """Verify Mode B recursive engine preserves code dependencies across functions."""
    code_context = (
        "```python\ndef helper_hash(val):\n    return hash(val)\n```\n\n"
        "```python\ndef validate_user(u):\n    h = helper_hash(u)\n    return h != 0\n```\n\n"
        "```python\ndef unused_printer(msg):\n    print(msg)\n```"
    )
    query = "How does validate_user work?"
    resp = compress_context("Assistant", code_context, query, target_tokens=80, mode="recursive")

    assert "validate_user" in resp.compressed_context
    assert "helper_hash" in resp.compressed_context
    assert "unused_printer" not in resp.compressed_context


def test_recursive_mode_preserves_corrections_and_negations():
    """Verify Mode B recursive engine captures later corrections and negations."""
    context = (
        "Schedule v1: meeting on Tuesday at 10:00.\n\n"
        "Schedule v2: Revised! Meeting postponed to Thursday at 14:00.\n\n"
        "Security rule: Do NOT open port 22 under any circumstance."
    )
    query = "When is the meeting and what is the rule for port 22?"
    resp = compress_context("Assistant", context, query, target_tokens=60, mode="recursive")

    assert "Thursday" in resp.compressed_context
    assert "NOT" in resp.compressed_context or "not" in resp.compressed_context
    assert "22" in resp.compressed_context


def test_case_40_long_document_needle_retrieval():
    """Verify benchmark case_40 needle extraction in recursive mode."""
    context = (
        "Enterprise Data Center Operations Manual (Revision 2026.4):\n"
        "Chapter 1 - Overview: The Global Data Network connects 14 primary regional hubs.\n\n"
        "Chapter 2 - Visitor Protocols: External contractors must submit badge request form SEC-019.\n\n"
        "Chapter 3 - Environmental Controls: Server rack ambient temperature 18.0 to 22.5 Celsius.\n\n"
        "Chapter 4 - Network Routing: Core routing utilizes redundant BGP peering across AS-65001.\n\n"
        "Chapter 5 - Vault Credential Rotation: To access the cold-storage disaster recovery repository, "
        "the secondary authorization key is SEC-KEY-77891 with checksum SHA-9081.\n\n"
        "Chapter 6 - Fire Suppression: Inergen gas suppression systems armed.\n\n"
        "Chapter 7 - Weekly Maintenance Schedules: Standard maintenance windows Saturday 23:00 UTC."
    )
    query = "What is the secondary authorization key and checksum for the cold-storage disaster recovery repository?"
    resp = compress_context("Security Auditor", context, query, target_tokens=60, mode="recursive")

    assert "SEC-KEY-77891" in resp.compressed_context
    assert "SHA-9081" in resp.compressed_context
    assert resp.reduction_percentage > 50.0
    assert resp.mode == "recursive"


def test_api_compress_mode_parameter():
    """Verify POST /api/compress supports mode parameter and returns recursive metrics."""
    payload = {
        "system_prompt": "Assistant",
        "context": "Incident INC-900 at 08:00.\n\nResolution team deployed at 09:30.\n\nCafeteria closed.",
        "query": "What time was incident INC-900?",
        "target_tokens": 30,
        "mode": "recursive"
    }
    res = client.post("/api/compress", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "recursive"
    assert "INC-900" in data["compressed_context"]
    assert data["recursive_trace"] is not None
    assert len(data["recursive_trace"]) >= 2
    assert data["recursive_stats"] is not None
    assert data["recursive_stats"]["iterations"] >= 1
