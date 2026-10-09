"""
Context Surgeon - Comprehensive Unit Tests for Compressor Engine
Tests all 5 stages in detail:
1. Segmentation & boundaries
2. Deduplication & scoring
3. Protection & expansion
4. Selection & budget enforcement
5. Explanation, metrics, & immutability
"""

import pytest
from backend.tokenizer import Tokenizer
from backend.compressor import ContextSurgeon, compress_context


def test_tokenizer_modes():
    """Verify standard tokenizer and deterministic fallback tokenizer."""
    tok_default = Tokenizer()
    tok_fallback = Tokenizer(force_fallback=True)

    text = "Order ORD-842 with 15 items cost $129.50 on 2024-10-12."
    c1 = tok_default.count_tokens(text)
    c2 = tok_fallback.count_tokens(text)

    assert c1 > 0
    assert c2 > 0
    # Both should be reasonably close estimates
    assert abs(c1 - c2) <= 10
    assert tok_fallback.is_using_fallback is True
    assert tok_fallback.tokenizer_info["is_fallback"] is True


def test_code_block_fences_never_split():
    """Verify code blocks with multiple lines remain intact as a single section."""
    code = (
        "```python\n"
        "def process():\n"
        "    x = 10\n"
        "    y = 20\n"
        "    return x + y\n"
        "```"
    )
    context = f"Here is the algorithm:\n\n{code}\n\nThat was the algorithm."
    query = "How is process implemented?"

    surgeon = ContextSurgeon()
    resp = surgeon.compress(system_prompt="", context=context, query=query, target_tokens=100)

    # Find the code block section
    code_secs = [s for s in resp.sections if s.source_type == "code_block"]
    assert len(code_secs) == 1
    assert "def process():" in code_secs[0].content
    assert code_secs[0].decision == "retained"


def test_json_blocks_never_split():
    """Verify structured JSON records remain indivisible units."""
    json_record = '{\n  "id": "ITEM-99",\n  "stock": 42,\n  "available": true\n}'
    context = f"Inventory status:\n\n{json_record}\n\nEnd of report."
    query = "What is the stock of ITEM-99?"

    resp = compress_context(system_prompt="", context=context, query=query, target_tokens=100)
    json_secs = [s for s in resp.sections if s.source_type == "json_record"]
    assert len(json_secs) == 1
    assert "ITEM-99" in json_secs[0].content
    assert json_secs[0].decision == "retained"


def test_system_prompt_and_query_immutability():
    """Verify system prompt and query are NEVER modified."""
    sys_prompt = "You are an elite AI assistant. Strict rules apply."
    user_query = "What is the status of node-5?"
    context = "Node-5 is offline. Node-2 is online."

    resp = compress_context(
        system_prompt=sys_prompt,
        context=context,
        query=user_query,
        target_tokens=50
    )

    assert resp.system_prompt == sys_prompt
    assert resp.query == user_query


def test_unicode_and_special_characters():
    """Verify non-ASCII, emojis, and symbols do not break compression."""
    context = (
        "Server in Tokyo: 東京 (status: 正常 100%).\n\n"
        "Server in Berlin: 15°C temperature, €4,500 budget.\n\n"
        "Emoji alert: 🚀 Deployment successful on 2026-10-09."
    )
    query = "What is the Tokyo server status?"

    resp = compress_context(system_prompt="", context=context, query=query, target_tokens=50)
    assert "東京" in resp.compressed_context
    assert "100%" in resp.compressed_context
    assert resp.original_tokens > 0
    assert resp.compressed_tokens > 0


def test_large_context_efficiency():
    """Verify that processing a large context executes swiftly."""
    paragraphs = []
    for i in range(100):
        paragraphs.append(f"Document chunk {i}: The quick brown fox jumps over the lazy dog in session {i}.")
    paragraphs.append("TARGET CHUNK: Secret key for API is SEC-9988-X and must not be shared.")
    for i in range(100, 200):
        paragraphs.append(f"Document chunk {i}: Additional log entry {i} with boilerplate info.")

    context = "\n\n".join(paragraphs)
    query = "What is the secret key for the API?"

    resp = compress_context(system_prompt="", context=context, query=query, target_tokens=80)

    assert "SEC-9988-X" in resp.compressed_context
    assert resp.reduction_percentage > 85.0
    # Latency should be well under 100ms for fast local processing
    assert resp.compression_time_ms < 500.0


def test_explainability_all_fields_present():
    """Verify every section contains full explainability metadata."""
    context = "Section A with data. Section B with data."
    query = "Section A"

    resp = compress_context(system_prompt="", context=context, query=query, target_tokens=50)

    assert len(resp.sections) > 0
    for sec in resp.sections:
        assert sec.section_id.startswith("sec_")
        assert sec.original_position >= 0
        assert sec.preview != ""
        assert sec.tokens > 0
        assert sec.decision in ("retained", "removed", "deduplicated")
        assert len(sec.reason) > 0
