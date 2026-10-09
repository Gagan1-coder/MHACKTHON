"""
Comprehensive Regression Tests for Genuine Automatic Context Compression.
Verifies:
1. UI validation: No manual reduction sliders or 70%/85%/95%/98% preset buttons in frontend.
2. Scenario A: Repeated paragraphs are deduplicated.
3. Scenario B: Irrelevant paragraphs are removed when not needed for the query.
4. Scenario C: Important dates, numerical values, and units are preserved.
5. Scenario D: Negations and later corrections remain accurate.
6. Scenario E: Code dependencies and structured JSON records are preserved.
7. Scenario F: Unique, essential context legitimately shows low or zero reduction.
8. Scenario G: Empty input does not cause division-by-zero error.
9. Actual token calculations:
   tokens_removed == original_tokens - compressed_tokens
   reduction_percentage == (tokens_removed / original_tokens) * 100 (if original_tokens > 0)
10. Input sensitivity: Changing input changes measured reduction results dynamically.
"""

from pathlib import Path
import pytest
from backend.compressor import ContextSurgeon, compress_context
from backend.tokenizer import Tokenizer, default_tokenizer
from backend.schemas import CompressRequest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


# ------------------------------------------------------------------------------
# 1. UI VERIFICATION: NO MANUAL TARGET SLIDERS OR TARGET BUTTONS
# ------------------------------------------------------------------------------
def test_ui_no_manual_reduction_slider_or_target_buttons():
    """Verify frontend/index.html has no slider or 70/85/95/98 target buttons."""
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    index_html = (frontend_dir / "index.html").read_text(encoding="utf-8")
    app_js = (frontend_dir / "app.js").read_text(encoding="utf-8")

    # Slider and manual percentage controls must NOT exist in HTML
    assert "reduction-target" not in index_html, "Found reduction-target in index.html"
    assert "Preferred Reduction Target" not in index_html, "Found 'Preferred Reduction Target' in index.html"
    assert "target-chips" not in index_html, "Found target-chips in index.html"
    assert "data-target=\"70\"" not in index_html
    assert "data-target=\"85\"" not in index_html
    assert "data-target=\"95\"" not in index_html
    assert "data-target=\"98\"" not in index_html

    # Functional action buttons must exist
    assert 'id="btn-compress"' in index_html, "btn-compress must exist in index.html"
    assert 'id="btn-reset"' in index_html, "btn-reset must exist in index.html"

    # app.js must not contain event listeners or state for manual reduction targets
    assert "reductionTarget" not in app_js, "Found reductionTarget in app.js"
    assert "targetChips" not in app_js, "Found targetChips in app.js"
    assert "inputReductionTarget" not in app_js, "Found inputReductionTarget in app.js"


# ------------------------------------------------------------------------------
# 2. SCENARIO A: REPEATED PARAGRAPHS DEDUPLICATION
# ------------------------------------------------------------------------------
def test_scenario_a_repeated_paragraphs_deduplicated():
    """Repeated paragraphs should be deduplicated, reducing tokens while keeping one copy."""
    paragraph = (
        "Deployment event DEPL-4091 initiated by engineer Alice at 14:00 UTC. "
        "Database migration executed successfully with schema version v4.2.1."
    )
    # Context with the exact paragraph repeated 5 times, plus an irrelevant note
    context = "\n\n".join([paragraph] * 5 + ["The office coffee machine will be serviced on Friday."])
    query = "What is the schema version for deployment DEPL-4091?"

    result = compress_context(
        system_prompt="You are a DevOps assistant.",
        context=context,
        query=query,
        target_tokens=100
    )

    compressed = result.compressed_context
    assert "DEPL-4091" in compressed
    assert "v4.2.1" in compressed
    # The repeated paragraph should only appear once in compressed context
    assert compressed.count("DEPL-4091") == 1
    assert "coffee machine" not in compressed
    # Significant genuine reduction from deduplication
    assert result.reduction_percentage > 60.0
    assert result.summary["deduplicated_sections"] >= 4


# ------------------------------------------------------------------------------
# 3. SCENARIO B: IRRELEVANT PARAGRAPHS REMOVED BASED ON QUERY
# ------------------------------------------------------------------------------
def test_scenario_b_irrelevant_paragraphs_removed():
    """Irrelevant paragraphs must be removed when not needed to answer the query."""
    context = (
        "Server SRV-01 is located in Rack A3 of the Frankfurt Data Center.\n\n"
        "The cafeteria menu for Tuesday includes minestrone soup and grilled chicken salad.\n\n"
        "Server SRV-01 CPU temperature reached 88 degrees Celsius at 13:45 UTC, causing thermal throttling.\n\n"
        "Annual company picnic will be held at Riverside Park on July 20th. Transportation provided."
    )
    query = "Why did server SRV-01 throttle and what was its temperature?"

    result = compress_context(
        system_prompt="You are an infrastructure monitoring assistant.",
        context=context,
        query=query,
        target_tokens=60
    )

    compressed = result.compressed_context
    # Query-relevant evidence must be kept
    assert "SRV-01" in compressed
    assert "88 degrees Celsius" in compressed or "88" in compressed
    assert "thermal throttling" in compressed

    # Irrelevant content must be pruned
    assert "cafeteria" not in compressed
    assert "minestrone soup" not in compressed
    assert "picnic" not in compressed
    assert "Riverside Park" not in compressed


# ------------------------------------------------------------------------------
# 4. SCENARIO C: IMPORTANT DATES AND NUMERICAL VALUES PRESERVED
# ------------------------------------------------------------------------------
def test_scenario_c_important_dates_numbers_units_preserved():
    """Important dates, numerical values, and units must be preserved accurately."""
    context = (
        "Transaction TXN-8820 processed on 14 March 2026 at 09:15:32 EST.\n\n"
        "The customer transferred exactly $4,850.75 USD with a wire fee of 25.00 EUR.\n\n"
        "Daily weather in Zurich was partly cloudy with light precipitation of 2.1 mm.\n\n"
        "Account balance after settlement is 120,400.50 USD."
    )
    query = "What was the transfer amount, transaction ID, and date for transaction TXN-8820?"

    result = compress_context(
        system_prompt="You are a banking assistant.",
        context=context,
        query=query,
        target_tokens=80
    )

    compressed = result.compressed_context
    assert "TXN-8820" in compressed
    assert "14 March 2026" in compressed
    assert "$4,850.75" in compressed or "4,850.75 USD" in compressed
    assert "weather" not in compressed
    assert "Zurich" not in compressed


# ------------------------------------------------------------------------------
# 5. SCENARIO D: NEGATIONS AND LATER CORRECTIONS PRESERVED
# ------------------------------------------------------------------------------
def test_scenario_d_negations_and_corrections_preserved():
    """Negations and later corrections must remain accurate and not inverted."""
    context = (
        "Initial dispatch estimated delivery on May 10th.\n\n"
        "Update from logistics manager: delivery was rescheduled and will NOT arrive on May 10th. "
        "The revised confirmed delivery date is May 14th.\n\n"
        "Customer representative John noted that the customer likes green envelopes.\n\n"
        "Do NOT open port 8080 under any circumstances without VP approval."
    )
    query = "What is the confirmed delivery date and can port 8080 be opened?"

    result = compress_context(
        system_prompt="You are an operations assistant.",
        context=context,
        query=query,
        target_tokens=90
    )

    compressed = result.compressed_context
    # Correction preserved
    assert "May 14th" in compressed or "May 14" in compressed
    # Negations preserved
    assert "NOT" in compressed or "not" in compressed
    assert "8080" in compressed
    # Irrelevant chatter removed
    assert "green envelopes" not in compressed


# ------------------------------------------------------------------------------
# 6. SCENARIO E: CODE DEPENDENCIES AND STRUCTURED RECORDS PRESERVED
# ------------------------------------------------------------------------------
def test_scenario_e_code_dependencies_and_structured_records():
    """Code dependencies and structured JSON records must be preserved when required."""
    code_context = (
        "Here are utility functions:\n\n"
        "```python\n"
        "def compute_hash(data: str) -> str:\n"
        "    return hashlib.sha256(data.encode()).hexdigest()\n"
        "```\n\n"
        "```python\n"
        "def verify_token(token: str, secret: str) -> bool:\n"
        "    expected = compute_hash(secret)\n"
        "    return token == expected\n"
        "```\n\n"
        "```python\n"
        "def unused_formatting_helper(s: str) -> str:\n"
        "    return s.strip().title()\n"
        "```\n\n"
        "Here is the user record:\n\n"
        "```json\n"
        "{\"user_id\": \"USR-901\", \"role\": \"admin\", \"permissions\": [\"read\", \"write\", \"delete\"]}\n"
        "```"
    )
    query = "Show the implementation of verify_token and the record for USR-901."

    result = compress_context(
        system_prompt="You are a software engineer assistant.",
        context=code_context,
        query=query,
        target_tokens=150
    )

    compressed = result.compressed_context
    # verify_token and its dependency compute_hash must be retained
    assert "verify_token" in compressed
    assert "compute_hash" in compressed
    # The structured JSON record must be retained
    assert "USR-901" in compressed
    assert "admin" in compressed
    # Unused function should be removed
    assert "unused_formatting_helper" not in compressed


# ------------------------------------------------------------------------------
# 7. SCENARIO F: UNIQUE, ESSENTIAL CONTEXT SHOWS LOW OR ZERO REDUCTION
# ------------------------------------------------------------------------------
def test_scenario_f_dense_unique_context_honest_low_reduction():
    """A context containing only unique, essential information may legitimately show low or zero reduction."""
    dense_context = "Flight AA-104 departs JFK at 08:30 and arrives at LHR at 20:45. Seat assignment is 14B."
    query = "What is the departure and arrival time for flight AA-104?"

    result = compress_context(
        system_prompt="You are a flight concierge.",
        context=dense_context,
        query=query,
        target_tokens=100  # Budget is generous enough to retain all essential facts
    )

    # Every fact is relevant to the query; nothing should be dropped
    assert "AA-104" in result.compressed_context
    assert "08:30" in result.compressed_context
    assert "20:45" in result.compressed_context
    # Because it is dense and essential, reduction percentage is low (or 0)
    assert result.reduction_percentage <= 25.0
    assert result.tokens_removed >= 0
    assert result.original_tokens == result.compressed_tokens or result.tokens_removed < 10


# ------------------------------------------------------------------------------
# 8. SCENARIO G: EMPTY INPUT DOES NOT CAUSE ZERO DIVISION
# ------------------------------------------------------------------------------
def test_scenario_g_empty_context_no_zero_division():
    """Empty or whitespace input must return 0 tokens with zero division safety."""
    for empty_input in ["", "   ", "\n\t\n  "]:
        result = compress_context(
            system_prompt="You are an assistant.",
            context=empty_input,
            query="Where is my order?",
            target_tokens=50
        )
        assert result.original_tokens == 0
        assert result.compressed_tokens == 0
        assert result.tokens_removed == 0
        assert result.reduction_percentage == 0.0
        assert result.compressed_context == ""


# ------------------------------------------------------------------------------
# 9. TOKEN REDUCTION CALCULATION AND TOKENIZER CONSISTENCY
# ------------------------------------------------------------------------------
def test_actual_token_reduction_formula():
    """Verify that tokens_removed and reduction_percentage are mathematically exact."""
    context = (
        "Alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november. "
        "The quick brown fox jumps over the lazy dog. "
        "A stitch in time saves nine. To be or not to be, that is the question."
    )
    query = "Find the quick brown fox."

    result = compress_context(
        system_prompt="Assistant",
        context=context,
        query=query,
        target_tokens=15
    )

    assert result.original_tokens > 0
    assert result.compressed_tokens > 0
    # Exact formula verification
    assert result.tokens_removed == (result.original_tokens - result.compressed_tokens)
    expected_pct = round(((result.tokens_removed) / result.original_tokens) * 100, 2)
    assert result.reduction_percentage == expected_pct


# ------------------------------------------------------------------------------
# 10. INPUT SENSITIVITY: CHANGING INPUT CHANGES MEASURED RESULTS
# ------------------------------------------------------------------------------
def test_changing_input_changes_measured_result():
    """Changing input text or budget dynamically changes the measured tokens and reduction."""
    query = "What is the secret code?"
    base_text = "The secret code is OMEGA-77."
    noise_short = "\n\nJust casual conversation here."
    noise_long = "\n\n" + "\n\n".join([f"Filler paragraph #{i} with random details." for i in range(15)])

    res1 = compress_context("Assistant", base_text + noise_short, query, target_tokens=20)
    res2 = compress_context("Assistant", base_text + noise_long, query, target_tokens=20)

    # Measured tokens and reduction percentages must differ significantly
    assert res1.original_tokens != res2.original_tokens
    assert res2.reduction_percentage > res1.reduction_percentage
    assert "OMEGA-77" in res1.compressed_context
    assert "OMEGA-77" in res2.compressed_context


# ------------------------------------------------------------------------------
# 11. API DATA FLOW INTEGRITY
# ------------------------------------------------------------------------------
def test_api_compression_data_flow():
    """Verify POST /api/compress returns real tokens, no manual target requested."""
    payload = {
        "system_prompt": "Assistant",
        "context": (
            "Ticket TKT-100 created at 10:00.\n\n"
            "Ticket TKT-100 status updated to Resolved at 11:30 by Agent Sam.\n\n"
            "Unrelated weather forecast: cloudy with showers tomorrow."
        ),
        "query": "Who resolved ticket TKT-100 and at what time?",
        "target_tokens": 30
    }
    response = client.post("/api/compress", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["original_tokens"] > data["compressed_tokens"]
    assert data["tokens_removed"] == data["original_tokens"] - data["compressed_tokens"]
    assert data["reduction_percentage"] == round((data["tokens_removed"] / data["original_tokens"]) * 100, 2)
    assert "TKT-100" in data["compressed_context"]
    assert "Agent Sam" in data["compressed_context"]
    assert "weather" not in data["compressed_context"]
    # Evidence status must be present
    assert "evidence_preservation_status" in data
    assert data["preferred_reduction_target"] is None
