"""
Context Surgeon - Critical Correctness Tests (Section 10)
Tests:
- Test A: Delivery-date example
- Test B: Negation preservation
- Test C: Later correction
- Test D: Code dependency
- Test E: Structured data
- Test F: Impossible budget
- Additional tests: empty input, duplicate content, unicode text, large inputs
"""

import pytest
from backend.compressor import compress_context, ContextSurgeon


def test_a_delivery_date_delay_calculation():
    """
    Test A: Delivery-date example
    Context has original promised date (12 October) and later revision (15 October).
    Query asks how many days late order ORD-842 is.
    Both 12 October and 15 October along with ORD-842 must be preserved.
    Irrelevant packaging and greeting should be removed.
    """
    context = (
        "Order ORD-842 was promised for 12 October. "
        "The delivery was later revised to 15 October. "
        "The customer also prefers blue packaging. "
        "Hello, thank you for contacting support."
    )
    query = "How many days late is order ORD-842?"
    
    result = compress_context(
        system_prompt="You are a helpful logistics assistant.",
        context=context,
        query=query,
        target_tokens=40
    )
    
    compressed = result.compressed_context
    assert "ORD-842" in compressed
    assert "12 October" in compressed
    assert "15 October" in compressed
    # Blue packaging and greeting should be removed
    assert "blue packaging" not in compressed
    assert "thank you for contacting" not in compressed
    assert result.reduction_percentage > 0.0
    assert result.system_prompt == "You are a helpful logistics assistant."
    assert result.query == query


def test_b_negation_preservation():
    """
    Test B: Negation preservation
    Context: Port 443 allowed, Port 22 must not be opened. Maintenance window.
    Query: Which port must remain closed?
    Port 22 negation must be preserved.
    """
    context = (
        "The server is configured to allow port 443. "
        "Port 22 must not be opened. "
        "The maintenance window is 02:00–03:00."
    )
    query = "Which port must remain closed?"
    
    result = compress_context(
        system_prompt="You are a network security assistant.",
        context=context,
        query=query,
        target_tokens=25
    )
    
    compressed = result.compressed_context
    assert "Port 22" in compressed or "port 22" in compressed.lower()
    assert "must not" in compressed
    # Irrelevant maintenance window removed
    assert "maintenance window" not in compressed


def test_c_later_correction():
    """
    Test C: Later correction
    Context: Meeting at 10 AM, Correction: moved to 2 PM.
    Query: What is the current meeting time?
    Expected: 2 PM preserved with later_correction rule.
    """
    context = (
        "The meeting is scheduled for 10 AM. "
        "Correction: the meeting has moved to 2 PM."
    )
    query = "What is the current meeting time?"
    
    result = compress_context(
        system_prompt="",
        context=context,
        query=query,
        target_tokens=30
    )
    
    compressed = result.compressed_context
    assert "2 PM" in compressed
    # Check section decisions
    corr_sec = [s for s in result.sections if "later_correction" in s.protection_rules]
    assert len(corr_sec) >= 1
    assert corr_sec[0].decision == "retained"


def test_d_code_dependency_expansion():
    """
    Test D: Code dependency
    A selected Python function depends on a helper function defined elsewhere in supplied code.
    Query asks to explain the selected function.
    Both the target function and its helper dependency must be retained.
    """
    context = (
        "```python\ndef get_tier_rate(tier):\n    return 0.15 if tier == 'gold' else 0.05\n```\n\n"
        "```python\ndef calculate_discount(price, tier):\n    rate = get_tier_rate(tier)\n    return price * (1 - rate)\n```\n\n"
        "```python\ndef send_telemetry_event(event_id, payload):\n    import logging\n    logging.info(f'Event {event_id}')\n```"
    )
    query = "Explain how the calculate_discount function calculates the result."
    
    result = compress_context(
        system_prompt="You are a code reviewer.",
        context=context,
        query=query,
        target_tokens=60
    )
    
    compressed = result.compressed_context
    assert "calculate_discount" in compressed
    assert "get_tier_rate" in compressed
    # Unrelated telemetry should be removed
    assert "send_telemetry_event" not in compressed


def test_e_structured_data_linking():
    """
    Test E: Structured data
    Multiple customer records with IDs, statuses, and unrelated records.
    Query asks for status of CUST-204.
    Preserve record for CUST-204 and status Suspended, remove other records.
    """
    context = (
        '{"customer_id": "CUST-101", "name": "Alice Smith", "status": "Active", "balance": 150.00}\n\n'
        '{"customer_id": "CUST-204", "name": "Bob Jones", "status": "Suspended", "balance": 0.00}\n\n'
        '{"customer_id": "CUST-309", "name": "Charlie Brown", "status": "Pending", "balance": 45.50}'
    )
    query = "Find the status of customer CUST-204."
    
    result = compress_context(
        system_prompt="Database assistant",
        context=context,
        query=query,
        target_tokens=40
    )
    
    compressed = result.compressed_context
    assert "CUST-204" in compressed
    assert "Suspended" in compressed
    assert "CUST-101" not in compressed
    assert "CUST-309" not in compressed


def test_f_impossible_budget():
    """
    Test F: Impossible budget
    Critical evidence provided, but target budget is too small (e.g. 5 tokens).
    Must NOT silently discard critical evidence to claim budget compliance.
    Must issue a budget warning and preserve the evidence.
    """
    context = (
        "Order ORD-842 delivery revised to 15 October."
    )
    query = "What is the delivery date for ORD-842?"
    
    result = compress_context(
        system_prompt="",
        context=context,
        query=query,
        target_tokens=5  # Impossibly small
    )
    
    # Must preserve evidence
    assert "ORD-842" in result.compressed_context
    assert "15 October" in result.compressed_context
    # Must signal budget failure and provide clear warning
    assert result.budget_met is False
    assert len(result.warnings) > 0
    assert "smaller than required critical evidence" in result.warnings[0]


def test_empty_and_whitespace_context():
    """Empty or whitespace context returns cleanly with 0 tokens and no crash."""
    result = compress_context(
        system_prompt="Be concise",
        context="   ",
        query="What is the answer?",
        target_tokens=100
    )
    assert result.compressed_context == ""
    assert result.original_tokens == 0
    assert result.compressed_tokens == 0
    assert result.reduction_percentage == 0.0
    assert result.budget_met is True


def test_exact_duplicate_detection():
    """Repeated identical blocks are marked as deduplicated."""
    context = (
        "Server CPU load is 88% on node-1.\n\n"
        "Server CPU load is 88% on node-1.\n\n"
        "Server CPU load is 88% on node-1."
    )
    query = "What is the CPU load on node-1?"
    
    result = compress_context(
        system_prompt="",
        context=context,
        query=query,
        target_tokens=50
    )
    
    dedup_sections = [s for s in result.sections if s.decision == "deduplicated"]
    assert len(dedup_sections) == 2
    assert result.summary["deduplicated_sections"] == 2


def test_delivery_delay_regression():
    """
    Regression Test (Section 3):
    Context:
    Customer ORD-842 placed an order on 8 October. The original promised delivery date was 9 October.
    Due to a logistics delay, the delivery date was revised to 12 October.
    The customer prefers blue packaging, but this preference is unrelated to the delivery delay.

    The support team confirmed that ORD-842 is currently in transit.
    The latest update confirms delivery is expected on 12 October.

    Repeated information: ORD-842 is currently in transit. The delivery date for ORD-842 is 12 October.

    The customer contacted support to ask how late order ORD-842 will arrive compared with the original promised date.

    Query:
    How many days late is order ORD-842, and what is its latest delivery date?

    Target budget: 50 tokens.
    """
    context = (
        "Customer ORD-842 placed an order on 8 October. "
        "The original promised delivery date was 9 October. "
        "Due to a logistics delay, the delivery date was revised to 12 October. "
        "The customer prefers blue packaging, but this preference is unrelated to the delivery delay.\n\n"
        "The support team confirmed that ORD-842 is currently in transit. "
        "The latest update confirms delivery is expected on 12 October.\n\n"
        "Repeated information: ORD-842 is currently in transit. "
        "The delivery date for ORD-842 is 12 October.\n\n"
        "The customer contacted support to ask how late order ORD-842 will arrive compared with the original promised date."
    )
    query = "How many days late is order ORD-842, and what is its latest delivery date?"

    result = compress_context(
        system_prompt="You are a helpful assistant.",
        context=context,
        query=query,
        target_tokens=50
    )

    # 1. Essential evidence preserved
    assert "ORD-842" in result.compressed_context
    assert "9 October" in result.compressed_context
    assert "12 October" in result.compressed_context

    # 2. Irrelevant packaging preference removed
    assert "blue packaging" not in result.compressed_context.lower()

    # 3. Repeated information removed
    assert "repeated information" not in result.compressed_context.lower()

    # 4. Budget met without false warning
    assert result.compressed_tokens <= 50
    assert result.budget_met is True
    assert len(result.warnings) == 0

    # 5. Meaningful reduction
    assert result.reduction_percentage > 50.0
    assert result.tokens_removed > 0
    assert result.original_tokens > 100

