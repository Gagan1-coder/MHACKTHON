import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.compressor import compress_context

print("--- DIAGNOSING TEST E ---")
context_e = (
    '{"customer_id": "CUST-101", "name": "Alice Smith", "status": "Active", "balance": 150.00}\n\n'
    '{"customer_id": "CUST-204", "name": "Bob Jones", "status": "Suspended", "balance": 0.00}\n\n'
    '{"customer_id": "CUST-309", "name": "Charlie Brown", "status": "Pending", "balance": 45.50}'
)
query_e = "Find the status of customer CUST-204."

res_e = compress_context("Database assistant", context_e, query_e, target_tokens=40)
print("Compressed text E:\n", res_e.compressed_context)
for sec in res_e.sections:
    print(sec.section_id, "is_prot:", sec.is_protected, "decision:", sec.decision, "score:", sec.relevance_score, repr(sec.content))

print("\n--- DIAGNOSING TEST DELIVERY DELAY ---")
context_del = (
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
query_del = "How many days late is order ORD-842, and what is its latest delivery date?"
res_del = compress_context("You are a helpful assistant.", context_del, query_del, target_tokens=50)
print("Compressed text DEL:\n", res_del.compressed_context)
for sec in res_del.sections:
    print(sec.section_id, "is_prot:", sec.is_protected, "decision:", sec.decision, "score:", sec.relevance_score, repr(sec.content))
