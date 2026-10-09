"""
Context Surgeon - Benchmark CLI Runner & Comparative Analysis
Executes benchmark cases across:
- Mode A: Fast Local Compression (Deterministic 5-stage algorithm)
- Mode B: Recursive Large-Context Processing (RLM-inspired externalized REPL exploration)

Usage:
    python benchmarks/run_benchmark.py [--mode fast_local|recursive|both]
"""

import sys
import os
import argparse

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.evaluator import BenchmarkEvaluator


def run_benchmark_for_mode(evaluator: BenchmarkEvaluator, cases, mode: str):
    print("\n" + "=" * 78)
    mode_label = "MODE A: FAST LOCAL COMPRESSION" if mode == "fast_local" else "MODE B: RECURSIVE RLM PROCESSING"
    print(f" {mode_label}")
    print("=" * 78)
    print(f"{'Case ID':<9} | {'Category':<30} | {'Orig':<5} | {'Comp':<5} | {'Red %':<6} | {'Facts':<6} | {'Lat(ms)':<7}")
    print("-" * 78)

    aggregate, results = evaluator.run_all(cases, mode=mode)

    for r in results:
        fact_str = f"{r.facts_retained}/{r.total_facts}"
        print(f"{r.case_id:<9} | {r.category[:30]:<30} | {r.original_tokens:<5} | {r.compressed_tokens:<5} | {r.reduction_percentage:>5.1f}% | {fact_str:<6} | {r.compression_time_ms:>7.2f}")

    print("\n" + "-" * 78)
    print(f"[{mode.upper()} AGGREGATE SUMMARY]")
    print(f"  - Total Cases:               {aggregate.total_cases}")
    print(f"  - Mean Token Reduction:      {aggregate.mean_token_reduction}%")
    print(f"  - Median Latency:            {aggregate.median_compression_latency_ms:.2f} ms")
    print(f"  - P95 Latency:               {aggregate.p95_compression_latency_ms:.2f} ms")
    print(f"  - Fact Retention Rate:       {aggregate.critical_fact_retention_rate}%")
    print("-" * 78)
    return aggregate, results


def main():
    parser = argparse.ArgumentParser(description="Context Surgeon Comparative Benchmark")
    parser.add_argument("--mode", choices=["fast_local", "recursive", "both"], default="both", help="Operating mode to benchmark")
    args = parser.parse_args()

    print("=" * 78)
    print(" CONTEXT SURGEON - COMPREHENSIVE BENCHMARK SUITE")
    print(" Testing Fast Local vs Recursive Language Model (RLM) Paradigm")
    print(" Reference: https://github.com/grishahq/recursive-llm (Grigori Gvadzabia)")
    print("=" * 78)

    evaluator = BenchmarkEvaluator()
    cases = evaluator.load_cases()
    print(f"[*] Loaded {len(cases)} benchmark cases from: {evaluator.cases_file_path}")

    if args.mode in ("fast_local", "both"):
        agg_a, res_a = run_benchmark_for_mode(evaluator, cases, mode="fast_local")

    if args.mode in ("recursive", "both"):
        agg_b, res_b = run_benchmark_for_mode(evaluator, cases, mode="recursive")

    if args.mode == "both":
        print("\n" + "=" * 78)
        print(" HEAD-TO-HEAD ARCHITECTURAL COMPARISON: MODE A vs MODE B")
        print("=" * 78)
        print(f"{'Metric':<35} | {'Mode A: Fast Local':<18} | {'Mode B: Recursive RLM':<18}")
        print("-" * 78)
        print(f"{'Mean Token Reduction':<35} | {agg_a.mean_token_reduction:>16}% | {agg_b.mean_token_reduction:>16}%")
        print(f"{'P95 Compression Latency':<35} | {agg_a.p95_compression_latency_ms:>13.2f} ms | {agg_b.p95_compression_latency_ms:>13.2f} ms")
        print(f"{'Median Latency':<35} | {agg_a.median_compression_latency_ms:>13.2f} ms | {agg_b.median_compression_latency_ms:>13.2f} ms")
        print(f"{'Critical Fact Retention':<35} | {agg_a.critical_fact_retention_rate:>16}% | {agg_b.critical_fact_retention_rate:>16}%")
        print(f"{'Context Placement':<35} | {'Inside prompt':>18} | {'Externalized store':>18}")
        print(f"{'REPL Inspection Trajectory':<35} | {'None (inline)':>18} | {'Full step audit':>18}")
        print(f"{'External LLM Dependency':<35} | {'None (100% offline)':>18} | {'Optional / offline':>18}")
        print("=" * 78)


if __name__ == "__main__":
    main()
