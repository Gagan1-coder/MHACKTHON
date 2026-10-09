"""
Context Surgeon - Unified Application Launcher
MITK AI VISION 24H Hackathon | PS-02 | CommVault Challenge

Usage:
    python run.py [--host 127.0.0.1] [--port 8000] [--no-eval]
"""

import sys
import os
import argparse
import uvicorn

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.evaluator import BenchmarkEvaluator


def main():
    parser = argparse.ArgumentParser(description="Context Surgeon - LLM Context Compression Engine")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--no-eval", action="store_true", help="Skip pre-running benchmarks on startup")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print("=" * 75)
    print("  CONTEXT SURGEON — Intelligent LLM Context Compression Layer")
    print("  Tagline: Compress LLM Context Without Losing Essential Meaning.")
    print("  Hackathon: MITK AI VISION 24H | PS-02 | CommVault Challenge")
    print("=" * 75)

    if not args.no_eval:
        print("[*] Pre-evaluating 30 benchmark cases to populate dashboard statistics...")
        try:
            evaluator = BenchmarkEvaluator()
            cases = evaluator.load_cases()
            agg, _ = evaluator.run_all(cases)
            print(f"[+] Evaluated {agg.total_cases} benchmark cases:")
            print(f"    - Mean context token reduction: {agg.mean_token_reduction}% (Target: >= 30%)")
            print(f"    - P95 compression latency:      {agg.p95_compression_latency_ms:.2f} ms (Target: < 100 ms)")
            print(f"    - Critical fact retention:      {agg.critical_fact_retention_rate}%")
            print(f"    - Results saved to benchmarks/results/results.json and results.csv")
        except Exception as e:
            print(f"[!] Warning during benchmark initialization: {e}")

    print("\n[+] Starting FastAPI server...")
    print(f"    -> Web Dashboard:       http://{args.host}:{args.port}/")
    print(f"    -> Interactive API:     http://{args.host}:{args.port}/docs")
    print(f"    -> Health Check:        http://{args.host}:{args.port}/api/health")
    print("=" * 75)
    print("Press CTRL+C to terminate server.\n")

    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
