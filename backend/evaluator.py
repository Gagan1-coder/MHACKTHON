"""
Context Surgeon - Benchmark Evaluator Module
Provides reproducible evaluation over benchmark cases, calculates statistical metrics,
checks fact preservation, exports JSON/CSV results, and validates target criteria.
"""

import os
import json
import csv
import time
import statistics
from typing import List, Dict, Any, Optional, Tuple
from backend.compressor import ContextSurgeon
from backend.tokenizer import Tokenizer, default_tokenizer
from backend.schemas import (
    BenchmarkCase,
    BenchmarkCaseResult,
    AggregateMetrics,
    CompressResponse
)


class BenchmarkEvaluator:
    """
    Executes benchmark evaluations and computes aggregate performance metrics.
    Works completely offline without requiring any paid API keys.
    """

    def __init__(
        self,
        cases_file_path: Optional[str] = None,
        results_dir: Optional[str] = None,
        tokenizer: Optional[Tokenizer] = None
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.cases_file_path = cases_file_path or os.path.join(base_dir, "benchmarks", "cases.json")
        self.results_dir = results_dir or os.path.join(base_dir, "benchmarks", "results")
        os.makedirs(self.results_dir, exist_ok=True)
        self.tokenizer = tokenizer or default_tokenizer
        self.surgeon = ContextSurgeon(tokenizer=self.tokenizer)

    def load_cases(self) -> List[BenchmarkCase]:
        """Loads benchmark cases from JSON file."""
        if not os.path.exists(self.cases_file_path):
            raise FileNotFoundError(f"Cases file not found at {self.cases_file_path}")
        with open(self.cases_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [BenchmarkCase(**item) for item in data]

    def evaluate_case(self, case: BenchmarkCase, mode: str = "fast_local") -> BenchmarkCaseResult:
        """Evaluates a single benchmark case."""
        resp: CompressResponse = self.surgeon.compress(
            system_prompt=case.system_prompt,
            context=case.context,
            query=case.query,
            target_tokens=case.target_tokens,
            preferred_reduction_target=case.preferred_reduction_target,
            mode=mode
        )

        compressed_text = resp.compressed_context.lower()
        facts_retained = 0
        missing_facts = []

        # Check retention of expected critical facts
        for fact in case.expected_critical_facts:
            fact_lower = fact.lower()
            if fact_lower in compressed_text:
                facts_retained += 1
            else:
                missing_facts.append(fact)

        total_facts = len(case.expected_critical_facts)
        fact_retention_rate = (
            round((facts_retained / total_facts) * 100, 2) if total_facts > 0 else 100.0
        )

        return BenchmarkCaseResult(
            case_id=case.case_id,
            category=case.category,
            original_tokens=resp.original_tokens,
            compressed_tokens=resp.compressed_tokens,
            tokens_removed=resp.tokens_removed,
            reduction_percentage=resp.reduction_percentage,
            mode=mode,
            preferred_reduction_target=resp.preferred_reduction_target,
            target_achievement_status=resp.target_achievement_status,
            evidence_preservation_status=resp.evidence_preservation_status,
            compression_time_ms=resp.compression_time_ms,
            target_tokens=case.target_tokens,
            budget_met=resp.budget_met,
            warnings=resp.warnings,
            facts_retained=facts_retained,
            total_facts=total_facts,
            fact_retention_rate=fact_retention_rate,
            missing_facts=missing_facts,
            original_answer=None,
            compressed_answer=None,
            original_score=None,
            compressed_score=None,
            degraded=None,
            answer_status="pending",
            scoring_explanation="LLM answer evaluation is pending (no external LLM provider configured; offline mode enabled)."
        )

    def run_all(self, cases: Optional[List[BenchmarkCase]] = None, mode: str = "fast_local") -> Tuple[AggregateMetrics, List[BenchmarkCaseResult]]:
        """
        Executes all benchmark cases, generates aggregate statistics,
        and saves results to JSON and CSV.
        """
        if cases is None:
            cases = self.load_cases()

        results: List[BenchmarkCaseResult] = []
        for case in cases:
            res = self.evaluate_case(case, mode=mode)
            results.append(res)

        # Compute aggregate metrics
        reductions = [r.reduction_percentage for r in results if r.original_tokens > 0]
        latencies = [r.compression_time_ms for r in results]
        fact_rates = [r.fact_retention_rate for r in results if r.total_facts > 0]

        mean_reduction = round(statistics.mean(reductions), 2) if reductions else 0.0
        median_reduction = round(statistics.median(reductions), 2) if reductions else 0.0
        cases_reaching_95 = sum(1 for r in results if r.reduction_percentage >= 95.0)
        cases_safety_limit = sum(1 for r in results if r.target_achievement_status == "Safety limit reached")
        median_latency = round(statistics.median(latencies), 2) if latencies else 0.0
        p95_latency = (
            round(statistics.quantiles(latencies, n=100)[94], 2)
            if len(latencies) >= 20 else round(max(latencies), 2)
        )
        mean_fact_retention = round(statistics.mean(fact_rates), 2) if fact_rates else 100.0
        budget_warnings = sum(1 for r in results if not r.budget_met or r.warnings)

        # Target validations
        t1_met = mean_reduction >= 30.0
        t2_status = "Pending (LLM provider not configured; core evidence retention verified)"
        t3_met = p95_latency < 100.0

        targets_status = {
            "target_1_token_reduction": {
                "target": ">= 30.0%",
                "measured": f"{mean_reduction}%",
                "achieved": t1_met
            },
            "target_2_answer_quality": {
                "target": "compressed full-correct rate within 5% of original",
                "measured": "Pending LLM Configuration",
                "status": t2_status
            },
            "target_3_latency_p95": {
                "target": "< 100.0 ms",
                "measured": f"{p95_latency} ms",
                "achieved": t3_met
            },
            "target_stretch_95_reduction": {
                "target": ">= 95.0% on suitable redundant inputs",
                "cases_reaching_95": cases_reaching_95,
                "status": f"{cases_reaching_95} cases achieved >= 95% reduction"
            }
        }

        # Categories breakdown
        categories_summary: Dict[str, Dict[str, Any]] = {}
        for r in results:
            cat = r.category
            if cat not in categories_summary:
                categories_summary[cat] = {"count": 0, "reductions": [], "fact_rates": []}
            categories_summary[cat]["count"] += 1
            if r.original_tokens > 0:
                categories_summary[cat]["reductions"].append(r.reduction_percentage)
            if r.total_facts > 0:
                categories_summary[cat]["fact_rates"].append(r.fact_retention_rate)

        cat_metrics = {}
        for cat, data in categories_summary.items():
            cat_metrics[cat] = {
                "count": data["count"],
                "avg_reduction": round(statistics.mean(data["reductions"]), 1) if data["reductions"] else 0.0,
                "avg_fact_retention": round(statistics.mean(data["fact_rates"]), 1) if data["fact_rates"] else 100.0
            }

        aggregate = AggregateMetrics(
            total_cases=len(results),
            mean_token_reduction=mean_reduction,
            median_token_reduction=median_reduction,
            cases_reaching_95_pct=cases_reaching_95,
            cases_safety_limit_count=cases_safety_limit,
            median_compression_latency_ms=median_latency,
            p95_compression_latency_ms=p95_latency,
            critical_fact_retention_rate=mean_fact_retention,
            budget_warning_count=budget_warnings,
            answer_quality_status=t2_status,
            original_full_correct_rate=None,
            compressed_full_correct_rate=None,
            diff_full_correct_rate=None,
            degraded_cases_count=0,
            targets_status=targets_status,
            categories_summary=cat_metrics
        )

        # Export results to JSON and CSV
        self._export_results(aggregate, results)

        return aggregate, results

    def _export_results(self, aggregate: AggregateMetrics, results: List[BenchmarkCaseResult]):
        """Saves evaluation results to JSON and CSV."""
        json_path = os.path.join(self.results_dir, "results.json")
        csv_path = os.path.join(self.results_dir, "results.csv")

        # Save JSON
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "aggregate_metrics": aggregate.model_dump(),
            "cases": [r.model_dump() for r in results]
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Save CSV
        fieldnames = [
            "case_id",
            "category",
            "original_tokens",
            "compressed_tokens",
            "tokens_removed",
            "reduction_percentage",
            "preferred_reduction_target",
            "target_achievement_status",
            "evidence_preservation_status",
            "compression_time_ms",
            "target_tokens",
            "budget_met",
            "warnings_count",
            "facts_retained",
            "total_facts",
            "fact_retention_rate",
            "missing_facts",
            "answer_status"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "case_id": r.case_id,
                    "category": r.category,
                    "original_tokens": r.original_tokens,
                    "compressed_tokens": r.compressed_tokens,
                    "tokens_removed": r.tokens_removed,
                    "reduction_percentage": r.reduction_percentage,
                    "preferred_reduction_target": r.preferred_reduction_target,
                    "target_achievement_status": r.target_achievement_status,
                    "evidence_preservation_status": r.evidence_preservation_status,
                    "compression_time_ms": r.compression_time_ms,
                    "target_tokens": r.target_tokens,
                    "budget_met": r.budget_met,
                    "warnings_count": len(r.warnings),
                    "facts_retained": r.facts_retained,
                    "total_facts": r.total_facts,
                    "fact_retention_rate": r.fact_retention_rate,
                    "missing_facts": "; ".join(r.missing_facts),
                    "answer_status": r.answer_status
                })

    def get_latest_results(self) -> Optional[Dict[str, Any]]:
        """Returns the latest saved evaluation results if available."""
        json_path = os.path.join(self.results_dir, "results.json")
        if not os.path.exists(json_path):
            return None
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
