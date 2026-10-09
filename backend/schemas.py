"""
Context Surgeon - Pydantic Schemas
Defines request, response, section detail, and benchmark data structures.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class CompressRequest(BaseModel):
    system_prompt: str = Field(default="", description="System prompt to preserve verbatim.")
    context: str = Field(..., description="Supporting context or conversation history to compress.")
    query: str = Field(..., description="User query driving relevance assessment.")
    target_tokens: int = Field(default=500, gt=0, description="Target context token budget ceiling (must be positive).")
    preferred_reduction_target: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Optional target benchmark percentage.")
    mode: str = Field(default="fast_local", description="Compression mode: 'fast_local' (default) | 'recursive' | 'auto'.")
    max_iterations: Optional[int] = Field(default=8, ge=1, le=50, description="Max iterations for recursive mode.")
    force_fallback_tokenizer: bool = Field(default=False, description="Whether to use offline fallback tokenizer.")

    @field_validator("target_tokens")
    @classmethod
    def validate_budget(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("target_tokens must be greater than 0")
        return v


class SectionInfo(BaseModel):
    section_id: str
    original_position: int
    source_type: str = Field(..., description="e.g. message_turn, code_block, json_record, paragraph, sentence")
    preview: str
    content: str
    tokens: int
    relevance_score: float
    decision: str = Field(..., description="'retained' | 'removed' | 'deduplicated'")
    reason: str
    protection_rules: List[str] = Field(default_factory=list)
    is_protected: bool = False


class CompressSummary(BaseModel):
    total_sections: int
    retained_sections: int
    removed_sections: int
    deduplicated_sections: int
    protected_sections: int
    preferred_reduction_target: Optional[float] = None
    target_achievement_status: Optional[str] = "Evaluated"
    evidence_preservation_status: str = "All critical evidence preserved"
    tokenizer_info: Dict[str, Any]
    mode: str = "fast_local"


class CompressResponse(BaseModel):
    system_prompt: str
    query: str
    original_context: str
    compressed_context: str
    original_tokens: int
    compressed_tokens: int
    tokens_removed: int
    reduction_percentage: float
    mode: str = Field(default="fast_local", description="Operating compression mode used.")
    preferred_reduction_target: Optional[float] = None
    target_achievement_status: Optional[str] = "Evaluated"
    evidence_preservation_status: str = "All critical evidence preserved"
    compression_time_ms: float
    target_tokens: int
    budget_met: bool
    warnings: List[str] = Field(default_factory=list)
    sections: List[SectionInfo] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    recursive_trace: Optional[List[Dict[str, Any]]] = Field(default=None, description="Step-by-step observable trajectory for recursive exploration.")
    recursive_stats: Optional[Dict[str, Any]] = Field(default=None, description="Detailed usage and sub-call metrics for recursive engine.")


class BenchmarkCase(BaseModel):
    case_id: str
    category: str
    system_prompt: str = ""
    context: str
    query: str
    target_tokens: int
    preferred_reduction_target: float = 95.0
    expected_critical_facts: List[str] = Field(default_factory=list)
    expected_identifiers: List[str] = Field(default_factory=list)
    notes: str = ""


class BenchmarkCaseResult(BaseModel):
    case_id: str
    category: str
    original_tokens: int
    compressed_tokens: int
    tokens_removed: int
    reduction_percentage: float
    mode: str = "fast_local"
    preferred_reduction_target: float = 95.0
    target_achievement_status: str = "Below target"
    evidence_preservation_status: str = "All critical evidence preserved"
    compression_time_ms: float
    target_tokens: int
    budget_met: bool
    warnings: List[str]
    facts_retained: int
    total_facts: int
    fact_retention_rate: float
    missing_facts: List[str]
    original_answer: Optional[str] = None
    compressed_answer: Optional[str] = None
    original_score: Optional[int] = None
    compressed_score: Optional[int] = None
    degraded: Optional[bool] = None
    answer_status: str = "pending"  # "pending" or "evaluated"
    scoring_explanation: Optional[str] = None


class AggregateMetrics(BaseModel):
    total_cases: int
    mean_token_reduction: float
    median_token_reduction: float = 0.0
    cases_reaching_95_pct: int = 0
    cases_safety_limit_count: int = 0
    median_compression_latency_ms: float
    p95_compression_latency_ms: float
    critical_fact_retention_rate: float
    budget_warning_count: int
    answer_quality_status: str
    original_full_correct_rate: Optional[float] = None
    compressed_full_correct_rate: Optional[float] = None
    diff_full_correct_rate: Optional[float] = None
    degraded_cases_count: int = 0
    targets_status: Dict[str, Any]
    categories_summary: Dict[str, Any]
