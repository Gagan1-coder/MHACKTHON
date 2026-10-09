"""
Context Surgeon - FastAPI Application
Provides REST API for context compression, health monitoring, and benchmark evaluation.
Serves the web dashboard static assets.
"""

import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.schemas import (
    CompressRequest,
    CompressResponse,
    BenchmarkCase,
    AggregateMetrics,
    BenchmarkCaseResult
)
from backend.compressor import ContextSurgeon
from backend.tokenizer import default_tokenizer
from backend.evaluator import BenchmarkEvaluator


app = FastAPI(
    title="Context Surgeon API",
    description="Intelligent Preprocessing Layer: Compress LLM Context Without Losing Essential Meaning",
    version="1.0.0"
)

# Enable CORS for local development and dashboard communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
surgeon = ContextSurgeon()
evaluator = BenchmarkEvaluator()

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")


# -----------------------------------------------------------------------------
# API ENDPOINTS
# -----------------------------------------------------------------------------

@app.get("/api/health", summary="Service health and tokenizer status")
async def health_check() -> Dict[str, Any]:
    """Returns the operational status and tokenizer configuration of Context Surgeon."""
    return {
        "status": "healthy",
        "service": "Context Surgeon",
        "tagline": "Compress LLM Context Without Losing Essential Meaning",
        "version": "1.0.0",
        "tokenizer": default_tokenizer.tokenizer_info,
        "docs_url": "/docs"
    }


@app.post("/api/compress", response_model=CompressResponse, summary="Compress supporting context")
async def compress_context_endpoint(request: CompressRequest) -> CompressResponse:
    """
    Compresses supporting context based on query relevance and token budget.
    Ensures that system_prompt and query remain unchanged.
    """
    try:
        response = surgeon.compress(
            system_prompt=request.system_prompt,
            context=request.context,
            query=request.query,
            target_tokens=request.target_tokens,
            preferred_reduction_target=request.preferred_reduction_target,
            mode=request.mode,
            max_iterations=request.max_iterations or 8,
            force_fallback_tokenizer=request.force_fallback_tokenizer
        )
        return response
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Compression engine failure: {str(exc)}"
        )


@app.get("/api/benchmarks", response_model=List[BenchmarkCase], summary="List available benchmark cases")
async def list_benchmarks() -> List[BenchmarkCase]:
    """Retrieves all pre-configured benchmark cases covering 30+ problem categories."""
    try:
        cases = evaluator.load_cases()
        return cases
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load benchmark cases: {str(exc)}"
        )


@app.post("/api/evaluate", summary="Run benchmark evaluation")
async def run_evaluation(
    selected_cases: Optional[List[BenchmarkCase]] = Body(default=None),
    mode: str = "fast_local"
) -> Dict[str, Any]:
    """
    Executes benchmark evaluation across test cases, generates aggregate statistics,
    and updates persisted JSON/CSV results.
    """
    try:
        aggregate, results = evaluator.run_all(cases=selected_cases, mode=mode)
        return {
            "status": "success",
            "aggregate_metrics": aggregate.model_dump(),
            "results_count": len(results),
            "cases": [r.model_dump() for r in results]
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation execution error: {str(exc)}"
        )


@app.get("/api/benchmarks/results", summary="Get latest benchmark evaluation results")
async def get_benchmark_results() -> Dict[str, Any]:
    """
    Returns the most recently computed benchmark results.
    If evaluation has not yet been executed, returns a clear pending indicator.
    """
    results_data = evaluator.get_latest_results()
    if results_data is None:
        return {
            "status": "not_evaluated",
            "message": "Benchmark evaluation has not been executed yet. Run POST /api/evaluate or execute via the Benchmark Lab tab.",
            "aggregate_metrics": None,
            "cases": []
        }
    return {
        "status": "available",
        **results_data
    }


# -----------------------------------------------------------------------------
# STATIC FILES & WEB DASHBOARD MOUNTING
# -----------------------------------------------------------------------------

if os.path.exists(FRONTEND_DIR):
    # Mount static assets directory and serve root SPA/dashboard
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

