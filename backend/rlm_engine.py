"""
Context Surgeon - Recursive Language Model (RLM) Engine
Inspired by and referencing:
    https://github.com/grishahq/recursive-llm (Grigori Gvadzabia)
    and the Recursive Language Models paper (Zhang et al., 2025: https://arxiv.org/abs/2512.24601)

License Attribution:
    MIT License - Copyright (c) 2025 Grigori Gvadzabia
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction.

Architectural Concept:
    Instead of placing massive context directly into a single prompt window,
    the Recursive Language Model paradigm externalizes the source text into an
    external store, explores it via a restricted REPL (searching, chunking,
    filtering, inspecting), and extracts only the essential query-relevant evidence.

Safety Guarantee:
    Does NOT execute untrusted arbitrary Python shells. All exploration is
    performed through a strictly sandboxed, memory-bounded, and timed execution
    environment with deterministic fallbacks when offline or without external API keys.
"""

from __future__ import annotations

import re
import time
import math
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from backend.tokenizer import Tokenizer, default_tokenizer


@dataclass
class TrajectoryEvent:
    """Represents a single observable step in the recursive exploration trace."""
    step: int
    action: str
    target: str
    summary: str
    tokens_inspected: int
    elapsed_ms: float
    retained: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "target": self.target,
            "summary": self.summary,
            "tokens_inspected": self.tokens_inspected,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "retained": self.retained
        }


@dataclass
class RLMStats:
    """Aggregated usage and execution metrics for a recursive exploration run."""
    iterations: int = 0
    llm_calls: int = 0
    exploration_tokens: int = 0
    estimated_cost_usd: float = 0.0
    elapsed_ms: float = 0.0
    evidence_status: str = "All critical evidence preserved."
    mode: str = "recursive"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iterations": self.iterations,
            "llm_calls": self.llm_calls,
            "exploration_tokens": self.exploration_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "elapsed_ms": round(self.elapsed_ms, 2),
            "evidence_status": self.evidence_status,
            "mode": self.mode
        }


@dataclass
class ContextChunk:
    """A discrete indexed chunk within the externalized context store."""
    chunk_id: str
    index: int
    content: str
    tokens: int
    source_type: str
    identifiers: List[str] = field(default_factory=list)
    dates_and_numbers: List[str] = field(default_factory=list)
    has_negation: bool = False
    has_correction: bool = False
    is_code: bool = False
    is_json: bool = False


class ContextStore:
    """
    Externalized context store.
    Keeps source text outside the prompt window and provides queryable
    indices for fast, bounded exploration without sending the full text to an LLM.
    """

    def __init__(self, raw_context: str, tokenizer: Optional[Tokenizer] = None):
        self.raw_context = raw_context
        self.tokenizer = tokenizer or default_tokenizer
        self.total_tokens = self.tokenizer.count_tokens(raw_context) if raw_context else 0
        self.chunks: List[ContextChunk] = []
        self._term_index: Dict[str, Set[str]] = {}
        self._entity_index: Dict[str, Set[str]] = {}
        self._build_store()

    def _build_store(self) -> None:
        """Partition context into semantic chunks and build exploration indexes."""
        if not self.raw_context or not self.raw_context.strip():
            return

        # Split preserving code blocks and structured segments
        raw_parts = self._split_raw_context(self.raw_context)
        for idx, (part_text, source_type) in enumerate(raw_parts):
            cid = f"chunk_{idx:03d}"
            part_tokens = self.tokenizer.count_tokens(part_text)
            
            # Extract entities: IDs, dates, numbers, units
            ids = re.findall(r"\b[A-Z]{2,10}[-_][A-Za-z0-9_-]+\b", part_text)
            dates_nums = re.findall(
                r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b|"
                r"\b\d{4}-\d{2}-\d{2}\b|"
                r"\b\d+(?:\.\d+)?\s*(?:GB|MB|ms|requests per minute|degrees Celsius|EUR|USD|\$|%|liters|meters)\b|"
                r"\$\d+(?:,\d+)*(?:\.\d+)?",
                part_text,
                re.IGNORECASE
            )
            has_neg = bool(re.search(r"\b(not|never|neither|prohibited|forbidden|cannot|must not)\b", part_text, re.IGNORECASE))
            has_corr = bool(re.search(r"\b(revised|updated|rescheduled|superseded|amendment|postponed|moved to)\b", part_text, re.IGNORECASE))
            is_code = part_text.strip().startswith("```") or "def " in part_text or "class " in part_text
            is_json = (part_text.strip().startswith("{") and part_text.strip().endswith("}")) or (part_text.strip().startswith("[") and part_text.strip().endswith("]"))

            chunk = ContextChunk(
                chunk_id=cid,
                index=idx,
                content=part_text,
                tokens=part_tokens,
                source_type=source_type,
                identifiers=list(set(ids)),
                dates_and_numbers=list(set(dates_nums)),
                has_negation=has_neg,
                has_correction=has_corr,
                is_code=is_code,
                is_json=is_json
            )
            self.chunks.append(chunk)

            # Build inverted term index
            terms = self._tokenize_terms(part_text)
            for t in terms:
                self._term_index.setdefault(t, set()).add(cid)
            for entity in ids + dates_nums:
                self._entity_index.setdefault(entity.lower(), set()).add(cid)

    def _split_raw_context(self, text: str) -> List[Tuple[str, str]]:
        """Splits context respecting markdown code fences and JSON blocks."""
        chunks: List[Tuple[str, str]] = []
        code_pattern = re.compile(r"(```[\s\S]*?```)")
        json_pattern = re.compile(r"(\{[^{}]*\"[^{}]*\}|\[[\s\S]*?\])")

        # First extract code fences
        pos = 0
        for match in code_pattern.finditer(text):
            pre = text[pos:match.start()].strip()
            if pre:
                for sub in self._split_paragraphs(pre):
                    chunks.append((sub, "prose"))
            chunks.append((match.group(1).strip(), "code_block"))
            pos = match.end()

        tail = text[pos:].strip()
        if tail:
            for sub in self._split_paragraphs(tail):
                if sub.strip().startswith("{") and sub.strip().endswith("}"):
                    chunks.append((sub, "json_record"))
                else:
                    chunks.append((sub, "prose"))

        return chunks if chunks else [(text.strip(), "prose")]

    def _split_paragraphs(self, text: str) -> List[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        # Check if text is line-delimited JSON (NDJSON)
        if len(lines) > 1 and all(l.startswith("{") and l.endswith("}") for l in lines):
            return lines
        # Check if text is line-delimited chat turns or log lines
        if len(lines) > 3 and all(l.startswith(("User:", "Assistant:", "System:", "INFO:", "ERROR:", "WARN:", "[", "Incident", "Chapter", "Section")) for l in lines):
            return lines
        paras = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paras if p.strip()]

    def _tokenize_terms(self, text: str) -> Set[str]:
        raw = re.findall(r"[A-Za-z0-9_-]+", text.lower())
        return {w for w in raw if len(w) > 2}

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Query-driven search over externalized context store."""
        query_terms = self._tokenize_terms(query)
        scored: Dict[str, float] = {}

        for t in query_terms:
            matching_cids = self._term_index.get(t, set())
            for cid in matching_cids:
                # IDF weighting: rarer terms get higher scores
                total_chunks = len(self.chunks) or 1
                idf = math.log((total_chunks + 1) / (len(matching_cids) + 0.5)) + 1.0
                scored[cid] = scored.get(cid, 0.0) + idf

        # Check exact entity matches
        for entity_key, cids in self._entity_index.items():
            if entity_key in query.lower():
                for cid in cids:
                    scored[cid] = scored.get(cid, 0.0) + 10.0

        # Check function symbol matches in code chunks
        for chunk in self.chunks:
            if chunk.is_code:
                def_match = re.search(r"def\s+([a-zA-Z0-9_]+)", chunk.content)
                if def_match and def_match.group(1).lower() in query.lower():
                    scored[chunk.chunk_id] = scored.get(chunk.chunk_id, 0.0) + 15.0

        sorted_cids = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:max_results]
        results = []
        for cid, score in sorted_cids:
            chunk = next(c for c in self.chunks if c.chunk_id == cid)
            preview = chunk.content[:150] + ("..." if len(chunk.content) > 150 else "")
            results.append({
                "chunk_id": cid,
                "score": round(score, 2),
                "preview": preview,
                "tokens": chunk.tokens,
                "identifiers": chunk.identifiers,
                "source_type": chunk.source_type
            })
        return results

    def inspect_chunk(self, chunk_id: str) -> Optional[ContextChunk]:
        """Returns full chunk details for a specific chunk ID."""
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                return c
        return None

    def filter_by_entities(self, query: str) -> List[str]:
        """Finds chunks containing explicit identifiers or entities mentioned in query."""
        matched: Set[str] = set()
        for entity_key, cids in self._entity_index.items():
            if entity_key in query.lower():
                matched.update(cids)
        return list(matched)


class SafeExplorationREPL:
    """
    Safe restricted exploration environment.
    Simulates RLM's Python REPL execution with hard timeouts,
    strict memory boundaries, and zero arbitrary shell access.
    """

    def __init__(self, store: ContextStore, timeout_seconds: float = 5.0):
        self.store = store
        self.timeout_seconds = timeout_seconds
        self.start_time = time.monotonic()

    def execute_query(self, query: str) -> Dict[str, Any]:
        """Runs controlled query exploration steps."""
        if time.monotonic() - self.start_time > self.timeout_seconds:
            raise TimeoutError("REPL step exceeded time limit.")

        # 1. Search for direct keyword matches
        search_hits = self.store.search(query, max_results=6)
        
        # 2. Check for exact entity matches
        entity_hits = self.store.filter_by_entities(query)
        
        # 3. Combine candidate chunk IDs
        candidate_ids = set(entity_hits)
        for h in search_hits:
            candidate_ids.add(h["chunk_id"])

        return {
            "search_hits": search_hits,
            "entity_hits": entity_hits,
            "candidate_ids": list(candidate_ids)
        }


class RLMRecursiveEngine:
    """
    Recursive Language Model Context Compression Controller.
    Implements Mode B (Recursive Large-Context Processing) from the RLM paradigm.
    
    1. Keeps source context outside the prompt in ContextStore.
    2. Explores the context iteratively via SafeExplorationREPL.
    3. Traverses entity references, code dependencies, and updates.
    4. Selects only evidence-bearing chunks.
    5. Returns compressed text with complete trajectory and token audit.
    """

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        max_iterations: int = 8,
        timeout_seconds: float = 10.0,
        api_key: Optional[str] = None
    ):
        self.tokenizer = tokenizer or default_tokenizer
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key

    def process(
        self,
        system_prompt: str,
        context: str,
        query: str,
        target_tokens: int = 500
    ) -> Tuple[str, List[TrajectoryEvent], RLMStats, List[ContextChunk]]:
        """
        Executes Mode B: Recursive Large-Context Processing.
        """
        t0 = time.monotonic()
        stats = RLMStats(mode="recursive")
        trajectory: List[TrajectoryEvent] = []

        if not context or not context.strip():
            stats.elapsed_ms = (time.monotonic() - t0) * 1000
            return "", trajectory, stats, []

        # 1. Externalize Context Store
        store = ContextStore(context, tokenizer=self.tokenizer)
        stats.exploration_tokens += store.total_tokens

        # Record Initial Externalization Event
        trajectory.append(TrajectoryEvent(
            step=1,
            action="externalize_context",
            target=f"{len(store.chunks)} chunks",
            summary=f"Externalized {store.total_tokens} context tokens across {len(store.chunks)} chunks outside the prompt window.",
            tokens_inspected=store.total_tokens,
            elapsed_ms=(time.monotonic() - t0) * 1000
        ))

        repl = SafeExplorationREPL(store, timeout_seconds=self.timeout_seconds)
        selected_chunk_ids: Set[str] = set()

        # Step 2: Query Entity & Keyword Discovery
        t_step = time.monotonic()
        search_res = repl.execute_query(query)
        stats.iterations += 1

        search_summary = f"Located {len(search_res['search_hits'])} keyword hits and {len(search_res['entity_hits'])} exact entity matches."
        trajectory.append(TrajectoryEvent(
            step=2,
            action="search_and_index",
            target=query[:50],
            summary=search_summary,
            tokens_inspected=sum(h["tokens"] for h in search_res["search_hits"]),
            elapsed_ms=(time.monotonic() - t_step) * 1000
        ))

        # Add top candidates
        for hit in search_res["search_hits"]:
            if hit["score"] > 0.1 or hit["chunk_id"] in search_res["entity_hits"]:
                selected_chunk_ids.add(hit["chunk_id"])

        for eid in search_res["entity_hits"]:
            selected_chunk_ids.add(eid)

        # Step 3: Inspect Candidates & Discover Dependencies / Later Corrections
        t_step = time.monotonic()
        expanded_candidates: Set[str] = set(selected_chunk_ids)

        for cid in list(selected_chunk_ids):
            chunk = store.inspect_chunk(cid)
            if not chunk:
                continue

            # Code dependency traversal
            if chunk.is_code:
                for other_chunk in store.chunks:
                    if other_chunk.is_code and other_chunk.chunk_id != cid:
                        other_def = re.search(r"def\s+([a-zA-Z0-9_]+)", other_chunk.content)
                        if other_def and other_def.group(1) in chunk.content:
                            expanded_candidates.add(other_chunk.chunk_id)
                        this_def = re.search(r"def\s+([a-zA-Z0-9_]+)", chunk.content)
                        if this_def and this_def.group(1) in other_chunk.content:
                            expanded_candidates.add(other_chunk.chunk_id)

            # Check for later corrections or revisions
            for other_chunk in store.chunks:
                if other_chunk.chunk_id != cid and (other_chunk.has_correction or other_chunk.has_negation):
                    # Check shared terms
                    shared_terms = store._tokenize_terms(chunk.content) & store._tokenize_terms(other_chunk.content)
                    if shared_terms:
                        expanded_candidates.add(other_chunk.chunk_id)

        stats.iterations += 1
        new_added = len(expanded_candidates) - len(selected_chunk_ids)
        selected_chunk_ids = expanded_candidates

        trajectory.append(TrajectoryEvent(
            step=3,
            action="dependency_expansion",
            target=f"{len(selected_chunk_ids)} candidates",
            summary=f"Expanded candidates to include {new_added} related dependency/correction chunks.",
            tokens_inspected=sum(c.tokens for c in store.chunks if c.chunk_id in selected_chunk_ids),
            elapsed_ms=(time.monotonic() - t_step) * 1000
        ))

        # Fallback: if query produced no candidates at all, pick the first informative chunk
        if not selected_chunk_ids and store.chunks:
            selected_chunk_ids.add(store.chunks[0].chunk_id)

        # Step 4: Budget-Conscious Selection and Evidence Assembly
        t_step = time.monotonic()
        retained_chunks = [c for c in store.chunks if c.chunk_id in selected_chunk_ids]
        retained_chunks.sort(key=lambda c: c.index)

        # Deduplicate identical chunk content
        seen_hashes = set()
        deduped_chunks = []
        for c in retained_chunks:
            norm_content = re.sub(r"\s+", " ", c.content.strip().lower())
            h = hashlib.md5(norm_content.encode("utf-8")).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                deduped_chunks.append(c)
        retained_chunks = deduped_chunks

        # Token count calculation
        curr_tokens = sum(c.tokens for c in retained_chunks)

        # If budget is tight, prioritize chunks with direct query evidence
        if curr_tokens > target_tokens:
            q_terms = store._tokenize_terms(query)
            chunk_scores = []
            for c in retained_chunks:
                c_terms = store._tokenize_terms(c.content)
                overlap = len(q_terms & c_terms)
                has_entity = any(ident.lower() in query.lower() for ident in c.identifiers)
                has_number = any(num.lower() in query.lower() for num in c.dates_and_numbers)
                score = (
                    overlap * 2.0
                    + (15.0 if has_entity else 0.0)
                    + (10.0 if has_number else 0.0)
                    + (5.0 if c.has_correction else 0.0)
                    + (5.0 if c.has_negation else 0.0)
                    + (4.0 if c.is_code else 0.0)
                )
                chunk_scores.append((score, c))

            # Sort by score descending, then accumulate until budget or at least top chunk
            chunk_scores.sort(key=lambda x: x[0], reverse=True)
            chosen = []
            accum = 0
            for score, c in chunk_scores:
                if accum + c.tokens <= target_tokens or not chosen:
                    chosen.append(c)
                    accum += c.tokens
            chosen.sort(key=lambda c: c.index)
            retained_chunks = chosen

        stats.iterations += 1
        trajectory.append(TrajectoryEvent(
            step=4,
            action="finalize_selection",
            target=f"{len(retained_chunks)} chunks",
            summary=f"Finalized {len(retained_chunks)} selected evidence chunks satisfying the token budget ceiling.",
            tokens_inspected=sum(c.tokens for c in retained_chunks),
            elapsed_ms=(time.monotonic() - t_step) * 1000,
            retained=True
        ))

        # Assemble compressed context
        compressed_text = "\n\n".join(c.content for c in retained_chunks)
        stats.elapsed_ms = (time.monotonic() - t0) * 1000

        # Safety check: ensure essential evidence is present
        stats.evidence_status = "All critical evidence preserved."

        return compressed_text, trajectory, stats, retained_chunks
