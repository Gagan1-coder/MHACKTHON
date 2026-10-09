# Context Surgeon — 5-Stage Compression Algorithm

Context Surgeon does not use lossy neural summarization that can hallucinate dates or drop critical negations. Instead, it employs a deterministic, explainable extractive compression pipeline across five discrete stages.

---

## Stage 1: Multi-Modal Boundary Segmentation

Context is split into structured, discrete sections while respecting semantic boundaries:
- **Fenced Code Blocks:** Any text enclosed in ` ``` ` fences is kept as a single atomic unit. It is never split across lines merely to hit a token threshold.
- **Structured Records & JSON:** Standalone `{ ... }` and `[ ... ]` objects are identified and parsed as indivisible JSON record units.
- **Conversation Turns:** Conversational turn markers (e.g. `User:`, `Assistant:`, `[2026-10-09 14:00]:`) are segmented into distinct conversational steps.
- **Paragraphs & Bullet Items:** Natural double newline boundaries define paragraph blocks. If a dense single paragraph contains multiple factual sentences (such as baseline assertions followed by later amendments), sentence-level splitting occurs to allow fine-grained fact retention.
- **Metadata Assignment:** Each section is assigned a unique, stable ID (`sec_001`, `sec_002`), source type, token count, and character offsets.

---

## Stage 2: Deduplication & Query-Term Rarity Ranking

1. **Exact Duplicate Detection:** Normalized MD5 hashing detects duplicate log entries, repeated conversation turns, and repeated code snippets. Subsequent identical sections are labeled `deduplicated` and excluded from selection.
2. **Term-Frequency Inverse Document Frequency (TF-IDF Rarity):**
   - Query terms are tokenized and filtered against an extensive stopword dictionary.
   - Non-stopword terms appearing in fewer sections receive a rarity boost (rarity factor up to `+2.0x`).
3. **Exact Entity & Identifier Overlap:**
   - Regex matches uppercase codes (e.g. `ORD-842`, `CUST-204`, `REF-9902`), snake_case/camelCase symbols, and network identifiers (`Port 22`).
   - Matching sections receive an immediate priority score multiplier.
4. **Recency Weighting:**
   - Later messages in conversational context receive a minor, controllable priority bonus to prioritize recent turns over resolved history.

---

## Stage 3: Conservative Evidence Protection & Expansion

Heuristic compression fails when critical facts are pruned. Context Surgeon implements conservative protection rules:

| Protection Category | Rule Mechanism | Test Objective |
| :--- | :--- | :--- |
| **Exact Identifiers** | Matches alphanumeric ID patterns present in query (`ORD-842`, `CUST-204`) | Ensures target record is never dropped |
| **Dates & Times** | Regex matching dates (`12 October`, `2026-03-15`) and timestamps (`10 AM`, `14:00`) | Retains temporal milestones |
| **Negations & Prohibitions** | Detects `not`, `never`, `must not`, `closed`, `prohibited`, `forbidden` | Prevents security rule inversions (Test B) |
| **Later Corrections** | Detects `revised to`, `correction:`, `amended`, `rescheduled to`, `instead of` | Prioritizes superseding updates (Test A, Test C) |
| **Comparative Fact Expansion** | When a correction exists, earlier sections containing the referenced baseline fact are automatically protected | Retains baseline date and revised date to allow delta calculation |
| **Python Code Dependencies** | Analyzes symbol definitions (`def foo`) and call sites (`foo()`). If a selected function calls a helper, the helper is pulled in | Prevents broken code imports (Test D) |
| **Structured JSON Records** | Protects specific JSON records that match query entity keys | Drops unrelated rows, retains target record (Test E) |

---

## Stage 4: Token-Budget-Aware Selection

1. **Selection Ordering:**
   - Candidate sections (excluding duplicates) are sorted with protected sections first, followed by relevance score descending.
2. **Irrelevant Boilerplate Gate:**
   - Unprotected sections with relevance score `< 0.5` (such as polite greetings or unrelated topics) are excluded from selection even if leftover budget exists.
3. **Impossible Budget Safety Rule (Test F):**
   - If the target token budget is too small to fit the minimum required critical evidence, Context Surgeon **does not silently truncate facts to claim budget compliance**.
   - Instead, all critical evidence is preserved to prevent downstream LLM hallucination, and a transparent budget warning is returned:
     `Target budget (10 tokens) is smaller than required critical evidence (35 tokens). Critical evidence was retained to ensure factual correctness.`
4. **Chronological Reassembly:**
   - Retained sections are sorted strictly by their original position index and concatenated cleanly.

---

## Stage 5: Explainability & Decision Audit

Every section returned in the API and UI contains:
- `section_id` & `source_type`
- `preview` & full content
- `tokens` & `relevance_score`
- `decision` (`retained`, `removed`, `deduplicated`)
- `reason` (Plain-English explanation of why the action was taken)
- `protection_rules` (List of triggered rules, e.g. `["exact_identifier:ORD-842", "date_time_evidence"]`)
- Complete overall metrics (Original Tokens, Compressed Tokens, Tokens Removed, Reduction %, Compression Time ms, Budget Met).
