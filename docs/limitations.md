# Context Surgeon — Known Engineering Limitations

In the spirit of honest engineering and hackathon rigor, this document outlines the operational boundaries and known edge cases of Context Surgeon.

---

## 1. Extractive Heuristic Nature
Context Surgeon is an **extractive** compression system, not an abstractive rewriting system.
- **Advantage:** Eliminates generative hallucination risks; exact dates, numbers, and code syntax remain 100% faithful to the original source text.
- **Limitation:** Cannot rephrase multiple verbose sentences into a single hyper-condensed summary sentence. If a section is retained, its original text is preserved in full.

## 2. Multi-Hop Semantic Reasoning
- Highly indirect facts that lack explicit lexical overlap, entity IDs, or direct function call dependencies may receive lower relevance scores unless they contain protected numbers or temporal markers.
- *Mitigation:* The user query should clearly express the core entities (e.g. `order ORD-842` or `customer CUST-204`).

## 3. Programming Language Dependency Support
- The current symbol definition-reference resolver actively traverses Python syntax (`def <name>`, function calls).
- Other languages (such as C++, Java, Rust, or JavaScript) will have their code blocks protected as indivisible units, but symbol cross-referencing between separate blocks defaults to lexical identifier matching rather than AST parsing.

## 4. Subword Tokenizer Approximation in Fallback Mode
- When running in offline fallback mode without `tiktoken`, the deterministic regex tokenizer approximates subwords (~3.8 to 4.2 characters per token). While within ~5-10% of GPT-4 BPE counts for English text, it does not emulate byte merges for non-Latin alphabets.
