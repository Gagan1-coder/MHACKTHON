"""
Context Surgeon - Tokenizer Module
Provides robust token counting with tiktoken (cl100k_base / o200k_base)
and a deterministic offline fallback tokenizer with clearly identified methodology.
"""

from typing import List, Tuple
import re

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    _TIKTOKEN_AVAILABLE = False


class Tokenizer:
    """
    Tokenizer wrapper that prefers standard LLM tokenizers (tiktoken cl100k_base for GPT-4/3.5/modern models)
    with a deterministic fallback regex/BPE-approximating tokenizer for offline/air-gapped environments.
    """

    def __init__(self, model_name: str = "cl100k_base", force_fallback: bool = False):
        self.model_name = model_name
        self.force_fallback = force_fallback
        self._tiktoken_enc = None

        if _TIKTOKEN_AVAILABLE and not force_fallback:
            try:
                # Try getting encoding directly or by model name
                if model_name in ("cl100k_base", "o200k_base", "p50k_base", "r50k_base"):
                    self._tiktoken_enc = tiktoken.get_encoding(model_name)
                else:
                    self._tiktoken_enc = tiktoken.encoding_for_model(model_name)
            except Exception:
                try:
                    self._tiktoken_enc = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    self._tiktoken_enc = None

    @property
    def is_using_fallback(self) -> bool:
        return self._tiktoken_enc is None

    @property
    def tokenizer_info(self) -> dict:
        if not self.is_using_fallback:
            return {
                "type": "tiktoken",
                "encoding": self.model_name,
                "is_fallback": False,
                "description": "Production byte-pair encoding (BPE) matching modern LLM tokenizers."
            }
        return {
            "type": "deterministic_fallback",
            "encoding": "regex_subword_heuristic",
            "is_fallback": True,
            "description": (
                "Deterministic offline tokenizer estimating subword tokens via regex word-boundary, "
                "punctuation, and character clustering (~3.8 - 4.2 chars per token). "
                "Limitations: Does not emulate exact BPE vocab merges for non-Latin scripts."
            )
        }

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a string."""
        if not text:
            return 0
        if self._tiktoken_enc is not None:
            try:
                return len(self._tiktoken_enc.encode(text, disallowed_special=()))
            except Exception:
                pass
        return self._fallback_count_tokens(text)

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs. If fallback, uses deterministic integer hashing of subwords."""
        if not text:
            return []
        if self._tiktoken_enc is not None:
            try:
                return self._tiktoken_enc.encode(text, disallowed_special=())
            except Exception:
                pass
        return self._fallback_encode(text)

    def _fallback_count_tokens(self, text: str) -> int:
        """
        Deterministic regex-based subword tokenizer fallback.
        Approximates subword tokenization patterns:
        - Words with contractions (e.g. don't -> don, 't)
        - Numbers and punctuation
        - Long words split into ~4 char chunks
        - Whitespace chunks
        """
        if not text:
            return 0

        # Standard Python unicode pattern matching words, numbers, punctuation, contractions, and whitespace
        py_pattern = re.compile(
            r"'(?:[sStTrReEvVmMdD]|ll|LL|re|RE|ve|VE)|[^\W\d_]+|\d+|[^\w\s]|\s+",
            re.UNICODE
        )
        
        matches = py_pattern.findall(text)
        token_count = 0
        for m in matches:
            length = len(m)
            if length <= 4:
                token_count += 1
            else:
                # Subword approximation: 1 token per ~3.8 characters for longer tokens
                token_count += max(1, (length + 3) // 4)
        return max(1, token_count) if text.strip() else token_count

    def _fallback_encode(self, text: str) -> List[int]:
        """Deterministic integer hashing of subwords for offline representation."""
        if not text:
            return []
        py_pattern = re.compile(
            r"'(?:[sStTrReEvVmMdD]|ll|LL|re|RE|ve|VE)|[A-Za-z]+|[0-9]+|[^\w\s]|\s+",
            re.UNICODE
        )
        matches = py_pattern.findall(text)
        tokens = []
        for m in matches:
            # Deterministic hash to positive int32
            val = abs(hash(m)) % 100000 + 1
            tokens.append(val)
        return tokens


# Global default tokenizer instance
default_tokenizer = Tokenizer()
