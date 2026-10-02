"""
PromptCompressor — Reduce token count in prompts before sending to AI.

Two modes:
  - "simple"    : Rule-based compression (no dependencies, instant)
  - "llmlingua" : LLMLingua model-based compression (requires: pip install tokensaver[llmlingua])

Example:
    from tokensaver import PromptCompressor

    # Simple (no model needed)
    c = PromptCompressor(mode="simple")
    result = c.compress("Your long prompt here...")
    print(result.ratio, result.compressed_text)

    # LLMLingua (requires optional install)
    c = PromptCompressor(mode="llmlingua", target_ratio=0.5)
    result = c.compress("Your long prompt here...")
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

from tokensaver.counter import TokenCounter


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class CompressionResult:
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    mode: str

    @property
    def ratio(self) -> float:
        """Compression ratio: 0.3 means 30% of original size (70% saved)."""
        if self.original_tokens == 0:
            return 1.0
        return self.compressed_tokens / self.original_tokens

    @property
    def saved_tokens(self) -> int:
        return self.original_tokens - self.compressed_tokens

    @property
    def saved_percent(self) -> float:
        return (1.0 - self.ratio) * 100

    def summary(self) -> str:
        return (
            f"[{self.mode}] {self.original_tokens:,} → {self.compressed_tokens:,} tokens "
            f"(saved {self.saved_tokens:,} = {self.saved_percent:.1f}%)"
        )


# ---------------------------------------------------------------------------
# PromptCompressor
# ---------------------------------------------------------------------------

class PromptCompressor:
    """
    Compress prompts to reduce token usage.

    Args:
        mode: "simple" or "llmlingua"
        target_ratio: For llmlingua — target compression ratio (0.5 = keep 50%)
        provider: Provider for token counting (default "anthropic")
        model: Model for token counting
    """

    def __init__(
        self,
        mode: Literal["simple", "llmlingua"] = "simple",
        target_ratio: float = 0.5,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-5",
    ):
        self.mode = mode
        self.target_ratio = target_ratio
        self._counter = TokenCounter(provider=provider, model=model)
        self._llmlingua = None  # lazy loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(self, text: str) -> CompressionResult:
        """Compress text and return a CompressionResult."""
        original_tokens = self._counter.count(text).tokens

        if self.mode == "llmlingua":
            compressed = self._compress_llmlingua(text)
        else:
            compressed = self._compress_simple(text)

        compressed_tokens = self._counter.count(compressed).tokens
        return CompressionResult(
            original_text=text,
            compressed_text=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            mode=self.mode,
        )

    def compress_messages(
        self, messages: list[dict], compress_roles: list[str] | None = None
    ) -> tuple[list[dict], CompressionResult]:
        """
        Compress content in a messages list.

        Args:
            messages: List of {"role": ..., "content": ...}
            compress_roles: Which roles to compress. Default: ["user", "system"]
        Returns:
            Tuple of (compressed_messages, overall_CompressionResult)
        """
        roles = compress_roles or ["user", "system"]
        full_original = " ".join(
            m.get("content", "") for m in messages if isinstance(m.get("content"), str)
        )
        original_tokens = self._counter.count(full_original).tokens

        compressed_messages = []
        for msg in messages:
            if msg.get("role") in roles and isinstance(msg.get("content"), str):
                compressed_content = self._do_compress(msg["content"])
                compressed_messages.append({**msg, "content": compressed_content})
            else:
                compressed_messages.append(msg)

        full_compressed = " ".join(
            m.get("content", "") for m in compressed_messages if isinstance(m.get("content"), str)
        )
        compressed_tokens = self._counter.count(full_compressed).tokens

        result = CompressionResult(
            original_text=full_original,
            compressed_text=full_compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            mode=self.mode,
        )
        return compressed_messages, result

    # ------------------------------------------------------------------
    # Simple compression (rule-based, no models)
    # ------------------------------------------------------------------

    def _compress_simple(self, text: str) -> str:
        """Apply a series of rule-based compressions."""
        text = self._remove_excess_whitespace(text)
        text = self._remove_repeated_punctuation(text)
        text = self._condense_code_comments(text)
        text = self._remove_filler_phrases(text)
        return text.strip()

    @staticmethod
    def _remove_excess_whitespace(text: str) -> str:
        # Normalize multiple blank lines to single blank line
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Normalize multiple spaces (but not indentation)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text

    @staticmethod
    def _remove_repeated_punctuation(text: str) -> str:
        # e.g. "....." → "..." | "!!!!" → "!"
        text = re.sub(r"\.{4,}", "...", text)
        text = re.sub(r"!{2,}", "!", text)
        text = re.sub(r"\?{2,}", "?", text)
        text = re.sub(r"-{4,}", "---", text)
        text = re.sub(r"={4,}", "===", text)
        return text

    @staticmethod
    def _condense_code_comments(text: str) -> str:
        # Remove single-line comments in common languages (# // --)
        # but only if line is ONLY a comment (preserves inline comments)
        lines = text.split("\n")
        result = []
        for line in lines:
            stripped = line.strip()
            is_pure_comment = (
                stripped.startswith("#")
                or stripped.startswith("//")
                or stripped.startswith("--")
            ) and not stripped.startswith("#!")  # keep shebangs
            if not is_pure_comment:
                result.append(line)
        return "\n".join(result)

    _FILLER_PATTERNS = [
        r"\bplease note that\b",
        r"\bit is important to note that\b",
        r"\bit should be noted that\b",
        r"\bas you can see\b",
        r"\bof course\b",
        r"\bcertainly\b",
        r"\babsolutely\b",
        r"\bI hope this helps\b",
        r"\bfeel free to\b",
        r"\bin conclusion\b",
        r"\bto summarize\b",
        r"\bbasically\b",
        r"\bessentially\b",
        r"\bactually\b",
        r"\bjust\b(?= \w)",  # "just" before a word (but not "just in time")
    ]

    def _remove_filler_phrases(self, text: str) -> str:
        for pattern in self._FILLER_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        # Clean up any double spaces left behind
        text = re.sub(r"  +", " ", text)
        return text

    # ------------------------------------------------------------------
    # LLMLingua compression (optional dependency)
    # ------------------------------------------------------------------

    def _compress_llmlingua(self, text: str) -> str:
        if self._llmlingua is None:
            self._llmlingua = self._load_llmlingua()
        if self._llmlingua is None:
            # Fallback to simple if llmlingua not available
            return self._compress_simple(text)
        try:
            result = self._llmlingua.compress_prompt(
                text,
                rate=self.target_ratio,
                force_tokens=["\n", "?", ".", "!"],
            )
            return result.get("compressed_prompt", text)
        except Exception:
            return self._compress_simple(text)

    @staticmethod
    def _load_llmlingua():
        try:
            from llmlingua import PromptCompressor as LLM  # type: ignore
            return LLM(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                use_llmlingua2=True,
                device_map="cpu",
            )
        except ImportError:
            raise ImportError(
                "LLMLingua is not installed. Run: pip install tokensaver[llmlingua]"
            )
        except Exception:
            return None

    def _do_compress(self, text: str) -> str:
        if self.mode == "llmlingua":
            return self._compress_llmlingua(text)
        return self._compress_simple(text)
