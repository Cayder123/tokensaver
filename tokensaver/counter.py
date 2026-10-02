"""
TokenCounter — Count tokens accurately per provider/model.

Supported providers:
  - anthropic  (uses cl100k_base via tiktoken, close approximation)
  - openai     (uses exact tiktoken encoding per model)
  - gemini     (estimate: 4 chars ≈ 1 token)

Example:
    from tokensaver import TokenCounter

    counter = TokenCounter(provider="anthropic", model="claude-sonnet-4-5")
    result = counter.count("Hello, how are you?")
    print(result.tokens, result.cost_usd)

    result = counter.count_messages([
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ])
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from tokensaver.pricing import PricingTable


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TokenCount:
    tokens: int
    provider: str
    model: str
    cost_usd: float = 0.0
    token_type: str = "input"  # "input" | "output" | "total"

    def __str__(self) -> str:
        return (
            f"TokenCount(tokens={self.tokens:,}, "
            f"model={self.model}, "
            f"cost=${self.cost_usd:.6f})"
        )


@dataclass
class ConversationCount:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    provider: str
    model: str
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    total_cost_usd: float = 0.0


# ---------------------------------------------------------------------------
# Provider encoding map
# ---------------------------------------------------------------------------

# tiktoken encoding per OpenAI model family
_OPENAI_ENCODINGS: dict[str, str] = {
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o3-mini": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3-5-turbo": "cl100k_base",
}

# Anthropic uses a custom BPE close to cl100k_base
_ANTHROPIC_ENCODING = "cl100k_base"

# Message overhead tokens (approximated from OpenAI cookbook)
_MSG_OVERHEAD_PER_MESSAGE = 4  # every message: <|im_start|>{role}\n ... <|im_end|>
_MSG_OVERHEAD_REPLY_PRIMER = 3  # every reply is primed with <|im_start|>assistant


# ---------------------------------------------------------------------------
# TokenCounter
# ---------------------------------------------------------------------------

class TokenCounter:
    """
    Count tokens for a prompt or conversation, optionally calculate cost.

    Args:
        provider: "anthropic" | "openai" | "gemini"
        model: Model name string (e.g. "claude-sonnet-4-5", "gpt-4o-mini")
        pricing: Optional PricingTable instance. Created automatically if None.
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-5",
        pricing: Optional[PricingTable] = None,
    ):
        self.provider = provider.lower()
        self.model = model.lower()
        self._pricing = pricing or PricingTable()
        self._enc = self.__load_encoder_impl()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count(self, text: str, token_type: str = "input") -> TokenCount:
        """Count tokens in a plain text string."""
        n = self._count_text(text)
        cost = self._cost(n, 0) if token_type == "input" else self._cost(0, n)
        return TokenCount(
            tokens=n,
            provider=self.provider,
            model=self.model,
            cost_usd=cost,
            token_type=token_type,
        )

    def count_messages(
        self,
        messages: list[dict],
        expected_output_tokens: int = 0,
    ) -> ConversationCount:
        """
        Count tokens in a conversation messages list.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": "..."}
            expected_output_tokens: Estimate of output tokens for cost calculation.
        """
        total = _MSG_OVERHEAD_REPLY_PRIMER
        for msg in messages:
            total += _MSG_OVERHEAD_PER_MESSAGE
            role = msg.get("role", "")
            content = msg.get("content", "")
            total += self._count_text(role)
            if isinstance(content, str):
                total += self._count_text(content)
            elif isinstance(content, list):
                # Handle multi-modal content blocks
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        total += self._count_text(block.get("text", ""))

        price = self._pricing.get(self.provider, self.model)
        in_cost = (price.input_per_1m * total / 1_000_000) if price else 0.0
        out_cost = (price.output_per_1m * expected_output_tokens / 1_000_000) if price else 0.0

        return ConversationCount(
            input_tokens=total,
            output_tokens=expected_output_tokens,
            total_tokens=total + expected_output_tokens,
            provider=self.provider,
            model=self.model,
            input_cost_usd=in_cost,
            output_cost_usd=out_cost,
            total_cost_usd=in_cost + out_cost,
        )

    def estimate_output_cost(self, output_text: str) -> float:
        """Estimate cost for a given output text."""
        n = self._count_text(output_text)
        return self._cost(0, n)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_text(self, text: str) -> int:
        if not text:
            return 0
        if self.provider == "gemini":
            # Google does not publish tokenizer; 4 chars ≈ 1 token is a common estimate
            return max(1, len(text) // 4)
        try:
            return len(self._enc.encode(text))
        except Exception:
            return max(1, len(text) // 4)


    def __load_encoder_impl(self):
        if self.provider == "gemini":
            return None
        try:
            import tiktoken
        except ImportError:
            return None

        if self.provider == "anthropic":
            return tiktoken.get_encoding(_ANTHROPIC_ENCODING)

        # OpenAI — find best encoding match
        for prefix, encoding in _OPENAI_ENCODINGS.items():
            if self.model.startswith(prefix):
                return tiktoken.get_encoding(encoding)
        # Default fallback
        try:
            return tiktoken.encoding_for_model(self.model)
        except Exception:
            return tiktoken.get_encoding("cl100k_base")

    def _cost(self, input_tokens: int, output_tokens: int) -> float:
        price = self._pricing.get(self.provider, self.model)
        if price is None:
            return 0.0
        return price.cost_usd(input_tokens, output_tokens)
