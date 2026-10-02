"""
BudgetGuard — Warn or block when token/cost budget is exceeded.

Example:
    from tokensaver import BudgetGuard, BudgetExceededError

    guard = BudgetGuard(session_limit_usd=1.0, warn_at_percent=0.8)
    guard.record(input_tokens=1200, output_tokens=350, provider="anthropic", model="claude-sonnet-4-5")

    # Or check before sending:
    guard.check(estimated_input_tokens=3000, provider="anthropic", model="claude-sonnet-4-5")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from tokensaver.pricing import PricingTable


# ---------------------------------------------------------------------------
# Exceptions & Data Classes
# ---------------------------------------------------------------------------

@dataclass
class BudgetUsage:
    used_tokens: int = 0
    used_usd: float = 0.0
    limit_tokens: Optional[int] = None
    limit_usd: Optional[float] = None
    warn_at_percent: float = 0.8

    @property
    def token_percent(self) -> Optional[float]:
        if self.limit_tokens:
            return self.used_tokens / self.limit_tokens
        return None

    @property
    def usd_percent(self) -> Optional[float]:
        if self.limit_usd:
            return self.used_usd / self.limit_usd
        return None

    @property
    def percent(self) -> Optional[float]:
        """Return the highest usage percent across limits."""
        percents = [p for p in [self.token_percent, self.usd_percent] if p is not None]
        return max(percents) if percents else None

    @property
    def is_warning(self) -> bool:
        p = self.percent
        return p is not None and p >= self.warn_at_percent

    @property
    def is_exceeded(self) -> bool:
        p = self.percent
        return p is not None and p >= 1.0

    def summary(self) -> str:
        parts = []
        if self.limit_tokens:
            pct = (self.token_percent or 0) * 100
            parts.append(f"Tokens: {self.used_tokens:,} / {self.limit_tokens:,} ({pct:.1f}%)")
        if self.limit_usd:
            pct = (self.usd_percent or 0) * 100
            parts.append(f"Cost: ${self.used_usd:.4f} / ${self.limit_usd:.2f} ({pct:.1f}%)")
        return " | ".join(parts) if parts else f"Used {self.used_tokens:,} tokens (${self.used_usd:.4f})"


class BudgetExceededError(Exception):
    """Raised when token/cost budget is exceeded."""

    def __init__(self, usage: BudgetUsage):
        self.usage = usage
        super().__init__(f"Budget exceeded: {usage.summary()}")


# ---------------------------------------------------------------------------
# BudgetGuard
# ---------------------------------------------------------------------------

class BudgetGuard:
    """
    Track cumulative token/cost usage and enforce limits.

    Args:
        session_limit_tokens: Max tokens per session (optional).
        session_limit_usd: Max cost in USD per session (optional).
        warn_at_percent: Fraction [0–1] at which to trigger warning callback (default 0.8).
        on_warning: Callable called with BudgetUsage when warn threshold is crossed.
            Default: prints a yellow warning to stdout.
        on_exceed: Callable called with BudgetUsage when budget is exceeded.
            Default: raises BudgetExceededError.
        pricing: Optional PricingTable instance.
    """

    DEFAULT_SESSION_TOKENS = 50_000
    DEFAULT_SESSION_USD = 1.0

    def __init__(
        self,
        session_limit_tokens: Optional[int] = DEFAULT_SESSION_TOKENS,
        session_limit_usd: Optional[float] = DEFAULT_SESSION_USD,
        warn_at_percent: float = 0.8,
        on_warning: Optional[Callable[[BudgetUsage], None]] = None,
        on_exceed: Optional[Callable[[BudgetUsage], None]] = None,
        pricing: Optional[PricingTable] = None,
    ):
        self._limit_tokens = session_limit_tokens
        self._limit_usd = session_limit_usd
        self._warn_at = warn_at_percent
        self._on_warning = on_warning or self._default_warning
        self._on_exceed = on_exceed or self._default_exceed
        self._pricing = pricing or PricingTable()

        self._used_tokens: int = 0
        self._used_usd: float = 0.0
        self._warned: bool = False  # only warn once until reset

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        input_tokens: int,
        output_tokens: int,
        provider: str,
        model: str,
    ) -> BudgetUsage:
        """Record actual usage after an API call."""
        price = self._pricing.get(provider, model)
        cost = price.cost_usd(input_tokens, output_tokens) if price else 0.0
        self._used_tokens += input_tokens + output_tokens
        self._used_usd += cost
        return self._evaluate()

    def check(
        self,
        estimated_input_tokens: int,
        provider: str,
        model: str,
        estimated_output_tokens: int = 0,
    ) -> BudgetUsage:
        """
        Check if a projected request would breach the budget.
        Does NOT record — call record() after the actual API call.
        Raises BudgetExceededError if already over budget (ignores estimate).
        """
        usage = self._build_usage()
        if usage.is_exceeded:
            self._on_exceed(usage)
        return usage

    def reset(self) -> None:
        """Reset session counters."""
        self._used_tokens = 0
        self._used_usd = 0.0
        self._warned = False

    @property
    def usage(self) -> BudgetUsage:
        return self._build_usage()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate(self) -> BudgetUsage:
        usage = self._build_usage()
        if usage.is_exceeded:
            self._on_exceed(usage)
        elif usage.is_warning and not self._warned:
            self._warned = True
            self._on_warning(usage)
        return usage

    def _build_usage(self) -> BudgetUsage:
        return BudgetUsage(
            used_tokens=self._used_tokens,
            used_usd=self._used_usd,
            limit_tokens=self._limit_tokens,
            limit_usd=self._limit_usd,
            warn_at_percent=self._warn_at,
        )

    # ------------------------------------------------------------------
    # Default callbacks
    # ------------------------------------------------------------------

    @staticmethod
    def _default_warning(usage: BudgetUsage) -> None:
        try:
            from rich.console import Console
            Console().print(
                f"[bold yellow]⚠️  TokenSaver Budget Warning:[/bold yellow] {usage.summary()}"
            )
        except ImportError:
            print(f"⚠️  TokenSaver Budget Warning: {usage.summary()}")

    @staticmethod
    def _default_exceed(usage: BudgetUsage) -> None:
        raise BudgetExceededError(usage)
