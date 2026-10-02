"""Tests for BudgetGuard."""
import pytest
from tokensaver.budget import BudgetGuard, BudgetExceededError, BudgetUsage


class TestBudgetGuard:

    def test_no_alert_under_threshold(self):
        warnings = []
        guard = BudgetGuard(
            session_limit_tokens=10_000,
            session_limit_usd=None,
            warn_at_percent=0.8,
            on_warning=lambda u: warnings.append(u),
            on_exceed=lambda u: (_ for _ in ()).throw(BudgetExceededError(u)),
        )
        guard.record(1000, 500, "anthropic", "claude-sonnet-4-5")
        assert len(warnings) == 0

    def test_warning_at_80_percent_tokens(self):
        warnings = []
        guard = BudgetGuard(
            session_limit_tokens=10_000,
            session_limit_usd=None,
            warn_at_percent=0.8,
            on_warning=lambda u: warnings.append(u),
            on_exceed=lambda u: None,
        )
        guard.record(8_001, 0, "anthropic", "claude-haiku-3-5")
        assert len(warnings) == 1
        assert isinstance(warnings[0], BudgetUsage)

    def test_exceed_raises_default(self):
        guard = BudgetGuard(
            session_limit_tokens=1_000,
            session_limit_usd=None,
            warn_at_percent=0.8,
        )
        with pytest.raises(BudgetExceededError):
            guard.record(1_001, 0, "anthropic", "claude-sonnet-4-5")

    def test_reset_clears_usage(self):
        warnings = []
        guard = BudgetGuard(
            session_limit_tokens=10_000,
            session_limit_usd=None,
            on_warning=lambda u: warnings.append(u),
            on_exceed=lambda u: None,
        )
        guard.record(9_000, 0, "openai", "gpt-4o-mini")
        assert len(warnings) == 1
        guard.reset()
        assert guard.usage.used_tokens == 0
        assert guard._warned is False

    def test_usd_limit(self):
        exceeded = []
        guard = BudgetGuard(
            session_limit_tokens=None,
            session_limit_usd=0.001,  # very small limit
            warn_at_percent=0.8,
            on_warning=lambda u: None,
            on_exceed=lambda u: exceeded.append(u),
        )
        # $3/1M input → 1000 tokens = $0.003 > $0.001
        guard.record(1_000, 0, "anthropic", "claude-sonnet-4-5")
        assert len(exceeded) == 1

    def test_usage_percent(self):
        guard = BudgetGuard(
            session_limit_tokens=10_000,
            session_limit_usd=None,
            on_warning=lambda u: None,
            on_exceed=lambda u: None,
        )
        guard.record(5_000, 0, "openai", "gpt-4o-mini")
        usage = guard.usage
        assert abs(usage.token_percent - 0.5) < 0.01

    def test_budget_usage_summary(self):
        usage = BudgetUsage(
            used_tokens=5000,
            used_usd=0.05,
            limit_tokens=10000,
            limit_usd=1.0,
        )
        summary = usage.summary()
        assert "5,000" in summary
        assert "10,000" in summary

    def test_no_limits_no_percent(self):
        guard = BudgetGuard(
            session_limit_tokens=None,
            session_limit_usd=None,
            on_warning=lambda u: None,
            on_exceed=lambda u: None,
        )
        guard.record(99999, 0, "openai", "gpt-4o")
        assert guard.usage.percent is None
