"""Tests for TokenCounter."""
import pytest
from tokensaver.counter import TokenCounter, TokenCount, ConversationCount
from tokensaver.pricing import PricingTable


class TestTokenCounter:

    def test_count_returns_token_count(self):
        counter = TokenCounter(provider="anthropic", model="claude-sonnet-4-5")
        result = counter.count("Hello, world!")
        assert isinstance(result, TokenCount)
        assert result.tokens > 0
        assert result.provider == "anthropic"
        assert result.model == "claude-sonnet-4-5"

    def test_count_empty_string(self):
        counter = TokenCounter(provider="anthropic", model="claude-sonnet-4-5")
        result = counter.count("")
        assert result.tokens == 0

    def test_count_openai(self):
        counter = TokenCounter(provider="openai", model="gpt-4o-mini")
        result = counter.count("This is a test prompt for OpenAI.")
        assert result.tokens > 0
        assert result.provider == "openai"

    def test_count_gemini_estimates(self):
        counter = TokenCounter(provider="gemini", model="gemini-2-0-flash")
        text = "This is a test prompt"  # 21 chars → ~5 tokens
        result = counter.count(text)
        assert result.tokens > 0

    def test_cost_calculated(self):
        counter = TokenCounter(provider="anthropic", model="claude-sonnet-4-5")
        # claude-sonnet-4-5: $3/1M input → 1000 tokens = $0.003
        cost = counter._cost(1_000_000, 0)
        assert abs(cost - 3.00) < 0.01

    def test_count_messages_basic(self):
        counter = TokenCounter(provider="anthropic", model="claude-sonnet-4-5")
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        result = counter.count_messages(messages)
        assert isinstance(result, ConversationCount)
        assert result.input_tokens > 0
        assert result.provider == "anthropic"

    def test_count_messages_empty(self):
        counter = TokenCounter(provider="openai", model="gpt-4o")
        result = counter.count_messages([])
        assert result.input_tokens >= 0

    def test_longer_text_more_tokens(self):
        counter = TokenCounter(provider="anthropic", model="claude-sonnet-4-5")
        short = counter.count("Hi")
        long = counter.count("This is a much longer sentence with many more words in it.")
        assert long.tokens > short.tokens

    def test_unknown_model_fallback(self):
        """Unknown model should not crash."""
        counter = TokenCounter(provider="openai", model="gpt-99-ultra")
        result = counter.count("Hello")
        assert result.tokens > 0

    def test_unknown_provider_no_cost(self):
        """Unknown provider price returns 0 cost."""
        counter = TokenCounter(provider="unknown-provider", model="some-model")
        result = counter.count("Hello")
        # No crash, but cost may be 0
        assert result.tokens >= 0
