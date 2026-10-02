"""Tests for PromptCompressor."""
import pytest
from tokensaver.compressor import PromptCompressor, CompressionResult


class TestPromptCompressor:

    def test_simple_compress_returns_result(self):
        c = PromptCompressor(mode="simple")
        result = c.compress("Hello world. This is a test.")
        assert isinstance(result, CompressionResult)
        assert result.original_text == "Hello world. This is a test."
        assert result.compressed_text  # not empty
        assert result.mode == "simple"

    def test_removes_excess_blank_lines(self):
        c = PromptCompressor(mode="simple")
        text = "Line one.\n\n\n\n\nLine two."
        result = c.compress(text)
        assert "\n\n\n" not in result.compressed_text

    def test_removes_repeated_dots(self):
        c = PromptCompressor(mode="simple")
        text = "Wait for it....... okay"
        result = c.compress(text)
        assert "......" not in result.compressed_text
        assert "..." in result.compressed_text

    def test_removes_pure_comment_lines(self):
        c = PromptCompressor(mode="simple")
        text = "def foo():\n    # This is a comment\n    return 42"
        result = c.compress(text)
        assert "# This is a comment" not in result.compressed_text
        assert "return 42" in result.compressed_text

    def test_keeps_shebang(self):
        c = PromptCompressor(mode="simple")
        text = "#!/usr/bin/env python\nprint('hello')"
        result = c.compress(text)
        assert "#!/usr/bin/env python" in result.compressed_text

    def test_removes_filler_phrases(self):
        c = PromptCompressor(mode="simple")
        text = "Please note that the answer is 42. Of course, you knew that."
        result = c.compress(text)
        assert "please note that" not in result.compressed_text.lower()
        assert "of course" not in result.compressed_text.lower()

    def test_ratio_property(self):
        c = PromptCompressor(mode="simple")
        # A text with lots of comments and whitespace
        text = (
            "# Comment 1\n# Comment 2\n# Comment 3\n" * 20
            + "actual_code = True\n"
        )
        result = c.compress(text)
        # Should save at least something
        assert result.ratio <= 1.0
        assert result.saved_tokens >= 0

    def test_compress_messages(self):
        c = PromptCompressor(mode="simple")
        messages = [
            {"role": "system", "content": "Please note that you are a helpful assistant."},
            {"role": "user", "content": "# Comment\nWhat is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        compressed_msgs, result = c.compress_messages(messages, compress_roles=["user", "system"])
        assert isinstance(result, CompressionResult)
        # assistant content should be unchanged
        assert compressed_msgs[2]["content"] == "4"

    def test_empty_string(self):
        c = PromptCompressor(mode="simple")
        result = c.compress("")
        assert result.original_tokens == 0
        assert result.compressed_tokens == 0

    def test_summary_string(self):
        c = PromptCompressor(mode="simple")
        result = c.compress("Hello " * 100)
        summary = result.summary()
        assert "simple" in summary
        assert "→" in summary
