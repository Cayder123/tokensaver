"""Tests for HTMLCleaner."""
import pytest
from tokensaver.cleaner import HTMLCleaner


SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Test Page</title>
  <style>body { color: red; }</style>
  <script>alert('hello');</script>
</head>
<body>
  <nav><a href="/">Home</a><a href="/about">About</a></nav>
  <main>
    <h1>Main Heading</h1>
    <p>This is the main content paragraph with useful information.</p>
    <h2>Section Two</h2>
    <ul>
      <li>Item one</li>
      <li>Item two</li>
    </ul>
    <code>print("hello")</code>
  </main>
  <footer>Copyright 2026</footer>
</body>
</html>
"""


class TestHTMLCleaner:

    def test_returns_markdown_string(self):
        cleaner = HTMLCleaner()
        result = cleaner.clean(SAMPLE_HTML)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_removes_scripts(self):
        cleaner = HTMLCleaner()
        result = cleaner.clean(SAMPLE_HTML)
        assert "alert" not in result
        assert "<script>" not in result

    def test_removes_styles(self):
        cleaner = HTMLCleaner()
        result = cleaner.clean(SAMPLE_HTML)
        assert "color: red" not in result
        assert "<style>" not in result

    def test_removes_nav(self):
        cleaner = HTMLCleaner()
        result = cleaner.clean(SAMPLE_HTML)
        assert "<nav>" not in result

    def test_removes_footer(self):
        cleaner = HTMLCleaner()
        result = cleaner.clean(SAMPLE_HTML)
        assert "<footer>" not in result

    def test_keeps_main_content(self):
        cleaner = HTMLCleaner()
        result = cleaner.clean(SAMPLE_HTML)
        assert "Main Heading" in result
        assert "useful information" in result

    def test_includes_title(self):
        cleaner = HTMLCleaner()
        result = cleaner.clean(SAMPLE_HTML)
        assert "Test Page" in result

    def test_result_is_shorter_than_input(self):
        cleaner = HTMLCleaner()
        result = cleaner.clean(SAMPLE_HTML)
        # Cleaned markdown should be significantly shorter
        assert len(result) < len(SAMPLE_HTML)

    def test_token_reduction_estimate(self):
        cleaner = HTMLCleaner()
        info = cleaner.token_reduction_estimate(SAMPLE_HTML)
        assert "original_chars" in info
        assert "cleaned_chars" in info
        assert "estimated_reduction_percent" in info
        assert info["estimated_reduction_percent"] > 0

    def test_empty_html(self):
        cleaner = HTMLCleaner()
        result = cleaner.clean("")
        assert isinstance(result, str)

    def test_plain_text_passthrough(self):
        cleaner = HTMLCleaner()
        result = cleaner.clean("<p>Hello world</p>")
        assert "Hello world" in result
