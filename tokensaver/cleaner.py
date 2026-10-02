"""
HTMLCleaner — Convert HTML pages/strings to clean Markdown.

Reduces token usage by 70–90% by stripping navigation, scripts,
ads, and other boilerplate before sending content to an AI.

Example:
    from tokensaver import HTMLCleaner

    cleaner = HTMLCleaner()

    # From string
    md = cleaner.clean("<html>...</html>")

    # From URL
    md = cleaner.clean_url("https://docs.example.com/api")
    print(md)
"""
from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# HTMLCleaner
# ---------------------------------------------------------------------------

# Tags to completely remove (including their content)
_REMOVE_TAGS = [
    "script", "style", "noscript", "nav", "footer", "header",
    "aside", "advertisement", "iframe", "form", "button",
    "svg", "figure", "figcaption", "picture", "meta", "link",
]

# Tags to strip (remove tag but keep inner text)
_UNWRAP_TAGS = [
    "span", "div", "section", "article", "main", "wrapper",
    "container",
]


class HTMLCleaner:
    """
    Convert HTML to clean Markdown for lower token usage.

    Args:
        remove_tags: Additional tags to remove.
        keep_images: If True, convert <img> to ![alt](src). Default False.
        keep_links: If True, preserve hyperlinks. Default True.
        timeout: HTTP request timeout in seconds for clean_url().
    """

    def __init__(
        self,
        remove_tags: Optional[list[str]] = None,
        keep_images: bool = False,
        keep_links: bool = True,
        timeout: int = 15,
    ):
        self._extra_remove = remove_tags or []
        self._keep_images = keep_images
        self._keep_links = keep_links
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, html: str, base_url: str = "") -> str:
        """Convert an HTML string to clean Markdown."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("beautifulsoup4 is required: pip install beautifulsoup4")

        soup = BeautifulSoup(html, "html.parser")

        # Remove unwanted tags entirely
        for tag in _REMOVE_TAGS + self._extra_remove:
            for el in soup.find_all(tag):
                el.decompose()

        # Remove elements hidden via display:none or visibility:hidden
        for el in soup.find_all(style=True):
            style = el.get("style", "")
            if "display:none" in style.replace(" ", "") or "visibility:hidden" in style.replace(" ", ""):
                el.decompose()

        # Extract title
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        # Convert to markdown
        body = soup.find("body") or soup
        md = self._to_markdown(body)

        # Prepend title if found
        if title:
            md = f"# {title}\n\n{md}"

        # Post-process: clean up excessive blank lines
        md = re.sub(r"\n{3,}", "\n\n", md)
        return md.strip()

    def clean_url(self, url: str) -> str:
        """Fetch a URL and convert its content to clean Markdown."""
        try:
            import requests
        except ImportError:
            raise ImportError("requests is required: pip install requests")

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; TokenSaver/0.1; "
                "+https://github.com/tokensaver)"
            )
        }
        resp = requests.get(url, headers=headers, timeout=self._timeout)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "html" in content_type:
            return self.clean(resp.text, base_url=url)
        # Plain text — return as-is (already clean)
        return resp.text

    def token_reduction_estimate(self, html: str) -> dict:
        """Estimate token reduction for reporting purposes."""
        md = self.clean(html)
        orig_chars = len(html)
        clean_chars = len(md)
        reduction = 1 - (clean_chars / orig_chars) if orig_chars else 0
        return {
            "original_chars": orig_chars,
            "cleaned_chars": clean_chars,
            "estimated_reduction_percent": round(reduction * 100, 1),
        }

    # ------------------------------------------------------------------
    # Markdown conversion helpers
    # ------------------------------------------------------------------

    def _to_markdown(self, soup) -> str:
        """Convert BeautifulSoup element to Markdown string."""
        try:
            from markdownify import markdownify as md
        except ImportError:
            raise ImportError("markdownify is required: pip install markdownify")

        options = {
            "heading_style": "ATX",
            "bullets": "-",
            "convert": None,
        }

        # Build strip/convert lists based on options
        strip = []
        if not self._keep_images:
            strip.append("img")
        if not self._keep_links:
            strip.append("a")

        html_str = str(soup)
        result = md(html_str, **{k: v for k, v in options.items() if v is not None},
                    strip=strip if strip else None)

        return result
