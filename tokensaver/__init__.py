"""
TokenSaver — Token optimization toolkit for AI projects.

Usage as library:
    from tokensaver import TokenCounter, BudgetGuard, PromptCompressor, HTMLCleaner

Usage as CLI:
    tokensaver count prompt.txt --provider anthropic --model claude-sonnet-4-5
    tokensaver report --today
"""

from tokensaver.counter import TokenCounter, TokenCount
from tokensaver.budget import BudgetGuard, BudgetExceededError, BudgetUsage
from tokensaver.compressor import PromptCompressor, CompressionResult
from tokensaver.cleaner import HTMLCleaner
from tokensaver.logger import SessionLogger
from tokensaver.reporter import Reporter
from tokensaver.pricing import PricingTable

__version__ = "0.1.0"
__all__ = [
    "TokenCounter",
    "TokenCount",
    "BudgetGuard",
    "BudgetExceededError",
    "BudgetUsage",
    "PromptCompressor",
    "CompressionResult",
    "HTMLCleaner",
    "SessionLogger",
    "Reporter",
    "PricingTable",
]
