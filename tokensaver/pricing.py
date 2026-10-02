"""
Pricing table for AI providers.
Loads from pricing.yaml bundled with the package.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelPrice:
    input_per_1m: float   # USD per 1M input tokens
    output_per_1m: float  # USD per 1M output tokens

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_1m / 1_000_000) + (
            output_tokens * self.output_per_1m / 1_000_000
        )


class PricingTable:
    """
    Load and query provider pricing. Falls back to bundled pricing.yaml.
    Can also load a user-supplied YAML for custom prices.

    Example:
        pt = PricingTable()
        price = pt.get("anthropic", "claude-sonnet-4-5")
        cost = price.cost_usd(input_tokens=1000, output_tokens=300)
    """

    _BUNDLED = Path(__file__).parent / "pricing.yaml"

    def __init__(self, custom_path: Optional[str | Path] = None):
        path = Path(custom_path) if custom_path else self._BUNDLED
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self._table: dict[str, dict[str, ModelPrice]] = {}
        for provider, models in data.get("providers", {}).items():
            self._table[provider.lower()] = {}
            for model, prices in models.items():
                self._table[provider.lower()][model.lower()] = ModelPrice(
                    input_per_1m=prices["input_per_1m"],
                    output_per_1m=prices["output_per_1m"],
                )

    def get(self, provider: str, model: str) -> Optional[ModelPrice]:
        """Return ModelPrice or None if not found.
        
        Normalizes model name: dots are treated the same as hyphens,
        so 'gemini-3.8-flash' matches 'gemini-3-8-flash' in pricing.yaml.
        """
        p = self._table.get(provider.lower(), {})
        # Normalize: replace dots with hyphens for consistent lookup
        model_norm = model.lower().replace(".", "-")
        # Try exact match first (normalized)
        if model_norm in p:
            return p[model_norm]
        # Also try raw model name
        if model.lower() in p:
            return p[model.lower()]
        # Prefix / suffix match (for versioned model names)
        for key in p:
            key_norm = key.replace(".", "-")
            if model_norm.startswith(key_norm) or key_norm.startswith(model_norm):
                return p[key]
        return None

    def list_providers(self) -> list[str]:
        return list(self._table.keys())

    def list_models(self, provider: str) -> list[str]:
        return list(self._table.get(provider.lower(), {}).keys())
