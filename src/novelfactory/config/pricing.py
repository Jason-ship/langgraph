"""DeepSeek model pricing for NovelFactory token usage tracking.

Pricing is per 1,000,000 tokens (1M tokens). The prices below reflect
DeepSeek Flash (deepseek-chat) rates as of 2026-06-14. Override via
env vars or register_model() to match your actual billing.

Pricing structure per model:
    input_cost_per_mtok  — prompt tokens
    output_cost_per_mtok — completion tokens

DeepSeek Flash (deepseek-chat):
    Input:  ~0.5 CNY / 1M tokens
    Output: ~2.0 CNY / 1M tokens

Tech stack: DeepSeek Flash (replaced MiniMax 2026-06-14).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ModelPricing:
    """Pricing for a single model in CNY per 1M tokens."""

    input_cost_per_mtok: float
    output_cost_per_mtok: float

    def calc_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost in CNY for given token counts."""
        return (prompt_tokens / 1_000_000.0) * self.input_cost_per_mtok + (
            completion_tokens / 1_000_000.0
        ) * self.output_cost_per_mtok


# ── Registry (module-level; populated at import) ───────────────────────────

_MODEL_REGISTRY: dict[str, ModelPricing] = {}
_DEFAULT_PRICING: ModelPricing = ModelPricing(
    input_cost_per_mtok=0.5,
    output_cost_per_mtok=2.0,
)


def _init_registry() -> None:
    """Populate _MODEL_REGISTRY with default rates + env overrides.

    Env var format (env values override the defaults baked in here):
        DEEPSEEK_PRICE_INPUT   = CNY per 1M input tokens
        DEEPSEEK_PRICE_OUTPUT  = CNY per 1M output tokens
    """
    global _DEFAULT_PRICING

    defaults = {
        "deepseek-chat": ModelPricing(
            input_cost_per_mtok=0.5,
            output_cost_per_mtok=2.0,
        ),
    }
    _MODEL_REGISTRY.update(defaults)
    _DEFAULT_PRICING = defaults["deepseek-chat"]

    # v6.1: 统一从 settings 读取
    from novelfactory.config.settings import settings as _st_price

    env_in = os.environ.get("DEEPSEEK_PRICE_INPUT") or (
        str(_st_price.DEEPSEEK_PRICE_INPUT) if _st_price.DEEPSEEK_PRICE_INPUT else None
    )
    env_out = os.environ.get("DEEPSEEK_PRICE_OUTPUT") or (
        str(_st_price.DEEPSEEK_PRICE_OUTPUT)
        if _st_price.DEEPSEEK_PRICE_OUTPUT
        else None
    )
    if env_in or env_out:
        base = _MODEL_REGISTRY.get("deepseek-chat", _DEFAULT_PRICING)
        _MODEL_REGISTRY["deepseek-chat"] = ModelPricing(
            input_cost_per_mtok=float(env_in) if env_in else base.input_cost_per_mtok,
            output_cost_per_mtok=float(env_out)
            if env_out
            else base.output_cost_per_mtok,
        )

    env_def_in = os.environ.get("DEEPSEEK_PRICE_DEFAULT_INPUT") or (
        str(_st_price.DEEPSEEK_PRICE_DEFAULT_INPUT)
        if _st_price.DEEPSEEK_PRICE_DEFAULT_INPUT
        else None
    )
    env_def_out = os.environ.get("DEEPSEEK_PRICE_DEFAULT_OUTPUT") or (
        str(_st_price.DEEPSEEK_PRICE_DEFAULT_OUTPUT)
        if _st_price.DEEPSEEK_PRICE_DEFAULT_OUTPUT
        else None
    )
    if env_def_in or env_def_out:
        _DEFAULT_PRICING = ModelPricing(
            input_cost_per_mtok=float(env_def_in)
            if env_def_in
            else _DEFAULT_PRICING.input_cost_per_mtok,
            output_cost_per_mtok=float(env_def_out)
            if env_def_out
            else _DEFAULT_PRICING.output_cost_per_mtok,
        )


# Run registry init on import.
_init_registry()


def get_model_pricing(model: str) -> ModelPricing:
    """Return pricing for a model, falling back to default."""
    return _MODEL_REGISTRY.get(model, _DEFAULT_PRICING)


def calc_cost(prompt_tokens: int, completion_tokens: int, model: str = "") -> float:
    """Calculate cost in CNY using the model's pricing.

    Args:
        prompt_tokens: Number of prompt tokens.
        completion_tokens: Number of completion tokens.
        model: Model name (e.g. "deepseek-chat"). If empty, uses the default.
    """
    if model:
        return get_model_pricing(model).calc_cost(prompt_tokens, completion_tokens)
    return _DEFAULT_PRICING.calc_cost(prompt_tokens, completion_tokens)


def register_model(model: str, pricing: ModelPricing) -> None:
    """Register or override pricing for a model at runtime."""
    _MODEL_REGISTRY[model] = pricing


# ── Per-model display names ─────────────────────────────────────────────────

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "deepseek-chat": "DeepSeek Flash (deepseek-chat)",
}
