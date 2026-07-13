"""Quota configuration for NovelFactory token budget enforcement.

Controls how aggressively the system reacts to low quota conditions.

Settings:
  QUOTA_CHECK_BEFORE_CALL — whether to check budget before each LLM call (default: False)
  QUOTA_CHECK_INTERVAL_SECONDS — minimum interval between two quota checks (default: 60s)
  QUOTA_WARN_THRESHOLD_PCT — warn when quota drops below this % (default: 20%)
  QUOTA_BLOCK_THRESHOLD_PCT — block LLM calls when quota drops below this % (default: 5%)
  QUOTA_TOKEN_BUDGET — per-run token budget; 0 = disabled, rely on API-level billing
  QUOTA_EST_MAX_TOKENS_PER_CALL — estimated max tokens per LLM call (default: 50_000)
  QUOTA_CHAPTER_COST_WARN_CNY — warn when estimated chapter cost exceeds this value in CNY

TODO: The billing backend is currently unavailable — DeepSeek has no public billing API.
      When a billing API becomes available, wire it into agents/infra/quota.py.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class QuotaSettings(BaseSettings):
    """Quota enforcement settings.

    The token budget check uses the accumulated usage tracked by
    agents/infra/usage.py (read_usage_tracking). Set QUOTA_TOKEN_BUDGET > 0
    and QUOTA_CHECK_BEFORE_CALL = True to enforce a per-run cap.
    """

    # ── Quota check behaviour ──────────────────────────────────────────────────
    QUOTA_CHECK_BEFORE_CALL: bool = Field(
        default=False,
        description="If True, check token budget before every LLM call and block if exhausted.",
    )
    QUOTA_CHECK_INTERVAL_SECONDS: float = Field(
        default=60.0,
        description="Minimum seconds between consecutive quota budget checks.",
    )
    QUOTA_WARN_THRESHOLD_PCT: float = Field(
        default=20.0,
        description="Emit a warning log when remaining budget < threshold %%.",
    )
    QUOTA_BLOCK_THRESHOLD_PCT: float = Field(
        default=5.0,
        description="Block LLM calls when remaining budget < threshold %% (QUOTA_CHECK_BEFORE_CALL=True).",
    )

    # ── Budget ─────────────────────────────────────────────────────────────────
    QUOTA_TOKEN_BUDGET: int = Field(
        default=0,
        description="Per-run token budget (0 = disabled, rely on API-level billing).",
    )

    # ── Conservative estimates ─────────────────────────────────────────────────
    QUOTA_EST_MAX_TOKENS_PER_CALL: int = Field(
        default=50_000,
        description="Estimated max tokens consumed by one LLM call.",
    )
    QUOTA_CHAPTER_COST_WARN_CNY: float = Field(
        default=0.50,
        description="Warn when estimated chapter cost exceeds this value (CNY).",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }


quota_settings = QuotaSettings()
