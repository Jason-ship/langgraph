"""Usage tracking — token counting, cost estimation, and per-chapter accumulator.

v5.4-fix: 从 threading.local() 改为进程级 dict + Lock，
解决 LangGraph 多线程/异步执行下 token 统计为 0 的问题。
"""

from __future__ import annotations

import threading
from typing import Any

from novelfactory.config.constants import (
    DEEPSEEK_COMPLETION_TOKEN_RATE,
    DEEPSEEK_PROMPT_TOKEN_RATE,
    TOKENS_PER_MILLION,
)

# DeepSeek Flash pricing — 唯一来源为 config.constants，此处保留别名
# 供 _retry_common.py 等模块的既有导入使用（向后兼容）。
_TOKENS_PER_MILLION = TOKENS_PER_MILLION
_DEEPSEEK_PROMPT_RATE = DEEPSEEK_PROMPT_TOKEN_RATE
_DEEPSEEK_COMPLETION_RATE = DEEPSEEK_COMPLETION_TOKEN_RATE

# 进程级共享存储（替代 threading.local，跨线程/协程可见）
_usage_data: dict = {}
_usage_lock = threading.Lock()

_pricing_module: Any = None


def _get_pricing() -> Any:
    """Lazy-load pricing module (optional dependency)."""
    global _pricing_module
    if _pricing_module is None:
        try:
            from novelfactory.config import (
                pricing as _pricing_module,
            )
        except ImportError:
            _pricing_module = None
    return _pricing_module


def _record_usage(
    step_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    model: str = "deepseek-v4-flash",
) -> None:
    """Append one LLM call's token usage to the shared accumulator."""
    p = int(prompt_tokens or 0)
    c = int(completion_tokens or 0)
    with _usage_lock:
        if "calls" not in _usage_data:
            _usage_data["calls"] = []
            _usage_data["prompt_tokens"] = 0
            _usage_data["completion_tokens"] = 0
            _usage_data["model_totals"] = {}
        _usage_data["calls"].append(
            {
                "step": step_name,
                "prompt_tokens": p,
                "completion_tokens": c,
                "model": model,
            }
        )
        _usage_data["prompt_tokens"] += p
        _usage_data["completion_tokens"] += c
        m = _usage_data["model_totals"].setdefault(
            model, {"prompt_tokens": 0, "completion_tokens": 0}
        )
        m["prompt_tokens"] += p
        m["completion_tokens"] += c


def reset_usage_tracking() -> None:
    """Clear the shared usage accumulator."""
    with _usage_lock:
        _usage_data.clear()
        _usage_data["calls"] = []
        _usage_data["prompt_tokens"] = 0
        _usage_data["completion_tokens"] = 0
        _usage_data["model_totals"] = {}


def read_usage_tracking() -> dict:
    """Return a snapshot of accumulated usage from the shared accumulator."""
    pricing = _get_pricing()
    with _usage_lock:
        if "calls" not in _usage_data:
            return {
                "calls": [],
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_cny": 0.0,
                "model_breakdown": {},
            }
        p = _usage_data["prompt_tokens"]
        c = _usage_data["completion_tokens"]
        calls_snapshot = list(_usage_data.get("calls", []))
        model_totals: dict = _usage_data.get("model_totals", {})
        # Deep copy model_totals to avoid mutation during iteration
        model_totals_copy = {
            m: {"prompt_tokens": d["prompt_tokens"],
                "completion_tokens": d["completion_tokens"]}
            for m, d in model_totals.items()
        }
    total_cost = 0.0
    model_breakdown = {}
    for model, mdata in model_totals_copy.items():
        mp = mdata["prompt_tokens"]
        mc = mdata["completion_tokens"]
        if pricing:
            cost = pricing.calc_cost(mp, mc, model)
        else:
            # Fallback: DeepSeek Flash rate
            cost = (mp / _TOKENS_PER_MILLION) * _DEEPSEEK_PROMPT_RATE + (
                mc / _TOKENS_PER_MILLION
            ) * _DEEPSEEK_COMPLETION_RATE
        model_breakdown[model] = {
            "prompt_tokens": mp,
            "completion_tokens": mc,
            "total_tokens": mp + mc,
            "estimated_cost_cny": round(cost, 4),
        }
        total_cost += cost
    return {
        "calls": calls_snapshot,
        "prompt_tokens": p,
        "completion_tokens": c,
        "total_tokens": p + c,
        "estimated_cost_cny": round(total_cost, 4),
        "model_breakdown": model_breakdown,
    }


def _extract_tokens_from_dict(u: dict) -> tuple[int, int]:
    """Extract (prompt_tokens, completion_tokens) from a usage dict.

    Supports both key naming conventions:
      - prompt_tokens / completion_tokens (OpenAI-style)
      - input_tokens / output_tokens (Anthropic-style)
    """
    prompt = int(u.get("prompt_tokens", u.get("input_tokens", 0)) or 0)
    completion = int(u.get("completion_tokens", u.get("output_tokens", 0)) or 0)
    return prompt, completion


def _try_extract_usage(obj: Any) -> dict | None:
    """Try to extract a usage dict from LangChain result object using common access patterns."""
    if not obj:
        return None
    # Patterns 1-2: check object attributes for usage info
    for attr_name in ("response_metadata", "usage_metadata"):
        meta = getattr(obj, attr_name, None)
        if meta is None:
            continue
        if attr_name == "response_metadata":
            u = meta.get("usage") or meta.get("token_usage")
        else:
            u = meta
        if isinstance(u, dict):
            return u
    # Pattern 3: dict with usage or usage_metadata keys
    if isinstance(obj, dict):
        u = obj.get("usage") or obj.get("usage_metadata")
        if isinstance(u, dict):
            return u
    # Pattern 4: messages list — check last message's usage
    for msg in reversed(getattr(obj, "messages", None) or []):
        u = getattr(msg, "usage_metadata", None) or {}
        if isinstance(u, dict):
            return u
        meta = getattr(msg, "response_metadata", None) or {}
        u = meta.get("usage")
        if isinstance(u, dict):
            return u
    return None


def _extract_usage_from_result(result: Any) -> tuple[int, int]:
    """Pull (prompt_tokens, completion_tokens) from a LangChain result."""
    usage = _try_extract_usage(result)
    if usage is not None:
        return _extract_tokens_from_dict(usage)
    return 0, 0


# ── Token Counting ────────────────────────────────────────────────────────────

_tiktoken_encoder: Any = None
_TIKTOKEN_LOAD_ERROR: str | None = None


def _get_tiktoken_encoder() -> tuple[Any, str | None]:
    global _tiktoken_encoder, _TIKTOKEN_LOAD_ERROR
    if _tiktoken_encoder is not None or _TIKTOKEN_LOAD_ERROR:
        return _tiktoken_encoder, _TIKTOKEN_LOAD_ERROR
    try:
        import tiktoken

        _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
        _TIKTOKEN_LOAD_ERROR = None
        return _tiktoken_encoder, None
    except ImportError:
        _TIKTOKEN_LOAD_ERROR = "tiktoken not installed (pip install tiktoken)"
        _tiktoken_encoder = None
        return None, _TIKTOKEN_LOAD_ERROR
    except (ValueError, OSError) as e:
        _TIKTOKEN_LOAD_ERROR = f"tiktoken load failed: {e}"
        _tiktoken_encoder = None
        return None, _TIKTOKEN_LOAD_ERROR


def count_tokens(text: str) -> int:
    """Accurate token count using tiktoken. Falls back to regex heuristic."""
    import re as _re

    if not text:
        return 0

    encoder, err = _get_tiktoken_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text, disallowed_special=()))
        except (ValueError, TypeError):
            pass

    cjk = len(_re.findall(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", text))
    english = len(_re.findall(r"[a-zA-Z]+", text))
    other = len(text) - cjk - english
    return int(cjk * 2 + english * 1.3 + other * 0.3)


def count_tokens_dict(data: dict) -> int:
    """Count total tokens across all string values in a dict."""
    total = 0
    for v in data.values():
        if isinstance(v, str):
            total += count_tokens(v)
        elif isinstance(v, dict):
            total += count_tokens_dict(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    total += count_tokens(item)
    return total
