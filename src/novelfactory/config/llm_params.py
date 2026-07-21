"""集中化 LLM 参数调优中心 (v7.0)。

设计原则：
  1. 参数分层 — 全局默认值 → Tier 默认值 → Agent 覆盖 → 运行时覆盖
  2. 一处修改，全局生效 — 所有 LLM 调用从本中心读取参数
  3. 运行时可调 — 通过 RunnableConfig / env 变量实时调优
  4. 完整审计 — 所有参数变更可追踪

参数层级（优先级从低到高）：
  Layer 0: 代码硬编码默认值 (LLMParams.__init__)
  Layer 1: 全局环境变量覆盖 (NOVELFACTORY_LLM_*)
  Layer 2: Tier 注册值 (register_tier)
  Layer 3: Agent 注册值 (register_agent)
  Layer 4: 运行时覆盖 (get_params override 参数)

使用示例：
  >>> from novelfactory.config.llm_params import center
  >>> # 获取写作 tier 参数
  >>> params = center.get_params("worker")
  >>> params.temperature
  0.7
  >>> # 获取带运行时覆盖的参数
  >>> params = center.get_params("worker", temperature=0.9)
  >>> params.temperature
  0.9
  >>> # 获取指定 Agent 的参数（继承 tier 默认值）
  >>> params = center.get_params("worker", agent="chapter_writer")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

from novelfactory.agents.infra.logger import get_logger

logger = get_logger("novelfactory.config.llm_params")


# ═══════════════════════════════════════════════════════════════════════════════
#  参数模型
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class LLMParams:
    """单次 LLM 调用的完整参数集。

    所有字段都有合理默认值，可增量覆盖。
    """

    # ── 模型生成参数 ──────────────────────────────────────────────────────
    temperature: float = 0.7
    max_tokens: int = 65536
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: list[str] | None = None

    # ── 重试策略 ──────────────────────────────────────────────────────────
    max_retries: int = 3
    timeout_seconds: float = 300.0
    retry_policy: str = "default"

    # ── LLM 缓存 ──────────────────────────────────────────────────────────
    cache_enabled: bool = False
    cache_ttl_seconds: int = 3600

    # ── Provider 策略 ─────────────────────────────────────────────────────
    preferred_provider: str = "ark"
    fallback_enabled: bool = True

    # ── 元信息 ────────────────────────────────────────────────────────────
    description: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
#  参数注册中心
# ═══════════════════════════════════════════════════════════════════════════════


class LLMParameterCenter:
    """LLM 参数注册中心。

    集中管理所有 LLM 调用参数，支持分层覆盖和运行时调优。
    """

    def __init__(self) -> None:
        # tier → LLMParams
        self._tier_params: dict[str, LLMParams] = {}
        # (tier, agent) → LLMParams
        self._agent_params: dict[tuple[str, str], LLMParams] = {}
        # 环境变量覆盖
        self._env_overrides: dict[str, Any] = self._load_env_overrides()

    # ── 注册 API ──────────────────────────────────────────────────────────

    def register_tier(self, name: str, params: LLMParams) -> None:
        """注册 Tier 级默认参数。"""
        merged = self._apply_env(name, params)
        self._tier_params[name] = merged
        logger.info(
            "[params] tier=%s temp=%.2f max_tokens=%d timeout=%.0f policy=%s",
            name,
            merged.temperature,
            merged.max_tokens,
            merged.timeout_seconds,
            merged.retry_policy,
        )

    def register_agent(self, tier: str, agent: str, params: LLMParams) -> None:
        """注册 Agent 级参数覆盖。"""
        merged = self._apply_env(f"{tier}.{agent}", params)
        self._agent_params[(tier, agent)] = merged
        logger.info(
            "[params] agent=%s/%s temp=%.2f timeout=%.0f",
            tier,
            agent,
            merged.temperature,
            merged.timeout_seconds,
        )

    # ── 查询 API ──────────────────────────────────────────────────────────

    def get_params(
        self,
        tier: str,
        agent: str | None = None,
        **runtime_overrides: Any,
    ) -> LLMParams:
        """获取合并后的参数（优先级：运行时 > Agent > Tier > 全局默认值）。

        Args:
            tier: Tier 名称 (supervisor/worker/reviewer/review/writing)
            agent: Agent 名称 (可选，如 chapter_writer)
            **runtime_overrides: 运行时覆盖字段

        Returns:
            合并后的 LLMParams
        """
        # Layer 0: 全局默认值
        base = LLMParams()

        # Layer 2: Tier 注册值
        if tier in self._tier_params:
            base = self._merge_params(base, self._tier_params[tier])

        # Layer 3: Agent 注册值
        if agent and (tier, agent) in self._agent_params:
            base = self._merge_params(base, self._agent_params[(tier, agent)])

        # Layer 4: 运行时覆盖
        if runtime_overrides:
            safe_overrides = {
                k: v
                for k, v in runtime_overrides.items()
                if k in LLMParams.__dataclass_fields__ and v is not None
            }
            if safe_overrides:
                base = replace(base, **safe_overrides)

        return base

    def get_tier_params(self, tier: str) -> LLMParams | None:
        """获取 Tier 级参数。"""
        return self._tier_params.get(tier)

    def get_agent_params(self, tier: str, agent: str) -> LLMParams | None:
        """获取 Agent 级参数。"""
        return self._agent_params.get((tier, agent))

    # ── 调优 API ──────────────────────────────────────────────────────────

    def update_tier(self, name: str, **overrides: Any) -> LLMParams | None:
        """更新 Tier 参数（运行时调优用）。"""
        if name not in self._tier_params:
            return None
        current = self._tier_params[name]
        safe = {
            k: v for k, v in overrides.items() if k in LLMParams.__dataclass_fields__
        }
        updated = replace(current, **safe)
        self._tier_params[name] = updated
        return updated

    def update_agent(self, tier: str, agent: str, **overrides: Any) -> LLMParams | None:
        """更新 Agent 参数（运行时调优用）。"""
        key = (tier, agent)
        if key not in self._agent_params:
            return None
        current = self._agent_params[key]
        safe = {
            k: v for k, v in overrides.items() if k in LLMParams.__dataclass_fields__
        }
        updated = replace(current, **safe)
        self._agent_params[key] = updated
        return updated

    # ── 诊断 API ──────────────────────────────────────────────────────────

    def list_params(self) -> dict[str, Any]:
        """列出所有注册参数，用于调优面板。"""
        result: dict[str, Any] = {
            "tiers": {},
            "agents": {},
            "env_overrides": dict(self._env_overrides),
        }
        for name, params in sorted(self._tier_params.items()):
            result["tiers"][name] = {
                "temperature": params.temperature,
                "max_tokens": params.max_tokens,
                "timeout_seconds": params.timeout_seconds,
                "max_retries": params.max_retries,
                "retry_policy": params.retry_policy,
                "cache_enabled": params.cache_enabled,
                "fallback_enabled": params.fallback_enabled,
            }
        for (tier, agent), params in sorted(self._agent_params.items()):
            key = f"{tier}/{agent}"
            result["agents"][key] = {
                "temperature": params.temperature,
                "timeout_seconds": params.timeout_seconds,
                "max_retries": params.max_retries,
            }
        return result

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _merge_params(self, base: LLMParams, override: LLMParams) -> LLMParams:
        """合并两个 LLMParams，override 中的非默认值覆盖 base。"""
        updates: dict[str, Any] = {}
        default = LLMParams()
        for field_name in LLMParams.__dataclass_fields__:
            base_val = getattr(base, field_name)
            override_val = getattr(override, field_name)
            default_val = getattr(default, field_name)
            if override_val != default_val:
                updates[field_name] = override_val
            else:
                updates[field_name] = base_val
        return LLMParams(**updates)

    def _load_env_overrides(self) -> dict[str, Any]:
        """从 NOVELFACTORY_LLM_* 环境变量加载覆盖。"""
        overrides: dict[str, Any] = {}
        prefix = "NOVELFACTORY_LLM_"
        # 字段名 → 类型映射
        type_map: dict[str, type] = {
            "temperature": float,
            "max_tokens": int,
            "top_p": float,
            "frequency_penalty": float,
            "presence_penalty": float,
            "max_retries": int,
            "timeout_seconds": float,
            "cache_ttl_seconds": int,
        }
        for key, val in os.environ.items():
            if not key.startswith(prefix):
                continue
            field_name = key[len(prefix) :].lower()
            if field_name in type_map:
                cast = type_map[field_name]
                try:
                    overrides[field_name] = cast(val)
                except (ValueError, TypeError):
                    logger.warning("[params] 环境变量 %s=%s 解析失败，跳过", key, val)
        if overrides:
            logger.info("[params] 全局环境覆盖: %s", overrides)
        return overrides

    def _apply_env(self, scope: str, params: LLMParams) -> LLMParams:
        """将全局环境变量覆盖应用到参数上。"""
        if not self._env_overrides:
            return params
        scope_key = scope.upper().replace(".", "_")
        scope_overrides: dict[str, Any] = {}
        for field_name, value in self._env_overrides.items():
            env_key = f"NOVELFACTORY_LLM_{scope_key}_{field_name.upper()}"
            if env_key in os.environ:
                try:
                    type_map = {
                        "temperature": float,
                        "max_tokens": int,
                        "top_p": float,
                        "max_retries": int,
                        "timeout_seconds": float,
                    }
                    cast = type_map.get(field_name, str)
                    scope_overrides[field_name] = cast(os.environ[env_key])
                except (ValueError, TypeError):
                    continue
        if scope_overrides:
            logger.info("[params] %s 环境覆盖: %s", scope, scope_overrides)
            return replace(params, **scope_overrides)
        return params


# ═══════════════════════════════════════════════════════════════════════════════
#  全局单例 — 应用启动时由 llm.py 初始化
# ═══════════════════════════════════════════════════════════════════════════════

center = LLMParameterCenter()


# ═══════════════════════════════════════════════════════════════════════════════
#  初始化注册 — 在模块导入时执行
# ═══════════════════════════════════════════════════════════════════════════════


def _initialize_defaults() -> None:
    """注册系统默认参数。"""
    # ── Tier 级注册 ───────────────────────────────────────────────────────
    center.register_tier(
        "supervisor",
        LLMParams(
            temperature=0.3,
            max_tokens=65536,
            timeout_seconds=300.0,
            max_retries=3,
            retry_policy="default",
            description="编排调度 — 低温度确保路由决策确定性",
        ),
    )
    center.register_tier(
        "worker",
        LLMParams(
            temperature=0.7,
            max_tokens=65536,
            timeout_seconds=300.0,
            max_retries=3,
            retry_policy="writer",
            description="创作写作 — 中等温度平衡创意与稳定",
        ),
    )
    center.register_tier(
        "reviewer",
        LLMParams(
            temperature=0.2,
            max_tokens=65536,
            timeout_seconds=300.0,
            max_retries=3,
            retry_policy="reviewer",
            description="结构评分 — 低温度确保评分一致性",
        ),
    )
    center.register_tier(
        "review",
        LLMParams(
            temperature=0.2,
            max_tokens=65536,
            timeout_seconds=300.0,
            max_retries=3,
            retry_policy="default",
            description="人工终审 — 低温度确保审核判断稳定",
        ),
    )
    center.register_tier(
        "writing",
        LLMParams(
            temperature=0.75,
            max_tokens=65536,
            timeout_seconds=300.0,
            max_retries=3,
            retry_policy="writer",
            description="叙事写作 — 最高温度追求文学创意",
        ),
    )

    # ── Agent 级覆盖 ──────────────────────────────────────────────────────
    center.register_agent(
        "worker",
        "chapter_writer",
        LLMParams(
            temperature=0.75,
            timeout_seconds=600.0,
            description="章节创作 — 较长超时，较高创造性",
        ),
    )
    center.register_agent(
        "worker",
        "chapter_refiner",
        LLMParams(
            temperature=0.5,
            timeout_seconds=300.0,
            description="定向修复 — 中低温度确保精确性",
        ),
    )
    center.register_agent(
        "worker",
        "chapter_planner",
        LLMParams(
            temperature=0.3,
            timeout_seconds=120.0,
            description="章节规划 — 低温度确保结构化输出",
        ),
    )
    center.register_agent(
        "worker",
        "illustrator",
        LLMParams(
            temperature=0.8,
            timeout_seconds=120.0,
            description="插画生成 — 高温度增加创意多样性",
        ),
    )
    center.register_agent(
        "reviewer",
        "four_dim_review",
        LLMParams(
            temperature=0.15,
            timeout_seconds=180.0,
            description="四维评分 — 最低温度确保评分最严格",
        ),
    )
    center.register_agent(
        "reviewer",
        "editor_review",
        LLMParams(
            temperature=0.3,
            timeout_seconds=180.0,
            description="编辑评审 — 略高温度允许视角多样性",
        ),
    )
    center.register_agent(
        "reviewer",
        "reader_review",
        LLMParams(
            temperature=0.35,
            timeout_seconds=180.0,
            description="读者评审 — 最高温度允许主观判断",
        ),
    )

    logger.info(
        "[params] 注册完成: %d tiers, %d agents",
        len(center._tier_params),
        len(center._agent_params),
    )


# 模块导入时自动初始化
_initialize_defaults()
