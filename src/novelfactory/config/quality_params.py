"""质量参数动态管理中心。

将分散在 constants.py / engine.py / calibration.py / ai_style_analyzer.py /
old_reader_reviewer.py / debate/engine.py 中的质量参数集中管理，
支持运行时动态调整 + 持久化 + 变更历史 + 回滚 + 反馈参数定位。

分层优先级：
  Layer 0: 代码默认值（从各模块导入作为基线）
  Layer 1: Redis 覆盖（运行时持久化）
  Layer 2: 运行时内存覆盖（最高优先级，即时生效）

版本：v1.0.0
创建日期：2026-08-12
"""

from __future__ import annotations

import json
import logging
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from novelfactory.config.constants import (
    AI_STYLE_THRESHOLD,
    COMPOSITE_THRESHOLD,
    LAO_SHU_THRESHOLD,
    MAX_REWRITE_ATTEMPTS,
    QUALITY_SCORE_THRESHOLD,
    REFINE_MAX_ATTEMPTS,
    VERDICT_ITERATION_BONUS_MAX,
    VERDICT_ITERATION_BONUS_REFINE,
    VERDICT_ITERATION_BONUS_REWRITE,
    VERDICT_PASS_THRESHOLD,
    VERDICT_REFINE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  参数规格定义
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ParamSpec:
    """单个质量参数的规格定义。"""

    key: str
    module: str
    value_type: type
    default: Any
    min_val: Any
    max_val: Any
    description: str
    category: str
    feedback_tags: tuple[str, ...] = ()


def _bool_range() -> tuple[bool, bool]:
    """bool 类型的 min/max 占位（实际不用于范围校验）。"""
    return (False, True)


# ═══════════════════════════════════════════════════════════════════════════════
#  参数注册表
# ═══════════════════════════════════════════════════════════════════════════════

PARAM_REGISTRY: dict[str, ParamSpec] = {}


def _register(spec: ParamSpec) -> ParamSpec:
    """注册参数并返回（便于模块级批量注册）。"""
    PARAM_REGISTRY[spec.key] = spec
    return spec


# ── 通过阈值 ──
_register(ParamSpec("verdict.pass_threshold", "verdict", float, VERDICT_PASS_THRESHOLD, 50.0, 95.0,
                    "融合分通过线（≥此值直接 PASS）", "thresholds",
                    ("verdict.pass_threshold", "pass_threshold", "thresholds")))
_register(ParamSpec("verdict.refine_threshold", "verdict", float, VERDICT_REFINE_THRESHOLD, 30.0, 80.0,
                    "融合分润色/重写分界（≥此值润色，<此值重写）", "thresholds",
                    ("verdict.refine_threshold", "refine_threshold", "thresholds")))

# ── 迭代加分 ──
_IT = "iteration"
_register(ParamSpec("verdict.iteration_bonus.rewrite", "verdict", float,
                    VERDICT_ITERATION_BONUS_REWRITE, 0.0, 10.0,
                    "每次重写加分", _IT, ("iteration",)))
_register(ParamSpec("verdict.iteration_bonus.refine", "verdict", float,
                    VERDICT_ITERATION_BONUS_REFINE, 0.0, 10.0,
                    "每次润色加分", _IT, ("iteration",)))
_register(ParamSpec("verdict.iteration_bonus.max", "verdict", float,
                    VERDICT_ITERATION_BONUS_MAX, 0.0, 20.0,
                    "迭代加分封顶", _IT, ("iteration",)))

# ── 重试/迭代限制 ──
_register(ParamSpec("iteration.max_rewrite", "iteration", int,
                    MAX_REWRITE_ATTEMPTS, 1, 10,
                    "最大重写次数", _IT,
                    ("iteration.max_rewrite", "iteration", "重写次数太多", "重写次数不够")))
_register(ParamSpec("iteration.max_refine_high", "iteration", int,
                    REFINE_MAX_ATTEMPTS["high"], 0, 3,
                    "80-89 分润色次数", _IT, ("iteration",)))
_register(ParamSpec("iteration.max_refine_mid", "iteration", int,
                    REFINE_MAX_ATTEMPTS["mid"], 0, 5,
                    "60-79 分润色次数", _IT,
                    ("iteration.max_refine", "iteration", "润色不够")))

# ── 旧版评分阈值（兼容） ──
_register(ParamSpec("threshold.quality_score", "verdict", float,
                    QUALITY_SCORE_THRESHOLD, 50.0, 100.0,
                    "四维评分通过线（旧版兼容）", "thresholds", ()))
_register(ParamSpec("threshold.composite", "verdict", float,
                    COMPOSITE_THRESHOLD, 0.0, 1.0,
                    "综合指标通过线（旧版兼容）", "thresholds", ()))
_register(ParamSpec("threshold.ai_style", "ai_style", float,
                    AI_STYLE_THRESHOLD, 0.0, 1.0,
                    "AI 味指数合格线", "thresholds", ("ai_style", "AI味")))
_register(ParamSpec("threshold.lao_shu", "old_reader", float,
                    LAO_SHU_THRESHOLD, 0.0, 100.0,
                    "老书虫评分合格线", "thresholds", ()))

# ── 统一评审参数（v8.2 新增） ──
_U = "unified"
_register(ParamSpec("unified.max_retries", "unified", int, 1, 0, 3,
                    "统一评审失败最大重试次数", _U, ("unified",)))
_register(ParamSpec("unified.fallback_score", "unified", float, 60.0, 0.0, 100.0,
                    "统一评审失败降级分（REFINE 档）", _U, ("unified",)))


# ═══════════════════════════════════════════════════════════════════════════════
#  反馈-参数映射表
# ═══════════════════════════════════════════════════════════════════════════════

FEEDBACK_PARAM_MAP: dict[str, list[str]] = {
    # ── 评分标准相关（v8.2 保留阈值类） ──
    "评分虚高": ["unified.fallback_score"],
    "分数偏高": ["unified.fallback_score"],
    "太松了": ["verdict.pass_threshold", "verdict.refine_threshold"],
    "太严了": ["verdict.pass_threshold", "verdict.refine_threshold"],
    "垃圾章节通过了": ["verdict.pass_threshold", "unified.fallback_score"],
    "好章节被重写": ["verdict.pass_threshold", "verdict.refine_threshold"],

    # ── 内容质量（v8.2 统一评审承接，映射到 unified 评审提示词参数） ──
    "AI味": ["unified"],
    "机器味": ["unified"],
    "套话": ["unified"],
    "模板化": ["unified"],
    "句式重复": ["unified"],
    "毒点": ["unified"],
    "虐主": ["unified"],
    "被虐": ["unified"],
    "憋屈": ["unified"],
    "压抑": ["unified"],
    "圣母": ["unified"],
    "降智": ["unified"],
    "主角降智": ["unified"],
    "反派降智": ["unified"],
    "NTR": ["unified"],
    "战力崩坏": ["unified"],
    "水文": ["unified"],
    "煽情": ["unified"],
    "三观不正": ["unified"],

    # ── 爽点相关 ──
    "爽点不够": ["unified"],
    "不够爽": ["unified"],
    "打脸不够": ["unified"],
    "装逼不够": ["unified"],
    "逆袭不够": ["unified"],
    "升级不够": ["unified"],

    # ── 质量衰减（统一评审 decay_hint 承接） ──
    "高开低走": ["unified"],
    "后段质量下降": ["unified"],
    "虎头蛇尾": ["unified"],

    # ── 辩论相关（分歧仲裁承接） ──
    "辩论不够深入": ["unified"],
    "辩论太多轮": ["unified"],

    # ── 重写/润色 ──
    "重写次数太多": ["iteration.max_rewrite"],
    "重写次数不够": ["iteration.max_rewrite"],
    "润色不够": ["iteration.max_refine_mid", "iteration.max_refine_high"],

    # ── 评审口径 ──
    "评审标准": ["unified.fallback_score", "verdict.pass_threshold"],
}


# ═══════════════════════════════════════════════════════════════════════════════
#  变更记录
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ChangeRecord:
    """单次参数变更记录。"""

    change_id: str
    timestamp: float
    key: str
    old_value: Any
    new_value: Any
    reason: str
    operator: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_id": self.change_id,
            "timestamp": self.timestamp,
            "key": self.key,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason": self.reason,
            "operator": self.operator,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  QualityParameterCenter 核心类
# ═══════════════════════════════════════════════════════════════════════════════


def get_param(key: str, default: Any = None) -> Any:
    """模块级便捷读取：quality_center 覆盖优先，缺失/异常时返回 default。

    v8.3: 供 evaluation/verdict 与 evaluation/unified 共用，
    替代各自重复实现的 _get_*_param 私有函数。
    """
    try:
        from novelfactory.config.quality_params import quality_center

        val = quality_center.get(key)
        if val is not None:
            return val
    except Exception:
        pass
    return default


class QualityParameterCenter:
    """质量参数动态管理中心（单例）。

    使用方式::

        from novelfactory.config.quality_params import quality_center

        # 读取
        val = quality_center.get("verdict.pass_threshold")

        # 更新
        quality_center.update("verdict.pass_threshold", 80.0, reason="用户反馈太松")

        # 批量更新（v8.2: 统一评审，无权重归一化）
        quality_center.update_many(
            {"verdict.pass_threshold": 75.0, "unified.max_retries": 2},
            reason="提高通过标准",
        )

        # 回滚
        quality_center.rollback(change_id)

        # 根据反馈定位参数
        params = quality_center.find_params_by_feedback("AI味太重了")
    """

    _REDIS_KEY_CURRENT = "quality_params:current"
    _REDIS_KEY_HISTORY = "quality_params:history"
    _JSON_PATH = Path(".novelfactory") / "quality" / "params.json"
    _MAX_HISTORY = 100

    def __init__(self) -> None:
        self._overrides: dict[str, Any] = {}
        self._history: list[ChangeRecord] = []
        # RLock 支持可重入（update 内部调用 get 时避免死锁）
        self._lock = threading.RLock()
        self._redis_store: Any = None

    # ── 读取 ──────────────────────────────────────────────────────────────

    def get(self, key: str) -> Any:
        """获取参数当前值（Layer 2 > Layer 1 > Layer 0 默认）。"""
        with self._lock:
            if key in self._overrides:
                return self._overrides[key]
        spec = PARAM_REGISTRY.get(key)
        if spec is None:
            logger.warning("[QualityParams] Unknown key: %s", key)
            return None
        return spec.default

    def get_override(self, key: str) -> Any:
        """直接读取运行时覆盖值（不经过注册表，用于题材阈值等动态 key）。

        Returns:
            覆盖值；未设置时返回 None。
        """
        with self._lock:
            return self._overrides.get(key)

    def set_override(self, key: str, value: Any, reason: str = "", operator: str = "") -> bool:
        """直接写入运行时覆盖值（跳过注册表校验，用于题材阈值等动态 key）。"""
        with self._lock:
            old_value = self._overrides.get(key)
            self._overrides[key] = value
            record = ChangeRecord(
                change_id=uuid.uuid4().hex[:12],
                timestamp=time.time(),
                key=key,
                old_value=old_value,
                new_value=value,
                reason=reason,
                operator=operator,
            )
            self._history.insert(0, record)
            if len(self._history) > self._MAX_HISTORY:
                self._history = self._history[: self._MAX_HISTORY]
        logger.info(
            "[QualityParams] Override %s: %s -> %s (reason=%s)",
            key, old_value, value, reason,
        )
        return True

    def get_many(self, keys: list[str]) -> dict[str, Any]:
        """批量获取参数。"""
        return {k: self.get(k) for k in keys}

    def get_category(self, category: str) -> dict[str, Any]:
        """获取指定分类的所有参数。"""
        return {
            spec.key: self.get(spec.key)
            for spec in PARAM_REGISTRY.values()
            if spec.category == category
        }

    def get_all(self) -> dict[str, Any]:
        """获取所有参数。"""
        return {key: self.get(key) for key in PARAM_REGISTRY}

    def get_snapshot(self) -> dict[str, Any]:
        """获取全量参数快照（给 Agent 看，含元信息）。"""
        return {
            key: {
                "value": self.get(key),
                "default": spec.default,
                "min": spec.min_val,
                "max": spec.max_val,
                "type": spec.value_type.__name__,
                "description": spec.description,
                "category": spec.category,
                "modified": key in self._overrides,
            }
            for key, spec in PARAM_REGISTRY.items()
        }

    # ── 更新 ──────────────────────────────────────────────────────────────

    def update(self, key: str, value: Any, reason: str = "", operator: str = "") -> bool:
        """更新单个参数。"""
        ok, msg = self._validate(key, value)
        if not ok:
            logger.warning("[QualityParams] Validation failed for %s: %s", key, msg)
            return False

        with self._lock:
            old_value = self.get(key)
            self._overrides[key] = value

            record = ChangeRecord(
                change_id=uuid.uuid4().hex[:12],
                timestamp=time.time(),
                key=key,
                old_value=old_value,
                new_value=value,
                reason=reason,
                operator=operator,
            )
            self._history.insert(0, record)
            if len(self._history) > self._MAX_HISTORY:
                self._history = self._history[: self._MAX_HISTORY]

        logger.info(
            "[QualityParams] Updated %s: %s -> %s (reason=%s, id=%s)",
            key, old_value, value, reason, record.change_id,
        )
        return True

    def update_many(
        self,
        updates: dict[str, Any],
        reason: str = "",
        operator: str = "",
    ) -> dict[str, bool]:
        """批量更新参数。

        v8.2: 权重归一化校验已移除（统一评审无权重融合参数）。

        Returns:
            {key: success} 字典
        """
        results: dict[str, bool] = {}
        change_ids: list[str] = []
        for key, value in updates.items():
            ok = self.update(key, value, reason=reason, operator=operator)
            results[key] = ok
            if ok and self._history:
                change_ids.append(self._history[0].change_id)

        # 异步持久化
        self._save_async()

        return results

    def reset(self, key: str) -> bool:
        """恢复单个参数到默认值。"""
        with self._lock:
            if key in self._overrides:
                old_value = self._overrides[key]
                del self._overrides[key]
                logger.info("[QualityParams] Reset %s to default (was %s)", key, old_value)
                self._save_async()
                return True
            return False

    def reset_all(self) -> None:
        """恢复所有参数到默认值。"""
        with self._lock:
            count = len(self._overrides)
            self._overrides.clear()
            logger.info("[QualityParams] Reset all %d params to defaults", count)
            self._save_async()

    # ── 变更历史 ──────────────────────────────────────────────────────────

    def get_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """获取最近 N 条变更记录。"""
        with self._lock:
            return [r.to_dict() for r in self._history[:limit]]

    def rollback(self, change_id: str) -> bool:
        """回滚指定变更。"""
        with self._lock:
            for record in self._history:
                if record.change_id == change_id:
                    if record.old_value is None:
                        del self._overrides[record.key]
                    else:
                        self._overrides[record.key] = record.old_value
                    logger.info(
                        "[QualityParams] Rolled back %s: %s -> %s",
                        record.key, record.new_value, record.old_value,
                    )
                    self._save_async()
                    return True
        logger.warning("[QualityParams] Change ID not found: %s", change_id)
        return False

    def rollback_last(self) -> bool:
        """回滚最近一次变更。"""
        with self._lock:
            if not self._history:
                return False
            record = self._history[0]
        return self.rollback(record.change_id)

    # ── 反馈参数定位 ──────────────────────────────────────────────────────

    def find_params_by_feedback(self, feedback: str) -> list[ParamSpec]:
        """根据用户反馈文本自动定位相关参数。

        匹配策略：
        1. 关键词包含匹配 -> 返回对应标签的所有参数
        2. 无匹配 -> 返回空列表（交给 LLM 做语义分析）
        """
        matched_tags: set[str] = set()
        for keyword, tags in FEEDBACK_PARAM_MAP.items():
            if keyword in feedback:
                matched_tags.update(tags)

        results: list[ParamSpec] = []
        seen_keys: set[str] = set()
        for spec in PARAM_REGISTRY.values():
            if spec.key in seen_keys:
                continue
            for tag in spec.feedback_tags:
                if tag in matched_tags or tag in feedback:
                    results.append(spec)
                    seen_keys.add(spec.key)
                    break
        return results

    # ── 校验 ──────────────────────────────────────────────────────────────

    def _validate(self, key: str, value: Any) -> tuple[bool, str]:
        """校验参数值是否合法。"""
        spec = PARAM_REGISTRY.get(key)
        if spec is None:
            return False, f"Unknown parameter key: {key}"

        if not isinstance(value, spec.value_type):
            # int 可以接受 float（向下取整）
            if spec.value_type is int and isinstance(value, (int, float)):
                value = int(value)
            elif spec.value_type is float and isinstance(value, (int, float)):
                pass
            else:
                return False, f"Type mismatch: expected {spec.value_type.__name__}, got {type(value).__name__}"

        if spec.value_type in (int, float):
            if value < spec.min_val:
                return False, f"Value {value} < min {spec.min_val}"
            if value > spec.max_val:
                return False, f"Value {value} > max {spec.max_val}"

        return True, "ok"

    # ── 持久化 ────────────────────────────────────────────────────────────

    async def load_from_redis(self) -> None:
        """从 Redis 加载持久化参数。"""
        try:
            from novelfactory.store.redis_store import get_redis_store

            store = get_redis_store()
            if not store or not store.is_connected():
                logger.info("[QualityParams] Redis not connected, skipping load")
                return

            data = await store.hgetall(self._REDIS_KEY_CURRENT)
            if data:
                with self._lock:
                    for key, raw_val in data.items():
                        val: Any = raw_val
                        if isinstance(raw_val, str):
                            try:
                                val = json.loads(raw_val)
                            except json.JSONDecodeError:
                                val = raw_val
                        spec = PARAM_REGISTRY.get(key)
                        if spec:
                            if spec.value_type is float:
                                val = float(val)
                            elif spec.value_type is int:
                                val = int(val)
                            elif spec.value_type is bool:
                                val = str(val).lower() in ("true", "1", "yes")
                            self._overrides[key] = val
                logger.info("[QualityParams] Loaded %d overrides from Redis", len(data))

            # 加载历史
            history_raw = await store.lrange(self._REDIS_KEY_HISTORY, 0, self._MAX_HISTORY - 1)
            if history_raw:
                with self._lock:
                    self._history = []
                    for item in history_raw:
                        try:
                            d = json.loads(item) if isinstance(item, str) else item
                            self._history.append(ChangeRecord(**d))
                        except (json.JSONDecodeError, TypeError):
                            continue

        except Exception as e:
            logger.warning("[QualityParams] Failed to load from Redis: %s", e)

    async def save_to_redis(self) -> None:
        """保存参数到 Redis。"""
        try:
            from novelfactory.store.redis_store import get_redis_store

            store = get_redis_store()
            if not store or not store.is_connected():
                return

            with self._lock:
                if self._overrides:
                    for k, v in self._overrides.items():
                        await store.hset(self._REDIS_KEY_CURRENT, k, json.dumps(v))

                if self._history:
                    items = [json.dumps(r.to_dict()) for r in self._history[: self._MAX_HISTORY]]
                    await store.delete(self._REDIS_KEY_HISTORY)
                    for item in items:
                        await store.lpush(self._REDIS_KEY_HISTORY, item)

        except Exception as e:
            logger.warning("[QualityParams] Failed to save to Redis: %s", e)

    def load_from_json(self) -> None:
        """从 JSON 文件加载（降级方案）。"""
        path = self._JSON_PATH
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                with self._lock:
                    for key, raw_val in raw.get("overrides", {}).items():
                        val: Any = raw_val
                        if isinstance(raw_val, str):
                            try:
                                val = json.loads(raw_val)
                            except json.JSONDecodeError:
                                val = raw_val
                        spec = PARAM_REGISTRY.get(key)
                        if spec:
                            if spec.value_type is float:
                                val = float(val)
                            elif spec.value_type is int:
                                val = int(val)
                            elif spec.value_type is bool:
                                val = str(val).lower() in ("true", "1", "yes")
                            self._overrides[key] = val
                logger.info("[QualityParams] Loaded %d overrides from JSON", len(raw.get("overrides", {})))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("[QualityParams] Failed to load JSON: %s", e)

    def save_to_json(self) -> None:
        """保存到 JSON 文件（降级方案）。"""
        path = self._JSON_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {
                "overrides": self._overrides,
                "history": [r.to_dict() for r in self._history[: self._MAX_HISTORY]],
            }
        fd = tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, suffix=".tmp", delete=False, encoding="utf-8",
        )
        try:
            json.dump(data, fd, indent=2, ensure_ascii=False, default=str)
            fd.close()
            Path(fd.name).replace(path)
        except BaseException:
            fd.close()
            Path(fd.name).unlink(missing_ok=True)
            raise

    def _save_async(self) -> None:
        """异步保存（先 JSON 即时写，Redis 异步写）。"""
        # JSON 即时写（保证数据不丢）
        try:
            self.save_to_json()
        except Exception as e:
            logger.warning("[QualityParams] JSON save failed: %s", e)

        # Redis 异步写（通过事件循环或线程）
        try:
            import asyncio

            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.save_to_redis())
            else:
                loop.run_until_complete(self.save_to_redis())
        except Exception:
            # 无事件循环时跳过 Redis
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  全局单例
# ═══════════════════════════════════════════════════════════════════════════════

quality_center = QualityParameterCenter()

# 启动时从 JSON 加载（保证立即可用）
quality_center.load_from_json()
