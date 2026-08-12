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
    CALIBRATION_LLM_VIRTUAL_HIGH,
    CALIBRATION_PROGRAMMATIC_LOW,
    CALIBRATION_SEVERE_TOXIC_CAP,
    CALIBRATION_SHORT_TEXT_LLM_WEIGHT,
    COMPOSITE_THRESHOLD,
    FALLBACK_AI_STYLE_SCORE,
    FALLBACK_COMPOSITE_SCORE,
    FALLBACK_LAO_SHU_SCORE,
    FALLBACK_QUALITY_SCORE,
    LAO_SHU_THRESHOLD,
    MAX_REWRITE_ATTEMPTS,
    QUALITY_SCORE_THRESHOLD,
    REFINE_MAX_ATTEMPTS,
    VERDICT_DEBATE_PENALTY_CAP,
    VERDICT_DEBATE_PENALTY_PER_ISSUE,
    VERDICT_DEBATE_PENALTY_PER_SEVERE,
    VERDICT_ITERATION_BONUS_MAX,
    VERDICT_ITERATION_BONUS_REFINE,
    VERDICT_ITERATION_BONUS_REWRITE,
    VERDICT_LENGTH_NORMALIZE,
    VERDICT_NORMALIZE_BASE,
    VERDICT_PASS_THRESHOLD,
    VERDICT_REFINE_THRESHOLD,
    VERDICT_WEIGHTS,
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


# ── 融合权重（总和必须 = 1.0） ──
_W = "weights"
for _k, _v in VERDICT_WEIGHTS.items():
    _register(ParamSpec(
        key=f"verdict.weights.{_k}",
        module="verdict",
        value_type=float,
        default=_v,
        min_val=0.0,
        max_val=1.0,
        description=f"融合权重 - {_k}",
        category=_W,
        feedback_tags=(f"verdict.weights.{_k}", "verdict.weights"),
    ))

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

# ── 质量衰减 ──
# v9.0: 与 evaluation/verdict/engine.py _DECAY_* 同步
_D = "decay"
_register(ParamSpec("verdict.decay.head_ratio", "verdict", float, 0.30, 0.1, 0.5,
                    "前段比例（前 N% 为前段）", _D, ("decay",)))
_register(ParamSpec("verdict.decay.tail_ratio", "verdict", float, 0.30, 0.1, 0.5,
                    "后段比例（后 N% 为后段）", _D, ("decay",)))
_register(ParamSpec("verdict.decay.penalty_per_point", "verdict", float, 0.5, 0.0, 2.0,
                    "每 1 分衰减扣分系数", _D, ("decay",)))
_register(ParamSpec("verdict.decay.max_penalty", "verdict", float, 8.0, 0.0, 30.0,
                    "衰减惩罚上限", _D, ("decay",)))

# ── 辩论惩罚 ──
_DP = "debate_penalty"
_register(ParamSpec("verdict.debate_penalty.cap", "verdict", float,
                    VERDICT_DEBATE_PENALTY_CAP, 0.0, 50.0,
                    "辩论惩罚上限", _DP, ("debate_penalty",)))
_register(ParamSpec("verdict.debate_penalty.per_issue", "verdict", float,
                    VERDICT_DEBATE_PENALTY_PER_ISSUE, 0.0, 10.0,
                    "每个问题扣分", _DP, ("debate_penalty",)))
_register(ParamSpec("verdict.debate_penalty.per_severe", "verdict", float,
                    VERDICT_DEBATE_PENALTY_PER_SEVERE, 0.0, 15.0,
                    "严重问题额外扣分", _DP, ("debate_penalty",)))

# ── 长度归一化 ──
_LN = "length"
_register(ParamSpec("verdict.length_normalize", "verdict", bool,
                    VERDICT_LENGTH_NORMALIZE, *_bool_range(),
                    "是否启用长度归一化", _LN, ("length",)))
_register(ParamSpec("verdict.normalize_base", "verdict", int,
                    VERDICT_NORMALIZE_BASE, 1000, 10000,
                    "基准字数（中文字符）", _LN, ("length",)))

# ── 校准阈值 ──
_CAL = "calibration"
_register(ParamSpec("calibration.llm_virtual_high", "calibration", float,
                    CALIBRATION_LLM_VIRTUAL_HIGH, 80.0, 100.0,
                    "LLM 虚高阈值（quality≥此值且程序化低时压分）", _CAL,
                    ("calibration.llm_virtual_high", "calibration", "评分虚高", "分数偏高")))
_register(ParamSpec("calibration.programmatic_low", "calibration", float,
                    CALIBRATION_PROGRAMMATIC_LOW, 0.0, 1.0,
                    "程序化分过低阈值（<此值触发虚高校准）", _CAL,
                    ("calibration.programmatic_low", "calibration")))
_register(ParamSpec("calibration.short_text_llm_weight", "calibration", float,
                    CALIBRATION_SHORT_TEXT_LLM_WEIGHT, 0.0, 1.0,
                    "短文本 LLM 权重", _CAL, ("calibration",)))
_register(ParamSpec("calibration.severe_toxic_cap", "calibration", float,
                    CALIBRATION_SEVERE_TOXIC_CAP, 0.0, 80.0,
                    "严重毒点分数封顶值", _CAL,
                    ("calibration.severe_toxic_cap", "calibration", "毒点")))

# ── 辩论参数 ──
_DB = "debate"
_register(ParamSpec("debate.max_rounds", "debate", int, 3, 1, 5,
                    "最大辩论轮数", _DB,
                    ("debate.max_rounds", "debate", "辩论不够深入", "辩论太多轮")))
_register(ParamSpec("debate.convergence_idle_rounds", "debate", int, 2, 1, 3,
                    "连续无新问题收敛轮数", _DB, ("debate",)))

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

# ── AI 味 8 维权重（总和必须 = 1.0） ──
# v9.0: 与 analysis/ai_style_analyzer.py AI_WEIGHTS 同步
_AI_W = {
    "repetition_ngram": 0.25,
    "sentence_length_variance": 0.18,
    "lexical_diversity": 0.16,
    "cliche_ratio": 0.20,
    "punctuation_rhythm": 0.08,
    "dialogue_ratio": 0.05,
    "sensory_emotion_density": 0.03,
    "semantic_smoothness": 0.05,
}
for _k, _v in _AI_W.items():
    _register(ParamSpec(
        key=f"ai_style.weights.{_k}",
        module="ai_style",
        value_type=float,
        default=_v,
        min_val=0.0,
        max_val=1.0,
        description=f"AI 味权重 - {_k}",
        category="ai_weights",
        feedback_tags=(f"ai_style.{_k}", "ai_style", "AI味", "机器味"),
    ))

# ── 毒点权重 ──
_TOXIC_W = {
    "NTR": 50,
    "SHENGMU": 30,
    "PROTAGONIST_STUPID": 30,  # v9.0: 原25
    "NUE_ZHU": 30,  # v9.0: 原25
    "POWER_BREAK": 20,
    "ANTAGONIST_STUPID": 15,
    "CHARACTER_DEATH": 12,  # v9.0: 原15
    "WATER_CONTENT": 15,
    "SENTIMENTAL_TORTURE": 10,
    "MORAL_WRONG": 10,
}
for _k, _v in _TOXIC_W.items():
    _register(ParamSpec(
        key=f"toxic.weights.{_k}",
        module="old_reader",
        value_type=int,
        default=_v,
        min_val=0,
        max_val=100,
        description=f"毒点权重 - {_k}",
        category="toxic_weights",
        feedback_tags=(f"toxic.{_k}", "toxic", "毒点", _k),
    ))

# ── 爽点权重 ──
# v9.0: 与 analysis/old_reader_reviewer.py SHUANGDIAN_POINTS 同步
_SHUANGDIAN_W = {
    "打脸": 1.0,
    "装逼": 0.9,
    "逆袭": 0.90,  # v9.0: 原0.85
    "升级": 0.85,  # v9.0: 原0.8
    "感情": 0.70,  # v9.0: 原0.75
    "悬念": 0.75,  # v9.0: 原0.7
}
for _k, _v in _SHUANGDIAN_W.items():
    _register(ParamSpec(
        key=f"shuangdian.weights.{_k}",
        module="old_reader",
        value_type=float,
        default=_v,
        min_val=0.0,
        max_val=2.0,
        description=f"爽点权重 - {_k}",
        category="shuangdian_weights",
        feedback_tags=(f"shuangdian.{_k}", "shuangdian", "爽点", _k),
    ))

# ── 降级默认值 ──
_FB = "fallback"
_register(ParamSpec("fallback.quality_score", "verdict", float,
                    FALLBACK_QUALITY_SCORE, 0.0, 100.0,
                    "降级默认四维评分", _FB, ()))
_register(ParamSpec("fallback.composite_score", "verdict", float,
                    FALLBACK_COMPOSITE_SCORE, 0.0, 1.0,
                    "降级默认综合指标", _FB, ()))
_register(ParamSpec("fallback.ai_style_score", "verdict", float,
                    FALLBACK_AI_STYLE_SCORE, 0.0, 1.0,
                    "降级默认 AI 味指数", _FB, ()))
_register(ParamSpec("fallback.lao_shu_score", "verdict", float,
                    FALLBACK_LAO_SHU_SCORE, 0.0, 100.0,
                    "降级默认老书虫评分", _FB, ()))

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

del _k, _v  # 清理循环变量


# ═══════════════════════════════════════════════════════════════════════════════
#  反馈-参数映射表
# ═══════════════════════════════════════════════════════════════════════════════

FEEDBACK_PARAM_MAP: dict[str, list[str]] = {
    # ── AI 味相关 ──
    "AI味": ["ai_style", "llm_human_like"],
    "机器味": ["ai_style", "llm_human_like"],
    "套话": ["ai_style.cliche_ratio", "llm_human_like"],
    "模板化": ["ai_style.cliche_ratio", "llm_human_like"],
    "句式重复": ["ai_style.repetition_ngram", "ai_style.sentence_length_variance"],
    "词汇单调": ["ai_style.lexical_diversity"],
    "标点节奏": ["ai_style.punctuation_rhythm"],
    "对话比例": ["ai_style.dialogue_ratio"],

    # ── 评分标准相关 ──
    "评分虚高": ["calibration.llm_virtual_high", "calibration.programmatic_low"],
    "分数偏高": ["calibration.llm_virtual_high"],
    "太松了": ["verdict.pass_threshold", "verdict.refine_threshold"],
    "太严了": ["verdict.pass_threshold", "verdict.refine_threshold"],
    "垃圾章节通过了": ["verdict.pass_threshold", "calibration.severe_toxic_cap"],
    "好章节被重写": ["verdict.pass_threshold", "verdict.refine_threshold"],

    # ── 毒点相关 ──
    "毒点": ["toxic"],
    "虐主": ["toxic.NUE_ZHU"],
    "被虐": ["toxic.NUE_ZHU"],
    "憋屈": ["toxic.NUE_ZHU"],
    "压抑": ["toxic.NUE_ZHU"],
    "圣母": ["toxic.SHENGMU"],
    "降智": ["toxic.PROTAGONIST_STUPID"],
    "主角降智": ["toxic.PROTAGONIST_STUPID"],
    "反派降智": ["toxic.ANTAGONIST_STUPID"],
    "NTR": ["toxic.NTR"],
    "战力崩坏": ["toxic.POWER_BREAK"],
    "水文": ["toxic.WATER_CONTENT"],
    "煽情": ["toxic.SENTIMENTAL_TORTURE"],
    "三观不正": ["toxic.MORAL_WRONG"],

    # ── 爽点相关 ──
    "爽点不够": ["shuangdian"],
    "不够爽": ["shuangdian"],
    "打脸不够": ["shuangdian.打脸"],
    "装逼不够": ["shuangdian.装逼"],
    "逆袭不够": ["shuangdian.逆袭"],
    "升级不够": ["shuangdian.升级"],

    # ── 质量衰减 ──
    "高开低走": ["decay"],
    "后段质量下降": ["decay"],
    "虎头蛇尾": ["decay"],

    # ── 辩论相关 ──
    "辩论不够深入": ["debate.max_rounds"],
    "辩论太多轮": ["debate.max_rounds"],

    # ── 重写/润色 ──
    "重写次数太多": ["iteration.max_rewrite"],
    "重写次数不够": ["iteration.max_rewrite"],
    "润色不够": ["iteration.max_refine_mid", "iteration.max_refine_high"],

    # ── 融合权重 ──
    "程序化分析权重": ["verdict.weights.programmatic"],
    "LLM评分权重": ["verdict.weights.quality"],
    "跨章一致性权重": ["verdict.weights.cross_chapter"],
    "吸引力权重": ["verdict.weights.attraction"],
    "老书虫权重": ["verdict.weights.llm_old_reader"],
}


# ═══════════════════════════════════════════════════════════════════════════════
#  权重组定义（哪些参数变更时需要校验总和 = 1.0）
# ═══════════════════════════════════════════════════════════════════════════════

WEIGHT_GROUPS: dict[str, list[str]] = {
    "verdict_weights": [
        "verdict.weights.quality",
        "verdict.weights.programmatic",
        "verdict.weights.llm_old_reader",
        "verdict.weights.llm_human_like",
        "verdict.weights.cross_chapter",
        "verdict.weights.debate_penalty",
        "verdict.weights.attraction",
    ],
    "ai_style_weights": [
        "ai_style.weights.repetition_ngram",
        "ai_style.weights.sentence_length_variance",
        "ai_style.weights.lexical_diversity",
        "ai_style.weights.cliche_ratio",
        "ai_style.weights.punctuation_rhythm",
        "ai_style.weights.dialogue_ratio",
        "ai_style.weights.sensory_emotion_density",
        "ai_style.weights.semantic_smoothness",
    ],
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


class QualityParameterCenter:
    """质量参数动态管理中心（单例）。

    使用方式::

        from novelfactory.config.quality_params import quality_center

        # 读取
        val = quality_center.get("verdict.weights.quality")

        # 更新
        quality_center.update("verdict.pass_threshold", 80.0, reason="用户反馈太松")

        # 批量更新（自动校验权重组总和）
        quality_center.update_many(
            {"verdict.weights.quality": 0.25, "verdict.weights.programmatic": 0.20},
            reason="提高 LLM 评分权重",
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

    def get_weights(self, group_name: str) -> dict[str, float]:
        """获取权重组的 {key: value} 字典。"""
        keys = WEIGHT_GROUPS.get(group_name, [])
        return {k: self.get(k) for k in keys}

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
        """批量更新参数。自动校验权重组总和。

        Returns:
            {key: success} 字典
        """
        # 预校验权重组总和
        for group_name, group_keys in WEIGHT_GROUPS.items():
            affected = {k: v for k, v in updates.items() if k in group_keys}
            if not affected:
                continue
            merged = {k: self.get(k) for k in group_keys}
            merged.update(affected)
            total = sum(float(v) for v in merged.values())
            if abs(total - 1.0) > 0.001:
                logger.warning(
                    "[QualityParams] Weight group '%s' sum=%.4f != 1.0, auto-normalizing",
                    group_name, total,
                )
                # 归一化整个权重组（含未变更项），保证总和 = 1.0
                factor = 1.0 / total
                for k in group_keys:
                    if k in affected:
                        updates[k] = round(float(merged[k]) * factor, 4)
                    else:
                        # 未变更项也按比例归一化（写入 overrides 保持组内一致）
                        updates[k] = round(float(self.get(k)) * factor, 4)

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
                    for key, val in data.items():
                        if isinstance(val, str):
                            try:
                                val = json.loads(val)
                            except json.JSONDecodeError:
                                pass
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
                    for key, val in raw.get("overrides", {}).items():
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
