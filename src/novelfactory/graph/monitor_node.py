"""智能监控节点 —— LLM 分析 + 飞书通知 + LangGraph 集成。

本模块是 NovelFactory 智能监控系统的核心，功能：
  1. 作为 LangGraph 节点嵌入根图，连接在 writing_crew→main_supervisor 之间
  2. 每次写入章节后，调用 LLM 分析当前写作状态（质量趋势、Token消耗、成本、
     异常检测等），生成人类可读的分析报告
  3. 通过飞书推送智能分析报告
  4. 将分析结果写入 state（给后续路由节点做决策参考）

架构约束：
  - 轻量级：单次 LLM 调用 + 飞书推送，<5s 完成
  - 无状态：所有分析数据从 state 读取
  - 容错：LLM 或飞书失败不影响主流程
"""

import logging
import os
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import RunnableConfig

from novelfactory.config.llm import get_supervisor_llm
from novelfactory.integrations.feishu.feishu_api import send_lark_message

logger = logging.getLogger(__name__)

# ── 监控分析常量 ────────────────────────────────────────
_MIN_SCORE_TREND_WINDOW = 3  # 趋势分析最小窗口
_TREND_THRESHOLD_UP = 5  # 上升趋势阈值（分）
_TREND_THRESHOLD_DOWN = -5  # 下降趋势阈值（分）
_MIN_TOKEN_TREND_WINDOW = 5  # Token 趋势分析最小窗口
_MIN_QUALITY_ANOMALY_WINDOW = 2  # 质量异常检测窗口
_QUALITY_ANOMALY_THRESHOLD = 60  # 质量异常阈值（分）
_FIRST_ANALYSIS_CHAPTER = 2  # 首次分析触发章节

# ── 监控提示词 ──────────────────────────────────────────

ANALYSIS_SYSTEM_PROMPT = ""  # v5.4: 从 config/prompts 加载, 此处仅作为 fallback 默认值


def _get_analysis_prompt() -> str:
    """获取监控分析 System Prompt (优先从模板文件加载, 不可用时回退硬编码)。"""
    from novelfactory.config.prompts import get_prompt

    prompt = get_prompt("monitoring", "analysis_system")
    if prompt:
        return prompt
    # 终极回退
    return """你是一名小说创作数据分析师，帮助分析 NovelFactory AI 小说的自动生成状态。

你收到的数据包括：
- 已完成章节数 / 目标总章数
- 最近章节的质量评分历史（滑动窗口）
- 累计 Token 消耗和估算费用
- 最近的错误日志（如有）
- 当前写作阶段

请输出简短（<200字）的分析报告，包含：
1. **状态摘要**：进度百分比，预计剩余时间
2. **质量趋势**：近几章评分是上升/下降/稳定，如有下降趋势给出预警
3. **成本提示**：当前总花费，如果有异常消耗提醒
4. **健康检查**：是否有任何异常（连续低分、Token 突增、错误）
5. **一句话建议**：当前状态需要关注什么

用中文回复，不要用 Markdown 格式，直接返回纯文本。"""


def _count_volumes(volume_structure: dict | list | None) -> int:
    """Count volumes from volume_structure (dict or list, v5.9 fix)."""
    if isinstance(volume_structure, dict):
        return len(volume_structure.get("volumes", []))
    if isinstance(volume_structure, list):
        return len(volume_structure)
    return 0


def build_monitoring_snapshot(state: dict) -> str:
    """从 state 中提取监控数据，构建 LLM 可读的快照。"""
    completed = state.get("completed_chapters", [])
    total = state.get("target_chapters") or 500
    current_ch = state.get("current_chapter", 1)
    usage = state.get("total_usage", {})
    chapter_usages = usage.get("chapter_usages", [])
    errors = state.get("monitor_errors", [])

    # 提取最近质评
    recent_scores = []
    for ch in completed[-10:]:
        qs = ch.get("quality_score", 0)
        if qs:
            recent_scores.append(qs)

    # Token 和成本
    total_tokens = usage.get("total_tokens", 0)
    total_cost = usage.get("estimated_cost_cny", 0.0)

    # 最近的错误
    recent_errors = errors[-3:] if errors else []

    lines = [
        "【基本信息】",
        f"  项目：{state.get('project_name', '未知')}",
        f"  阶段：{state.get('current_phase', '?')}",
        f"  章节：{current_ch}/{total} 章",
        f"  卷结构：{_count_volumes(state.get('volume_structure'))} 卷",
        "",
        f"【质量趋势】（最近 {len(recent_scores)} 章）",
    ]
    if recent_scores:
        scores_str = " → ".join(f"{s:.0f}" for s in recent_scores)
        avg = sum(recent_scores) / len(recent_scores)
        lines.append(f"  分数：{scores_str}")
        lines.append(f"  平均：{avg:.1f}/100")
        # 趋势判断
        if len(recent_scores) >= _MIN_SCORE_TREND_WINDOW:
            first_half = recent_scores[: len(recent_scores) // 2]
            second_half = recent_scores[len(recent_scores) // 2 :]
            trend = (sum(second_half) / len(second_half)) - (
                sum(first_half) / len(first_half)
            )
            if trend > _TREND_THRESHOLD_UP:
                lines.append(f"  趋势：[UP] 上升中 (+{trend:.1f})")
            elif trend < _TREND_THRESHOLD_DOWN:
                lines.append(f"  趋势：[DOWN] 下降中 ({trend:.1f})")
            else:
                lines.append(f"  趋势：[STABLE] 稳定 (±{abs(trend):.1f})")
    else:
        lines.append("  （尚无评分数据）")

    lines.extend(
        [
            "",
            "【Token 消耗】",
            f"  累计：{total_tokens:,} tokens",
            f"  费用：¥{total_cost:.4f}",
        ]
    )

    # Token 趋势
    if len(chapter_usages) >= _MIN_TOKEN_TREND_WINDOW:
        recent_tokens = [u.get("total_tokens", 0) for u in chapter_usages[-5:]]
        avg_tokens = sum(recent_tokens) / len(recent_tokens)
        lines.append(f"  最近5章平均：{avg_tokens:,.0f} tokens/章")

    if recent_errors:
        lines.extend(
            [
                "",
                "[WARN] 最近异常：",
            ]
        )
        for e in recent_errors:
            lines.append(f"  - {str(e)[:120]}")

    return "\n".join(lines)


async def analyze_with_llm(snapshot: str) -> str | None:
    """调用 LLM 分析监控快照，返回自然语言报告（v5.4: 异步化）。"""
    try:
        from novelfactory.agents.infra.async_retry import async_llm_call_with_retry

        llm = get_supervisor_llm()
        messages = [
            SystemMessage(content=_get_analysis_prompt()),
            HumanMessage(content=f"请分析以下创作数据：\n\n{snapshot}"),
        ]

        async def _invoke():
            result = await llm.ainvoke(messages)
            return {"messages": [result]}

        result = await async_llm_call_with_retry(
            _invoke,
            step_name="intelligent_monitor",
            timeout_seconds=120,
        )
        msgs = result.get("messages", []) if isinstance(result, dict) else []
        if msgs:
            content = (
                msgs[-1].content if hasattr(msgs[-1], "content") else str(msgs[-1])
            )
            return content.strip() if content else "(分析结果为空)"
        return "(分析结果为空)"
    except Exception as e:
        logger.warning("[monitor] LLM analysis failed: %s", e)
        return None


def push_to_feishu(report: str) -> bool:
    """推送监控报告到飞书。"""
    # v6.1: 统一从 settings 读取
    from novelfactory.config.settings import settings as _st

    target = _st.FEISHU_USER_OPEN_ID or os.environ.get("FEISHU_USER_OPEN_ID", "")
    if not target:
        logger.warning("[monitor] 未配置 FEISHU_USER_OPEN_ID，跳过飞书推送")
        return False
    header = f"[Monitor Report] {datetime.now().strftime('%H:%M')}\n\n"
    id_type = "chat_id" if target.startswith("oc_") else "open_id"
    return send_lark_message(target, header + report, receive_id_type=id_type)


def is_milestone_chapter(chapter: int, total: int) -> bool:
    """判断当前章节是否里程碑（10%、25%、50%、75%、90%、100%）。

    v6.0.1 fix: 对小数 chapter 目标使用 math.ceil 确保至少触发 1。
    例如 total=7 时，ceil(7 * 0.1) = 1 而非 int(7 * 0.1) = 0。
    """
    import math

    ratios = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    for r in ratios:
        milestone = max(1, math.ceil(total * r))
        if chapter == milestone:
            return True
    return False


# ── LangGraph 节点 ─────────────────────────────────────


async def intelligent_monitor_node(
    state: dict,
    config: RunnableConfig | None = None,
) -> dict:
    """LangGraph 监控节点 —— 分析写作状态并推送智能报告。

    插入位置：
      writing_crew → [intelligent_monitor_node] → main_supervisor

    行为：
      - 每章完成后自动执行 LLM 分析
      - 仅在里程碑章节（10%/25%/50%/75%/90%/100%）或质量异常时推送飞书
      - 分析结果写入 state['_last_monitor_report'] 供调试
      - 绝不阻塞主流程（所有异常内部消化）
      - 每章完成后 fire-and-forget 清理旧检查点

    Returns:
      携带 monitor 信息的 state update dict。
    """
    updates: dict = {}
    current_ch = state.get("current_chapter", 1)
    total = state.get("target_chapters") or 500
    phase = state.get("current_phase", "setup")
    completed_ch = (
        current_ch - 1
    )  # 刚完成的章节（current_chapter 已被 writing_crew +1）

    # 仅在 writing/sync 阶段执行分析
    if phase not in ("writing", "sync"):
        return updates

    # 构建快照
    snapshot = build_monitoring_snapshot(state)
    logger.debug("[monitor] snapshot built (%d chars)", len(snapshot))

    # 判断是否需要推送
    should_push = False
    reason = ""

    # 条件1：里程碑章节（用刚完成的章节号判定）
    if is_milestone_chapter(completed_ch, total):
        should_push = True
        pct = int(completed_ch / max(total, 1) * 100)
        reason = f"里程碑 {pct}%"

    # 条件2：质量异常（最近连续两章 < 60）
    completed = state.get("completed_chapters", [])
    recent_scores = [
        ch.get("quality_score", 0) for ch in completed[-3:] if ch.get("quality_score")
    ]
    if len(recent_scores) >= _MIN_QUALITY_ANOMALY_WINDOW and all(
        s < _QUALITY_ANOMALY_THRESHOLD
        for s in recent_scores[-_MIN_QUALITY_ANOMALY_WINDOW:]
    ):
        should_push = True
        reason = f"质量预警（连续{len(recent_scores)}章<60分）"

    # 条件3：首次分析（第1章完成时，current_ch=2 意味着第1章已完成）
    if current_ch == _FIRST_ANALYSIS_CHAPTER + 1 and not state.get(
        "_first_analysis_done"
    ):
        should_push = True
        reason = "首次分析"
        updates["_first_analysis_done"] = True

    if not should_push:
        logger.debug("[monitor] ch%d skipped (not milestone/not anomaly)", completed_ch)
        return updates

    # LLM 分析 (v5.4: 异步化)
    report = await analyze_with_llm(snapshot)
    if report:
        prefixed = f"[Monitor] {reason}\n\n{report}"
        ok = push_to_feishu(prefixed)
        logger.info("[monitor] pushed for ch%d (%s): ok=%s", completed_ch, reason, ok)
    else:
        # LLM 失败时的兜底：直接推送快照
        push_to_feishu(f"{reason}\n\n{snapshot}")

    # 写入 state 供调试
    updates["_last_monitor_report"] = report or snapshot
    updates["_last_monitor_chapter"] = completed_ch

    # ── Chat UI message ──
    updates["messages"] = [
        AIMessage(
            content=f"[Status] {completed_ch}/{total} ({completed_ch * 100 // max(total, 1)}%) | "
            f"{reason}",
            name="intelligent_monitor",
        )
    ]

    # ── Checkpoint GC ──
    # v6.1: agc_checkpoints removed (official AsyncPostgresSaver doesn't support
    # per-checkpoint deletion). Use cleanup_thread_full for terminal-phase cleanup.

    return updates


def build_monitor_node() -> dict:
    """构建监控节点的外观函数（兼容 graph.add_node 调用约定）。

    LangGraph 的节点函数签名是 (state) -> dict，
    所以 intelligent_monitor_node 可以直接用。
    """
    return {"node": intelligent_monitor_node}
