"""质量参数调优对话子代理。

接收用户反馈，自动定位参数，生成变更方案，可选时间旅行回溯验证。

能力：
  1. 自动参数定位：根据用户自然语言反馈匹配反馈-参数映射表
  2. 参数变更：解析 LLM 输出的 <params_update> JSON 块并应用
  3. 时间旅行验证：用新参数重新评审历史章节，对比调参前后效果
  4. 自动回滚：验证结果变差时自动回滚参数变更

版本：v1.0.0
创建日期：2026-08-12
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage

from novelfactory.agents.infra import async_llm_call_with_retry
from novelfactory.config.llm import get_worker_llm
from novelfactory.config.quality_params import quality_center
from novelfactory.evaluation.replay import (
    CheckpointReplayService,
    ReplayReport,
)
from novelfactory.graph.chat.agents.utils import get_last_user_message
from novelfactory.state.lead_agent_state import LeadAgentState

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Prompt 模板
# ═══════════════════════════════════════════════════════════════════════════════


def _build_tuner_prompt(
    user_msg: str,
    snapshot: dict[str, Any],
    history: list[dict[str, Any]],
    matched_info: str,
) -> str:
    """构建调优 Agent 的 Prompt。"""
    # 只展示当前值和默认值，精简快照避免 token 爆炸
    snapshot_lines = []
    for key, info in snapshot.items():
        marker = " (已修改)" if info.get("modified") else ""
        snapshot_lines.append(
            f"  - {key}: {info['value']} (默认 {info['default']}, 范围 {info['min']}-{info['max']})"
            f"{marker} | {info['description']}"
        )
    snapshot_text = "\n".join(snapshot_lines)

    history_text = "无变更记录"
    if history:
        history_lines = []
        for h in history:
            history_lines.append(
                f"  - [{h.get('change_id', '')}] {h.get('key', '')}: "
                f"{h.get('old_value')} -> {h.get('new_value')} (原因: {h.get('reason', '')})"
            )
        history_text = "\n".join(history_lines)

    return f"""你是 NovelFactory 的质量参数调优助手。你可以查看和调整小说创作系统的质量评审参数，
还能通过"时间旅行"功能用新参数重新评审历史章节来验证调整效果。

## 当前参数快照
{snapshot_text}

## 最近变更历史
{history_text}

## 自动定位结果
根据用户反馈关键词预匹配到的相关参数：
{matched_info}

## 操作规则

### 规则 1：理解用户反馈
分析用户反馈，确定问题类别和调整方向。如果预匹配结果为空，根据你的理解自行选择最相关的参数。

### 规则 2：生成参数变更方案
当需要调整参数时，输出 JSON 操作块：
<params_update>
{{"verdict.pass_threshold": 75.0, "unified.max_retries": 2}}
</params_update>

重要约束：
- v8.2 统一 LLM 评审后，可调参数以阈值/迭代类为主（verdict.pass_threshold、verdict.refine_threshold、verdict.iteration_bonus.*、iteration.max_rewrite、iteration.max_refine_mid、unified.max_retries、unified.fallback_score）。
- 已移除融合权重类参数（verdict.weights.*、ai_style.weights.* 等），不要再输出此类 key。
- 数值必须落在参数快照标注的范围内。
- 如果只是查看/咨询，不要输出 <params_update> 块。

### 规则 3：时间旅行验证
当用户要求验证效果（如"验证一下""用最近N章验证""回溯重评"），
或你判断本次变更影响较大时，**必须**输出 <replay_request> 块：
<replay_request>
{{"action": "replay_recent", "count": 3}}
</replay_request>
或指定章节：
<replay_request>
{{"action": "replay_chapter", "chapter_number": 5}}
</replay_request>

强制要求（防止只口头承诺不执行）：
- 只要你的回复中打算提及"验证/复评/回溯/重新评审"相关结果，
  就**必须先**输出 <replay_request> 块，再写验证说明。
- 用户消息含"验证""回溯""重评"等词时，无论你是否调整参数，都必须输出该块。
- 不要输出空块或删除该块。系统将严格按照块内容执行时间旅行回溯。

### 规则 4：回复用户
用通俗易懂的语言解释：
1. 变更了什么参数，为什么
2. 时间旅行验证结果（如果执行了）
3. 预期对后续章节的影响

## 用户消息
{user_msg}
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM 输出解析
# ═══════════════════════════════════════════════════════════════════════════════

_PARAMS_UPDATE_RE = re.compile(
    r"<params_update>\s*(.*?)\s*</params_update>", re.DOTALL
)
_REPLAY_REQUEST_RE = re.compile(
    r"<replay_request>\s*(.*?)\s*</replay_request>", re.DOTALL
)

# 时间旅行验证意图关键词（用户消息或 LLM 回复命中时自动触发 replay）
_REPLAY_INTENT_KEYWORDS: tuple[str, ...] = (
    "时间旅行", "回溯", "重评", "复评", "验证效果", "验证一下",
    "重新评审", "回测", "用新参数", "用最新参数", "replay",
    "验证", "复看",
)


def _detect_replay_intent(text: str) -> dict[str, Any] | None:
    """检测文本中的时间旅行验证意图。

    匹配优先级：
    1. "最近N章" → replay_recent count=N
    2. "第N章" → replay_chapter chapter_number=N
    3. 通用验证关键词 → 默认 replay_recent count=3
    """
    for kw in _REPLAY_INTENT_KEYWORDS:
        if kw not in text:
            continue
        # "最近3章" / "最近 N 章"
        m = re.search(r"最近\s*(\d{1,3})\s*章", text)
        if m:
            return {"action": "replay_recent", "count": int(m.group(1))}
        # "第5章" / "第 N 章"
        m = re.search(r"第\s*(\d{1,3})\s*章", text)
        if m:
            return {"action": "replay_chapter", "chapter_number": int(m.group(1))}
        # 默认：最近 3 章
        return {"action": "replay_recent", "count": 3}
    return None


def _resolve_replay_request(user_msg: str, content: str) -> dict[str, Any] | None:
    """解析时间旅行请求。

    三级回退策略：
    1. LLM 结构化输出 <replay_request> 块 → 按块执行
    2. 用户消息含验证意图关键词 → 自动构造默认请求
    3. LLM 回复中承诺验证（含验证关键词）→ 自动构造默认请求

    确保用户说"验证一下效果"时即使 LLM 未输出结构化块也能自动触发。
    """
    # 1. LLM 结构化输出（最高优先级，可指定章节/数量）
    structured = _parse_replay_request(content)
    if structured:
        return structured

    # 2. 用户消息明确要求验证
    user_intent = _detect_replay_intent(user_msg)
    if user_intent:
        return user_intent

    # 3. LLM 回复承诺"将进行验证/复评"
    llm_intent = _detect_replay_intent(content)
    if llm_intent:
        return llm_intent

    return None


def _parse_params_update(content: str) -> dict[str, Any] | None:
    """解析 LLM 输出中的 <params_update> JSON 块。"""
    m = _PARAMS_UPDATE_RE.search(content)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError as e:
        logger.warning("[QualityTuner] Failed to parse params_update: %s", e)
    return None


def _parse_replay_request(content: str) -> dict[str, Any] | None:
    """解析 LLM 输出中的 <replay_request> JSON 块。"""
    m = _REPLAY_REQUEST_RE.search(content)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError as e:
        logger.warning("[QualityTuner] Failed to parse replay_request: %s", e)
    return None


def _format_matched_params(params: list[Any]) -> str:
    """格式化预匹配到的参数。"""
    if not params:
        return "无直接匹配，需要语义分析"
    lines = []
    for spec in params:
        lines.append(
            f"  - {spec.key}: {spec.description} (当前 {quality_center.get(spec.key)}, "
            f"范围 {spec.min_val}-{spec.max_val})"
        )
    return "\n".join(lines)


def _build_explanation(
    content: str,
    apply_msg: str,
    replay_reports: ReplayReport | list[ReplayReport] | None,
) -> str:
    """构建最终回复文本。

    优先使用 LLM 的回复正文；若 LLM 只输出操作块，则用程序化说明兜底。
    """
    # 去除 LLM 输出中的操作块，保留正文
    cleaned = _PARAMS_UPDATE_RE.sub("", content)
    cleaned = _REPLAY_REQUEST_RE.sub("", cleaned)
    cleaned = cleaned.strip()

    parts: list[str] = []
    if cleaned:
        parts.append(cleaned)
    if apply_msg:
        parts.append(apply_msg)
    if replay_reports:
        parts.append(_format_replay_reports(replay_reports))

    return "\n\n".join(parts) if parts else "已完成处理。"


def _format_replay_reports(
    reports: ReplayReport | list[ReplayReport],
) -> str:
    """格式化重评对比报告。"""
    if isinstance(reports, ReplayReport):
        reports = [reports]

    lines = ["\n━━━━ 时间旅行验证报告 ━━━━"]
    for r in reports:
        if r.success:
            lines.append(r.summary)
        else:
            lines.append(f"第{r.chapter_number}章：{r.error}")

    # 总体判断
    success_reports = [r for r in reports if r.success]
    if success_reports:
        improved = sum(1 for r in success_reports if r.is_improved)
        total = len(success_reports)
        if improved >= total / 2:
            lines.append(f"\n✅ 总体判断：{improved}/{total} 章评分提升，参数变更效果正面")
        else:
            lines.append(f"\n⚠️ 总体判断：仅 {improved}/{total} 章评分提升，建议关注")
    lines.append("━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  自动回滚判定
# ═══════════════════════════════════════════════════════════════════════════════


def _should_auto_rollback(
    reports: ReplayReport | list[ReplayReport],
) -> bool:
    """判断是否需要自动回滚参数变更。

    判定规则：
    - 多章：超过半数章节未改善 -> 回滚
    - 多章：任何章节从 PASS 变为 REWRITE -> 回滚
    - 单章：评分下降 > 5 分 -> 回滚
    """
    if isinstance(reports, ReplayReport):
        reports = [reports]

    success_reports = [r for r in reports if r.success]
    if not success_reports:
        return False

    # 任何章节从 PASS 变为 REWRITE -> 回滚
    for r in success_reports:
        if r.old_level.upper() == "PASS" and r.new_level.upper() == "REWRITE":
            logger.info("[QualityTuner] Auto-rollback: PASS -> REWRITE on chapter %d", r.chapter_number)
            return True

    worse_count = sum(1 for r in success_reports if not r.is_improved)
    total = len(success_reports)
    if worse_count > total / 2:
        logger.info("[QualityTuner] Auto-rollback: %d/%d chapters not improved", worse_count, total)
        return True

    # 单章评分下降 > 5 分
    if len(success_reports) == 1 and success_reports[0].delta < -5.0:
        logger.info("[QualityTuner] Auto-rollback: score dropped by %.1f", success_reports[0].delta)
        return True

    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  节点函数
# ═══════════════════════════════════════════════════════════════════════════════


async def quality_tuner_agent_node(state: LeadAgentState) -> dict[str, Any]:
    """质量参数调优对话节点。

    接收用户反馈，自动定位参数，生成变更方案，可选时间旅行验证，
    验证失败自动回滚。
    """
    llm = get_worker_llm()
    messages = state.get("messages", [])
    last_user_msg = get_last_user_message(messages)

    if not last_user_msg:
        return {
            "current_agent": "quality_tuner_agent",
            "messages": [
                AIMessage(
                    content=(
                        "我是质量参数调优助手。我可以帮你：\n\n"
                        "1. 🔧 **调整质量参数** — 如\"AI味太重了，加强检测\"\n"
                        "2. 📊 **查看参数** — 如\"当前融合权重是多少\"\n"
                        "3. ⏪ **时间旅行验证** — 如\"调整后用历史章节验证效果\"\n"
                        "4. ↩️ **回滚变更** — 如\"撤销刚才的调整\"\n\n"
                        "你想做什么？"
                    ),
                    name="quality_tuner_agent",
                )
            ],
        }

    # 1. 获取当前参数快照和变更历史
    snapshot = quality_center.get_snapshot()
    history = quality_center.get_history(limit=5)

    # 2. 自动参数定位（关键词预匹配）
    matched_params = quality_center.find_params_by_feedback(last_user_msg)
    matched_info = _format_matched_params(matched_params)

    # 3. 构建 Prompt
    prompt = _build_tuner_prompt(last_user_msg, snapshot, history, matched_info)

    # 4. LLM 调用（带重试保护）
    try:
        response = await async_llm_call_with_retry(
            llm.ainvoke, prompt,
            step_name="quality_tuner",
            retry_policy="default",
            timeout_seconds=180.0,
        )
        content = getattr(response, "content", None) or str(response)
    except Exception as e:
        logger.exception("[QualityTuner] LLM call failed: %s", e)
        return {
            "current_agent": "quality_tuner_agent",
            "messages": [
                AIMessage(
                    content=f"处理反馈时遇到问题: {e}，请重试。",
                    name="quality_tuner_agent",
                )
            ],
        }

    # 5. 解析参数变更并应用
    updates = _parse_params_update(content)
    apply_msg = ""
    change_ids: list[str] = []

    if updates:
        ok_results = quality_center.update_many(
            updates, reason=f"用户反馈: {last_user_msg[:100]}", operator="quality_tuner",
        )
        applied_keys = [k for k, ok in ok_results.items() if ok]
        rejected_keys = [k for k, ok in ok_results.items() if not ok]

        if applied_keys:
            applied_lines = []
            for k in applied_keys:
                spec_text = f"{k} = {quality_center.get(k)}"
                applied_lines.append(f"  - {spec_text}")
            apply_msg = "✅ 已应用参数变更：\n" + "\n".join(applied_lines)
            change_ids = [h["change_id"] for h in quality_center.get_history(limit=5)
                          if h["key"] in applied_keys][:1]
        if rejected_keys:
            apply_msg += f"\n⚠️ 以下参数校验失败（可能超出范围）：{rejected_keys}"

    # 6. 解析时间旅行请求并执行
    # 三级回退：LLM 结构化块 → 用户消息验证意图 → LLM 回复承诺验证。
    # 验证请求独立触发（无需本次有参数变更），确保"验证一下效果"必触发。
    replay_reports: ReplayReport | list[ReplayReport] | None = None
    replay_request = _resolve_replay_request(last_user_msg, content)

    if replay_request:
        thread_id = state.get("thread_id", "")
        if not thread_id:
            logger.warning("[QualityTuner] No thread_id for replay, skipping")
            no_thread_msg = (
                "\n\nℹ️ 已识别到验证请求，但当前对话未绑定项目线程（thread_id），"
                "无法执行时间旅行回溯。请通过项目对话发起验证，或在 Webhook 请求中传入 thread_id。"
            )
            if apply_msg:
                apply_msg += no_thread_msg
            else:
                apply_msg = no_thread_msg
        else:
            replay_service = CheckpointReplayService()
            action = replay_request.get("action", "replay_recent")
            try:
                if action == "replay_chapter":
                    replay_reports = await replay_service.replay_chapter(
                        thread_id, int(replay_request.get("chapter_number", 1))
                    )
                else:
                    replay_reports = await replay_service.replay_recent(
                        thread_id, count=int(replay_request.get("count", 3))
                    )
            except Exception as e:
                logger.exception("[QualityTuner] Replay failed: %s", e)
                replay_reports = None

            # 7. 如果验证结果变差，自动回滚（仅当本次会话有参数变更）
            if replay_reports and _should_auto_rollback(replay_reports):
                if change_ids:
                    for cid in change_ids:
                        quality_center.rollback(cid)
                elif updates:
                    # 兜底：回滚最近一次变更
                    quality_center.rollback_last()
                apply_msg += "\n\n⚠️ 时间旅行验证显示参数变更效果负面，已自动回滚。"
                replay_reports = None

    # 8. 构建回复
    explanation = _build_explanation(content, apply_msg, replay_reports)

    return {
        "current_agent": "quality_tuner_agent",
        "messages": [AIMessage(content=explanation, name="quality_tuner_agent")],
    }
