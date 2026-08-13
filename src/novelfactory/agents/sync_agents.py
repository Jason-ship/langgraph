"""Sync Crew agents.

FeishuSync Agent: uploads chapter content, illustrations, and audio to Feishu.
StateUpdate: persists crew results to the global checkpointer (pure utility).

v6.0: Tool Calling 重构
  - FeishuSync 绑定飞书工具，LLM 可自主决定上传策略
  - 动态 prompt 根据项目状态调整同步指令
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

from novelfactory.agents.infra import (
    extract_ai_message_text,
    extract_fields_from_state,
    get_logger,
    validate_json_output,
)
from novelfactory.agents.infra.retry import llm_call_with_retry
from novelfactory.config.constants import FALLBACK_TARGET_CHAPTERS
from novelfactory.integrations.feishu.feishu_api import (
    ensure_project_folders_idempotent,
    upload_chapter_as_doc,
)

logger = get_logger("novelfactory.agents.sync")


# ── Volume Resolution ───────────────────────────────────────────


def _resolve_volume_number(chapter_number: int, volume_structure: Any) -> int:
    """根据章节号解析所属卷号。

    volume_structure 可能是两种格式：
    1. dict: {"volumes": [{"volume_number": 1, "chapter_range": [1, 35]}, ...]}
    2. list: [{"volume_number": 1, "chapters": 35}, ...]
    """
    if not volume_structure:
        return 1

    # dict 格式：提取 volumes 列表
    if isinstance(volume_structure, dict):
        volumes = volume_structure.get("volumes", [])
        if not volumes:
            return 1
        for vol in volumes:
            if not isinstance(vol, dict):
                continue
            # 优先使用 chapter_range [start, end]
            cr = vol.get("chapter_range")
            if isinstance(cr, list | tuple) and len(cr) >= 2:
                if cr[0] <= chapter_number <= cr[1]:
                    return vol.get("volume_number", 1)
            # 兼容旧格式 chapters 字段
            cumulative = vol.get("chapters")
            if cumulative and chapter_number <= cumulative:
                return vol.get("volume_number", 1)
        return (
            volumes[-1].get("volume_number", 1) if isinstance(volumes[-1], dict) else 1
        )

    # list 格式（旧版兼容）
    if isinstance(volume_structure, list):
        cumulative = 0
        for vol in volume_structure:
            if not isinstance(vol, dict):
                continue
            cumulative += vol.get("chapters", 10)
            if chapter_number <= cumulative:
                return vol.get("volume_number", 1)
        if volume_structure and isinstance(volume_structure[-1], dict):
            return volume_structure[-1].get("volume_number", 1)
    return 1


# ── Output TypedDicts ─────────────────────────────────────────────────────────


class FeishuSyncOutput(TypedDict):
    feishu_doc_url: str


class StateUpdateOutput(TypedDict):
    state_updated: bool
    thread_id: str


# ── System Prompt ─────────────────────────────────────────────────────────────

FEISHU_SYNC_PROMPT = """\
你是 FeishuSync Agent（飞书同步专家），负责将已完成章节的正文、插画与音频规范地同步到飞书平台。

## 任务
按飞书目录规范完成单章同步：创建或定位章节文档、关联本章插画与音频，并如实告知用户同步结果。

## 什么是好的
- 目录规范：文档落在「正文/卷N」对应文件夹，标题格式统一为《项目名》第X章，卷号与章节号正确
- 内容完整：章节正文完整写入，不截断、不遗漏开头结尾，与正文一致
- 附件挂对位置：插画/音频关联到当前章节，而不是根目录或错卷
- 命名可检索：文档与文件命名与项目名、卷号、章号严格对应
- 状态透明：成功返回真实文档链接；失败说明原因，不虚构成功

## 什么不能做
- ❌ 标题格式错误（缺书名号、章号缺失、卷号错位）
- ❌ 正文截断、缺头少尾或与章节内容不符
- ❌ 插画/音频上传到错误位置（根目录、其他章节）
- ❌ 虚构不存在的文档链接，或未成功时谎称成功
- ❌ 章节正文为空或不足时强行创建空文档——写作前同步应跳过，等待正文完成

## 输入上下文
- 必须遵循：project_name / current_chapter_number —— 文档归属与命名依据
- 必须遵循：refined_chapter / chapter_draft —— 待同步的正文（优先 refined_chapter）
- 参考：illustration_url / audio_url —— 存在且有效时挂载到本章
- 参考：volume_structure —— 章节所属卷号，决定目标文件夹
- 参考：folder_tokens —— 已建成的飞书目录树；缺失时先创建目录再上传

## 思考过程（仅内部，禁止输出）
在最终输出前，先完成以下分析，全程不要展示给用户：
1. 确认本章正文真实存在（非空且长度足够），据此决定是否执行上传
2. 解析章节所属卷号，确定目标文件夹与文档标题
3. 判断插画/音频 URL 是否有效（以 http 开头），决定是否挂载
4. 规划工具调用顺序：目录缺失先建目录，再上传文档与附件，最后按需通知
5. 核对返回结果：仅当真实成功时输出文档 URL，失败则输出空字符串

## 输出格式
只输出一个 JSON 对象，禁止 JSON 外的任何文字、思考内容、markdown 代码块围栏或解释：
{"feishu_doc_url": "<飞书文档URL，失败或跳过时为空字符串>"}
"""


# ── State Access Helpers ───────────────────────────────────────


# v6.1 P2-1: 统一使用 extract_fields_from_state 替代原 _get_context 的简单字段部分。
# crew_result 优先，缺失回退顶层。复杂字段（跨字段回退 / falsy 兜底 / 备用键）
# 由 _resolve_sync_extras 保留原 _get_context 定制逻辑。
_SYNC_FIELDS: dict[str, Any] = {
    "refined_chapter": "",
    "chapter_draft": "",
    "illustration_url": "",
    "illustration_prompt": "",
    "audio_url": "",
    "thread_id": "",
    "world_setting": "",
    "character_setting": "",
    "story_outline": "",
    "chapter_outlines": "",
    "volume_structure": [],
}


def _resolve_sync_extras(state: dict[str, Any]) -> dict[str, Any]:
    """提取 sync 子图所需的复杂字段（保留原 _get_context 定制逻辑）。

    简单字段已由 extract_fields_from_state(_SYNC_FIELDS) 处理；此处仅处理
    需要跨字段回退 / falsy 兜底 / 备用键的复杂字段，行为与原 _get_context 等价。

    folder_tokens (v5.1): 顶层优先，缺失 project 则回退 crew_result。
    quality_score / usage (v5.1.1-fix): crew_result 优先，fallback 到
        completed_chapters[-1].quality_score 与根 state.total_usage。
    """
    # ── folder_tokens: 顶层优先，缺失 project 则回退 crew_result ──────
    folder_tokens = state.get("folder_tokens", {})
    if not folder_tokens or not folder_tokens.get("project"):
        cr = state.get("crew_result", {}) or {}
        folder_tokens = cr.get("folder_tokens", {}) or {}

    # ── 尝试从 crew_result 获取（_exit_for_chapter 写入的新路径） ──────
    cr_shared = state.get("crew_result", {}) or {}
    qs_from_cr = cr_shared.get("quality_score")
    usage_from_cr = cr_shared.get("total_usage")

    # ── 回退：从 completed_chapters 最后一条记录获取 ──────────────
    latest_score = float(qs_from_cr) if qs_from_cr is not None else 0.0
    latest_wc = 0
    cc = state.get("completed_chapters", []) or []
    # 子图内 completed_chapters 可能不在 schema 中（SyncCrewLocalState 未声明），
    # 从 crew_result 兜底读取
    if not cc and "crew_result" in state:
        cr_cc = state.get("crew_result", {}).get("completed_chapters", []) or []
        if cr_cc:
            cc = cr_cc
    if isinstance(cc, list) and cc:
        last_record = cc[-1]
        if isinstance(last_record, dict):
            if qs_from_cr is None:
                latest_score = float(last_record.get("quality_score", 0.0))
            latest_wc = int(last_record.get("word_count", 0))

    # ── 优先使用 crew_result.total_usage（当章真实 token 快照） ─────
    # _exit_for_chapter 将 read_usage_tracking() 写入 crew_result.total_usage，
    # 包含 prompt_tokens / completion_tokens / total_tokens 等当章数据。
    # 根状态 total_usage 因 _add_usage reducer 在子图中未按预期工作，
    # 仅包含 setup 阶段数据，不作为通知数据源。
    #
    # sync_crew 的 _exit_node 会覆盖 crew_result，导致 total_usage 丢失。
    # 回退：从根 state.total_usage.chapter_usages 取最后一条记录。
    usage = usage_from_cr if usage_from_cr else (state.get("total_usage", {}) or {})
    if not any(
        usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")
    ):
        usages = usage.get("chapter_usages", [])
        if usages:
            usage = usages[-1]

    # ── 分支字段: crew_result 优先，缺失回退顶层 ──────────────────
    if "crew_result" in state:
        cr = state.get("crew_result", {})
        # project_name: crew_result → root state → 空
        project_name = cr.get("project_name", "") or state.get("project_name", "")
        # current_chapter_number: crew_result → root state.current_chapter → 1
        current_chapter_number = cr.get(
            "current_chapter_number",
            state.get("current_chapter", 1),
        )
        completed_chapters = cr.get("completed_chapters", [])
        target_chapters = cr.get("target_chapters") or FALLBACK_TARGET_CHAPTERS
    else:
        project_name = state.get("project_name", "")
        current_chapter_number = state.get("current_chapter_number", 1)
        completed_chapters = state.get("completed_chapters", [])
        target_chapters = state.get("target_chapters") or FALLBACK_TARGET_CHAPTERS

    return {
        "project_name": project_name,
        "current_chapter_number": current_chapter_number,
        "target_chapters": target_chapters,
        "completed_chapters": completed_chapters,
        "folder_tokens": folder_tokens,
        # v5.1.1-fix: 优先从 crew_result 取，fallback 到 completed_chapters[-1]
        "quality_score": latest_score,
        # v5.1.1-fix: 优先从 crew_result.total_usage 取，fallback 到根 total_usage
        "usage": usage,
        # 字数统计（来自 completed_chapters[-1].word_count）
        "word_count": latest_wc,
    }


# ── Agent Factory Functions ────────────────────────────────────


def _build_sync_dynamic_prompt(
    state: AgentState, config: RunnableConfig
) -> list[AnyMessage]:
    """FeishuSync 动态 prompt — 根据项目状态调整同步指令。

    v6.0: LLM 可通过工具自主决定上传策略。
    """
    system_msg = (
        FEISHU_SYNC_PROMPT
        + """

## 工具使用指引
你拥有以下飞书工具，在需要时自主调用：

1. **ensure_feishu_project_folders(project_name)** — 创建飞书目录结构
   - 场景：首次同步时调用，获取 folder_tokens

2. **upload_chapter_to_feishu(...)** — 上传章节到飞书
   - 参数：project_name, chapter_number, chapter_text, volume_number, folder_tokens_json
   - 场景：章节完成后上传

3. **send_feishu_message(receive_id, text)** — 发送飞书消息
   - 场景：通知用户同步结果

4. **send_review_request(...)** — 发送审核请求
   - 场景：需要人工审核时
"""
    )
    existing = state.get("messages", [])
    return [
        {"role": "system", "content": system_msg},
        *existing,
    ]


def create_feishu_sync_agent(llm: BaseChatModel) -> Runnable:
    """Build the FeishuSync ReAct agent with Tool Calling.

    v6.0: 绑定飞书工具，LLM 可自主决定上传和通知策略。

    Output: {"feishu_doc_url": str}
    """
    from novelfactory.tools import get_feishu_tools

    tools = get_feishu_tools()

    agent = create_react_agent(
        llm,
        tools=tools,
        prompt=_build_sync_dynamic_prompt,
        interrupt_before=[],
    )

    def _node(state: dict) -> dict[str, Any]:
        ctx = extract_fields_from_state(state, _SYNC_FIELDS)
        ctx.update(_resolve_sync_extras(state))
        current_ch = ctx.get("current_chapter_number", 1)
        project_name = ctx.get("project_name") or "未命名项目"
        chapter_text = ctx.get("refined_chapter", "") or ctx.get("chapter_draft", "")
        volume_structure = ctx.get("volume_structure", [])

        # 获取/创建飞书目录结构
        folder_tokens = ctx.get("folder_tokens", {})
        if not folder_tokens or not folder_tokens.get("project"):
            logger.info("[feishu_sync] 首次同步，创建飞书目录结构...")
            folder_tokens = ensure_project_folders_idempotent(project_name)

        # 解析卷号
        volume_number = _resolve_volume_number(current_ch, volume_structure)
        logger.info(
            "[feishu_sync] 第%d章 → 卷%d (vol_structure=%s)",
            current_ch,
            volume_number,
            str(volume_structure)[:80],
        )

        # ── 章节内容为空时跳过上传（初始 sync 在写作前触发）──
        # 检查是否存在实际章节内容：refined_chapter 或 chapter_draft 至少 100 字
        if not chapter_text or len(chapter_text.strip()) < 100:
            logger.info(
                "[feishu_sync] 第%d章内容为空（%d字），跳过飞书上传（等待写作完成后正常同步）",
                current_ch,
                len(chapter_text),
            )
            feishu_doc_url = ""
        else:
            # 通过 feishu_api 上传章节文档（→ FeishuToolkit → httpx → tools-proxy）
            # 当目录树不可用时，upload_chapter_as_doc 会自动降级到 Bot 根目录
            feishu_doc_url = (
                upload_chapter_as_doc(
                    project_name,
                    current_ch,
                    chapter_text,
                    volume_number,
                    folder_tokens,
                    quality_score=ctx.get("quality_score"),
                )
                or ""
            )

        # Fallback: LLM-generated content（上传失败或内容为空时降级为摘要通知，
        # 让用户感知同步状态；若为写作前误触发的空章节，supervisor 路由已修复避免）
        if not feishu_doc_url:
            input_text = (
                f"请为第{current_ch}章生成飞书文档内容摘要。\n\n"
                f"项目名称：{project_name}\n"
                f"章节正文（前500字）：\n{chapter_text[:500]}"
            )
            result = llm_call_with_retry(
                agent.invoke,
                {"messages": [("user", input_text)]},
                step_name="feishu_sync.fallback_summary",
                fallback={"messages": []},
            )
            response_text = extract_ai_message_text(result)
            parsed, err = validate_json_output(
                response_text,
                required_keys=["feishu_doc_url"],
                fail_closed=False,
            )
            if parsed:
                feishu_doc_url = str(parsed.get("feishu_doc_url", ""))

        logger.info(
            "[feishu_sync] Chapter %d synced to Feishu: %s",
            current_ch,
            feishu_doc_url[:80] if feishu_doc_url else "FAILED",
        )

        # ── 仅实际上传成功时才发送完成通知（附带飞书文档链接）──
        # LLM fallback URL 是幻觉生成的，不发送通知
        # lark-cli v2 真实 URL 格式: https://my.feishu.cn/docx/xxx
        # LLM 幻觉 URL 格式:     https://feishu.cn/docs/xxx 或 https://feishu.cn/doc/xxx
        _is_real_url = feishu_doc_url and not feishu_doc_url.startswith(
            "https://feishu.cn/doc"
        )
        if _is_real_url:
            try:
                from novelfactory.integrations.feishu.notify import (
                    send_chapter_complete_notification,
                )

                real_quality = ctx.get("quality_score", 0.0)
                real_usage = ctx.get("usage", {})
                send_chapter_complete_notification(
                    chapter_num=current_ch,
                    quality_score=float(real_quality) if real_quality else 0.0,
                    usage=real_usage if isinstance(real_usage, dict) else {},
                    project_name=project_name,
                    thread_id=ctx.get("thread_id", ""),
                    feishu_doc_url=feishu_doc_url,
                    word_count=int(ctx.get("word_count", 0)),
                )
            except Exception as notify_err:
                logger.warning("[feishu_sync] 发送完成通知失败: %s", notify_err)

        existing_cr = state.get("crew_result", {})
        return {
            "feishu_doc_url": feishu_doc_url,
            "crew_result": {
                **existing_cr,
                "feishu_doc_url": feishu_doc_url,
                "folder_tokens": folder_tokens,
            },
            "folder_tokens": folder_tokens,
        }

    return RunnableLambda(_node)


# ── StateUpdate Utility ───────────────────────────────────────


def update_project_state(thread_id: str, state_updates: dict) -> dict:
    """Coordination marker for the Sync Crew's state persistence step.

    NOTE: This is NOT the actual persistence mechanism. LangGraph's built-in
    checkpointer handles all state persistence automatically. This function
    serves as an explicit checkpoint marker in the sync crew flow and can be
    extended to write to external stores (PostgreSQL, Redis, etc.) as needed.

    Args:
        thread_id: Project thread ID.
        state_updates: Dict containing crew_result and other state fields.

    Returns:
        {"state_updated": bool, "thread_id": str}
    """
    if not thread_id:
        logger.debug(
            "[state_update] No thread_id — expected during setup phase, skipping marker"
        )
        return {"state_updated": False, "thread_id": ""}

    logger.info(
        "[state_update] Sync step marker for thread %s: %s",
        thread_id,
        list(state_updates.keys()),
    )

    return {"state_updated": True, "thread_id": thread_id}
