"""Media Crew ReAct agents.

Each agent is built with create_react_agent and a typed system prompt.
Agents are invoked by the Media Crew supervisor in parallel:

    illustrator + tts_generator  (independent, no dependencies)

All @tool wrappers are synchronous (LangGraph tool calling requirement).
"""

from __future__ import annotations

import subprocess
from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.prebuilt import create_react_agent

from novelfactory.agents.infra import (
    extract_ai_message_text,
    extract_fields_from_state,
    get_logger,
    llm_call_with_retry,
    validate_json_output,
)

logger = get_logger("novelfactory.agents.media")

_MIN_PROMPT_LEN = 10  # 有意义的 prompt 最小长度


# ── Output TypedDicts ─────────────────────────────────────────────────────────


class IllustratorOutput(TypedDict):
    illustration_url: str
    illustration_prompt: str


class TTSGeneratorOutput(TypedDict):
    audio_url: str


# ── System Prompts ─────────────────────────────────────────────────────────────

ILLUSTRATOR_PROMPT = """\
你是 Illustrator（插画生成师），负责为小说章节提炼最具视觉冲击力的场景，并产出可被文生图模型直接执行的高质量插画描述。

## 任务
阅读章节正文，选定 1 个情感张力最强或视觉冲击力最大的真实情节瞬间，输出：① 图像结果 URL；② 可复现该画面的英文画面描述。

## 什么是好的
- 忠于叙事：画面必须是本章真实发生的情节瞬间，人物动作与表情有原文依据，不是脱离剧情的"美图"
- 描述可执行：英文画面描述 100-200 words，按"主体 → 环境 → 构图 → 光线 → 风格"分层组织，文生图模型可逐条还原
- 人物忠于设定：服装、发型、气质、武器与角色设定一致，古装人物不会穿现代服装
- 氛围统一：光线与色彩服务于剧情情绪（诀别用冷色逆光、重逢用暖色侧光），且与世界观自洽
- 构图有张力：主体明确、前后景有层次、留白合理，适配 16:9 横版插画
- 场景可识别：读者看到插画能立刻认出是哪一段情节

## 什么不能做
- ❌ 生成与情节无关的风景/人像"美图"
- ❌ 画面中出现任何可读文字（书名、对话气泡、标题、水印）
- ❌ 人物与设定矛盾（如仙侠世界出现现代科技，角色服装发型与设定不符）
- ❌ 场景与世界观冲突（如修真界出现枪械、古代背景出现现代建筑）
- ❌ 一幅画塞入多个不相干场景，主体含糊
- ❌ 画面描述用中文或空泛词汇（如"漂亮的风景"）——必须是具体、可执行的英文

## 输入上下文
- 必须遵循：章节正文（refined_chapter / chapter_draft）——场景的唯一来源
- 必须遵循：角色设定（character_setting）——人物外观与气质不可违背
- 参考：世界观设定（world_setting）——背景氛围与时代风貌
- 参考：current_chapter_number —— 插画情绪需与章节在全书中的位置匹配

## 思考过程（仅内部，禁止输出）
在最终输出前，先完成以下分析，全程不要展示给用户：
1. 通读章节，定位情感/视觉峰值段落，确认该场景确实是情节的一部分
2. 明确画面构成：谁、在哪、做什么、什么情绪
3. 对照角色设定与世界观设定，检查一致性
4. 设计画面：构图、光线、色彩、艺术风格（如 Chinese ink painting / epic fantasy / cinematic），按"主体-环境-光线-风格"组织英文描述
5. 准备失败预案：图像生成可能失败，但画面描述必须始终给出，便于复现与重试

## 输出格式
只输出一个 JSON 对象，禁止 JSON 外的任何文字、思考内容、markdown 代码块围栏或解释：
{"illustration_url": "<图像URL，失败时为空字符串>", "illustration_prompt": "<英文画面描述，100-200 words，可直接用于文生图模型>"}
"""


TTS_GENERATOR_PROMPT = """\
你是 TTSGenerator（语音生成师），负责为小说章节挑选最适合朗读的内容并产出有声文本，让听众"用耳朵读小说"。

## 任务
从章节正文中选出对话丰富、情感充沛的片段作为朗读内容，兼顾章节标题朗读与角色声音差异化，输出音频结果 URL。

## 什么是好的
- 选段抓人：优先对话密集、情绪饱满的段落（冲突、告白、揭底），而非大段环境描写与说明性旁白
- 听感自然：朗读文本适合出声朗读——句子短促有停顿、口语化，删掉仅适合阅读的书面修饰与冗长铺陈
- 结构完整：保留章节标题；选段有起有收，单独收听也成立，不突然断在对话半句
- 角色有辨识度：男女角色、主角配角通过语速/语调/情绪区分，同一角色前后一致
- 篇幅克制：选段聚焦最精彩的 2000 字以内，不贪多求全，避免音频冗长拖沓
- 与小说类型匹配：选段情绪符合题材气质（玄幻重张力、都市重日常感），不出现风格错位

## 什么不能做
- ❌ 全章逐句朗读，包括所有环境描写与过渡段，导致音频冗长
- ❌ 选段缺乏冲突或情绪起伏，通篇平淡无起伏
- ❌ 朗读文本残留 markdown 符号、编号、书名号装饰等不适配语音的内容
- ❌ 省略章节标题
- ❌ 所有角色统一音色、语速与情绪，听众无法区分谁在说话
- ❌ 无视小说类型选择朗读风格（如玄幻小说用现代新闻腔）

## 输入上下文
- 必须遵循：章节正文（refined_chapter / chapter_draft）——朗读内容的唯一来源
- 必须遵循：current_chapter_number —— 标题朗读与选段定位依据
- 参考：章节内对话与旁白分布——对话密集处优先

## 思考过程（仅内部，禁止输出）
在最终输出前，先完成以下分析，全程不要展示给用户：
1. 通读章节，圈出对话密集、情感充沛的段落，剔除旁白堆砌段
2. 判断哪些对话推动情节或揭示人物，作为必选内容
3. 精简书面化表述，改为适合出声朗读的口语化句式，确认无 markdown/特殊符号残留
4. 设计角色声音：主角、配角、旁白在音色与语速上的差异化
5. 检查选段完整性：起止自然，章节标题已保留

## 输出格式
只输出一个 JSON 对象，禁止 JSON 外的任何文字、思考内容、markdown 代码块围栏或解释：
{"audio_url": "<音频URL，生成失败时为空字符串>"}
"""


# ── State Access Helpers ───────────────────────────────────────────────────────

# v6.1 P2-1: 统一使用 extract_fields_from_state 替代原 _get_context。
# crew_result 优先，缺失回退顶层。
_MEDIA_FIELDS: dict[str, Any] = {
    "refined_chapter": "",
    "chapter_draft": "",
    "world_setting": "",
    "character_setting": "",
    "current_chapter_number": 1,
    "project_name": "",
}


# ── Image Generation Helper ───────────────────────────────────────────────────


def _generate_image_via_matrix(prompt: str, project_name: str, chapter: int) -> str:
    """Generate image via matrix MCP.

    Uses subprocess to call mavis mcp call matrix matrix_generate_image.
    Returns URL or empty string on failure.
    """
    import json

    # Build MCP call command
    # mavis mcp call matrix matrix_generate_image '{"prompt": "...", ...}'
    cmd = [
        "mavis",
        "mcp",
        "call",
        "matrix",
        "matrix_generate_image",
        "--arg",
        json.dumps(
            {
                "prompt": prompt,
                "model": "MiniMax-Image-01",
                "aspect_ratio": "16:9",
                "resolution": "1280x720",
            }
        ),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode == 0 and result.stdout:
            # Parse output — may be JSON or plain URL
            try:
                data = json.loads(result.stdout.strip())
                return data.get("url", "")
            except Exception:
                # May return URL directly
                return result.stdout.strip()
        logger.warning("[illustrator] matrix_generate_image failed: %s", result.stderr)
    except Exception as e:
        logger.warning("[illustrator] Image generation error: %s", e)

    return ""


# ── TTS Generation Helper ─────────────────────────────────────────────────────


def _generate_tts_via_matrix(text: str, project_name: str, chapter: int) -> str:
    """Generate TTS via matrix MCP.

    Uses subprocess to call mavis mcp call matrix matrix_batch_text_to_audio.
    Returns URL or empty string on failure.
    """
    import json

    # Truncate text to reasonable length for TTS
    tts_text = text[:3000]

    cmd = [
        "mavis",
        "mcp",
        "call",
        "matrix",
        "matrix_batch_text_to_audio",
        "--arg",
        json.dumps(
            {
                "text": tts_text,
                "model": "MiniMax-TTS",
                "voice_id": "male-qn-qingse",
                "output_format": "mp3",
                "speed": 1.0,
            }
        ),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode == 0 and result.stdout:
            try:
                data = json.loads(result.stdout.strip())
                return data.get("url", "")
            except Exception:
                return result.stdout.strip()
        logger.warning("[tts] matrix_batch_text_to_audio failed: %s", result.stderr)
    except Exception as e:
        logger.warning("[tts] TTS generation error: %s", e)

    return ""


# ── Agent Factory Functions ────────────────────────────────────────────────────


def create_illustrator_agent(llm: BaseChatModel) -> Runnable:
    """Build the Illustrator ReAct agent.

    Output: {"illustration_url": str, "illustration_prompt": str}"""
    agent = create_react_agent(
        llm,
        tools=[],
        prompt=ILLUSTRATOR_PROMPT,
        interrupt_before=[],
    )

    def _node(state: dict) -> dict[str, Any]:
        ctx = extract_fields_from_state(state, _MEDIA_FIELDS)
        current_ch = ctx.get("current_chapter_number", 1)
        chapter_text = ctx.get("refined_chapter", "") or ctx.get("chapter_draft", "")

        if not chapter_text:
            logger.warning("[illustrator] No chapter text to illustrate")
            return {
                "illustration_url": "",
                "illustration_prompt": "章节内容为空，无法生成插画",
            }

        logger.info("[illustrator] Generating illustration for chapter %d", current_ch)

        input_text = (
            f"请为第{current_ch}章生成插画。\n\n"
            f"【章节正文】\n{chapter_text[:3000]}\n\n"
            f"【世界观设定】\n{ctx['world_setting'][:1000]}\n\n"
            f"【角色设定】\n{ctx['character_setting'][:500]}\n\n"
            f"项目名称：{ctx['project_name']}"
        )

        result = llm_call_with_retry(
            agent.invoke,
            {"messages": [("user", input_text)]},
            step_name="illustrator_agent",
            timeout_seconds=180,
            fallback={"messages": [], "crew_result": {}},
        )
        response_text = extract_ai_message_text(result)

        # Parse JSON from response (fail-open: fallback to response_text on parse error)
        parsed, err = validate_json_output(
            response_text,
            required_keys=["illustration_url", "illustration_prompt"],
            fail_closed=False,
        )
        if parsed:
            illustration_url = str(parsed.get("illustration_url", ""))
            illustration_prompt = str(parsed.get("illustration_prompt", ""))
        else:
            # Fallback: try to extract from response text
            illustration_prompt = response_text or "生成失败"
            illustration_url = ""

        # If no URL but we have a prompt, try to generate the image
        if illustration_url and illustration_url.startswith("http"):
            pass  # Already has URL
        elif illustration_prompt and len(illustration_prompt) > _MIN_PROMPT_LEN:
            # Try matrix generation
            matrix_url = _generate_image_via_matrix(
                illustration_prompt,
                ctx.get("project_name", ""),
                current_ch,
            )
            if matrix_url:
                illustration_url = matrix_url

        logger.info(
            "[illustrator] Chapter %d illustration complete (url=%s)",
            current_ch,
            bool(illustration_url),
        )

        existing_cr = state.get("crew_result", {})
        return {
            "crew_result": {
                **existing_cr,
                "illustration_url": illustration_url,
                "illustration_prompt": illustration_prompt,
            }
        }

    return RunnableLambda(_node)


def create_tts_generator_agent(llm: BaseChatModel) -> Runnable:
    """Build the TTSGenerator ReAct agent.

    Tools: None (direct text-to-TTS)

    Output: {"audio_url": str}
    """
    agent = create_react_agent(
        llm,
        tools=[],
        prompt=TTS_GENERATOR_PROMPT,
        interrupt_before=[],
    )

    def _node(state: dict) -> dict[str, Any]:
        ctx = extract_fields_from_state(state, _MEDIA_FIELDS)
        current_ch = ctx.get("current_chapter_number", 1)
        chapter_text = ctx.get("refined_chapter", "") or ctx.get("chapter_draft", "")

        if not chapter_text:
            logger.warning("[tts] No chapter text for TTS")
            existing_cr = state.get("crew_result", {})
            return {"crew_result": {**existing_cr, "audio_url": ""}}

        logger.info("[tts] Generating TTS for chapter %d", current_ch)

        input_text = (
            f"请为第{current_ch}章生成语音。\n\n【章节正文】\n{chapter_text[:2000]}"
        )

        result = llm_call_with_retry(
            agent.invoke,
            {"messages": [("user", input_text)]},
            step_name="tts_generator_agent",
            timeout_seconds=180,
            fallback={"messages": [], "crew_result": {}},
        )
        response_text = extract_ai_message_text(result)

        # Parse JSON from response (fail-open: fallback to empty URL on parse error)
        parsed, err = validate_json_output(
            response_text,
            required_keys=["audio_url"],
            fail_closed=False,
        )
        if parsed:
            audio_url = str(parsed.get("audio_url", ""))
        else:
            audio_url = ""

        # If no URL, try matrix TTS generation
        if not audio_url or not audio_url.startswith("http"):
            matrix_url = _generate_tts_via_matrix(
                chapter_text,
                ctx.get("project_name", ""),
                current_ch,
            )
            if matrix_url:
                audio_url = matrix_url

        logger.info(
            "[tts] Chapter %d TTS complete (url=%s)", current_ch, bool(audio_url)
        )

        existing_cr = state.get("crew_result", {})
        return {"crew_result": {**existing_cr, "audio_url": audio_url}}

    return RunnableLambda(_node)
