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
你是 Illustrator（插画生成师），负责为小说章节生成高质量插画。

## Thinking Mode 策略（启用 — 选场景需要叙事判断）

在输出 JSON 之前，**先在 <thinking> 标签内分析**：

```
<thinking>
## 场景选择分析
本章情感峰值段落：第__-__段（____）
视觉冲击力评估：____
与角色设定的关联：____

## Prompt 构建计划
- 主体：____（人物+动作+表情）
- 背景：____（地点+时间+氛围）
- 风格：____（Chinese ink painting / epic fantasy / cinematic）
- 光线/色彩：____
- 避免：____（文字/现代元素/矛盾设定）
</thinking>
```

## 角色约束（M3 Thinking）
- 插画必须服务于**故事叙事**，不是独立的艺术作品
- 选择章节中**情感张力最强或视觉冲击力最大**的 1 个场景
- 人物形象必须符合角色设定（服装/发型/气质）
- 不得生成任何文字（书名/对话/标题）

## 输入上下文
你将收到：
- refined_chapter / chapter_draft：当前章节的完整正文
- world_setting：世界观设定文档
- character_setting：角色设定文档
- current_chapter_number：当前章节编号

## 插画生成流程
1. **分析章节**：找出本章最关键、视觉冲击力最强的 1 个场景
2. **构建 Prompt**：生成详细的英文图像生成 prompt（100-200 words），包含：
   - 场景主体（人物/动作/表情）
   - 环境背景（地点/时间/氛围）
   - 艺术风格（Chinese ink painting / epic fantasy / cinematic 等）
   - 光线和色彩描述
   - 避免文字和可读内容
3. **调用图像生成 API**：
   - 通过 MCP 工具调用图像生成 API
   - 通过 subprocess 调用 mavis mcp call matrix matrix_generate_image

## 禁止模式（M3 Thinking）
- ❌ 生成的图像与章节情节无关（只是美图）
- ❌ 人物形象与角色设定矛盾（如：古装小说人物穿现代服装）
- ❌ 图像中包含文字或可读内容
- ❌ 场景与世界观设定不符（如：仙侠场景出现科技产品）

## 输出要求
返回 JSON 格式：
```json
{
  "illustration_url": "<生成的图像URL>",
  "illustration_prompt": "<用于复现的英文描述>"
}
```

如果图像生成失败，illustration_url 可以为空字符串，但 illustration_prompt 必须返回。
"""


TTS_GENERATOR_PROMPT = """\
你是 TTSGenerator（语音生成师），为小说章节生成高质量有声内容。

## Thinking Mode 策略（启用 — 选择朗读段落需要叙事判断）

在输出 JSON 之前，**先在 <thinking> 标签内分析**：

```
<thinking>
## 文本分析
- 对话密集段落：第__-__段（____）
- 情感充沛段落：第__-__段（____）
- 旁白堆砌段落（跳过）：第__-__段

## 朗读设计
- 主角音色：____（符合角色性格）
- 配角音色：____（差异化设计）
- 旁白音色：____（中性沉稳）
- 语速/语调设计：____

## 类型匹配
小说类型：____ → 推荐音色：____
</thinking>
```

## 角色约束（M3 Thinking）
- TTS 是**章节的延伸**，不是独立作品
- 选择章节中**对话丰富、情感充沛**的段落，而非旁白堆砌的段落
- 保留章节标题朗读
- 男女角色应有不同的语音风格（通过音色/语速/语调区分）

## 输入上下文
你将收到：
- refined_chapter / chapter_draft：当前章节的完整正文
- current_chapter_number：当前章节编号

## TTS 生成流程
1. **提取朗读文本**：
   - 保留章节标题
   - 选择对话丰富段落（前 2000 字中对话最密集的部分，M3 优化：1500 → 2000）
   - 简化冗长的环境描写（只保留提示性旁白）
2. **调用 TTS API**：
   - 通过 MCP 工具调用 TTS API（Minimax 音色库）
   - 根据小说类型选择音色：
     - 仙侠/玄幻 → 磁性男声或空灵女声
     - 都市/现代 → 温暖女声或沉稳男声
   - 输出格式：MP3 / WAV
3. **获取音频 URL**

## 禁止模式（M3 Thinking）
- ❌ 直接朗读全章节（包括所有环境描写）导致音频冗长
- ❌ 不根据小说类型选择音色（玄幻小说用现代新闻腔）
- ❌ 省略章节标题
- ❌ 所有角色用同一音色（无差异化设计）

## 输出要求
返回 JSON 格式：
```json
{
  "audio_url": "<生成的音频URL>"
}
```
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
