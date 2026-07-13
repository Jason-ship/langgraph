"""Setup Crew ReAct agents.

Each agent is built with create_react_agent and a typed system prompt.
Agents are invoked by the Setup Crew supervisor in sequence:
  WorldBuilder → CharacterDesigner → OutlineWriter

v6.0: Tool Calling 重构
  - OutlineWriter / VolumeDetailWriter 绑定 Neo4j 工具，LLM 可自主查询已有角色关系
  - 动态 prompt 根据项目上下文实时组装
  - WorldBuilder / CharacterDesigner 为纯 LLM 推理（无需外部查询）
"""

from __future__ import annotations

import json
import re
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
)
from novelfactory.agents.infra.helpers import make_retry_agent_invoke
from novelfactory.config.constants import FALLBACK_TARGET_CHAPTERS

_logger = get_logger("novelfactory.agents.setup")

_retry_agent_invoke = make_retry_agent_invoke("setup_agents")


# ── Output TypedDicts ─────────────────────────────────────────────────────────


class WorldBuilderOutput(TypedDict):
    world_setting: str


class CharacterDesignerOutput(TypedDict):
    character_setting: str


class OutlineWriterOutput(TypedDict):
    story_outline: str
    volume_structure: dict


class VolumeDetailOutput(TypedDict):
    volume_number: int
    chapter_outlines_detail: list


# ── System Prompts ─────────────────────────────────────────────────────────────

WORLD_BUILDER_PROMPT = """\
你是一位资深世界观架构师（WorldBuilder），构建的世界观必须能支撑 100+ 章节的剧情容量。

## Thinking Mode 策略（强制启用 — 世界架构是高复杂度推理任务）

在输出世界观文档之前，**先在 <thinking> 标签内进行架构设计**：

```
<thinking>
## 核心架构决策
1. 世界规模：____（决定章节容量上限）
2. 力量体系顶层设计：____（必须有天花板和代价）
3. 主要势力格局：____（至少3股，互相制衡）
4. 历史事件锚点：____（2-3个，决定当前世界格局）
5. 独特卖点：____（区别于同类作品的核心差异）

## 内部一致性检查
- 力量体系 × 地理环境：是否自洽？____
- 势力格局 × 历史事件：是否有因果？____
- 文化禁忌 × 力量体系：是否影响情节？____
</thinking>
```

## 角色约束（M3 Thinking）
- 你的输出是**后续所有章节写作的基础**，必须足够详细和自洽
- 世界观中的**力量体系**必须有明确的边界和代价（否则主角成长无悬念）
- 每个势力必须有**明确的利益诉求**（不能只是"正派vs反派"的二元对立）

## 必须覆盖的维度（1M 上下文 — 完整输出无截断）
1. **地理与地形**：世界有几块大陆/海域？有什么独特的地标、生态或气候？
2. **力量体系与修炼等级**：力量从何而来？修炼等级分几层？有什么独特的功法或天赋？**必须有代价和瓶颈**
3. **社会结构与势力关系**：有哪些主要势力？它们之间的矛盾与合作是什么？**每个势力有独立的利益逻辑**
4. **历史背景与重大事件**：发生过哪些塑造世界的重大事件？**这些事件必须影响现实格局**
5. **文化、语言、风俗禁忌**：主角所在群体的文化特色是什么？**必须有实际影响情节的禁忌**

## 禁止模式（M3 Thinking）
- ❌ 力量体系无上限（主角可以无限升级）
- ❌ 所有势力都是"正派"或"反派"（非黑即白）
- ❌ 世界观设定无法支撑超过 50 章（容量不足）
- ❌ 地理/势力/文化之间互相矛盾

## 字数要求（1M 上下文 — 无截断压力）
- 地理+力量体系≥2000字
- 社会势力≥1500字
- 历史+文化≥1500字
- 总计≥5000字（1M 上下文允许完整输出）

## 输出格式
输出一份完整的「世界观设定文档」，结构清晰，层次分明，便于后续角色设计和大纲写作直接引用。
"""


CHARACTER_DESIGNER_PROMPT = """\
你是一位资深角色设计师（CharacterDesigner），设计的角色必须能撑起一部长篇小说的情感张力。

## Thinking Mode 策略（强制启用 — 角色设计需要深度的心理建模）

在输出角色设定文档之前，**先在 <thinking> 标签内进行角色架构**：

```
<thinking>
## 角色矩阵设计
1. 主角：起点____ → 转折____ → 终点____
2. 反派（镜像主角）：起点____ → 转折____ → 终点____
3. 核心配角1：核心功能____；与主角的关系____
4. 核心配角2：核心功能____；与主角的关系____

## 人物关系网络
- 核心冲突对：____（推动主线）
- 情感羁绊：____（引发情感高潮）
- 潜在背叛者：____（制造悬念）

## 说话风格差异化设计
- 主角语言特点：____
- 反派语言特点：____
- 配角语言特点：____（至少2种不同风格）
</thinking>
```

## 角色约束（M3 Thinking）
- **每个主角/核心配角必须有明确的成长弧线**（起点→转折点→终点）
- **主角与核心反派必须有镜像关系**（相似的起点，不同的选择，导致不同的结局）
- **每个角色必须有独特的说话方式**（对话必须性格化，禁止千人一言）
- **主角必须有明确的弱点/恐惧/执念**（没有弱点的角色无法成长）

## 输入上下文
- seed_idea：用户提供的种子想法
- world_setting：已构建好的世界观设定（来自 WorldBuilder）

## 你的任务
基于世界观，设计 3-5 个主要角色的完整设定。

## 每个角色必须包含（1M 上下文 — 完整输出无截断）
1. **姓名**（中文名，有寓意）
2. **身份**：在故事中的位置（主角、配角、反派等）
3. **背景故事**：为何会成为现在的样子？**必须有创伤性事件**
4. **性格特征**：MBTI 或关键词描述
5. **修炼天赋/能力**：与世界观设定呼应
6. **关键人际关系**：与谁有羁绊？恩怨情仇？
7. **成长弧线**：起点（当前状态）→ 转折（触发事件）→ 终点（最终状态）
8. **弱点/恐惧/执念**（三选一，深入挖掘）

## 禁止模式（M3 Thinking）
- ❌ 角色设定可以套用到任何故事（缺乏独特性）
- ❌ 反派没有动机，只是"为了坏而坏"
- ❌ 主角没有弱点（无敌=无聊）
- ❌ 所有配角都是为主角服务的工具人
- ❌ 角色说话方式千人一面

## 字数要求（1M 上下文 — 无截断压力）
每个角色≥600字（背景+性格+成长弧线+说话风格），总计≥2500字

## 输出要求
输出一份完整的「角色设定文档」，每个角色独立成节，后续章节写作可直接引用。
"""


OUTLINE_WRITER_PROMPT = """\
你是一位资深小说大纲架构师（OutlineWriter），负责规划长篇小说的卷级故事架构。

## 你的任务
基于世界观设定和角色设定，将整部小说划分为若干「卷」，每卷覆盖 30-50 章。
输出 JSON 格式的卷级大纲结构，而非逐章大纲。

## Thinking Mode 策略（强制启用 — 大纲架构需要因果推理）

在输出 JSON 之前，**先在 <thinking> 标签内进行架构设计**：

```
<thinking>
## 核心矛盾曲线设计
- 起点（第1卷）：____冲突引入
- 承（第__-__卷）：矛盾升级3次（每次升级的原因：____）
- 转（第__卷）：核心反转____（因果：____）
- 合（最后1-2卷）：最终对决/解决____

## 卷级节奏规划
- 第1卷：开篇/引入 — ____
- 第__卷：小高潮（冲突型）— ____
- 第__卷：中高潮（揭示型）— ____
- 第__卷：大高潮（情感型）— ____
- 最终卷：高潮顶点 — ____

## 伏笔埋设计划
- 伏笔A：埋于第__卷，回收于第__卷，形式：____
- 伏笔B：埋于第__卷，回收于第__卷，形式：____
（至少3条跨卷伏笔，有头有尾）
</thinking>
```

## 大纲约束
- **每卷必须有独立的主题和情绪弧线**（不能只是章节的简单堆叠）
- **卷与卷之间必须有因果链**（前卷的结局推动后卷的开端）
- **核心矛盾的升级必须符合指数曲线**（越到后期冲突越剧烈）
- **伏笔必须在埋下后的2-3卷内回收**（禁止埋而不用）
- 每卷覆盖 30-50 章，总卷数根据 target_chapters 计算（如1000章约20-30卷）

## 输入上下文
- seed_idea：用户提供的种子想法
- world_setting：已构建好的世界观设定
- character_setting：已设计好的角色设定
- genre：题材类型（如：仙侠、玄幻、都市等）
- target_chapters：总章节数

## 输出格式（严格 JSON）

你必须输出有效的 JSON。**重要：JSON 字符串值必须始终用 ASCII 双引号 " 包裹。不要在字符串值内部使用未转义的 "（可用「」或Unicode代替）。严禁在 JSON 中添加任何注释（// 或 /* */），纯 JSON 不支持注释。**
JSON 格式如下：

{
  "story_theme": "整部小说的核心主题（一句话概括）",
  "total_volumes": 20,
  "volumes": [
    {
      "volume_number": 1,
      "title": "卷标题（如：废铁觉醒）",
      "theme": "本卷主题（如：底层挣扎与觉醒）",
      "chapter_range": [1, 40],
      "summary": "本卷概要（100-200字，描述主要情节走向）",
      "key_arcs": ["关键剧情线1", "关键剧情线2"]
    }
  ]
}

## 禁止模式
- 卷数过少（如1000章只分5卷，每卷200章过大）
- 卷与卷之间缺乏因果（像独立故事拼凑）
- 伏笔有头无尾
- story_theme 过于笼统（如"主角成长"）
"""


VOLUME_OUTLINE_PROMPT = """\
你是一位资深小说章节规划师，负责为单卷生成详细的逐章大纲。

## 你的任务
基于故事大纲、世界观设定和角色设定，为指定的「一卷」生成详细的逐章大纲。
只生成本卷范围内的章节，不要超出 chapter_start 到 chapter_end 的范围。

## Thinking Mode 策略（强制启用）

在输出 JSON 之前，**先在 <thinking> 标签内进行章节设计**：

```
<thinking>
## 本卷章节节奏
- 第__章（开篇）：____
- 第__章：小高潮 — ____
- 第__章：中高潮 — ____
- 第__章（卷末）：大高潮/悬念 — ____

## 因果链验证
- 第1章→第2章因果：____
- 每5章设置一个"钩子"（悬念/反转/重大揭示）
- 确保无孤立事件

## 伏笔分布
- 伏笔A：埋于第__章，回收于第__章
- 伏笔B：埋于第__章，回收于第__章
</thinking>
```

## 章节约束
- **每章必须有明确的核心事件**（1-2句话描述）
- **章节之间必须有因果链**（前因→后果，禁止孤立事件）
- **每章必须有钩子/悬念**（cliffhanger，吸引读者继续阅读）
- **importance 评分**：1-3=日常/过渡，4-6=常规推进，7-8=重要剧情，9-10=重大转折
- 如有前一卷摘要，需自然衔接前一卷的结局

## 输入上下文
- story_outline：整体故事主线和卷级大纲
- world_setting：世界观设定
- character_setting：角色设定
- volume_number：当前卷号
- volume_title：当前卷标题
- chapter_start：本卷起始章节号
- chapter_end：本卷结束章节号
- previous_volume_summary：前一卷的摘要（如有）

## 输出格式（严格 JSON 数组）

你必须输出有效的 JSON 数组，不要输出 JSON 以外的任何内容（thinking 标签内容除外）。**严禁在 JSON 中添加任何注释（// 或 /* */）。**
JSON 格式如下：

[
  {
    "chapter_number": 1,
    "title": "章节标题（简洁有力）",
    "core_events": "本章核心事件（1-2句话）",
    "cliffhanger": "本章结尾悬念/钩子（1句话）",
    "importance": 7
  }
]

## 禁止模式
- 章节之间是"然后...然后...然后"（缺乏因果）
- 所有冲突都是同一层次（缺乏递进）
- 高潮章节之间没有铺垫（突然高潮很突兀）
- 章节数量与 chapter_range 不符
- cliffhanger 为空或无意义
"""


# ── State Access Helpers ───────────────────────────────────────────────────────

# v6.1 P2-1: 统一使用 extract_fields_from_state 替代原 _get_context。
# crew_result 优先，缺失回退顶层；target_chapters 的 falsy 兜底在使用处显式处理。
_SETUP_FIELDS: dict[str, Any] = {
    "seed_idea": "",
    "genre": "",
    "project_name": "",
    "target_chapters": FALLBACK_TARGET_CHAPTERS,
    "world_setting": "",
    "character_setting": "",
}


# _to_messages → extract_ai_message_text (v5.1.1: 统一到 agents/infra/)


# ── JSON Parsing Helpers ──────────────────────────────────────────────────────


def _fix_llm_json(raw: str) -> str | None:
    """修复 LLM 输出 JSON 的常见问题，提高解析成功率。

    v6.1-fix: 移除了全局 ``//`` 注释移除 — 它会破坏 JSON 字符串值中的
    ``//`` 代码片段（如 ``// system.format("earth")``），导致字符串被截断。
    改为在逐行修复中对非字符串值行安全移除 ``//``。
    """
    if not raw:
        return None
    # 1. 移除非 JSON 前缀/后缀（如 markdown 代码块标记）
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw)

    # 2. 修复 LLM 用「」代替 "" 包裹字符串值的情况
    raw = re.sub(
        r'"(title|theme|summary|story_theme)"\s*:\s*「「([^」]*)」」',
        r'"\1": "\2"',
        raw,
    )
    raw = re.sub(
        r'"(volume_number|chapter_range|key_arcs|total_volumes)"\s*:\s*「「([^」]*)」」',
        r'"\1": \2',
        raw,
    )
    raw = re.sub(r":\s*「「([^」]*)」」", r': "\1"', raw)

    # 3. 尝试标准解析
    for s in (raw, raw.strip()):
        try:
            json.loads(s)
            return s
        except json.JSONDecodeError:
            pass

    # 4. 逐行修复：转义字符串值内的未转义双引号 + 安全移除非字符串行的 // 注释
    # v6.1-fix: 之前全局 re.sub(r"//[^\n]*", "", raw) 会破坏字符串值中的 //
    # 代码片段。现在只对非字符串值行（结构行）移除 //。
    lines = raw.split("\n")
    fixed_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            fixed_lines.append(line)
            continue
        kv_match = re.match(r'^(\s*"[^"]*"\s*:\s*)"(.*)"(,?\s*)$', stripped)
        if kv_match and kv_match.group(2):
            key_prefix = kv_match.group(1)
            raw_value = kv_match.group(2)
            trailing = kv_match.group(3)
            escaped_value = raw_value.replace('"', '\\"')
            fixed_line = f'{key_prefix}"{escaped_value}"{trailing}'
            indent = line[: len(line) - len(line.lstrip())]
            fixed_lines.append(indent + fixed_line.lstrip())
        else:
            # 非字符串值行：安全移除 // 单行注释
            cleaned_line = re.sub(r"//[^\n]*", "", line)
            fixed_lines.append(cleaned_line)
    fixed = "\n".join(fixed_lines)

    try:
        json.loads(fixed)
        return fixed
    except json.JSONDecodeError:
        pass

    # 5. 移除多行 /* */ 注释（作为降级手段）
    fixed2 = re.sub(r"/\*[\s\S]*?\*/", "", fixed)
    if fixed2 != fixed:
        try:
            json.loads(fixed2)
            return fixed2
        except json.JSONDecodeError:
            pass

    # 6. 尝试更激进的修复：直接用 ast.literal_eval 替代 json
    try:
        import ast

        raw_python = (
            raw.replace("true", "True")
            .replace("false", "False")
            .replace("null", "None")
        )
        parsed = ast.literal_eval(raw_python)
        if isinstance(parsed, dict) and "volumes" in parsed:
            return json.dumps(parsed, ensure_ascii=False)
    except (ValueError, SyntaxError, TypeError):
        pass

    return None


def _parse_volume_structure_json(text: str) -> dict | None:
    """从 LLM 输出中解析卷级大纲 JSON 结构。

    支持包含 <thinking> 标签的输出，自动提取 JSON 对象。
    支持修复常见 LLM JSON 格式问题。
    返回解析后的字典，失败时返回 None。
    """
    if not text:
        return None
    # 移除 <thinking> 标签内容
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    # 尝试提取 JSON 对象
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None
    raw_json = match.group()

    # 尝试修复后解析
    fixed = _fix_llm_json(raw_json)
    if fixed is not None:
        raw_json = fixed

    try:
        data = json.loads(raw_json)
        # 基本校验：必须包含 volumes 列表
        if "volumes" not in data or not isinstance(data["volumes"], list):
            return None
        return data
    except (json.JSONDecodeError, TypeError):
        return None


def _parse_chapter_outlines_json(
    text: str, chapter_start: int, chapter_end: int
) -> list[dict]:
    """从 LLM 输出中解析单卷章节大纲 JSON 数组。

    支持包含 <thinking> 标签的输出，自动提取 JSON 数组。
    返回规范化后的章大纲列表，失败时返回空列表。
    """
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    match = re.search(r"\[[\s\S]*\]", cleaned)
    if not match:
        return []
    raw_json = match.group()
    fixed = _fix_llm_json(raw_json)
    if fixed is not None:
        raw_json = fixed
    try:
        data = json.loads(raw_json)
        if not isinstance(data, list):
            return []
        # 校验并规范化每条章大纲
        result: list[dict] = []
        for ch in data:
            if not isinstance(ch, dict):
                continue
            ch_num = ch.get("chapter_number", 0)
            # 确保章节号在合理范围内
            if ch_num < chapter_start or ch_num > chapter_end:
                continue
            result.append(
                {
                    "chapter_number": ch_num,
                    "title": ch.get("title", f"第{ch_num}章"),
                    "core_events": ch.get("core_events", ""),
                    "cliffhanger": ch.get("cliffhanger", ""),
                    "importance": max(1, min(10, int(ch.get("importance", 5)))),
                }
            )
        return result
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


# ── Agent Factory Functions ────────────────────────────────────────────────────


def create_world_builder_agent(llm: BaseChatModel) -> Runnable:
    """Build the WorldBuilder ReAct agent (LLM-only, no RAG tools).

    Output: {"world_setting": str}

    Returns a RunnableLambda so callers can use .invoke() consistently.
    """
    agent = create_react_agent(
        llm,
        tools=[],
        prompt=WORLD_BUILDER_PROMPT,
        interrupt_before=[],
    )

    def _node(state: dict) -> dict[str, Any]:
        ctx = extract_fields_from_state(state, _SETUP_FIELDS)
        input_text = (
            f"请为以下小说项目构建世界观：\n"
            f"项目名称：{ctx['project_name']}\n"
            f"题材类型：{ctx['genre']}\n"
            f"种子想法：{ctx['seed_idea']}"
        )
        result = _retry_agent_invoke(
            agent, {"messages": [("user", input_text)]}, "WorldBuilder"
        )
        world_setting = extract_ai_message_text(result) or "世界观构建失败"
        existing_cr = state.get("crew_result", {})
        return {"crew_result": {**existing_cr, "world_setting": world_setting}}

    return RunnableLambda(_node)


def create_character_designer_agent(llm: BaseChatModel) -> Runnable:
    """Build the CharacterDesigner ReAct agent (LLM-only).

    Output: {"character_setting": str}

    Returns a RunnableLambda so callers can use .invoke() consistently.
    """
    agent = create_react_agent(
        llm,
        tools=[],
        prompt=CHARACTER_DESIGNER_PROMPT,
        interrupt_before=[],
    )

    def _node(state: dict) -> dict[str, Any]:
        ctx = extract_fields_from_state(state, _SETUP_FIELDS)
        input_text = (
            f"请为以下小说项目设计主要角色：\n"
            f"项目名称：{ctx['project_name']}\n"
            f"种子想法：{ctx['seed_idea']}\n\n"
            f"【已构建的世界观】\n{ctx['world_setting']}"
        )
        result = _retry_agent_invoke(
            agent, {"messages": [("user", input_text)]}, "CharacterDesigner"
        )
        character_setting = extract_ai_message_text(result) or "角色设计失败"
        existing_cr = state.get("crew_result", {})
        return {"crew_result": {**existing_cr, "character_setting": character_setting}}

    return RunnableLambda(_node)


def _build_outline_dynamic_prompt(
    state: AgentState, config: RunnableConfig
) -> list[AnyMessage]:
    """动态 prompt：根据项目上下文附加工具使用指引。"""
    from langchain_core.messages import SystemMessage

    tool_guidance = """
## 可用工具
你拥有 Neo4j 人物关系图谱查询工具，可在规划大纲时使用：

- `get_character_network(character_name, max_depth)` — 查询指定角色的关系网络
- `get_all_characters()` — 获取所有已有角色列表
- `get_character_info(character_name)` — 查询角色详细信息

### 工具使用建议
1. 若世界观或角色设定中已有角色名称，可调用 `get_all_characters()` 确认角色库
2. 若需规划跨卷伏笔或关系冲突，可调用 `get_character_network()` 获取关系拓扑
3. 工具返回 JSON 字符串，可直接引用其中信息辅助大纲规划
"""

    messages: list[AnyMessage] = [
        SystemMessage(content=OUTLINE_WRITER_PROMPT + tool_guidance),
    ]
    # 追加已有消息历史
    for msg in state.get("messages", []):
        messages.append(msg)
    return messages


def create_outline_writer_agent(llm: BaseChatModel) -> Runnable:
    """Build the OutlineWriter ReAct agent with Neo4j tools.

    绑定 Neo4j 工具，LLM 可自主查询角色关系辅助大纲规划。
    动态 prompt 根据项目上下文组装工具使用指引。

    Output: {"story_outline": str, "volume_structure": dict}

    Returns a RunnableLambda so callers can use .invoke() consistently.
    """
    from novelfactory.tools import get_neo4j_tools

    tools = get_neo4j_tools()
    agent = create_react_agent(
        llm,
        tools=tools,
        prompt=_build_outline_dynamic_prompt,
        interrupt_before=[],
    )

    def _node(state: dict) -> dict[str, Any]:
        ctx = extract_fields_from_state(state, _SETUP_FIELDS)
        # target_chapters: 保留原 _get_context 的 falsy 兜底逻辑
        if not ctx.get("target_chapters"):
            ctx["target_chapters"] = FALLBACK_TARGET_CHAPTERS
        input_text = (
            f"请为以下小说项目创作卷级故事大纲：\n"
            f"项目名称：{ctx['project_name']}\n"
            f"题材类型：{ctx['genre']}\n"
            f"种子想法：{ctx['seed_idea']}\n"
            f"目标总章节数：{ctx['target_chapters']}\n\n"
            f"【已构建的世界观】\n{ctx['world_setting']}\n\n"
            f"【已设计的角色】\n{ctx['character_setting']}"
        )
        result = _retry_agent_invoke(
            agent, {"messages": [("user", input_text)]}, "OutlineWriter"
        )
        full_output = extract_ai_message_text(result) or "大纲创作失败"

        # 解析 JSON 格式的卷级大纲结构
        volume_structure = _parse_volume_structure_json(full_output)
        story_theme = (
            volume_structure.get("story_theme", "") if volume_structure else ""
        )
        volumes = volume_structure.get("volumes", []) if volume_structure else []

        # story_outline 保留为可读文本（向后兼容 + 质量门控评分用）
        if story_theme:
            story_outline = (
                f"核心主题：{story_theme}\n\n"
                f"共 {volume_structure.get('total_volumes', len(volumes))} 卷：\n"
            )
            for vol in volumes:
                ch_range = vol.get("chapter_range", [0, 0])
                story_outline += (
                    f"  第{vol.get('volume_number', 0)}卷《{vol.get('title', '')}》"
                    f"（第{ch_range[0]}-{ch_range[1]}章）：{vol.get('summary', '')}\n"
                )
        else:
            # JSON 解析失败时的降级处理
            story_outline = full_output

        existing_cr = state.get("crew_result", {})
        return {
            "crew_result": {
                **existing_cr,
                "story_outline": story_outline,
                "volume_structure": volume_structure or {},
            }
        }

    return RunnableLambda(_node)


def _build_volume_detail_dynamic_prompt(
    state: AgentState, config: RunnableConfig
) -> list[AnyMessage]:
    """动态 prompt：为卷级详情大纲附加工具使用指引。"""
    from langchain_core.messages import SystemMessage

    tool_guidance = """
## 可用工具
你拥有 Neo4j 人物关系图谱查询工具，可在规划逐章大纲时使用：

- `get_character_network(character_name, max_depth)` — 查询角色关系网络
- `get_all_characters()` — 获取所有角色列表
- `get_character_info(character_name)` — 查询角色详细信息

### 工具使用建议
1. 若本卷涉及特定角色的成长或关系变化，先查询其关系网络
2. 规划伏笔线时，确认角色间的现有关系拓扑
"""

    messages: list[AnyMessage] = [
        SystemMessage(content=VOLUME_OUTLINE_PROMPT + tool_guidance),
    ]
    for msg in state.get("messages", []):
        messages.append(msg)
    return messages


def create_volume_detail_writer_agent(llm: BaseChatModel) -> Runnable:
    """构建卷级详情大纲 ReAct agent（单卷逐章大纲）。

    绑定 Neo4j 工具，LLM 可自主查询角色关系辅助逐章规划。
    为单卷生成逐章大纲，输出 JSON 数组。
    每章包含：chapter_number, title, core_events, cliffhanger, importance。

    输入 crew_result 需包含：
      - volume_number, volume_title, chapter_start, chapter_end
      - story_outline, world_setting, character_setting
      - previous_volume_summary（可选）

    Output: {"volume_number": int, "chapter_outlines_detail": list[dict]}

    Returns a RunnableLambda so callers can use .invoke() consistently.
    """
    from novelfactory.tools import get_neo4j_tools

    tools = get_neo4j_tools()
    agent = create_react_agent(
        llm,
        tools=tools,
        prompt=_build_volume_detail_dynamic_prompt,
        interrupt_before=[],
    )

    def _node(state: dict) -> dict[str, Any]:
        ctx = extract_fields_from_state(state, _SETUP_FIELDS)
        cr = state.get("crew_result", {})
        volume_number = cr.get("volume_number", 1)
        volume_title = cr.get("volume_title", "")
        chapter_start = cr.get("chapter_start", 1)
        chapter_end = cr.get("chapter_end", 40)
        previous_volume_summary = cr.get("previous_volume_summary", "")
        story_outline = cr.get("story_outline", "")

        input_text = (
            f"请为以下卷生成详细的逐章大纲：\n"
            f"卷号：第{volume_number}卷\n"
            f"卷标题：{volume_title}\n"
            f"章节范围：第{chapter_start}章 - 第{chapter_end}章\n\n"
            f"【故事大纲】\n{story_outline}\n\n"
            f"【世界观设定】\n{ctx['world_setting']}\n\n"
            f"【角色设定】\n{ctx['character_setting']}\n"
        )
        if previous_volume_summary:
            input_text += f"\n【前一卷摘要】\n{previous_volume_summary}\n"

        result = _retry_agent_invoke(
            agent,
            {"messages": [("user", input_text)]},
            f"VolumeDetailWriter_V{volume_number}",
        )
        full_output = extract_ai_message_text(result) or "[]"

        # 解析 JSON 数组格式的章节大纲
        chapter_outlines = _parse_chapter_outlines_json(
            full_output, chapter_start, chapter_end
        )

        existing_cr = state.get("crew_result", {})
        return {
            "crew_result": {
                **existing_cr,
                "volume_number": volume_number,
                "chapter_outlines_detail": chapter_outlines,
            }
        }

    return RunnableLambda(_node)
