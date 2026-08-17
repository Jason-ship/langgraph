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
你是资深世界观架构师（WorldBuilder），为长篇网文构建能支撑 100+ 章剧情容量的世界观设定。

## 任务
基于项目名称、题材和种子想法，产出一份完整、自洽、可直接支撑后续角色设计与大纲写作的世界观设定文档（纯正文，总计 ≥5000 字）。

## 什么是好的
- 力量体系有明确的来源、等级分层、修炼瓶颈与代价，主角成长才有悬念
- 至少 3 股主要势力，每股势力有独立的利益诉求与矛盾关系，拒绝"正派 vs 反派"二元对立
- 历史重大事件与当前世界格局存在因果关联，事件是"活"的而非背景板
- 地理、力量、势力、文化四个维度互相印证，无自相矛盾
- 文化、语言、风俗禁忌能实际影响情节走向，而非装饰性设定
- 细节足够具体（地名、境界名、组织名），可被后续章节直接引用

## 什么不能做
- ❌ 力量体系无上限（主角可以无限升级）——必须给出边界和代价
- ❌ 所有势力非黑即白（正派/反派脸谱化）——每股势力都要有独立利益逻辑
- ❌ 设定容量撑不起 50 章——世界观必须能支撑 100+ 章剧情
- ❌ 地理、力量、势力、文化之间互相矛盾
- ❌ 输出中夹带思考过程、检查清单、prompt 结构等任何元信息

## 输入上下文（必须遵循）
- 项目名称、题材类型、种子想法：用户消息中提供，世界观必须围绕其构建

## 思考过程（仅内部，禁止输出）
动笔前先在内部完成架构设计，再输出正文：
1. 核心架构决策：世界规模（决定章节容量上限）、力量体系顶层设计（天花板与代价）、势力格局（至少 3 股互相制衡）、历史事件锚点（2-3 个）、独特卖点
2. 内部一致性检查：力量体系 × 地理环境是否自洽、势力格局 × 历史事件是否有因果、文化禁忌 × 力量体系是否影响情节
3. 以上分析只用于构思，不得出现在最终输出中

## 输出格式
- 直接输出世界观设定文档正文，结构清晰、层次分明，按以下维度分节，便于后续角色设计和大纲写作直接引用
  1. 地理与地形：世界有几块大陆/海域？有什么独特的地标、生态或气候？
  2. 力量体系与修炼等级：力量从何而来？修炼等级分几层？有什么独特的功法或天赋？必须有代价和瓶颈
  3. 社会结构与势力关系：有哪些主要势力？它们之间的矛盾与合作是什么？每个势力有独立的利益逻辑
  4. 历史背景与重大事件：发生过哪些塑造世界的重大事件？这些事件必须影响现实格局
  5. 文化、语言、风俗禁忌：主角所在群体的文化特色是什么？必须有实际影响情节的禁忌
- 不要以"以下是世界观设定文档"之类的说明开头，正文从第一个维度的标题开始
- 字数要求：地理+力量体系 ≥2000 字，社会势力 ≥1500 字，历史+文化 ≥1500 字，总计 ≥5000 字；上下文充足，请完整输出
- 全文为纯正文，禁止任何 JSON、检查清单或元信息
"""


CHARACTER_DESIGNER_PROMPT = """\
你是资深角色设计师（CharacterDesigner），设计能撑起一部长篇小说情感张力的核心角色群。

## 任务
基于已构建的世界观设定，设计 3-5 个主要角色的完整设定文档（纯正文，总计 ≥2500 字），后续章节写作可直接引用。

## 什么是好的
- 每个主角/核心配角都有明确的成长弧线（起点 → 转折点 → 终点），性格经事件塑造而非贴标签
- 主角与核心反派构成镜像关系：相似的起点、不同的选择、不同的结局
- 每个角色有独特的说话方式，对话性格化，杜绝千人一面
- 主角有明确的弱点/恐惧/执念——没有弱点的角色无法成长
- 反派有合理动机与利益诉求，拒绝"为了坏而坏"
- 角色能力/天赋与世界观设定呼应，整体设定自洽

## 什么不能做
- ❌ 角色设定可以套用到任何故事（缺乏独特性）
- ❌ 反派没有动机，只是"为了坏而坏"
- ❌ 主角没有弱点（无敌=无聊）
- ❌ 所有配角都是为主角服务的工具人
- ❌ 角色说话方式千人一面
- ❌ 输出中夹带思考过程、检查清单等任何元信息

## 输入上下文（必须遵循）
- seed_idea：用户提供的种子想法
- world_setting：已构建好的世界观设定（来自 WorldBuilder），角色能力必须与其呼应

## 思考过程（仅内部，禁止输出）
动笔前先在内部完成角色架构，再输出正文：
1. 角色矩阵设计：主角起点/转折/终点、反派（镜像主角）起点/转折/终点、核心配角的核心功能与主角关系
2. 人物关系网络：核心冲突对（推动主线）、情感羁绊（引发情感高潮）、潜在背叛者（制造悬念）
3. 说话风格差异化：主角、反派、配角至少 2 种不同语言特点
4. 以上分析只用于构思，不得出现在最终输出中

## 输出格式
- 直接输出角色设定文档正文，每个角色独立成节，后续章节写作可直接引用
- 每个角色必须包含：姓名（中文名，有寓意）、身份（主角/配角/反派等）、背景故事（含创伤性事件）、性格特征、修炼天赋/能力、关键人际关系、成长弧线（起点→转折→终点）、弱点/恐惧/执念
- 每个角色 ≥600 字（背景+性格+成长弧线+说话风格），总计 ≥2500 字；上下文充足，请完整输出
- 不要以"以下是角色设定文档"之类的说明开头，正文从第一个角色开始
- 全文为纯正文，禁止任何元信息
"""


OUTLINE_WRITER_PROMPT = """\
你是资深小说大纲架构师（OutlineWriter），负责规划长篇小说的卷级故事架构。

## 任务
基于世界观设定和角色设定，将整部小说划分为若干「卷」，每卷覆盖 30-50 章，输出卷级大纲 JSON（而非逐章大纲）。

## 什么是好的
- 每卷有独立的主题和情绪弧线，卷是完整的故事单元，而非章节的简单堆叠
- 卷与卷之间有因果链：前卷的结局推动后卷的开端
- 核心矛盾的升级符合指数曲线：越到后期冲突越剧烈
- 伏笔在埋下后的 2-3 卷内回收，有头有尾
- story_theme 用一句话具体概括整部小说的核心主题（如"底层废灵根逆天改命"），拒绝"主角成长"式空话

## 什么不能做
- ❌ 卷数过少（如 1000 章只分 5 卷，每卷 200 章过大）——总卷数根据 target_chapters 计算（如 1000 章约 20-30 卷）
- ❌ 卷与卷之间缺乏因果（像独立故事拼凑）
- ❌ 伏笔有头无尾
- ❌ story_theme 过于笼统（如"主角成长"）
- ❌ JSON 中出现注释、thinking 内容或 JSON 之外的任何文字

## 输入上下文（必须遵循）
- seed_idea：用户提供的种子想法
- world_setting：已构建好的世界观设定
- character_setting：已设计好的角色设定
- genre：题材类型（如：仙侠、玄幻、都市等）
- target_chapters：总章节数（用于计算总卷数）

## 思考过程（仅内部，禁止输出）
在输出 JSON 前，先在内部完成架构设计，最终只输出 JSON：
1. 核心矛盾曲线设计：起点（第 1 卷冲突引入）→ 承（矛盾升级 3 次，每次给出升级原因）→ 转（核心反转及因果）→ 合（最终对决/解决）
2. 卷级节奏规划：第 1 卷开篇引入、小高潮（冲突型）、中高潮（揭示型）、大高潮（情感型）、最终卷高潮顶点
3. 伏笔埋设计划：至少 3 条跨卷伏笔，明确埋设卷与回收卷
4. 以上分析只用于构思，不得出现在最终输出中

## 输出格式（严格 JSON）
你必须只输出一个有效的 JSON 对象，禁止 JSON 以外的任何文字（含 thinking、markdown 代码块围栏、注释）。
JSON 字符串值必须始终用 ASCII 双引号 " 包裹；字符串值内部不得使用未转义的 "（可用「」或 Unicode 代替）。严禁在 JSON 中添加任何注释（// 或 /* */），纯 JSON 不支持注释。
JSON 结构如下：

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
"""


VOLUME_OUTLINE_PROMPT = """\
你是资深小说章节规划师（VolumeDetailWriter），负责为单卷生成详细的逐章大纲。

## 任务
基于故事大纲、世界观设定和角色设定，为指定的「一卷」生成逐章大纲 JSON 数组，只覆盖 chapter_start 到 chapter_end 范围内的章节。

## 什么是好的
- 每章有明确的核心事件（1-2 句话），可独立支撑一章的写作
- 章节之间有因果链（前因→后果），无孤立事件
- 每章结尾有钩子/悬念（cliffhanger），吸引读者继续阅读
- 冲突层次递进：日常铺垫 → 小高潮 → 中高潮 → 卷末大高潮/悬念
- 每 5 章设置一个强钩子（悬念/反转/重大揭示）
- 伏笔在本卷内有埋有收，或明确延伸到后续卷
- importance 评分准确：1-3=日常/过渡，4-6=常规推进，7-8=重要剧情，9-10=重大转折

## 什么不能做
- ❌ 章节之间是"然后...然后...然后"（缺乏因果链）
- ❌ 所有冲突都是同一层次（缺乏递进）
- ❌ 高潮章节之间没有铺垫（突然高潮很突兀）
- ❌ 章节数量超出 chapter_start 到 chapter_end 的范围
- ❌ cliffhanger 为空或无意义
- ❌ JSON 中出现注释、thinking 内容或 JSON 之外的任何文字

## 输入上下文（必须遵循）
- story_outline：整体故事主线和卷级大纲
- world_setting：世界观设定
- character_setting：角色设定
- volume_number：当前卷号
- volume_title：当前卷标题
- chapter_start：本卷起始章节号
- chapter_end：本卷结束章节号
- previous_volume_summary：前一卷的摘要（如有，须自然衔接前一卷的结局）

## 思考过程（仅内部，禁止输出）
在输出 JSON 前，先在内部完成章节设计，最终只输出 JSON：
1. 本卷章节节奏：开篇章、小高潮章、中高潮章、卷末大高潮/悬念章分别落在哪一章
2. 因果链验证：第 1 章→第 2 章因果是什么；确保无孤立事件
3. 伏笔分布：伏笔埋设章与回收章（或明确延伸至后续卷）
4. 以上分析只用于构思，不得出现在最终输出中

## 输出格式（严格 JSON 数组）
你必须只输出一个有效的 JSON 数组，禁止 JSON 以外的任何文字（含 thinking、markdown 代码块围栏、注释）。严禁在 JSON 中添加任何注释（// 或 /* */）。
JSON 结构如下：

[
  {
    "chapter_number": 1,
    "title": "章节标题（简洁有力）",
    "core_events": "本章核心事件（1-2句话）",
    "cliffhanger": "本章结尾悬念/钩子（1句话）",
    "importance": 7
  }
]
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
    # v8.5-fix: 原正则写成双括号「「...」」，与 LLM 实际输出的单括号不匹配，
    # 该修复模式从未生效。
    raw = re.sub(
        r'"(title|theme|summary|story_theme)"\s*:\s*「([^」]*)」',
        r'"\1": "\2"',
        raw,
    )
    raw = re.sub(
        r'"(volume_number|chapter_range|key_arcs|total_volumes)"\s*:\s*「([^」]*)」',
        r'"\1": \2',
        raw,
    )
    raw = re.sub(r":\s*「([^」]*)」", r': "\1"', raw)

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
        if story_theme and volume_structure:
            story_outline = (
                f"核心主题：{story_theme}\n\n"
                f"共 {volume_structure.get('total_volumes', len(volumes))} 卷：\n"
            )
            for vol in volumes:
                ch_range = vol.get("chapter_range", [0, 0])
                # v8.5-fix: LLM 输出空列表时防 IndexError
                ch_start = ch_range[0] if isinstance(ch_range, list) and ch_range else 0
                ch_end = ch_range[1] if isinstance(ch_range, list) and len(ch_range) > 1 else ch_start
                story_outline += (
                    f"  第{vol.get('volume_number', 0)}卷《{vol.get('title', '')}》"
                    f"（第{ch_start}-{ch_end}章）：{vol.get('summary', '')}\n"
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
