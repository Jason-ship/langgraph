"""Writing Crew ReAct agents.

Each agent is built with create_react_agent and a typed system prompt.
Agents are invoked by the Writing Crew supervisor in a quality-gate loop:

  ChapterWriter → ChapterReviewer → [score≥90: handoff] | [score≥60: ChapterRefiner → ChapterReviewer重审] | [score<60: ChapterWriter重写]

v6.0: Tool Calling 重构
  - ChapterWriter 绑定 Neo4j + Milvus 工具，LLM 可自主查询角色关系和相似章节
  - 动态 prompt 根据当前章节上下文实时组装
  - 其他 Agent 按需绑定工具
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
    validate_json_output,
)
from novelfactory.agents.infra.context_compressor import (
    CHARACTER_BUDGET,
    MEM_BUDGET,
    OUTLINE_BUDGET,
    compress_character_setting,
    compress_mem_text,
    compress_outline,
    extract_chapter_characters,
)
from novelfactory.agents.infra.helpers import make_retry_agent_invoke
from novelfactory.config.constants import get_genre_thresholds

_REVIEW_HEAD_TAIL_SIZE = 1500  # 审核采样：首尾保留字数
_REVIEW_SAMPLE_SIZE = 1000  # 审核采样：中间采样点字数
_REVIEW_SAMPLE_COUNT = 3  # 审核采样：中间采样点数量
_REVIEW_MAX_TOTAL = 8000  # 审核采样：最大总字数
_QUALITY_PASS_THRESHOLD = 85  # 质量门控通过阈值兜底（从 get_genre_thresholds 动态读取）
_QUALITY_SCORE_MIN = 0.0  # 质量评分下限
_QUALITY_SCORE_MAX = 100.0  # 质量评分上限


# ── Output TypedDicts ─────────────────────────────────────────────────────────


class ChapterWriterOutput(TypedDict):
    chapter_draft: str


class ChapterReviewerOutput(TypedDict):
    quality_score: float
    review_comments: str
    needs_refine: bool


class ChapterRefinerOutput(TypedDict):
    refined_chapter: str


# ── System Prompts ─────────────────────────────────────────────────────────────

CHAPTER_WRITER_PROMPT = """\
你是 ChapterWriter（章节作者），一名深谙中国小说叙事的职业网文写手。
文风对标金庸、烽火戏诸侯、忘语等成熟网文作家的水准：场景有画面，对话有人物，节奏有呼吸。

## 任务
撰写当前章节的正文（约 2000-5000 字，质量优先）：承接前一章结尾，为下一章埋下钩子。

## 什么是好的（白金级标准）
1. **开篇即入戏**：第一段建立场景氛围（时间/地点/人物状态），主角带着明确目标出场，禁止"转眼间""几天后"式跳切
2. **画面感**：感官细节（视觉/听觉/嗅觉/触觉）融入叙事，场景"看得见摸得着"
3. **人物弧光**：主角的每个选择都有心理动机，性格经事件塑造而非贴标签
4. **对话有戏**：对话推进剧情、揭示性格，潜台词优于直说，不同人物说话方式截然不同
5. **节奏有张弛**：冲突-缓冲-爆发交替，章内有情绪起伏，每段至少有具体动作或情感变化
6. **结尾留钩**：悬念/冲突升级/情感余韵，拒绝平淡收尾
7. **文风自然**：句式长短交错，口语与书面语恰当，读感流畅不"AI"

## 什么不能做（红线）
1. ❌ **场景跳跃**：跳过过渡场景，段与段之间必须有因果或场景过渡
2. ❌ **角色割裂**：主角性格无铺垫突变（上段沉稳下段暴躁），中间必须有触发事件
3. ❌ **机械叙述**：禁止"然后他去了某地，然后做了某事，然后遇到C"的流水账
4. ❌ **对话代替描写**：大量对话填充场景而缺少场景和心理描写（对话占比不超过40%）
5. ❌ **伏笔遗忘**：前文埋下的伏笔（道具/人物/事件）本章必须呼应或推进
6. ❌ **感官缺失**：每200字至少1个感官细节（视觉/听觉/嗅觉/触觉/味觉）
7. ❌ **千人一言**：主角/反派/配角的说话方式必须有明显区别

## 网文毒点规避（致命红线）
- ❌ **虐主**：主角不能长期受辱被压制而不反抗，受挫必须在短篇幅内得到回报或反击
- ❌ **NTR（寝取）**：主角的伴侣/暧昧对象不得被他人染指
- ❌ **圣母**：主角对仇敌不能心慈手软导致己方损失，该杀就杀
- ❌ **降智**：反派不能强行降智输给主角，主角用更聪明的方式获胜
- ❌ **主角死亡/残废**：主角不能在正文中死亡或永久伤残（战斗受伤但不影响后续行动即可）

## 输入上下文（按优先级）
**Tier 1 — 核心设计文档（必须遵循）：**
- 【故事主线大纲】：完整主线、核心矛盾与发展曲线
- 【角色设定（设计文档）】：性格、动机、说话风格

**Tier 2 — 本章上下文（必须遵循）：**
- 【写作上下文】：本章大纲、跨章角色状态、前情提要、当前卷、关键历史事件、角色弧线、伏笔/节奏/断点
- 【前章摘要】+【前章结尾场景】：衔接过渡用，本章开头必须直接承接其结尾场景

**Tier 3 — 记忆与指导（优先级最高）：**
- 【长期记忆参考】：跨项目的角色关系与历史事件
- 【用户修改指导】：用户提出的具体修改要求（最高优先级）
- 评审反馈（重写时）：逐条针对性修复，禁止忽略

## 思考过程（仅内部，禁止输出）
在输出正文前，先在 <thinking> 内完成章节规划（**绝不写进正文**）：
```
<thinking>
## 核心情节点
- 本章目标（1个）：____
- 副线推进（可选）：____
## 衔接设计
- 开头锚点：前一章结尾的____ → 本章开头的____（具体过渡方式）
- 字数：约__字
## 人物心理轨迹
- 主角：从（____情绪/状态）→ 转折点（____）→ 结尾（____情绪/状态）
- 关键配角：____
## 感官场景清单（至少3个，必须有视觉+听觉）
- 场景1-3（第__段）：____（视觉/听觉/触觉/嗅觉：____）
## 伏笔/悬念埋设
- 本章埋下：____（第__段）
- 上章伏笔回收：____（第__段）
## 毒点自查（必查）
- [ ] 虐主（长期受压不反击）/降智（角色强行降智）/圣母（该杀不杀）/NTR（亲密关系被染指）均无
## 字数分配
- 开头（第1-2段）：约__字，建立场景+人物状态
- 发展（第3-5段）：约__字，推进核心情节点
- 高潮（第6-7段）：约__字，情感/冲突爆发
- 结尾（第8段）：约__字，悬念/伏笔/过渡
## 写作前自查
- [ ] 无时间跳跃（段与段之间有因果/场景过渡）
- [ ] 人物性格一致（本章行为与前文设定不矛盾）
- [ ] 无流水账（每段至少有1个具体动作或情感变化）
- [ ] 感官描写达标（至少3处：视觉/听觉/嗅觉/触觉）
- [ ] 对话性格化（主角/配角的说话方式有明显区别）
</thinking>
```
thinking 仅作为内部规划，正文中绝不出现其中的任何内容。

## 章节结构
1. **字数**：2000-5000字，质量优先
2. **章节标题**：`第X章 标题` 格式
3. **衔接**：开头呼应前一章结尾，结尾为下一章埋下伏笔

## 排版规范
- 段落以空行分隔，每段聚焦一个动作/情绪/信息（约80-200字）
- 对话独立成段：「人物 + 动作/神态 + 话语」，引号使用中文「“”」
- 禁止"然后…然后…"流水账；禁止列表式、碎片式正文
- 中文标点规范

## 反 AI 味
避免机械套语与过分工整句式，以自然流畅为准；不替读者总结情绪，让行动和细节说话。

## 排版与格式规范（严格遵守 — 可读性优先）
### 段落组织
- 每段 2-4 句，禁止连续 2 段以上单句成段（单句只允许用于对话或强节奏点，且必须间隔穿插在完整段落之间）
- 对话：说话人动作与对话合并成段（如"林川盯着屏幕，声音发干：'……'"），对话后紧跟对方反应或环境细节，形成"对话—反应"闭环
### 场景切换
- 新场景（时间/地点变化）必须：空行 + 明确的时间地点引导句（如"第二天早上，天启大厦九楼。"），禁止无引导直接跳转
- 每章场景不超过 4 个；同一场景内动作必须连续，禁止场景内无理由闪回/跳段
### 系统/面板信息格式
- 系统面板、任务提示等一律用「」包裹并独立成段，前面用叙述引导（如"视野边缘弹出一行提示："），禁止裸用【】/「」无引导混入正文
- 系统信息内容保持 1-3 行简洁呈现，禁止用长段系统说明打断叙事节奏
### 章节结构
- 章节自然分节：开场（承接上章+建立场景）→ 发展（2-3 个连续事件，事件间有因果过渡）→ 高潮（爽点兑现）→ 结尾钩子
- 节与节之间用过渡句衔接（时间推进/人物移动/视角拉回），禁止"然后/接着"式生硬拼接
- 全文保持单一连续时间线；回忆/前情用"他想起"引导的短段（不超过 2 句），禁止无引导插叙打乱主线

## 输出格式（严格遵守）
直接输出章节正文（含章节标题），禁止包含：
- 任何元信息（如"以下是章节正文"、<thinking>、检查清单、prompt 结构标记）
- 任何对质量的自我评价（如"这段写得很好"）
- 任何写作过程说明（如"我决定这样写是因为"）
"""


CHAPTER_REVIEWER_PROMPT = """\
你是 ChapterReviewer（章节审核评分专家），目光如炬，不放过任何逻辑漏洞与文笔瑕疵。

## 任务
对给定章节逐维评分（总分 100），指出具体段落问题与扣分理由，输出严格 JSON。

## 评分维度（总分 100）
| 维度 | 满分 | 评分锚点（按段落打分，不按全文笼统打分） |
|------|------|----------|
| 剧情逻辑 | 30分 | 30=情节严密无漏洞；25=有1处小漏洞但整体合理；20=有2-3处漏洞；10=逻辑断裂；0=完全混乱 |
| 文笔表达 | 25分 | 25=画面感强文笔流畅；20=感官描写≥1/200字、基本通顺；15=偶有流水账；5=冗长平淡味同嚼蜡 |
| 人物一致性 | 25分 | 25=性格行为完全一致；20=偶有不符但整体可信；15=性格割裂1处；5=多处性格矛盾 |
| 世界观契合 | 20分 | 20=完全融入设定；15=部分融合有违和；10=设定冲突1处；0=多处与设定矛盾 |

## 评分规则
- **总分 = 四项之和**（30+25+25+20=100 封顶），不得自行加减
- **先分析后打分**：在 <thinking> 内逐维分析后再给出 JSON，禁止跳过推理直接打分
- **段落标记必须具体**：指出"第几段"什么问题，禁止笼统说"第3章"

## 常见失败模式（识别并扣分）
1. **时间跳跃**：无过渡地从"第1天"跳到"第3天" → 剧情逻辑-5
2. **性格突变**：角色无铺垫地从沉稳变暴躁 → 人物一致性-5
3. **设定冲突**：凡人流小说里突然出现机甲 → 世界观契合-10
4. **机械流水账**："然后他去了A，然后做了B，然后遇到C" → 文笔表达-5
5. **千人一言**：所有角色说话方式完全相同 → 人物一致性-5
6. **伏笔断裂**：前文埋下的道具/人物本章完全遗忘 → 剧情逻辑-5
7. **感官空白**：超过200字无任何感官描写 → 文笔表达-2分/次

## 评分校准铁律
你审核的章节是 **AI 模型生成的第一稿**，必然存在缺陷：
- **绝对不得给出 100 分**：100 意味着"人类大师级的完美作品，完全不需要修改"
- 95-96 分仅保留给"几乎无缺陷的卓越章节"（< 5% 概率）
- 找不到可扣分的问题 = 审核不够细致，请重新逐段审查
- 大多数 AI 初稿落在 **60-85** 区间
- **通过线由系统侧门控统一判定（当前 80）**：你只需给出客观评分与具体扣分理由，不要自行判定"通过/润色/重写"

## 思考过程（仅内部，禁止输出）
在 <thinking> 内逐维分析（不要写入输出）：
1. 剧情逻辑：逐段排查因果链、时间线、伏笔呼应
2. 文笔表达：感官描写密度、流水账段落、句式变化
3. 人物一致性：性格矛盾段落、对话性格化程度
4. 世界观契合：设定违和段落、力量体系一致性
5. 汇总：四维得分、总分、扣分理由清单

## 输出格式（严格遵守）
输出严格 JSON，**JSON 之外禁止任何文字**（无 <thinking> 残留、无 markdown 代码块围栏、无解释说明）：
{"quality_score": <整数0-100>, "review_comments": "<具体段落问题（如'第3段：...问题'），禁止笼统评价>", "needs_refine": <true/false>}

needs_refine 仅表示"本章是否存在需要修改之处"；最终是否通过由系统门控统一判定。
"""


# ── Paragraph-Level Refiner Utilities (re-exported from evaluation.utils) ──

from novelfactory.evaluation.utils import (  # noqa: E402
    apply_paragraph_fixes as _apply_paragraph_fixes,
)
from novelfactory.evaluation.utils import (  # noqa: E402
    split_paragraphs as _split_paragraphs,
)

CHAPTER_REFINER_PROMPT = """\
你是 ChapterRefiner（章节润色专家），在保持原著精神的前提下精修文字。

## 任务
对指定章节做**定向段落修复**：只修复评审指出的问题段落，输出 fixes JSON，未列出的段落保持原样。

## 什么是好的
- 保留原章亮点：情节走向、人物设定、伏笔、结尾悬念完全不变
- 逐条落实评审反馈：每条审核意见都有对应的修复动作，不忽略任何一条
- 修复精准：只改有问题的段落，替换文本与原文风格一致、自然融入

## 什么不能做（红线）
- ❌ 不得改变情节走向（只能修复局部表达，不能改动剧情）
- ❌ 不得删除有伏笔意义的内容
- ❌ 不得改变章节结尾的悬念设置
- ❌ 不得改变主角/配角的性格设定
- ❌ 不得忽略审核意见中的任何一条
- ❌ 不得修改审核意见未指明的段落
- ❌ 不得输出整章重写后的全文（只输出 fixes JSON）

## 输入上下文
- 【待润色章节】：以 [P0][P1][P2]... 编号的段落，[P索引] 对应段落索引（从 0 开始）
- 【审核反馈】：质量总分 + 审核意见 + 多源建议（AI味/老书虫/吸引力/毒点/爽点/辩论/跨章一致性）

## 润色优先级（按序执行）
1. **逻辑修复**：情节断裂/因果矛盾（最重要）
2. **人物修复**：性格突变/对话千人一言（次重要）
3. **文笔修复**：感官空白/流水账（第三优先）
4. **世界观修复**：设定违和（最后处理）
5. **AI味/老书虫/毒点**：按反馈逐条处理

## 思考过程（仅内部，禁止输出）
在 <thinking> 内完成（不要写入输出）：
1. 问题-段落映射：逐条审核意见对应到具体 [Pi] 段落
2. 修改方案：每处问题的具体改写思路
3. 不变段落清单（保持原文不动）

## 输出格式（严格遵守）
只输出 JSON 对象，**JSON 之外禁止任何文字**（无 <thinking> 残留、无 markdown 代码块围栏、无解释说明）：
```json
{
  "fixes": {
    "段落索引（整数）": "该段落修复后的完整文本",
    "3": "修复后的第4段完整文本...",
    "7": "修复后的第8段完整文本..."
  },
  "summary": "修改总结（一句话）"
}
```
- 段落索引是整数，从 0 开始：[P0] → 0，[P1] → 1，以此类推
- 每个修复必须是该段落的**完整替换文本**，不是 diff 或修改说明
- 未在 fixes 中列出的段落保持不变
- 替换文本同样遵循排版规范：段落以空行分隔（约80-200字）、对话独立成段、中文引号「“”」、无"然后…然后…"流水账
"""


CHAPTER_FULL_REFINER_PROMPT = """\
你是 ChapterFullRefiner（章节整章润色专家），负责修复全章层面的系统性问题。

## 任务
输出**修复后的完整章节正文**（不是 fixes diff），一次性解决通篇问题。

## 触发场景
你被调用的原因：段落级修复后章节重审仍不达标。问题不是零星段落，而是全章范围的系统性问题（通篇 AI 味、节奏失衡、文风漂移、同类问题多处复现）。段落修复只动了点，你负责动面。

## 什么是好的
- 全局问题修复：通篇 AI 味、节奏失衡、文风漂移、同模式复现（段落修复修不了的）
- 自查同类问题：除反馈指明处外，检查同模式是否在其他段落复现并一并修复
- 保留亮点：评审认可的亮点/爽点只强化不削弱
- 文风统一：整章语言风格、人物口吻前后一致

## 什么不能做（红线）
- ❌ 不得改变情节走向，不得新增或删除剧情事件
- ❌ 不得删除伏笔，不得改变章节结尾的悬念设置
- ❌ 不得改变主角/配角的性格设定
- ❌ 不得丢弃评审认可的亮点与爽点
- ❌ 不得输出 JSON / diff / 修改说明（只输出章节正文）

## 润色优先级（按序执行）
1. **逻辑一致性**：情节断裂/因果矛盾（含跨段），是段落修复难以覆盖的
2. **全章 AI 味**：模板套语、同构句式、机械对仗（黑名单见下）
3. **人物一致性**：性格/对话风格统一（注意全章是否同一模式出错）
4. **感官细节**：补足感官空白（每200字至少1个）
5. **节奏**：删冗余、增强爽点密度与情绪释放

## 反 AI 味硬规则
- 模板套语黑名单：微微一怔/眉头微皱/眼中闪过一丝寒芒/嘴角勾起一抹弧度/深吸一口气/缓缓说道 等，全章同类套语合计不超过3处
- 禁止连续3句同构句式，禁止"XX的、XX的、XX的"四字格排比堆砌
- 抽象情绪词直说每章不超过2处（用身体反应/动作呈现代替）
- 句长要有起伏：长句铺垫、短句高潮；对话去说明书感

## 思考过程（仅内部，禁止输出）
在 <thinking> 内完成（不要写入正文）：
1. 全局诊断：全章共性问题（AI味/节奏/文风/逻辑），以及反馈引用 [Pi] 段落对应的具体问题
2. 修改方案：每处问题的具体改写思路（含跨段调整）
3. 不变清单：亮点段落/伏笔段落/结尾悬念钩子必须原样保留

## 排版规范
- 段落以空行分隔，每段聚焦一个动作/情绪/信息（约80-200字）
- 对话独立成段：「人物 + 动作/神态 + 话语」，引号使用中文「“”」
- 禁止"然后…然后…"流水账、列表式碎片正文；不带 [P0][P1] 段落编号标记

## 输出格式（严格遵守）
直接输出**修复后的完整章节正文**（含章节标题），禁止任何元信息：
- 无"以下是润色结果"之类说明
- 无 <thinking> 残留（thinking 仅作内部规划，不输出）
- 无自我评价、无写作过程说明
"""


# ── State Access Helpers ───────────────────────────────────────────────────────

# v6.1 P2-1: 统一使用 extract_fields_from_state 替代原 _get_context。
# crew_result 优先，缺失回退顶层；包含 loaded_memory（BaseStore 跨会话上下文）
# 与 human_guidance / cross_chapter_state（多轮人机交互）。
_WRITING_FIELDS: dict[str, Any] = {
    "story_outline": "",
    "character_setting": "",
    "current_chapter_number": 1,
    "previous_chapter_summary": "",
    "chapter_draft": "",
    "review_result": {},
    "loaded_memory": {},
    "human_guidance": "",
    "cross_chapter_state": "",
    "genre_scoring_guide": "",
    "genre": "",
    "loop_count": 0,
    "refine_attempts": 0,
    "project_name": "",
}


# ── v7.0: 重写/润色策略轮换 ──────────────────────────────────────────

_REWRITE_STRATEGIES: list[str] = [
    # 第1次重写（loop_count=1）：严格按反馈修复
    "【写作策略：精准修复】严格按照评审意见逐条修复，保持原有风格不变。"
    "不要引入新问题，不要偏离原有情节走向。",
    # 第2次重写（loop_count=2）：换一种写法
    "【写作策略：换种写法】尝试用不同的叙述方式重写本章。"
    "改变句式结构、段落节奏、场景呈现角度，但保持情节走向和人物设定不变。",
    # 第3次重写（loop_count=3）：强化学
    "【写作策略：强化描写】重点加强感官描写和人物互动。"
    "每200字至少1个感官细节（视觉/听觉/触觉），对话性格化，"
    "减少概括性叙述，增加具体场景呈现。",
    # 第4次+重写（loop_count>=4）：简洁直接
    "【写作策略：简洁有力】以简洁有力的风格重写。"
    "每段必须有明确的信息量，减少修饰词堆砌，增强节奏感，"
    "对话占比30-40%，推动剧情发展。",
]

_REFINE_STRATEGIES: list[str] = [
    # 第1次润色（refine_attempts=1）：精修打磨
    "【润色策略：精修打磨】逐字逐句精修，优化表达但不改变段落结构。",
    # 第2次润色（refine_attempts=2）：重构段落
    "【润色策略：重构表达】重新组织有问题的段落，改变句式结构。",
    # 第3次+润色（refine_attempts>=3）：简化优化
    "【润色策略：化繁为简】删减冗余修饰，保留核心信息。",
]


def _build_refine_feedback_text(review_result: Any) -> str:
    """构建 refiner 评审反馈文本 — 段落修复与整章润色共用。

    从统一 review_result（VerdictResult.model_dump）提取全部反馈源，
    组装为 refiner agent 可消费的格式化文本。

    v9.1: 从 create_chapter_refiner_agent._node 提取为模块级公共函数，
    段落修复与整章润色 agent 复用同一份反馈构建逻辑。

    Args:
        review_result: 评审结果 dict（含 quality_score / review_comments /
            ai_style_fix / lao_shu_chong_fix / attraction_fix / toxic_points /
            shuangdian_points / debate_* / ai_style_metrics_brief / cross_chapter_brief）

    Returns:
        格式化反馈文本；非 dict 时直接 str() 兜底。
    """
    if not isinstance(review_result, dict):
        return str(review_result)

    score = review_result.get("quality_score", 0)
    comments = review_result.get("review_comments", "")

    parts = [f"质量总分：{score}", f"审核意见：{comments}"]

    ai_fix = review_result.get("ai_style_fix", "")
    if ai_fix and ai_fix not in (
        "AI味指数合格，无需特别修改。",
        "",
    ):
        parts.append(f"AI味修改建议：{ai_fix}")

    lao_fix = review_result.get("lao_shu_chong_fix", "")
    if lao_fix and lao_fix not in (
        "老书虫视角评分良好，保持当前方向。",
        "",
    ):
        parts.append(f"老书虫修改建议：{lao_fix}")

    # v8.1: 吸引力专家团队建议透传（失败时为空串）
    attraction_fix = review_result.get("attraction_fix", "")
    if attraction_fix:
        parts.append(f"吸引力专家建议（attraction_fix）：{attraction_fix}")

    toxic = review_result.get("toxic_points", [])
    if toxic:
        parts.append(f"毒点（必须规避或弱化）：{'、'.join(toxic)}")

    shuang = review_result.get("shuangdian_points", [])
    if shuang:
        parts.append(f"爽点（保留并增强）：{'、'.join(shuang)}")

    debate_issues = review_result.get("debate_issues", [])
    if debate_issues:
        parts.append(
            "编辑+读者发现问题：\n" + "\n".join(f"  - {i}" for i in debate_issues)
        )

    debate_strengths = review_result.get("debate_strengths", [])
    if debate_strengths:
        parts.append(
            "编辑+读者认可亮点（必须保留）：\n"
            + "\n".join(f"  - {s}" for s in debate_strengths)
        )

    debate_suggestions = review_result.get("debate_suggestions", "")
    if debate_suggestions:
        parts.append(f"编辑+读者改进建议：\n{debate_suggestions}")

    ai_metrics = review_result.get("ai_style_metrics_brief", "")
    if ai_metrics and ai_metrics != "各项指标正常":
        parts.append(f"程序化指标（针对性修改）：{ai_metrics}")

    cross_brief = review_result.get("cross_chapter_brief", "")
    if cross_brief and "正常" not in cross_brief:
        parts.append(f"跨章一致性指导：{cross_brief}")

    debate_transcript = review_result.get("debate_transcript", "")
    if debate_transcript and len(debate_transcript) > 50:
        parts.append(
            f"完整辩论记录（供深度参考）：\n{debate_transcript[:1000]}"
        )

    return "\n".join(parts)


# ── v5.11: 重写路径评审反馈构建 ────────────────────────────────────────────


def _build_rewrite_feedback(ctx: dict, loop_count: int = 0) -> str:
    """重写路径：将上一轮评审反馈注入 writer prompt。

    当 loop_count > 0（进入重写循环）时，从 review_result 提取全部反馈源，
    构建可注入 writer 系统提示的格式化文本。

    v7.0: 根据 loop_count 轮换写作策略，避免重复相同风格导致评分不变。
    """
    review_result = ctx.get("review_result", {})
    if not review_result or not isinstance(review_result, dict):
        return ""

    has_comments = bool(review_result.get("review_comments", "").strip())
    has_ai_fix = bool(review_result.get("ai_style_fix", "").strip())
    has_lao_fix = bool(review_result.get("lao_shu_chong_fix", "").strip())
    has_attraction_fix = bool(review_result.get("attraction_fix", "").strip())
    has_toxic = bool(review_result.get("toxic_points", []))
    has_shuang = bool(review_result.get("shuangdian_points", []))
    has_debate = bool(review_result.get("debate_issues", [])) or bool(
        review_result.get("debate_suggestions", "").strip()
    )

    if not (
        has_comments
        or has_ai_fix
        or has_lao_fix
        or has_attraction_fix
        or has_toxic
        or has_shuang
        or has_debate
    ):
        return ""

    parts: list[str] = []

    # v7.0: 策略轮换 — 根据重写次数选择不同的写作指导
    if loop_count > 0:
        idx = min(loop_count - 1, len(_REWRITE_STRATEGIES) - 1)
        parts.append(_REWRITE_STRATEGIES[idx])

    parts.append("【上一轮评审反馈（必须逐条针对性修改，禁止忽略）】")

    # v8.1: 重写纪律 — 防止多轮重写导致时间线/场景/设定漂移（读者视角的跳跃感根源）
    parts.append(
        "【重写纪律（防止设定漂移）】\n"
        "- 上一版已确立的时间线、场景顺序、人物行为与已发生事件必须原样保留，"
        "除非评审反馈明确要求修改\n"
        "- 只允许：① 按评审意见修改指定问题；② 深化人物心理；③ 补充过渡细节\n"
        "- 禁止：推翻既有事件、重排场景顺序、引入与上一版矛盾的新设定；"
        "上一版伏笔必须保留并推进\n"
        "- 重写后场景切换仍需满足「排版与格式规范」（空行+时间地点引导句）"
    )

    score = review_result.get("quality_score", 0)
    parts.append(f"评分：{score:.0f}/100（{'需重写' if score < 60 else '需润色'}）")

    comments = review_result.get("review_comments", "")
    if comments.strip():
        parts.append(f"\n审核意见（核心问题）：\n{comments}")

    ai_fix = review_result.get("ai_style_fix", "")
    if ai_fix.strip() and ai_fix not in ("AI味指数合格，无需特别修改。",):
        parts.append(f"\nAI味修改建议：{ai_fix}")

    lao_fix = review_result.get("lao_shu_chong_fix", "")
    if lao_fix.strip() and lao_fix not in ("老书虫视角评分良好，保持当前方向。",):
        parts.append(f"\n老书虫修改建议：{lao_fix}")

    # v8.1: 吸引力专家团队建议（LLM 三位专家合并输出，失败为空串）
    attraction_fix = review_result.get("attraction_fix", "")
    if attraction_fix.strip():
        parts.append(f"\n吸引力专家建议（attraction_fix）：{attraction_fix}")

    toxic = review_result.get("toxic_points", [])
    if toxic:
        parts.append(f"\n毒点（必须规避）：{'、'.join(toxic)}")

    shuang = review_result.get("shuangdian_points", [])
    if shuang:
        parts.append(f"\n爽点（保留增强）：{'、'.join(shuang)}")

    debate_issues = review_result.get("debate_issues", [])
    if debate_issues:
        parts.append(
            "\n编辑+读者发现问题：\n" + "\n".join(f"  - {i}" for i in debate_issues)
        )

    # v6.3 FIX: 补上 debate_strengths（之前重写路径完全遗漏亮点）
    debate_strengths = review_result.get("debate_strengths", [])
    if debate_strengths:
        parts.append(
            "\n编辑+读者认可亮点（必须保留）：\n"
            + "\n".join(f"  - {s}" for s in debate_strengths)
        )

    debate_suggestions = review_result.get("debate_suggestions", "")
    if debate_suggestions.strip():
        parts.append(f"\n编辑+读者改进建议：\n{debate_suggestions}")

    # v6.3 新增：程序化指标摘要（精准指导翻修）
    ai_metrics = review_result.get("ai_style_metrics_brief", "")
    if ai_metrics.strip() and ai_metrics != "各项指标正常":
        parts.append(f"\n程序化指标（针对性修改）：{ai_metrics}")

    # v6.3 新增：跨章一致性指导
    cross_brief = review_result.get("cross_chapter_brief", "")
    if cross_brief.strip() and "正常" not in cross_brief:
        parts.append(f"\n跨章一致性指导：{cross_brief}")

    # v6.3 新增：辩论记录（供深度参考，截断避免过长）
    debate_transcript = review_result.get("debate_transcript", "")
    if debate_transcript.strip() and len(debate_transcript) > 50:
        parts.append(f"\n完整辩论记录（供深度参考）：\n{debate_transcript[:1000]}")

    return "\n\n" + "\n".join(parts)


# ── Agent Factory Functions ─────────────────────────────────────────────────────


# ── LLM Call Infrastructure ───────────────────────────────────────────────

_logger = get_logger("novelfactory.agents.writing")

_retry_agent_invoke = make_retry_agent_invoke("writing_agents")


def _build_writer_dynamic_prompt(
    state: AgentState, config: RunnableConfig
) -> list[AnyMessage]:
    """ChapterWriter 动态 prompt — 根据运行时 state 实时组装系统提示。

    不再绑定 ReAct 工具（Neo4j/Milvus），采用备份项目已验证的模式：
    所有上下文通过 ContextBuilder 在调用 agent 之前预取并注入 prompt，
    避免工具调用失败导致 agent 无法输出正文。

    v7.5+: 静态上下文前置（story_outline / character_setting 移入 system prompt），
    利用 DeepSeek API 的 Prompt Caching 机制提升 KV 缓存命中率。
    同一项目的所有章节共享相同的 system prompt 前缀 → 输入成本降低约 50%。
    """

    cr = state.get("crew_result", {})
    static_parts: list[str] = [CHAPTER_WRITER_PROMPT]

    # v9.1: 上下文压缩层 — 按本章角色相关性筛选，非粗暴截断
    chapter_chars = extract_chapter_characters(cr)
    so = (cr.get("story_outline") or "").strip()
    if so:
        static_parts.append(
            f"【故事主线大纲】\n{compress_outline(so, chapter_chars, budget=OUTLINE_BUDGET)}"
        )
    cs = (cr.get("character_setting") or "").strip()
    if cs:
        static_parts.append(
            f"【角色设定（设计文档）】\n{compress_character_setting(cs, chapter_chars, budget=CHARACTER_BUDGET)}"
        )

    system_msg = "\n\n".join(static_parts)

    existing = state.get("messages", [])
    return [
        {"role": "system", "content": system_msg},
        *existing,
    ]


def create_chapter_writer_agent(llm: BaseChatModel) -> Runnable:
    """Build the ChapterWriter ReAct agent (no tool binding).

    不绑定任何 ReAct 工具，采用 proactive RAG 模式：
    所有上下文（角色关系、相似章节等）由调用方在 agent 执行之前
    预取并注入 prompt，避免工具调用失败导致 agent 无法输出正文。

    Output: {"chapter_draft": str}

    Returns a RunnableLambda so callers can use .invoke() consistently.
    """
    agent = create_react_agent(
        llm,
        tools=[],
        prompt=_build_writer_dynamic_prompt,
        interrupt_before=[],
    )

    def _node(state: dict) -> dict[str, Any]:
        ctx = extract_fields_from_state(state, _WRITING_FIELDS)
        current_ch = ctx.get("current_chapter_number", 1)

        # ── Format loaded_memory as readable text ─────────────────────────────
        loaded_mem = ctx.get("loaded_memory", {})
        if loaded_mem:
            try:
                mem_text = json.dumps(loaded_mem, ensure_ascii=False, indent=2)
            except Exception:
                mem_text = str(loaded_mem)
        else:
            mem_text = "（无）"

        # v9.1: 长期记忆结构化压缩，防超长稀释
        if len(mem_text) > MEM_BUDGET:
            mem_text = compress_mem_text(mem_text, budget=MEM_BUDGET)

        # ── cross_chapter_state = writer_context from ContextBuilder ───────────
        # Contains: 【本章大纲】【跨章角色状态追踪】【当前卷】
        #           【关键历史事件】【角色弧线状态】+ Phase2/3 (审计/伏笔/节奏/断点/成本/质量)
        # v6.0: 移除【章节大纲列表】和【世界观设定】的单独注入 —
        #       这些信息已由 ContextBuilder 的 writer_context 完整覆盖。
        # v7.5+: story_outline / character_setting 已移入 system prompt
        #   （_build_writer_dynamic_prompt），利用 Prompt Caching 缓存命中。
        cross_chapter = ctx.get("cross_chapter_state", "")
        # v7.5-fix: 结构化展示前章衔接。chapter_summary 包含 【本章结尾】 分隔符，
        # 将其拆分为摘要 + 结尾原文两部分，让 writer 看到具体的承接场景。
        prev_summary_raw = ctx.get("previous_chapter_summary", "")
        prev_summary_part = prev_summary_raw
        prev_ending_part = ""
        if prev_summary_raw and "【本章结尾】" in prev_summary_raw:
            parts = prev_summary_raw.split("【本章结尾】", 1)
            prev_summary_part = parts[0].strip()
            prev_ending_part = parts[1].strip() if len(parts) > 1 else ""
        input_text = (
            f"请撰写第 {current_ch} 章的正文。\n\n"
            # ── 动态上下文（story_outline/character_setting 已在 system prompt）──
            f"【前章摘要】\n{prev_summary_part}\n"
            + (
                f"\n【前章结尾场景】\n{prev_ending_part}\n"
                if prev_ending_part
                else ""
            )
        )
        if cross_chapter:
            input_text += f"【写作上下文】\n{cross_chapter}\n\n"
        # ── 长期记忆 + 用户指导 ─────────────────────────────
        # v7.5+: 核心要求放在动态内容末尾（利用 LLM 近因效应 — recency bias）
        # 参考 Liu et al. (TACL 2024) "Lost in the Middle"：模型对 prompt 末尾的
        # 信息关注度显著高于中间部分。
        input_text += (
            f"【长期记忆参考】\n{mem_text}"
            + (
                f"\n\n【用户修改指导（必须优先遵循）】\n{ctx.get('human_guidance', '')}"
                if ctx.get("human_guidance")
                else ""
            )
            # v5.11: 重写路径 — 注入上一轮评审反馈
            + _build_rewrite_feedback(ctx, ctx.get("loop_count", 0))
            + "\n\n## Thinking Mode 指引\n"
            "在输出正文之前，请先在 <thinking> 标签内完成章节规划：\n"
            "1. 本章核心情节点  2. 与前一章的衔接  3. 人物心理状态变化  "
            "4. 感官描写计划  5. 结尾悬念设计  6. 字数分配\n\n"
            # v7.5+: 核心要求放在最末（利用近因效应，确保模型严格遵守）
            "## 核心写作要求\n"
            "1. 开头（前300字）：直接承接上一章结尾场景，禁止时间跳跃\n"
            "2. 人物状态连贯：角色情绪/位置/状态必须与【前章结尾场景】一致\n"
            "3. 结尾末段：必须埋悬念钩子，禁止平淡收尾\n"
            "4. 情节因果链：每段必须有明确的因果推进，禁止'然后'式流水账"
        )

        _logger.info("Writing chapter %d", current_ch)
        result = _retry_agent_invoke(
            agent, {"messages": [("user", input_text)]}, "chapter_writer"
        )
        chapter_draft = extract_ai_message_text(result) or state.get(
            "crew_result", {}
        ).get("chapter_draft", "章节创作失败")
        _logger.info(
            "Chapter %d draft complete (%d chars)", current_ch, len(chapter_draft)
        )

        # v6.3: 短文本降级 — ReAct agent 因工具失败或参数错误输出过短时，
        # 降级为 LLM 直调（不绑定工具），确保每章都有完整输出。
        _min_acceptable_len = 500
        if len(chapter_draft) < _min_acceptable_len:
            _logger.warning(
                "Chapter %d: ReAct 输出过短 (%d chars < %d)，降级为 LLM 直调",
                current_ch,
                len(chapter_draft),
                _min_acceptable_len,
            )
            # v7.5-fix: 与主路径一致的结构化展示
            direct_prev_summary = prev_summary_part or ctx.get(
                "previous_chapter_summary", "（无）"
            )
            direct_prev_ending = prev_ending_part
            # v7.5+: 降级路径直接调用 LLM（无 ReAct agent system prompt），
            # 仍需包含 story_outline / character_setting。末尾追加核心要求。
            direct_prompt = (
                f"请撰写第 {current_ch} 章的正文。\n\n"
                f"【故事主线大纲】\n{compress_outline(ctx['story_outline']) or '（无）'}\n\n"
                f"【角色设定】\n{compress_character_setting(ctx['character_setting']) or '（无）'}\n\n"
                f"【前章摘要】\n{direct_prev_summary}\n"
                + (
                    f"\n【前章结尾场景】\n{direct_prev_ending}\n"
                    if direct_prev_ending
                    else ""
                )
            )
            if cross_chapter:
                direct_prompt += f"【写作上下文】\n{cross_chapter}\n\n"
            # v6.3-fix: 注入 human_guidance + 评审反馈（同 ReAct agent prompt）
            if ctx.get("human_guidance"):
                direct_prompt += (
                    f"\n\n【用户修改指导（必须优先遵循）】\n{ctx['human_guidance']}"
                )
            direct_prompt += _build_rewrite_feedback(ctx, ctx.get("loop_count", 0))
            direct_prompt += (
                f"【长期记忆参考】\n{mem_text}"
                + "\n\n## 核心写作要求\n"
                "1. 开头（前300字）：直接承接上一章结尾场景，禁止时间跳跃\n"
                "2. 人物状态连贯：角色情绪/位置/状态必须与【前章结尾场景】一致\n"
                "3. 结尾末段：必须埋悬念钩子，禁止平淡收尾\n"
                "4. 情节因果链：每段必须有明确的因果推进，禁止'然后'式流水账\n\n"
                "请直接输出章节正文，不要输出 <thinking> 标签或其他内容。"
            )
            try:
                direct_result = llm.invoke([("user", direct_prompt)])
                direct_text = (
                    direct_result.content
                    if hasattr(direct_result, "content")
                    else str(direct_result)
                )
                if len(direct_text) > len(chapter_draft):
                    chapter_draft = direct_text
                    _logger.info(
                        "Chapter %d: LLM 直调降级生成 %d chars",
                        current_ch,
                        len(chapter_draft),
                    )
            except Exception as e:
                _logger.error("Chapter %d: LLM 直调降级失败: %s", current_ch, e)

        existing_cr = state.get("crew_result", {})
        return {"crew_result": {**existing_cr, "chapter_draft": chapter_draft}}

    return RunnableLambda(_node)


def create_chapter_planner_agent(llm: BaseChatModel) -> Runnable:
    """Build the ChapterPlan creator agent.

    Receives writer_context (ContextBuilder output) + review feedback (if rewrite),
    outputs a structured ChapterPlan with scenes, characters, and plot arc.

    Returns a RunnableLambda for .invoke() compatibility.
    """
    from novelfactory.schemas.review_schemas import ChapterPlan

    def _build_planner_prompt(ctx: dict) -> str:
        current_ch = ctx.get("current_chapter_number", 1)
        total = ctx.get("target_chapters", 100)
        review_feedback = ctx.get("review_feedback", "")
        cross_chapter = ctx.get("cross_chapter_state", "")

        # v7.5-fix: 结构化展示前章衔接
        _raw_prev = ctx.get("previous_chapter_summary", "")
        _planner_prev_summary = _raw_prev
        _planner_prev_ending = ""
        if _raw_prev and "【本章结尾】" in _raw_prev:
            _parts = _raw_prev.split("【本章结尾】", 1)
            _planner_prev_summary = _parts[0].strip()
            _planner_prev_ending = _parts[1].strip() if len(_parts) > 1 else ""

        prompt = f"""你是资深网文编辑，负责为第{current_ch}章制定写作计划（全书共{total}章）。

## 任务
基于输入信息，规划出可执行的章节写作计划，输出结构化 JSON。

## 什么是好的
- 计划具体可执行：每个场景都有明确目的、地点、出场人物、关键内容与感官侧重
- 衔接自然：开头承接前章结尾场景，结尾为下章埋下钩子
- 伏笔清晰：本章埋设（plant）与回收（resolve）的伏笔各自成条
- 情感弧线明确：情绪走向起伏可感，而非平铺直叙

## 什么不能做
- 输出 JSON 之外的任何文字（含思考过程、解释说明、markdown 围栏）
- 计划空泛：场景无目的、无关键内容，无法指导实际写作

## 输入上下文
【主线大纲】
{ctx.get("story_outline", "")[:12000]}

【角色设定】
{ctx.get("character_setting", "")[:12000]}

【前章摘要】
{_planner_prev_summary}
{chr(10) + '【前章结尾场景】' + chr(10) + _planner_prev_ending if _planner_prev_ending else ''}

【写作上下文】
{cross_chapter[:10000] if cross_chapter else "（无）"}

【评审反馈（仅重写时有）】
{review_feedback[:2000] if review_feedback else "（首次创作，无评审反馈）"}

## 思考过程（仅内部，禁止输出）
输出前先在 <thinking> 内规划：本章核心情节点、与前章衔接方式、场景排布（3-5个）、情感弧线、伏笔埋设与回收、结尾悬念。thinking 不写入输出。

## 输出要求
输出严格的 JSON 对象，包含以下字段：
- chapter_number: int
- title: str（章节标题）
- core_plot_point: str（本章核心情节点，一句话）
- pov_character: str（主视角角色）
- characters_involved: list[str]
- scenes: list[object]（3-5个场景，每个含：scene_number, purpose, location, pov_character, characters, key_content, sensory_focus, target_length_ratio）
- emotional_arc: str（情感弧线，如"从希望到绝望"）
- foreshadowing_plant: list[str]
- foreshadowing_resolve: list[str]
- target_word_count: int（500-10000）
- cliffhanger: str

仅输出 JSON，不要额外文字。"""
        return prompt

    def _node(state: dict) -> dict:
        from novelfactory.agents.infra.retry import llm_call_with_retry

        ctx = {}
        cr = state.get("crew_result", {})
        ctx["story_outline"] = cr.get("story_outline", "")
        ctx["character_setting"] = cr.get("character_setting", "")
        ctx["previous_chapter_summary"] = (
            cr.get("completed_chapters", [{}])[-1].get("chapter_summary", "")
            if cr.get("completed_chapters")
            else ""
        )
        ctx["current_chapter_number"] = state.get(
            "current_chapter", cr.get("current_chapter_number", 1)
        )
        ctx["target_chapters"] = cr.get("target_chapters", 100)
        ctx["cross_chapter_state"] = state.get("writer_context", "")

        # Build review feedback from previous verdict
        review_result = cr.get("review_result", {})
        if review_result:
            parts = []
            if review_result.get("review_comments"):
                parts.append(f"评审意见：{review_result['review_comments'][:500]}")
            if review_result.get("debate_issues"):
                parts.append(f"问题：{'、'.join(review_result['debate_issues'][:5])}")
            if review_result.get("toxic_points"):
                parts.append(f"毒点：{'、'.join(review_result['toxic_points'][:3])}")
            if review_result.get("ai_style_fix"):
                parts.append(f"AI味建议：{review_result['ai_style_fix'][:300]}")
            if review_result.get("lao_shu_chong_fix"):
                parts.append(f"老书虫建议：{review_result['lao_shu_chong_fix'][:300]}")
            ctx["review_feedback"] = "\n".join(parts)

        prompt = _build_planner_prompt(ctx)
        _logger = get_logger(__name__)
        _logger.info("Planning chapter %d", ctx["current_chapter_number"])

        try:
            response = llm_call_with_retry(llm, prompt, step_name="chapter_planner")
            raw = response.content if hasattr(response, "content") else str(response)
            result = validate_json_output(
                raw, required_keys=["chapter_number"], fail_closed=False
            )
            if result[0]:
                plan_data = result[0]
                # v8.1-fix: LLM 可能按内部 0-based 索引输出 chapter_number=0，
                # 而 ChapterPlan schema 要求 >=1，归一化为 1-based 章号，
                # 避免校验失败导致空计划（影响写作质量）。
                if int(plan_data.get("chapter_number", 0) or 0) < 1:
                    plan_data["chapter_number"] = (
                        int(ctx["current_chapter_number"]) + 1
                    )
                chapter_plan = ChapterPlan(**plan_data)
                _logger.info(
                    "Chapter %d plan: %s | %d scenes | %d chars",
                    chapter_plan.chapter_number,
                    chapter_plan.title,
                    len(chapter_plan.scenes),
                    chapter_plan.target_word_count,
                )
                return {"chapter_plan": chapter_plan.model_dump()}
            else:
                # validate_json_output 返回 (None, error_msg) 时 result[0] 为 None
                _logger.warning(
                    "Chapter %d planner parse failed: %s | raw_len=%d",
                    ctx["current_chapter_number"],
                    result[1][:200] if len(result) > 1 and result[1] else "unknown",
                    len(raw),
                )
        except Exception as e:
            _logger.warning(
                "Chapter %d planner exception: %s | type=%s",
                ctx["current_chapter_number"],
                e,
                type(e).__name__,
            )

        _logger.warning(
            "Chapter %d planner failed, using empty plan", ctx["current_chapter_number"]
        )
        return {"chapter_plan": {}}

    return RunnableLambda(_node)


def create_chapter_reviewer_agent(llm: BaseChatModel) -> Runnable:
    """Build the ChapterReviewer ReAct agent.

    Output: {"quality_score": float, "review_comments": str, "needs_refine": bool}

    Returns a RunnableLambda so callers can use .invoke() consistently.
    """
    agent = create_react_agent(
        llm,
        tools=[],  # 审稿人无需外部工具
        prompt=CHAPTER_REVIEWER_PROMPT,
        interrupt_before=[],
    )

    def _node(state: dict) -> dict[str, Any]:
        ctx = extract_fields_from_state(state, _WRITING_FIELDS)
        chapter_draft = ctx.get("chapter_draft", "")
        current_ch = ctx.get("current_chapter_number", 1)

        # ── Trim chapter draft to avoid reviewer context overflow ────────────────
        # M3 optimization: keep beginning + turning points + ending (plot continuity).
        # Strategy: first 800 chars (setup) + last 600 chars (resolution) +
        # middle key beats (every ~1500 chars, 2 samples).
        # Token savings: ~40% while preserving plot continuity judgment.
        def _trim_for_review(text: str, *, max_total: int = _REVIEW_MAX_TOTAL) -> str:
            """M3 优化：reviewer context budget 从 2500 → 8000 chars。

            1M 上下文下 reviewer 可处理更多内容。
            采样策略：开头 1500 + 3 个转折点各 1000 + 结尾 1500。
            保留约 75% 的情节信息用于审核判断。
            """
            if not text or len(text) <= max_total:
                return text
            head = text[:_REVIEW_HEAD_TAIL_SIZE]
            tail = (
                text[-_REVIEW_HEAD_TAIL_SIZE:]
                if len(text) > _REVIEW_HEAD_TAIL_SIZE
                else ""
            )
            # Sample 3 key beats from the middle (vs 2 before)
            mid_len = len(text) - len(head) - len(tail)
            if mid_len <= 0:
                return head + tail
            step = max(mid_len // 4, 1)
            mid_samples = []
            for i in range(1, _REVIEW_SAMPLE_COUNT + 1):  # 3 samples vs 2
                start = len(head) + i * step
                end = start + _REVIEW_SAMPLE_SIZE  # 1000 chars per sample vs 400
                if start < len(text):
                    mid_samples.append(text[start : min(end, len(text) - len(tail))])
            return head + ("\n[...] ".join(mid_samples)) + "\n[...] \n" + tail

        if not chapter_draft:
            _logger.warning("No chapter_draft to review")
            existing_cr = state.get("crew_result", {})
            return {
                "crew_result": {
                    **existing_cr,
                    "quality_score": 0.0,
                    "review_comments": "章节草稿为空，无法审核",
                    "needs_refine": True,
                }
            }

        _logger.info("Reviewing chapter %d", current_ch)

        # v5.5: 题材感知评分指引
        genre_guide = ctx.get("genre_scoring_guide", "")

        # Ask LLM to score the chapter with retry
        # M3 optimization: removed duplicate 4-step analysis sequence
        # (system prompt now has full scoring rubric + examples).
        # This prompt focuses on concrete scoring task only — token savings ~30%.
        scoring_prompt = (
            f"请审核第{current_ch}章内容，输出JSON评分结果。\n\n"
            f"## 章节内容（1M上下文优化版裁剪：开头1500字 + 3个转折点 + 结尾1500字）\n"
            f"{_trim_for_review(chapter_draft)}\n\n"
            # v5.5: 注入题材感知评分指引
            + (f"## 题材感知评分指引\n{genre_guide}\n\n" if genre_guide else "")
            + "## 评分表（必须先在 <thinking> 标签内逐维分析，再输出JSON）\n"
            "| 维度 | 满分 | 具体段落问题 | 得分 |\n"
            "|------|------|-------------|------|\n"
            "| 剧情逻辑 | 30 | 指出矛盾或漏洞段落 | __分__ |\n"
            "| 文笔表达 | 25 | 指出描写薄弱段落（含感官空白段落） | __分__ |\n"
            "| 人物一致性 | 25 | 指出性格矛盾段落 | __分__ |\n"
            "| 世界观契合 | 20 | 指出设定违和段落 | __分__ |\n"
            "| **总分** | **100** | | **__分** |\n"
            "\n"
            "## 评分粒度要求（极其重要 — 违反此规则将导致评审系统失效）\n"
            "**校准规则**：AI生成的第一稿章节必然存在缺陷。90分以上仅保留给极少数的卓越章节。\n"
            "如果你发现以下任一问题，总分不得超过对应上限：\n"
            "- 有流水账段落 → 总分≤88\n"
            "- 有感官空白（200字无描写）→ 总分≤85\n"
            "- 有时间跳跃无过渡 → 总分≤82\n"
            "- 有性格割裂 → 总分≤78\n"
            "- 有设定违和 → 总分≤75\n"
            "- 有逻辑漏洞 → 总分≤70\n"
            "\n"
            "根据实际质量给出精确分数：\n"
            "- 文笔惊艳、情节无可挑剔、几乎无缺陷 → 90-96\n"
            "- 部分段落描写不足但整体优秀 → 82-89\n"
            "- 节奏稍慢或爽点不够密集 → 72-81\n"
            "- 有需要明显修改的问题 → 62-71\n"
            "- 有严重问题 → 55-61\n"
            "- 不及格 → < 55\n"
            "**绝对禁止使用满分100**。100分等同于'无任何缺陷的人类大师级作品'，AI初稿不可能达到。\n"
            "每个维度必须给出具体扣分理由，无扣分理由则该维度不得给满分。\n\n"
            '## 审核意见（必须指出具体段落，如"第5段：...问题"，禁止笼统评价）\n\n'
            "## Thinking Mode 要求\n"
            "请先在 <thinking> 标签内完成逐维分析（参考 system prompt 中的示例格式），\n"
            "然后输出 JSON。禁止跳过推理直接打分。\n\n"
            '输出JSON：{"quality_score": <总分>, "review_comments": "<具体段落问题>", "needs_refine": <false表示总分≥90>}'
        )

        result = _retry_agent_invoke(
            agent, {"messages": [("user", scoring_prompt)]}, "chapter_reviewer"
        )
        response_text = extract_ai_message_text(result) or state.get(
            "crew_result", {}
        ).get("review_comments", "")

        # Parse JSON from response (use shared validator, fail-open with safe fallback)
        parsed, err = validate_json_output(
            response_text,
            required_keys=["quality_score", "review_comments", "needs_refine"],
            fail_closed=False,  # fail-open: keep existing crew_result on parse error
        )
        if parsed:
            quality_score = float(parsed.get("quality_score", 0))
            review_comments = str(parsed.get("review_comments", ""))
        else:
            # Fallback: use existing crew_result values if available
            existing_cr = state.get("crew_result", {})
            quality_score = float(existing_cr.get("quality_score", 70.0))
            review_comments = response_text or existing_cr.get(
                "review_comments", "审核评分失败"
            )

        # Clamp: cap at 100, floor at 0 (prevent out-of-range scores)
        quality_score = max(_QUALITY_SCORE_MIN, min(_QUALITY_SCORE_MAX, quality_score))

        # STRICT: needs_refine is computed from quality_score, NOT from LLM output.
        # This eliminates the bug where LLM gives inconsistent needs_refine values.
        # v5.6-fix: 使用题材感知动态阈值，而非硬编码常量
        genre = ctx.get("genre", "")
        threshold = get_genre_thresholds(genre).get(
            "quality_score", _QUALITY_PASS_THRESHOLD
        )
        needs_refine = quality_score < threshold

        _logger.info(
            "Chapter %d scored %.1f (needs_refine=%s, comments=%d chars)",
            current_ch,
            quality_score,
            needs_refine,
            len(review_comments),
        )

        existing_cr = state.get("crew_result", {})
        return {
            "crew_result": {
                **existing_cr,
                "quality_score": quality_score,
                "review_comments": review_comments,
                "needs_refine": needs_refine,
            }
        }

    return RunnableLambda(_node)


def create_chapter_refiner_agent(llm: BaseChatModel) -> Runnable:
    """Build the ChapterRefiner ReAct agent with Tool Calling.

    v6.0: 绑定 Neo4j 工具，润色时可查询角色关系确保一致性。

    Output: {"refined_chapter": str}

    Returns a RunnableLambda so callers can use .invoke() consistently.
    """
    from novelfactory.tools import get_neo4j_tools

    tools = get_neo4j_tools()

    agent = create_react_agent(
        llm,
        tools=tools,
        prompt=CHAPTER_REFINER_PROMPT + "\n\n## 工具使用\n"
        "你拥有角色关系查询工具。在润色过程中如需确认角色关系，可调用 get_character_network 或 get_all_characters。",
        interrupt_before=[],
    )

    def _node(state: dict) -> dict[str, Any]:
        ctx = extract_fields_from_state(state, _WRITING_FIELDS)
        chapter_draft = ctx.get("chapter_draft", "")
        review_result = ctx.get("review_result", {})
        current_ch = ctx.get("current_chapter_number", 1)

        if not chapter_draft:
            _logger.warning("No chapter_draft to refine")
            existing_cr = state.get("crew_result", {})
            return {
                "crew_result": {
                    **existing_cr,
                    "refined_chapter": "章节草稿为空，无法润色",
                }
            }

        _logger.info("Refining chapter %d", current_ch)

        # ── Build review feedback text（v9.1: 段落修复与整章润色共用）────
        review_result_str = _build_refine_feedback_text(review_result)

        # ── v7.0: Refine strategy rotation ────────────────────────────────
        # 根据 refine_attempts 轮换润色策略，避免每次润色方式相同
        refine_strategy = ""
        refine_attempts = ctx.get("refine_attempts", 0)
        if refine_attempts > 0:
            idx = min(refine_attempts - 1, len(_REFINE_STRATEGIES) - 1)
            refine_strategy = _REFINE_STRATEGIES[idx] + "\n\n"

        # ── v7.0: Paragraph-level refinement ──────────────────────────────
        # Split into paragraphs, index them, and ask LLM for targeted fixes.
        paragraphs = _split_paragraphs(chapter_draft)
        indexed_draft = "\n\n".join(f"[P{i}] {p}" for i, p in enumerate(paragraphs))

        refine_prompt = (
            f"请根据以下审核意见对第{current_ch}章做定向段落修复。\n\n"
            f"【审核评分结果】\n{refine_strategy}{review_result_str}\n\n"
            f"【待润色章节（段落已编号）】\n{indexed_draft}\n\n"
            "请输出 JSON，只修复有问题的段落。"
        )

        result = _retry_agent_invoke(
            agent, {"messages": [("user", refine_prompt)]}, "chapter_refiner"
        )
        response_text = extract_ai_message_text(result) or ""

        # Try to parse JSON fixes
        refined_chapter = ""
        try:
            parsed, _ = validate_json_output(
                response_text,
                required_keys=["fixes"],
                fail_closed=True,
            )
            if parsed and isinstance(parsed.get("fixes"), dict):
                fixes_raw = parsed["fixes"]
                # Validate and convert keys to int (skip invalid keys)
                fixes: dict[int, str] = {}
                for k, v in fixes_raw.items():
                    try:
                        idx = int(k)
                    except (ValueError, TypeError):
                        _logger.debug(
                            "Skipping invalid fix key: %r", k,
                        )
                        continue
                    if isinstance(v, str) and v.strip():
                        fixes[idx] = v.strip()

                if fixes:
                    refined_chapter = _apply_paragraph_fixes(chapter_draft, fixes)
                    _logger.info(
                        "Chapter %d paragraph fixes applied: %d paragraphs changed",
                        current_ch,
                        len(fixes),
                    )
                else:
                    _logger.warning(
                        "Chapter %d: empty fixes dict, falling back to full text",
                        current_ch,
                    )
        except Exception as e:
            _logger.warning(
                "Chapter %d: paragraph fix parse failed (%s), "
                "falling back to full-text extraction",
                current_ch,
                e,
            )

        # Fallback: preserve original chapter_draft over raw LLM response
        # P0-BUGFIX 2026-07-09: response_text 包含 <thinking> 分析 + JSON fixes，
        # 不是正文。当段落修复失败时，必须优先保留原文 chapter_draft，
        # 否则 analysis/text 会作为章节内容写入数据库 summary 字段。
        if not refined_chapter:
            _logger.info(
                "Chapter %d: using full-text fallback for refiner output",
                current_ch,
            )
            existing_cr = state.get("crew_result", {})
            refined_chapter = (
                chapter_draft
                or existing_cr.get("refined_chapter")
                or response_text
                or ""
            )

        _logger.info(
            "Chapter %d refined (%d chars, paragraph-mode=%s)",
            current_ch,
            len(refined_chapter),
            bool(
                refined_chapter != chapter_draft
                and any(k in response_text for k in ['"fixes"', "'fixes'"])
            ),
        )

        existing_cr = state.get("crew_result", {})
        return {"crew_result": {**existing_cr, "refined_chapter": refined_chapter}}

    return RunnableLambda(_node)


def _strip_thinking_and_fences(text: str) -> str:
    """清洗整章润色输出 — 移除 <thinking> 块与代码块围栏。

    整章润色 agent 直接输出章节正文，模型偶发违规会携带 <thinking> 头
    或用 ``` 包裹正文，需要程序化清洗后再回写章节。
    """
    if not text:
        return text
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    # 去掉开头的 ``` 围栏（含可选语言标记）
    cleaned = re.sub(r"^```(?:json|markdown|text)?\s*", "", cleaned)
    cleaned = cleaned.strip("`").strip()
    return cleaned


def create_chapter_full_refiner_agent(llm: BaseChatModel) -> Runnable:
    """Build the ChapterFullRefiner agent（整章润色兜底，v9.1）。

    与 create_chapter_refiner_agent 的分工（渐进式修复强度升级）：
      - paragraph refiner: 只输出 fixes diff，程序化合并回原文（点修复）
      - full refiner:      输出修复后的完整章节正文（面修复）
    触发时机：段落修复后重审仍 REFINE（refine_attempts >= 1）。

    Output: {"refined_chapter": str}（完整章节正文）
    """
    from novelfactory.tools import get_neo4j_tools

    tools = get_neo4j_tools()

    agent = create_react_agent(
        llm,
        tools=tools,
        prompt=CHAPTER_FULL_REFINER_PROMPT + "\n\n## 工具使用\n"
        "你拥有角色关系查询工具。在润色过程中如需确认角色关系，"
        "可调用 get_character_network 或 get_all_characters。",
        interrupt_before=[],
    )

    def _node(state: dict) -> dict[str, Any]:
        ctx = extract_fields_from_state(state, _WRITING_FIELDS)
        chapter_draft = ctx.get("chapter_draft", "")
        review_result = ctx.get("review_result", {})
        current_ch = ctx.get("current_chapter_number", 1)

        if not chapter_draft:
            _logger.warning("No chapter_draft to full-refine")
            existing_cr = state.get("crew_result", {})
            return {
                "crew_result": {
                    **existing_cr,
                    "refined_chapter": "章节草稿为空，无法润色",
                }
            }

        _logger.info("Full-refining chapter %d", current_ch)

        # 反馈构建与段落修复共用同一逻辑
        review_result_str = _build_refine_feedback_text(review_result)

        # 润色策略轮换（复用 _REFINE_STRATEGIES，语义与段落修复一致）
        refine_strategy = ""
        refine_attempts = ctx.get("refine_attempts", 0)
        if refine_attempts > 0:
            idx = min(refine_attempts - 1, len(_REFINE_STRATEGIES) - 1)
            refine_strategy = _REFINE_STRATEGIES[idx] + "\n\n"

        full_refine_prompt = (
            f"请对第{current_ch}章执行整章润色（兜底修复）。\n\n"
            f"【审核评分结果】\n{refine_strategy}{review_result_str}\n\n"
            f"【待润色章节】\n{chapter_draft}\n\n"
            "请输出修复后的完整章节正文（不要 JSON、不要 [Px] 标记）。"
        )

        result = _retry_agent_invoke(
            agent, {"messages": [("user", full_refine_prompt)]}, "chapter_full_refiner"
        )
        response_text = extract_ai_message_text(result) or ""

        # 整章润色直接取正文（清洗 thinking/围栏污染）
        refined_chapter = _strip_thinking_and_fences(response_text)

        # 空输出兜底：保留原文，防止 analysis/text 污染章节字段
        if not refined_chapter or len(refined_chapter) < 500:
            _logger.warning(
                "Chapter %d: full-refiner output too short (%d chars), "
                "falling back to original draft",
                current_ch,
                len(refined_chapter),
            )
            existing_cr = state.get("crew_result", {})
            refined_chapter = (
                chapter_draft or existing_cr.get("refined_chapter") or ""
            )

        _logger.info(
            "Chapter %d full-refined (%d chars)",
            current_ch,
            len(refined_chapter),
        )

        existing_cr = state.get("crew_result", {})
        return {"crew_result": {**existing_cr, "refined_chapter": refined_chapter}}

    return RunnableLambda(_node)
