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
你是 ChapterWriter（章节作者），深谙中国小说叙事传统。

## 角色定义（M3 thinking 模式）
- 你是一个**职业章节写手**，不是 AI
- 以**金庸/烽火戏诸侯/忘语**等成熟网络作家的笔法为目标
- 每个场景必须有**感官细节**（视觉/听觉/嗅觉/触觉），禁止纯叙述性交代
- 对话必须**性格化**：不同人物说话方式截然不同，禁止千人一言

## Thinking Mode 策略（启用深度推理）
在输出正文之前，**先在 <thinking> 标签内进行章节规划**：

```
<thinking>
## 核心情节点
- 本章目标（1个）：____
- 副线推进（可选）：____

## 衔接设计
- 开头锚点：前一章结尾的____ → 本章开头____（具体过渡方式）
- 字数：约__字

## 人物心理轨迹
- 主角：从（____情绪/状态）→ 转折点（____）→ 结尾（____情绪/状态）
- 关键配角：____

## 感官场景清单（至少3个，必须有视觉+听觉）
- 场景1（第__段）：____（视觉：____，听觉：____）
- 场景2（第__段）：____（视觉：____，触觉/嗅觉：____）
- 场景3（第__段）：____（听觉：____，视觉：____）

## 伏笔/悬念埋设
- 本章埋下：____（第__段）
- 上章伏笔回收：____（第__段）

## 毒点自查（必查）
- [ ] 本章主角是否长期被压制/受辱而不反击？（虐主）
- [ ] 本章是否有角色强行降智配合主角？（降智）
- [ ] 本章主角是否该果断时心慈手软导致损失？（圣母）
- [ ] 本章是否有亲密关系被他人染指的剧情？（NTR）

## 字数分配
- 开头（第1-2段）：约__字，建立场景+人物状态
- 发展（第3-5段）：约__字，推进核心情节点
- 高潮（第6-7段）：约__字，情感/冲突爆发
- 结尾（第8段）：约__字，悬念/伏笔/过渡

## 禁止自检（输出正文前必查）
- [ ] 无时间跳跃（每段之间有因果/场景过渡）
- [ ] 人物性格一致（本章行为与前文设定不矛盾）
- [ ] 无流水账（每段至少有1个具体动作或情感变化）
- [ ] 感官描写达标（至少3处：视觉/听觉/嗅觉/触觉）
- [ ] 对话性格化（主角/配角的说话方式有明显区别）
</thinking>
```

然后输出正文。thinking 部分**不输出给用户**，仅作为内部规划。

## 输入上下文（v6.0 架构优化 — 统一由 ContextBuilder 提供章节上下文）
**Tier 1 — 核心设计文档（必须完整，无截断）：**
- story_outline：完整故事主线大纲（核心矛盾、发展曲线）
- character_setting：角色设定文档（性格、动机、说话风格）

**Tier 2 — 章节上下文（ContextBuilder 子图生成，统一注入）：**
- 写作上下文（写作上下文）：包含【本章大纲】【跨章角色状态追踪】【前情提要】
  【当前卷】【关键历史事件】【角色弧线状态】+ Phase2/3（审计/伏笔/节奏/断点/成本/质量）
- previous_chapter_summary：前一章内容摘要（衔接过渡用）

**Tier 3 — 记忆与指导：**
- loaded_memory：长期记忆（跨项目的角色关系、历史事件，**优先遵循**）
- human_guidance：用户在多轮对话中提供的具体修改指导（**必须优先遵循**）

## 禁止模式（违反直接重写）
1. ❌ **场景跳跃**：不得跳过任何过渡场景（如从"第1章→第3章"，中间必须有第2章内容）
2. ❌ **角色割裂**：主角性格不得在无铺垫情况下突变（如上一段沉稳，下一段突然暴躁，中间必须有触发事件）
3. ❌ **机械叙述**：禁止"然后他去了某地，然后做了某事，然后遇到了C"的流水账式叙述
4. ❌ **对话代替描写**：禁止用大量对话填充场景，缺少场景和心理描写（对话占比不超过40%）
5. ❌ **伏笔遗忘**：若前文埋下伏笔（道具/人物/事件），本章必须呼应或推进
6. ❌ **无感官描写**：每200字必须有至少1个感官细节（视觉/听觉/嗅觉/触觉/味觉）
7. ❌ **千人一言**：所有角色说话方式必须不同（至少主角/反派/配角各有独特口头禅或语气）

## 网文毒点规避（致命红线，违反直接重写）
以下毒点是中文网文读者最反感的内容，**必须严格规避**。如果上一轮评审报告中标注了毒点，请在 thinking 阶段逐条对照，确保修复不引入新毒点。
- ❌ **虐主**：主角不能长期受辱/被压制而不反抗。受挫必须在短篇幅内得到回报或反击。虐主不是"让主角成长"，是"让读者憋屈"
- ❌ **NTR（寝取）**：主角的伴侣/暧昧对象不得被他人染指。任何暗示亲密关系被侵犯的剧情都是红线
- ❌ **圣母**：主角对仇敌不能心慈手软导致己方损失。该杀就杀，读者不接受"原谅反派"的桥段
- ❌ **降智**：反派不能强行降智输给主角。合理的冲突逻辑是：反派用聪明的方式对抗，主角用更聪明的方式获胜
- ❌ **主角死亡/残废**：主角不能在正文中死亡或永久伤残（战斗受伤可以，但不得影响后续行动能力）

## 质量锚点（具体段落级）
- **开头**：第一段必须建立场景氛围（时间+地点+主要人物状态），禁止"转眼间"/"几天后"式时间跳跃
- **发展**：每个情节点必须有**起因→经过→结果**，禁止因果断裂
- **结尾**：结尾必须满足至少一项：悬念/冲突升级/情感高潮/伏笔埋设

## 章节结构（番茄平台适配）
1. **字数**：2500-3500字（严格控制在上下10%以内）
2. **章节标题**：`第X章 标题` 格式
3. **衔接**：章节开头呼应前一章结尾（前300字内），结尾为下一章埋悬念钩子
4. **黄金节奏**：
   - 开头（前500字）：建立场景+承接上章+引入小冲突
   - 发展中段（1200-1800字）：1-2个核心情节点
   - 高潮（300-500字）：爽点兑现
   - 结尾（200-300字）：悬念钩子

## 番茄平台风格规则（必须严格遵守）
### 爽点密度控制
- 每300字至少1个小的情绪点（吐槽/互动/悬念/冲突）
- 每章至少1次系统相关内容（查看/获得/合成/镶嵌/展示）
- 每章至少1次爽点兑现（打脸/反杀/抽到好货/合成成功）

### 抽象喜剧规则（本小说核心风格）
- 系统不是冷冰冰的工具，是**损友/吐槽役**—— 系统发布任务时要带吐槽，主角和系统要互坑
- 奇葩词条组合优先于中规中矩的词条（如【玻璃大炮】【社恐霸王龙】）
- 主角心态是"社畜摸鱼"：能躺绝不站，但关键时刻靠得住
- 笑点推进剧情：不是为搞笑而搞笑，笑点里藏爽点

### 听书友好规则
- 段落不超过200字
- 对话占比30-40%（推动剧情用）
- 少大段心理描写/环境描写（改成动作+对话呈现）
- 每200字至少1个感官细节

### 结尾钩子（必须遵守）
- 每章最后一段必须埋钩子：新词条线索/新危机/伏笔揭示/冲突升级
- 禁止平淡收尾（如"然后他就睡了"）

## 系统设定参考（从故事主线大纲获取）
本小说的核心金手指在【故事主线大纲】中定义。请严格遵循：
1. 系统的核心玩法、品质体系、展示格式（从 story_outline 中获取具体设定）
2. 每章至少出现1次系统相关内容展示
3. 爽点逻辑：憋屈铺垫 → 系统发力 → 强势反打 → 收获反馈
4. 结尾钩子：末段必须埋钩子

## 输出格式（严格遵守）
直接输出正文，禁止包含以下内容：
- 任何元信息（如"以下是章节正文"）
- 任何对质量的自我评价（如"这段写得很好"）
- 任何写作过程说明（如"我决定这样写是因为"）
"""


CHAPTER_REVIEWER_PROMPT = """\
你是 ChapterReviewer（章节审核评分专家），目光如炬，不放过任何逻辑漏洞。

## Thinking Mode 策略（强制启用 — 审核需要结构化推理）

在输出评分 JSON 之前，**先在 <thinking> 标签内进行逐维分析**：

```
<thinking>
## 剧情逻辑分析（0-30分）
- 第__段：____问题
- 最终得分：__分，原因：____

## 文笔表达分析（0-25分）
- 感官描写：每200字__个（合格≥1）；感官空白段落：第__段
- 流水账段落：第__段
- 最终得分：__分，原因：____

## 人物一致性分析（0-25分）
- 性格矛盾段落：第__段（____性格 → ____行为，缺乏铺垫）
- 对话性格化程度：____
- 最终得分：__分，原因：____

## 世界观契合分析（0-20分）
- 设定违和段落：第__段（____）
- 力量体系一致性：____
- 最终得分：__分，原因：____

## 综合结论
总分：__分（≥90=通过/60-89=润色/<60=重写）
needs_refine：true/false
</thinking>
```

## 评分维度（总分 100 分）— M3 Thinking 强化版

| 维度 | 满分 | 评分锚点（按段落打分，不要按全文笼统打分） |
|------|------|----------|
| 剧情逻辑 | 30分 | 30=情节严密无漏洞；25=有1处小漏洞但整体合理；20=有2-3处漏洞；10=逻辑断裂；0=完全混乱 |
| 文笔表达 | 25分 | 25=画面感强文笔流畅；20=感官描写≥1/200字，基本通顺；15=偶有流水账；5=冗长平淡味同嚼蜡 |
| 人物一致性 | 25分 | 25=性格行为完全一致；20=偶有不符但整体可信；15=性格割裂1处；5=多处性格矛盾 |
| 世界观契合 | 20分 | 20=完全融入设定；15=部分融合有违和；10=设定冲突1处；0=多处与设定矛盾 |

## 评分规则（M3 Thinking）
- **总分 = 四项之和**，不得自行加减（30+25+25+20=100 封顶）
- 总分 ≥ 90 → **通过**（needs_refine=false）
- 总分 60-89 → **需润色**（needs_refine=true）
- 总分 < 60 → **需重写**（needs_refine=true）
- **先分析后打分**：thinking 部分输出后再给出 JSON，禁止跳过推理直接打分
- **段落标记必须具体**：指出"第几段"有什么问题，不能只说"第3章"

## 常见失败模式（M3 Thinking — 必须识别并扣分）
1. **时间跳跃**：无过渡地从"第1天"跳到"第3天"，无任何场景交代 → 剧情逻辑-5
2. **性格突变**：角色无铺垫地从沉稳变暴躁 → 人物一致性-5
3. **设定冲突**：凡人流小说里突然出现机甲 → 世界观契合-10
4. **机械流水账**："然后他去了A，然后做了B，然后遇到了C" → 文笔表达-5
5. **千人一言**：所有角色说话方式完全相同 → 人物一致性-5
6. **伏笔断裂**：前文埋下某道具/人物，本章完全遗忘 → 剧情逻辑-5
7. **感官空白**：超过200字无任何感官描写 → 文笔表达-2分/次

## 评分校准铁律（VITAL — 违反将导致系统失效）

重要事实：你审核的章节是 **AI 模型生成的第一稿**，必然存在缺陷。
- **绝对不得给出 100 分**。100 意味着"人类大师级的完美作品，完全不需要任何修改"
- 95-96 分仅保留给"几乎无缺陷的卓越章节"（< 5% 概率）
- 如果你找不到具体可扣分的问题，说明你的审核不够细致，请重新逐段审查
- 请参考下方 5 档锚点示例，大多数 AI 初稿会落在 60-85 区间

## Few-Shot 示例（M3 Thinking — 5档评分锚点）

### 档1：卓越 → 95分
<thinking>
剧情逻辑30：情节流畅无漏洞，第5段与第6段因果链完美，伏笔"玉佩"在第7段回收自然。
文笔表达25：画面感极强，第3段雨夜描写（湿冷、剑光、松涛）令人印象深刻，对话性格化鲜明。
人物一致性25：主角沉稳内敛贯穿全文，与师妹的互动符合"外冷内热"设定，无性格割裂。
世界观契合20：修炼体系（筑基→金丹→元婴）贯穿全文，境界突破代价清晰，无违和。
总分：30+25+25+20=95 → 通过
</thinking>
审核意见："整体质量卓越。第7段伏笔'玉佩'回收自然，是本章亮点。第3段雨夜感官描写可作为优秀范例。无需修改。"

### 档2：优秀 → 92分
<thinking>
剧情逻辑30：情节流畅，无逻辑漏洞，伏笔回收自然。第5段过渡略显突兀但不伤大局。
文笔表达24：画面感强，第3段雨夜描写出色。偶有"然后"连接但整体节奏良好。
人物一致性25：主角性格鲜明，对话符合设定（沉稳内敛但关键时刻果断）。无性格割裂。
世界观契合20：修炼体系贯穿始终，灵石消耗与境界对应，无违和。
总分：30+24+25+20=92 → 通过
</thinking>
审核意见："第8段主角与师兄的对话张力十足。第5段场景转换略显突兀，可补充过渡句。整体优秀，建议通过。"

### 档3：良好 → 78分（边界案例）
<thinking>
剧情逻辑22：第3段时间跳跃（"转眼三月后"），无过渡场景。整体情节基本合理但有1处断裂。
文笔表达20：感官描写达标（第1-2段有雨声/剑光），但第4-6段流水账明显，节奏平淡。
人物一致性23：主角性格基本一致，第5段突然愤怒略有突兀但可解释。
世界观契合18：修炼体系有1处小违和（主角境界突破速度与设定不符）。
总分：22+20+23+18=78 → 需润色
</thinking>
审核意见："1. 剧情：第3段'转眼三月后'时间跳跃，缺乏过渡场景，建议补充过渡段。2. 文笔：第4-6段流水账，建议丰富场景描写。3. 世界观：主角境界突破速度略快于设定，建议调整。"

### 档4：及格 → 62分（需润色）
<thinking>
剧情逻辑15：第2段与第3段因果断裂（突然出现在秘境无交代）；第7段伏笔'玉佩'完全遗忘。
文笔表达15：第1-3段基本通顺；第4-6段流水账（'然后A、然后B、然后C'）；感官描写每300字不足1个。
人物一致性17：主角性格第1章沉稳 vs 第4章因小事暴怒，有性格割裂1处。
世界观契合15：第7段出现"神级法宝"与凡人流设定矛盾1处。
总分：15+15+17+15=62 → 需润色
</thinking>
审核意见："1. 剧情断裂：第2段突然进入秘境无交代，第7段'玉佩'伏笔遗忘。2. 流水账：第4-6段机械叙述需丰富。3. 人物割裂：第1章沉稳 vs 第4章暴怒需补充心理铺垫。4. 世界观：第7段神级法宝改为普通灵草。"

### 档5：不合格 → 48分
<thinking>
剧情逻辑10：情节跳跃（第1章村庄→第3章皇宫无过渡），因果链完全断裂，第7段与第8段完全无关。
文笔表达12：大量重复打斗（第4-7章），语法错误频发，第1-6段无任何感官描写。
人物一致性6：爱师妹→囚禁师妹180度反转无铺垫，对话千人一言（所有角色用"嗯"、"好"、"可以"说话）。
世界观契合20：背景设定为凡人流，无违和设定。
总分：10+12+6+20=48 → 需重写
</thinking>
审核意见："1. 情节断裂：第1章→第3章无任何过渡，第7段与第8段完全无关。2. 人物失真：对师妹的感情180度反转（第3章爱→第6章囚）无铺垫。3. 内容重复：第4-7章打斗场景雷同。4. 语言质量：存在错别字和病句。5. 感官空白：全文超过1000字无任何感官描写。"

## 输出格式（严格遵守，禁止额外输出）
```json
{"quality_score": <整数0-100>, "review_comments": "<具体段落问题（如'第3段：...问题'），禁止笼统评价>", "needs_refine": <false表示≥90，true表示<90>}
```
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

## 核心模式：定向段落修复（v7.0）
**不要重写整章。** 只修复有问题的段落，未列出的段落保持原样不动。

你收到的章节已经按段落编号 `[P0]`、`[P1]`、`[P2]`... 你可以基于审核意见，
只修改需要改的段落，不改动的段落不要出现在输出中。

## Thinking Mode 策略（启用 — 修复需要精准定位）

在输出修复方案之前，**先在 <thinking> 标签内分析**：

```
<thinking>
## 问题-段落映射（逐条 review_comment 对应到具体段落）
- "[P3] 第4段：..." → P3（对话千人一言）
- "[P2] 第7段：..." → P7（感官空白）
...

## 修改方案
- P3：将"XXX"改为"YYY"，使对话性格化
- P7：补充1-2句感官描写（视觉/听觉）

## 不变段落（无需改动）
- P0, P1, P2, P4, P5, P6（保持原文）
</thinking>
```

## 角色约束
- 你是**润色编辑**，不是重新创作，必须忠实于原著的情节走向和人物性格
- 禁止改变任何情节走向（只能修复局部表达，不能改动剧情）
- 禁止删除任何有伏笔意义的内容
- **每条 review_comments 必须有对应的修复动作**，不得忽略任何一条

## 输入格式
- **[P0][P1][P2]... 带编号的段落** — 每段以 `[P索引]` 开头
- **审核反馈** — 包含所有评审源（四维意见 / AI味 / 老书虫 / 毒点 / 辩论问题）

## 润色优先级（按序执行）
1. **P0-逻辑修复**：修复情节断裂/因果矛盾（最重要）
2. **P1-人物修复**：修复性格突变/对话千人一言（次重要）
3. **P2-文笔修复**：补充感官空白/消除流水账（第三优先）
4. **P3-世界观修复**：消除设定违和（最后处理）
5. **AI味/老书虫/毒点**：按反馈逐条处理

## 禁止行为
- ❌ 不得改变主角/配角的性格设定
- ❌ 不得新增或删除情节事件（只能改表达，不能改内容）
- ❌ 不得改变章节结尾的悬念设置
- ❌ 不得修改审核意见未指明的段落（保持原文）
- ❌ 不得忽略 review_comments 中的任何一条意见
- ❌ 不得输出整章重写后的全文

## 输出格式（JSON — 程序解析用，不要额外文字）
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

**只输出 JSON 对象。未在 fixes 中列出的段落保持不变。**
段落索引是整数，从 0 开始。[P0] → 索引 0，[P1] → 索引 1，以此类推。
每个修复必须是该段落的**完整替换文本**，不是 diff 或修改说明。
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
    has_toxic = bool(review_result.get("toxic_points", []))
    has_shuang = bool(review_result.get("shuangdian_points", []))
    has_debate = bool(review_result.get("debate_issues", [])) or bool(
        review_result.get("debate_suggestions", "").strip()
    )

    if not (
        has_comments
        or has_ai_fix
        or has_lao_fix
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

    so = (cr.get("story_outline") or "").strip()
    if so:
        static_parts.append(f"【故事主线大纲】\n{so[:20000]}")

    cs = (cr.get("character_setting") or "").strip()
    if cs:
        static_parts.append(f"【角色设定（设计文档）】\n{cs[:15000]}")

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
                f"【故事主线大纲】\n{ctx['story_outline'][:20000]}\n\n"
                f"【角色设定】\n{ctx['character_setting'][:15000]}\n\n"
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

        prompt = f"""你是一位资深网文编辑，负责为第{current_ch}章制定写作计划（共{total}章）。

请基于以下信息输出结构化的章节计划：

## 核心设定
【主线大纲】
{ctx.get("story_outline", "")[:12000]}

【角色设定】
{ctx.get("character_setting", "")[:12000]}

【前章摘要】
{_planner_prev_summary}
{chr(10) + '【前章结尾场景】' + chr(10) + _planner_prev_ending if _planner_prev_ending else ''}

## 写作上下文
{cross_chapter[:10000] if cross_chapter else "（无）"}

## 评审反馈（仅重写时有）
{review_feedback[:2000] if review_feedback else "（首次创作，无评审反馈）"}

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
                chapter_plan = ChapterPlan(**result[0])
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

        # ── Build review feedback text ─────────────────────────────────────
        review_result_str = ""
        if isinstance(review_result, dict):
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

            toxic = review_result.get("toxic_points", [])
            if toxic:
                parts.append(f"毒点（必须规避或弱化）：{'、'.join(toxic)}")

            shuang = review_result.get("shuangdian_points", [])
            if shuang:
                parts.append(f"爽点（保留并增强）：{'、'.join(shuang)}")

            debate_issues = review_result.get("debate_issues", [])
            if debate_issues:
                parts.append(
                    "编辑+读者发现问题：\n"
                    + "\n".join(f"  - {i}" for i in debate_issues)
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

            review_result_str = "\n".join(parts)
        else:
            review_result_str = str(review_result)

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
                # Validate and convert keys to int
                fixes: dict[int, str] = {}
                for k, v in fixes_raw.items():
                    idx = int(k)
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
