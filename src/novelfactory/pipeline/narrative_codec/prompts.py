"""Codec Engine 提示词定义。

所有 prompt 为硬编码字符串常量，遵循项目惯例。

参考论文:
- Beyond LLMs (ACL 2025) §3.1 — Vertices Extraction (句子精炼)
- Beyond LLMs (ACL 2025) §3.2 — STAC Categorization (叙事四分类)
- Beyond LLMs (ACL 2025) §3.4 — Graph Construction Iter3 (反事实剪枝)
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════
# Prompt 1: 句子精炼 (Sentence Refiner)
# 参考: Beyond LLMs §3.1 — Vertices Extraction
# ═══════════════════════════════════════════════════════════════════════

SENTENCE_REFINER_SYSTEM = """\
你是叙事文本句子精炼专家。

## 任务
将输入的句子精炼为适合因果分析的标准化形式。
输出JSON格式：{"refined": "...", "substitutions": [...], "changes": [...]}

## 精炼流程（Beyond LLMs §3.1）
请按以下4步处理每个句子：

### 1. Summarization（摘要精简）
- 去除冗余修饰词，保留核心语义
- 确保句子长度适中

### 2. Pronoun Substitution（代词替换）
- 将代词替换为具体的实体名称
- 注意区分多个实体间的指代关系

### 3. Clause Simplification（从句简化）
- 将复合句拆分为多个简单句
- 每个简单句表达一个完整且独立的事件

### 4. Active Voice + Continuous Flow（主动语态 + 连续流）
- 将被动语态转为主动语态
- 确保句子主语明确（agent-centered）

## Few-shot示例
输入: "他沿着溪流奔跑，寻找一个可以躲避的洞穴。"
输出: {"refined": "刘备沿着溪流奔跑。刘备寻找洞穴。",
      "substitutions": [{"他": "刘备"}],
      "changes": ["代词替换: 他→刘备", "从句拆分: 复合句→两个简单句"]}

输入: "她被长老们任命为新一任的族长。"
输出: {"refined": "长老们任命她为新一任族长。",
      "substitutions": [],
      "changes": ["语态转换: 被动→主动"]}

输入: "话未说完，便见一人从帐外闯入，身长八尺，面如重枣。"
输出: {"refined": "一人从帐外闯入。此人身长八尺。此人面如重枣。",
      "substitutions": [],
      "changes": ["从句拆分: 复合句→三个简单句", "代词补充: 此人指代闯入者"]}
"""

SENTENCE_REFINER_USER = """\
请精炼以下句子。

上下文(用于指代消解): {context}

句子的列表:
{text}

请返回JSON数组，每个元素对应一个句子的精炼结果。
"""


# ═══════════════════════════════════════════════════════════════════════
# Prompt 2: STAC 四分类 (STAC Classifier)
# 参考: Beyond LLMs §3.2 — STAC Categorization
# ═══════════════════════════════════════════════════════════════════════

STAC_CLASSIFIER_SYSTEM = """\
你是叙事功能分类专家。

## 任务
对句子进行叙事功能四分类：{Situation, Task, Action, Consequence}
输出JSON格式：{"label": "...", "confidence": 0.0-1.0, "reasoning": "..."}

## 分类定义
- Situation（情境/背景）: 提供背景上下文或为未来事件"搭台"
  示例: "天色渐暗，森林中弥漫着雾气。"
  "话说天下大势，分久必合，合久必分。"

- Task（任务/目标）: 明确表达需要完成的要求或责任
  示例: "他必须在天黑前找到庇护所。"
  "玄德曰：'吾必当讨贼，以安天下。'"

- Action（动作/行为）: 正在执行或刚完成的主动活动
  示例: "他沿着小溪奔跑，寻找洞穴。"
  "云长引军出城，直取华雄。"

- Consequence（结果/后果）: 描述先前事件导致的状态改变结果
  示例: "他终于找到了一个安全的山洞。"
  "张角大军遂溃，黄巾之势由此而衰。"
"""

STAC_CLASSIFIER_USER = """\
请对以下句子进行STAC分类：

句子: {text}

思考: 1.这句话主要描述了背景、目标、动作还是结果？
2.最匹配的STAC类别是？
3.你的确信度如何？

输出JSON格式: {{"label": "situation|task|action|consequence", "confidence": 0.0-1.0, "reasoning": "..."}}
"""


# ═══════════════════════════════════════════════════════════════════════
# Prompt 3: 反事实剪枝 (Counterfactual Pruning)
# 参考: Beyond LLMs §3.4 — Graph Construction Iter3
# ═══════════════════════════════════════════════════════════════════════

COUNTERFACTUAL_SYSTEM = """\
你是一个因果推理专家。

判断两个事件之间是否存在直接的因果关系。

## 判断标准
回答YES: 事件A发生后事件B才发生，且无A则无B
回答NO: 即使没有事件A，事件B也可能发生
回答MAYBE: 存在部分因果关联但不能完全确定
"""

COUNTERFACTUAL_USER = """\
事件A: {cause_text}
事件B: {effect_text}

问题: 如果事件A没有发生，事件B是否仍会发生？

请输出JSON格式: {{"answer": "YES|NO|MAYBE", "confidence": 0.0-1.0, "reasoning": "..."}}
"""
