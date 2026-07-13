"""LLM 增强型评价分析模块（v7.2）。

基于论文原文（已读）：
  - WebNovelBench (ACL 2025)：标签化输出 + COT 前置提取 + PCA 加权
  - WritingBench (NeurIPS 2025)：criteria-aware scoring
  - Fiction_Eval (ACL 2025)：题材特异性豁免 + 三层10维框架
  - ChineseHarm-Bench (2025)：知识规则迭代 + 中文规避手段
  - EQ-Bench v3 + Antislop: 14维评分 + slop 指纹检测

Narrative Codec Engine Re-factoring：
    混合架构：程序化分析 (快/稳) + LLM 语义分析 (深/灵活)
    LLM 只做语义判断，不做客观计量
"""

from novelfactory.evaluation.llm.ai_style_llm import llm_ai_style_analysis
from novelfactory.evaluation.llm.old_reader_llm import (
    get_discovered_toxic_rules,
    llm_old_reader_analysis,
)

__all__ = [
    "llm_old_reader_analysis",
    "llm_ai_style_analysis",
    "get_discovered_toxic_rules",
]
