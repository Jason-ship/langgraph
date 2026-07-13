# Narrative Codec Engine — Phase 0 编入引擎

将小说级原始文本转化为结构化叙事元数据的独立模块，零外部依赖（LLM 可选降级），纯 Python 实现。

---

## 架构图

```
                          ┌─────────────────────────────────────────────┐
                          │            Phase 0: Encode Pipeline         │
                          │           (文本 → 结构化元数据)               │
                          └─────────────────────────────────────────────┘

  ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────────┐   ┌──────────┐
  │  A1  │   │  A2  │   │  A3  │   │  A4  │   │  A5  │   │  A6  │   │  A7  │   │   A8     │   │   A9     │
  │ 原  │   │ 分  │   │ 场  │   │ 句  │   │ STAC│   │Expert│   │ 情  │   │ 因果图  │   │ 因果图  │
  │ 始  │──▶│ 句  │──▶│ 景  │──▶│ 子  │──▶│ 分  │──▶│Index │──▶│ 绪  │──▶│ Iter1  │──▶│ Iter3   │
  │ 文  │   │     │   │ 分  │   │ 精  │   │ 类  │   │ 提  │   │ 曲  │   │ +Iter2 │   │ +DAG化  │
  │ 本  │   │     │   │ 割  │   │ 炼  │   │     │   │ 取  │   │ 线  │   │ 候选边 │   │ +输出   │
  └──────┘   └──────┘   └──────┘   └──────┘   └──────┘   └──────┘   └──────┘   └──────────┘   └──────────┘
      │          │          │          │          │          │          │          │               │
      ▼          ▼          ▼          ▼          ▼          ▼          ▼          ▼               ▼
   raw      sentences    Scene[]    str[]     STACLabel  ExpertIndex EmotionArc   nx.DiGraph    CausalGraph
   text     (str list)                          edSentence
                                                   []

  ───────────────────────────────────────────────────────────────────────────────────────────────────────────
  依赖:     ✗ LLM     △ 可选 LLM    ✓ LLM      ✗ 纯规则    ✗ 纯规则    ✗ 纯规则    △ 可选 LLM
          (spaCy)     (回退降级)    (必要)                  (numpy/       (反事实剪枝)
                                                   scipy)
  ───────────────────────────────────────────────────────────────────────────────────────────────────────────

  Legend:
    ✓ = 必须调用 LLM        △ = 可选调用 LLM（有降级路径）        ✗ = 不依赖 LLM（纯规则/统计）
```

---

## 模块说明

| 模块 | 文件名 | 类名 | 功能描述 | 参考论文 | 依赖 LLM |
|------|--------|------|----------|----------|:--------:|
| A1 文本输入 | — | — | 接收原始中文小说文本 | — | ✗ |
| A2 分句 | `emotion_arc.py` | `EmotionArcExtractor._split_sentences` | 基于句末标点将文本拆分为句子，保留偏移量 | — | ✗ |
| A3 场景分割 | `scene_splitter.py` | `SceneSplitter` | 基于实体密度突变 + 时间/地点关键词的场景边界检测，支持 spaCy NER 或规则回退 | Zehe et al. (2021), Guhr et al. (2025) | ✗ |
| A4 句子精炼 | `sentence_refiner.py` | `SentenceRefiner` | LLM 驱动的 4 步精炼：摘要精简 → 代词替换 → 从句简化 → 主动语态 | Beyond LLMs §3.1 | ✓ |
| A5 STAC 分类 | `stac_classifier.py` | `STACClassifier` | 规则 + LLM 混合的叙事角色四分类（Situation / Task / Action / Consequence），置信度 ≥ 0.7 跳过 LLM | Beyond LLMs §3.2 | △ |
| A6 Expert Index | `expert_index.py` | `ExpertIndexExtractor` | 7 维语言学特征提取（通指性/事件性/有界性/主动性/起始时间/结束时间/影响性），输出 13 维 one-hot | Beyond LLMs §3.3 | ✗ |
| A7 情绪曲线 | `emotion_arc.py` | `EmotionArcExtractor` | 滑动窗口情感词典分析 + 高斯平滑 + Freytag 金字塔四阶段叙事弧划分 | Reagan et al. (2016), Shadow-Loom §A.5 | ✗ |
| A8 因果图 Iter1+2 | `causal_graph_builder.py` | `CausalGraphBuilder` | STAC Bond 学习 + O(n²/2) 候选边生成 + 反事实 LLM 剪枝 | Beyond LLMs §3.4 | △ |
| A9 因果图 Iter3+输出 | `causal_graph_builder.py` | `CausalGraphBuilder` | DAG 化（SCC 破环） + 因果完整性保证 → `CausalGraph` 输出 | Beyond LLMs §3.4 | ✗ |

---

## 参考文献

1. **Zehe, A., Hotho, A., & Jannidis, F.** Detecting Scenes in Fiction. In *Proceedings of the 5th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature*, 2021.

2. **Guhr, O., et al.** Rethinking Scene Segmentation: A Computational Perspective. In *Proceedings of the 7th Workshop on Narrative Understanding*, 2025.

3. **Reagan, A. J., Mitchell, L., Kiley, D., Danforth, C. M., & Dodds, P. S.** The Narrative Arc: Revealing Core Narrative Structures Through Text Analysis. *EPJ Data Science*, 5:25, 2016.

4. **Beyond LLMs: A Structured Causal Narrative Framework for Long-Form Story Understanding.** In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)*, 2025. — 提出 STAC 叙事角色四分类 + Expert Index 7 维语言学特征 + 3 轮迭代因果图构建。

5. **Shadow-Loom: A Narrative Causality and World State Framework for Long-Form Story Generation.** arXiv preprint, 2026. — 提出 WorldStateV1 核心类型系统 + 双时间轴理论 + Narrative Physics 情感评分器。

6. **LitVISTA: A VISTA Space Framework for Narrative Orchestration.** In *Proceedings of ACL*, 2026. — 提出 VISTA Space 叙事编排框架。

7. **ConStory-Bench: A Benchmark for Consistency Error Detection in Long-Form Narratives.** arXiv preprint, 2026. — 提出 5 × 19 一致性错误分类体系。

> **注**：项目中还参考了 NK Weaver (arXiv 2026, 多 Agent 叙事图谱构建) 和 LLM×MapReduce (ACL 2025, 结构化协议 + 置信度校准) 的思想，分别用于 Schema 的置信度校准设计和结构化协议定义。

---

## 快速开始

以下伪代码展示完整的 Phase 0 编入流水线：

```python
from novelfactory.pipeline.narrative_codec.encode.scene_splitter import SceneSplitter
from novelfactory.pipeline.narrative_codec.encode.sentence_refiner import SentenceRefiner
from novelfactory.pipeline.narrative_codec.encode.stac_classifier import STACClassifier
from novelfactory.pipeline.narrative_codec.encode.expert_index import ExpertIndexExtractor
from novelfactory.pipeline.narrative_codec.encode.emotion_arc import EmotionArcExtractor
from novelfactory.pipeline.narrative_codec.encode.causal_graph_builder import CausalGraphBuilder
from novelfactory.pipeline.narrative_codec.schemas import (
    Scene, STACLabeledSentence, ExpertIndex, EmotionArc, CausalGraph,
)

# ── 输入 ───────────────────────────────────────────────────────────
raw_text = """林月如站在城门前，望着高耸入云的城墙。
她深吸一口气，迈步走进了城门。
城内的繁华让她感到陌生而兴奋。" """

# ── A2: 分句（内置于 EmotionArcExtractor）──────────────────────
emotion_extractor = EmotionArcExtractor()
sentences = emotion_extractor._split_sentences(raw_text)  # 注意：内部方法
# 也可自行按标点分句
import re
sentences = [s.strip() for s in re.split(r"(?<=[。！？!?\n])", raw_text) if s.strip()]

# ── A3: 场景分割 ─────────────────────────────────────────────────
splitter = SceneSplitter()
scenes: list[Scene] = splitter.split(raw_text)

# ── A4: 句子精炼 ─────────────────────────────────────────────────
refiner = SentenceRefiner()  # 自动使用 get_worker_llm()
refined_sentences: list[str] = refiner.refine(sentences, context="城门前")

# ── 构建 STACLabeledSentence 列表 ────────────────────────────────
labeled_sentences = [
    STACLabeledSentence(text=orig, refined_text=refined)
    for orig, refined in zip(sentences, refined_sentences)
]

# ── A5: STAC 分类 ────────────────────────────────────────────────
classifier = STACClassifier()  # 自动使用 get_reviewer_llm()
labeled_sentences = classifier.classify(labeled_sentences)

# ── A6: Expert Index 提取 ───────────────────────────────────────
expert_extractor = ExpertIndexExtractor()
expert_indices: list[ExpertIndex] = expert_extractor.extract_batch(
    [s.refined_text or s.text for s in labeled_sentences]
)

# ── A7: 情绪曲线提取（逐场景）───────────────────────────────────
emotion_arcs: list[EmotionArc] = []
for scene in scenes:
    arc = emotion_extractor.extract(scene.text)
    emotion_arcs.append(arc)

# ── A8 + A9: 因果图构建 ─────────────────────────────────────────
builder = CausalGraphBuilder()  # 不传 llm 则跳过反事实剪枝
causal_graph: CausalGraph = builder.build(labeled_sentences, expert_indices)

# ── 输出 ──────────────────────────────────────────────────────────
print(f"场景数: {len(scenes)}")
print(f"句子数: {len(labeled_sentences)}")
for s in labeled_sentences:
    print(f"  [{s.stac_label.value}] {s.refined_text}  (置信度: {s.confidence})")
print(f"因果图: {causal_graph.metadata['node_count']} 节点, "
      f"{causal_graph.metadata['edge_count']} 边, "
      f"DAG={causal_graph.metadata['is_dag']}")
print(f"情绪曲线: {len(emotion_arcs)} 条, "
      f"阶段={[stage.value for _, stage in emotion_arcs[0].stages]}")
for ei in expert_indices:
    print(f"  ExpertIndex → onehot: {ei.to_onehot()}")
```

---

## 现有代码复用

| 项目组件 | 复用方式 | 使用方 |
|----------|----------|--------|
| `agents/infra/retry.py` — `llm_call_with_retry` | LLM 调用重试 + 超时保护 | `SentenceRefiner`, `STACClassifier` |
| `agents/infra/retry.py` — `TIMEOUT_EXTRACT` | 提取类超时常量 | `STACClassifier` |
| `agents/infra/serialization.py` — `validate_json_output` | LLM JSON 输出校验（fail_closed 模式） | `STACClassifier` |
| `agents/infra/serialization.py` — `_extract_json_from_text` | 从 LLM 原始响应中提取 JSON | `SentenceRefiner` |
| `agents/infra/llm_cache.py` — `LLMResponseCache` / `get_llm_cache` | LLM 响应缓存，减少重复调用 | `SentenceRefiner` |
| `agents/infra/logger.py` — `get_logger` | 统一日志记录 | `SentenceRefiner` |
| `config/llm.py` — `get_reviewer_llm` | 评审 LLM 实例（默认 0.3 温度） | `STACClassifier` |
| `config/llm.py` — `get_worker_llm` | 工作 LLM 实例（默认 0.2 温度） | `SentenceRefiner` |

---

## 设计要点

- **LLM 降级路径**：`STACClassifier`（规则置信度 ≥ 0.7 跳过 LLM）和 `CausalGraphBuilder`（无 LLM 时所有候选边保留）均提供无 LLM 的降级方案。
- **纯规则模块**：`SceneSplitter`、`ExpertIndexExtractor`、`EmotionArcExtractor` 完全基于规则/统计，零 LLM 依赖。
- **Schema 驱动**：所有结构化输出均使用 `schemas.py` 中定义的 Pydantic 模型，可直接序列化为 JSON。
- **容错设计**：每个 LLM 调用均有 try/except 保护和降级值，单模块失败不阻塞整体流水线。
