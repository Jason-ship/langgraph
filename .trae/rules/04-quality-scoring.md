---
alwaysApply: false
description: "评分体系规范，匹配graph/crews/writing_nodes/**、analysis/**、config/constants.py。评分不对、quality_score为0、阈值、辩论、重写/润色、composite计算、评分器故障、四维评分、editor/reader时触发。四维评分（文30/结25/角20/节15）、题材阈值、双条件通过、降级、重试（<55重写/55-79润2次/80-89润1次）、辩论（编辑vs读者/<10收敛/3轮）、故障检测（composite<0.1）。"
---
# 评分体系规范

**版本：** v2.1.0
**生效方式：** 智能生效
**优先级：** ⭐⭐⭐⭐
**匹配模式：** `graph/crews/writing_nodes/**`, `evaluation/**`, `analysis/**`, `config/constants.py`

---

## 一、评分维度

### 1.1 四维评分（100 分制）

| 维度 | 满分 | 评分锚点 |
|------|------|----------|
| 文学性 | 30 | 30=文笔精炼、修辞恰当、描写细腻；20=合格；10=平淡 |
| 结构 | 25 | 25=开篇钩子强、冲突递进、结尾悬念好；15=基本完整；5=松散 |
| 角色 | 20 | 20=角色性格鲜明、对话自然、行为自洽；12=基本一致；5=脸谱化 |
| 节奏 | 15 | 15=张弛有度、信息密度合适；8=基本还行；3=过慢或过快 |

### 1.2 综合指标（v7.0 VerdictEngine 融合评分）

VerdictEngine（`evaluation/verdict/engine.py`）融合多源评分后输出 `final_score`（0-100）：

```python
final_score = (
    quality_score * VERDICT_WEIGHTS["quality"]
    + programmatic_normalized * VERDICT_WEIGHTS["programmatic"]
    + llm_old_reader_score * VERDICT_WEIGHTS.get("llm_old_reader", 0.10)
    + llm_human_like_score * VERDICT_WEIGHTS.get("llm_human_like", 0.05)
    + cross_consistency * VERDICT_WEIGHTS["cross_chapter"]
    - debate_penalty * VERDICT_WEIGHTS["debate_penalty"]
    - decay_penalty  # 质量衰减惩罚
)
```

其中 `programmatic_normalized = (programmatic_score - 50) / 50`（映射到 0~1）。

质量衰减惩罚（VerdictEngine 内置 `_detect_quality_decay()` 函数，使用模块级私有常量 `_DECAY_HEAD_RATIO`/`_DECAY_TAIL_RATIO`/`_DECAY_PENALTY_PER_POINT`/`_DECAY_MAX_PENALTY`）：检测章节文本"高开低走"质量衰减，前段与后段对比扣分。

迭代宽松加分（`VERDICT_ITERATION_BONUS_*`）：重写/润色次数越多，加分越多：
- 重写轮：`VERDICT_ITERATION_BONUS_REWRITE`（默认 +2.0/次）
- 润色轮：`VERDICT_ITERATION_BONUS_REFINE`（默认 +1.0/次）
- 封顶：`VERDICT_ITERATION_BONUS_MAX`（默认 +4.0）

权重定义（`config/constants.py` -> `VERDICT_WEIGHTS`）：
- quality: 0.25, programmatic: 0.30, llm_old_reader: 0.10,
- llm_human_like: 0.05, cross_chapter: 0.20, debate_penalty: 0.10

### 1.3 维度详解

**文学性（30 分）**：评估语言表达质量，包括词汇丰富度、句式变化、修辞手法运用、场景描写的细腻程度。高分要求文笔精炼自然，修辞恰当而不做作，描写细腻而不冗长。

**结构（25 分）**：评估章节或卷的组织架构，包括开篇钩子强度、情节冲突的递进逻辑、高潮与铺垫的节奏安排、结尾悬念的设置。高分要求结构完整、起承转合清晰、信息释放节奏合理。

**角色（20 分）**：评估人物塑造的立体感，包括角色性格是否鲜明、对话是否符合人设、行为动机是否自洽、角色之间的互动是否自然。高分要求角色有辨识度、成长弧线合理、配角不脸谱化。

**节奏（15 分）**：评估阅读节奏的舒适度，包括张弛交替是否合理、战斗/文戏比例是否恰当、信息密度是否适应当前章节位置。

---

## 二、题材感知阈值

配置位置：`config/constants.py` → `GENRE_THRESHOLDS`，通过 `resolve_genre()` 统一解析。

| 题材 | quality 阈值 | composite 阈值 | 说明 |
|------|:-----------:|:--------------:|------|
| default | ≥85 | ≥0.65 | 通用题材 |
| 玄幻 | ≥85 | ≥0.65 | |
| 仙侠 | ≥88 | ≥0.65 | |
| 奇幻 | ≥82 | ≥0.60 | |
| 武侠 | ≥85 | ≥0.60 | |
| 都市 | ≥80 | ≥0.55 | |
| 历史 | ≥88 | ≥0.65 | |
| 科幻 | ≥85 | ≥0.60 | |
| 悬疑灵异 | ≥85 | ≥0.60 | |
| 游戏 | ≥78 | ≥0.55 | |
| 军事 | ≥88 | ≥0.65 | |
| 系统流 | ≥75 | ≥0.50 | |
| 重生 | ≥78 | ≥0.55 | |
| 穿越 | ≥80 | ≥0.55 | |
| 无敌流 | ≥72 | ≥0.45 | |
| 种田 | ≥82 | ≥0.58 | |
| 末世 | ≥80 | ≥0.55 | |
| 现代言情 | ≥82 | ≥0.55 | |
| 古代言情 | ≥85 | ≥0.58 | |
| 幻想言情 | ≥80 | ≥0.55 | |
| 耽美 | ≥85 | ≥0.58 | |
| 爽文 | ≥75 | ≥0.50 | |
| 脑洞 | ≥78 | ≥0.50 | |
| 同人 | ≥78 | ≥0.50 | |
| 二次元 | ≥76 | ≥0.50 | |

### 2.1 题材名统一解析

`constants.resolve_genre()` 匹配规则：
1. 精确匹配 `GENRE_THRESHOLDS` 键名
2. 关键词匹配（`_GENRE_KEYWORD_MAP`，按优先级排序）
3. 兜底返回 `"default"`

```python
def resolve_genre(genre: str | None) -> str:
    """根据 genre 字符串自动解析为 GENRE_THRESHOLDS 中的标准题材名。"""
```

### 2.2 阈值配置函数签名

```python
def get_genre_threshold(genre: str | None, key: str, default: float = 0.0) -> float:
    """获取指定题材的单个阈值。"""
    
def get_genre_thresholds(genre: str | None) -> dict:
    """获取指定题材的完整阈值字典。"""
```

---

## 三、双条件通过判定（VerdictEngine）

```
final_score >= 题材阈值 AND (lao_shu_chong_score / 100) >= 题材composite阈值 → 通过
```

两个条件必须**同时满足**才能判定为完全通过。任一条件不满足则进入降级机制或重试/润色流程。

### 3.1 降级机制

| 条件 | 判定结果 |
|------|---------|
| final_score >= 90 且 (lao_shu_chong_score / 100) >= 0.5 | **降级通过** |
| final_score >= 90 且 (lao_shu_chong_score / 100) < 0.1 | **评分器故障，强制通过** |
| 重试次数用尽（loop >= 3） | **强制通过** |

**降级优先级**：评分器故障检测 > 优秀降级 > 重试用尽强制通过。

### 3.2 判定流程伪代码

```
判定(quality, programmatic, llm_old_reader, debate, cross_chapter, loop, 题材):
    final = fuse(quality, programmatic, llm_old_reader, llm_human_like, debate, cross_chapter)
    q_threshold, c_threshold = get_genre_thresholds(题材)

    # step 1: 双条件标准通过
    如果 final >= q_threshold 且 (lao_shu_chong / 100) >= c_threshold:
        return PASS

    # step 2: 评分器故障检测
    如果 final >= 90 且 (lao_shu / 100) < 0.1:
        log_warning("评分器故障")
        return PASS (强制通过)

    # step 3: 优秀降级
    如果 final >= 90 且 (lao_shu / 100) >= 0.5:
        return PASS (降级通过)

    # step 4: 重试用尽
    如果 loop >= 3:
        return PASS (强制通过)

    # step 5: 三级决议
    如果 final < 55:    return REWRITE
    如果 final < 80:    return REFINE
    否则:               return PASS
```

---

## 四、重试与润色上限

### 4.1 阈值映射

| final_score | 操作 | 上限 |
|------------|------|------|
| < 55 | 完全重写 -> `chapter_planner` | `MAX_REWRITE_ATTEMPTS = 5` |
| 55 ≤ final < 80 | 润色 → `chapter_refiner` | 最多 2 次 |
| ≥ 80 | 通过 → `state_extractor` | 0 次 |

### 4.2 重试计数器管理

在 Writing Crew 子图中通过 `ManagedValue` 管理重试计数器：

- `rewrite_attempts`：完全重写计数器，每次重写后 +1，达到 `MAX_REWRITE_ATTEMPTS`（5）后不再重写
- `refine_attempts`：润色计数器，每次润色后 +1，根据质量分决定上限（<80 最多 2 次，<90 最多 1 次）
- 两个计数器独立计数，互不影响
- 重试/润次用尽后进入双条件判定流程的降级机制

### 4.3 verdict_router 路由逻辑

`evaluation/verdict/router.py` 中的 `verdict_router` 根据 `verdict_result.level` 决定下一节点：

- `chapter_planner`：REWRITE 级别（final < 55 且 loop_count < 3）
- `chapter_refiner`：REFINE 级别（55 ≤ final < 80）
- `state_extractor`：PASS 级别（通过/降级通过/强制通过）

---

## 五、VerdictEngine 评审引擎

### 5.1 评分来源

| 评分源 | 来源模块 | 权重键 |
|--------|---------|:------:|
| **四维 LLM 评分**（`evaluation/llm/`） | LLM 调用评分器 | `quality` |
| **程序化评分**（`evaluation/programmatic/runner.py`） | AI 味传感器 + 老书虫传感器 + 跨章传感器 | `programmatic` |
| **LLM 老书虫语义分**（`evaluation/llm/old_reader_llm.py`） | LLM 语义级毒点/爽点检测 | `llm_old_reader` |
| **LLM AI味语义分**（`evaluation/llm/ai_style_llm.py`） | LLM 语义级 AI 味检测 | `llm_human_like` |
| **辩论评分**（`evaluation/debate/engine.py`） | InformedDebateEngine 3 角色辩论 | `debate_penalty` |
| **跨章一致性**（`evaluation/programmatic/cross_chapter_sensor.py`） | 句长/对话/动作/词汇/伏笔/关键词 | `cross_chapter` |

### 5.2 辩论流程（InformedDebateEngine）

`evaluation/debate/engine.py` 中的 `InformedDebateEngine` 实现编辑（Editor）↔读者（Reader）↔点评师（Critic）3 角色辩论：

```
editor_review → reader_review → critic_review
                   ↓
              debate_router
                   ↓
     ┌─── 分歧大且 round < 3 ──→ 辩论轮
     │                              │
     │               editor_rebuttal → reader_rebuttal → critic_rebuttal
     │                                         │
     │                                         ↓
     │                                    debate_router
     │                                         │
     └─── 分歧小或 round ≥ 3 ──→ merge
```

**规则**：
- 评分差异 < 10 分 → 提前收敛
- 最多 3 轮辩论
- 连续 2 轮无新议题 → 空转收敛
- 3 轮后未收敛 → 取各角色评分平均值作为最终分

### 5.3 最终评分确定

```python
def _merge(state) -> dict:
    quality_mean = mean(all_perspectives.quality_score)
    debate_report.severity_weight = ...  # 辩论严重性加权
    return { "quality_score": quality_mean, "debate": debate_report }
```

---

## 六、评分器故障检测

### 6.1 检测条件

当同时满足以下两个条件时判定为评分器故障：

1. `quality_score >= 题材阈值`（即四维评分认为质量过关）
2. `composite_score < 0.1`（但综合指标极低，合理值不可能出现）

### 6.2 故障原因分析

composite_score 极低的原因通常包括：

- `lao_shu_chong_score` 异常低下（≤5），即使 `ai_style_score` 为 0 也无法合格
- `ai_style_score` 异常偏高（≥0.95），即使老书虫分高也被拉低
- LLM 输出格式异常，Schema 字段未正确填充
- 老书虫评分器或 AI 味检测器的调用/解析逻辑出现异常

### 6.3 故障处理

1. **强制通过**：跳过后续重试/润色流程，将章节标记为通过
2. **记录警告日志**：输出 `quality_score`、`composite_score`、`lao_shu_chong_score`、`ai_style_score` 等关键指标，用于后续排查
3. **不影响流程**：故障检测只影响当前章节的判定，不阻塞后续章节的写作流程

---

## 七、评分数据结构

### 7.1 核心 Schema

定义位置：`evaluation/schemas.py`（VerdictResult） + `schemas/review_schemas.py`（FourDimScores）

```python
# evaluation/schemas.py — 统一评审决议（唯一输出契约）
class VerdictLevel(str, Enum):
    PASS = "pass"
    REFINE = "refine"
    REWRITE = "rewrite"

class VerdictResult(BaseModel):
    level: VerdictLevel
    final_score: float = Field(ge=0.0, le=100.0)
    quality_score: float
    programmatic_score: float
    programmatic_report: ProgrammaticReport
    cross_chapter_signals: CrossChapterSignals
    four_dim: FourDimReviewResult
    llm_old_reader: LLMOldReaderResult | None
    llm_ai_style: LLMAIStyleResult | None
    debate: DebateReport
    feedback: FeedbackBundle
    iteration_bonus: float
    decay_penalty: float
    attempt_info: AttemptInfo
    deviation_report: dict | None

# schemas/review_schemas.py
class FourDimScores(BaseModel):
    literary: float = Field(ge=0.0, le=30.0)
    structure: float = Field(ge=0.0, le=25.0)
    character: float = Field(ge=0.0, le=20.0)
    pacing: float = Field(ge=0.0, le=15.0)

class ChapterPlan(BaseModel):
    chapter_number: int
    scenes: list[ScenePlan]
    emotional_arc: str = ""
    foreshadowing_plant: list[str] = []
    foreshadowing_resolve: list[str] = []
    cliffhanger: str = ""
    review_feedback: str = ""
```

### 7.2 状态字段映射

| 状态字段 | 类型 | 说明 | 来源 |
|---------|------|------|------|
| `verdict_result.level` | VerdictLevel | PASS/REFINE/REWRITE 三级决议 | VerdictEngine |
| `verdict_result.final_score` | float | 融合后最终评分 (0-100) | VerdictEngine |
| `verdict_result.quality_score` | float | 四维 LLM 质量评分 | evaluation/llm/ |
| `verdict_result.programmatic_score` | float | 程序化综合评分 | evaluation/programmatic/ |
| `verdict_result.debate.severity_weight` | float | 辩论严重性加权 | evaluation/debate/ |
| `crew_result.quality_score` | float | 质量总分 | coordinator |
| `crew_result.ai_style_score` | float | AI味指数 | analysis/ai_style_analyzer |
| `crew_result.lao_shu_chong_score` | float | 老书虫评分 | analysis/old_reader_reviewer |

---

## 八、相关代码文件

| 文件 | 职责 |
|------|------|
| `config/constants.py` | 阈值定义（GENRE_THRESHOLDS、VERDICT_WEIGHTS、VERDICT_ITERATION_BONUS_*、QUALITY_DECAY_* 等） |
| `evaluation/verdict/engine.py` | VerdictEngine 融合引擎 |
| `evaluation/verdict/calibration.py` | 评分校准/漂移检测 |
| `evaluation/verdict/feedback.py` | FeedbackBuilder 统一反馈包 |
| `evaluation/verdict/router.py` | verdict_router 三分支路由 |
| `evaluation/coordinator.py` | verdict_engine_node 图节点封装 |
| `evaluation/schemas.py` | 统一评审 Schema（VerdictResult） |
| `evaluation/debate/engine.py` | InformedDebate 辩论引擎 |
| `evaluation/debate/prompts.py` | 辩论 Prompt 模板 |
| `evaluation/llm/_shared.py` | 四维评分共享工具 |
| `evaluation/llm/old_reader_llm.py` | LLM 老书虫评审 |
| `evaluation/llm/ai_style_llm.py` | LLM AI味语义分析 |
| `evaluation/llm/prompts.py` | LLM 评审 Prompt |
| `evaluation/programmatic/runner.py` | 程序化分析运行器 |
| `evaluation/programmatic/ai_style_sensor.py` | AI 味 8 维检测 |
| `evaluation/programmatic/old_reader_sensor.py` | 老书虫传感器 |
| `evaluation/programmatic/cross_chapter_sensor.py` | 跨章一致性传感器 |
| `graph/crews/writing_nodes/routing.py` | _exit_for_chapter 出口 |
| `graph/crews/writing_nodes/reviewer.py` | chapter_refiner 润色节点 |
| `analysis/ai_style_analyzer.py` | AI 味程序化检测 |
| `analysis/old_reader_reviewer.py` | 老书虫规则检测 |
| `schemas/review_schemas.py` | 四维评分 Schema |

---

## 九、常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| composite 长期为 0 | ai_style_analyzer 或 old_reader_reviewer 异常 | 检查评分器日志；故障检测会触发强制通过 |
| 辩论持续分歧 | 编辑和读者对评分标准理解不一致 | 检查双方 prompt 中的评分锚点是否清晰；3 轮后自动收敛 |
| quality < 55 循环重写 | 章节内容与题材要求严重不匹配 | MAX_REWRITE_ATTEMPTS 用尽后强制通过 |
| 题材别名导致阈值错误 | genre_aliases 映射遗漏新题材别名 | 在 constants.py 中补充映射 |
| 四维分之和 != quality_score | FourDimScores 四个维度各有满分限制 | 确认 literary(30)+structure(25)+character(20)+pacing(15)=90，额外 10 分为整体加分项 |

---

**规则版本：** v2.1.0
**生效方式：** 智能生效
**最后更新：** 2026-07-04
