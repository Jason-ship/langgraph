---
alwaysApply: false
description: "评分体系规范，匹配graph/crews/writing_nodes/**、evaluation/**、config/constants.py、config/quality_params.py。评分不对、quality_score为0、阈值、重写/润色、仲裁、降级、四维评分、统一评审、VerdictEngine时触发。统一LLM评审（五视角+四维30/25/25/20）、自洽校验、分歧仲裁、三级决议（PASS≥80/REFINE≥55/REWRITE<55）、双次数用尽强制通过、评审失败降级PASS、迭代加分封顶4。"
---
# 评分体系规范

**版本：** v3.0.0
**生效方式：** 智能生效
**优先级：** ⭐⭐⭐⭐
**匹配模式：** `graph/crews/writing_nodes/**`, `evaluation/**`, `config/constants.py`, `config/quality_params.py`

---

## 一、统一 LLM 评审体系（v8.2+）

### 1.1 架构总览

v8.2 起评分职责 **100% 由统一 LLM 评审承担**，程序化传感器 / 5 LLM 维度并行 / 多轮辩论 / 加权融合 / 校准全部移除。

```
UnifiedReviewEngine（evaluation/unified/engine.py）
  单次 LLM 调用 → 五视角评审（老书虫/番茄编辑/读者/评论员 + 四维分项）
        ↓
  parse_review_output（parser.py，标签化双通道解析）
        ↓
  apply_consistency_check（parser.py，自洽校验硬约束）
        ↓
  分歧仲裁 arbitrate（arbitration.py，仅 severe 分歧触发 1 次）
        ↓
  VerdictEngine._fuse_unified（verdict/engine.py）
  综合分 + 迭代宽松加分 → 三级决议（PASS/REFINE/REWRITE）
```

### 1.2 四维评分（100 分制）

定义于 `evaluation/unified/schemas.py`（`UnifiedFourDim`）+ `parser.py`（`_DIM_MAX`）：

| 维度 | 满分 | 标签 | 评分锚点 |
|------|------|------|----------|
| 剧情逻辑 | 30 | `[四维-剧情逻辑] 26/30` | 情节逻辑自洽、冲突递进、因果链完整 |
| 文笔表达 | 25 | `[四维-文笔] 20/25` | 文笔精炼、修辞恰当、描写细腻 |
| 人物一致性 | 25 | `[四维-人物] 21/25` | 角色性格鲜明、对话自然、行为自洽 |
| 世界观契合 | 20 | `[四维-世界观] 17/20` | 世界观与角色行为自洽 |

**四维之和 = 100**。LLM 输出越界分数由 `parser._clamp()` 钳制（防击穿 pydantic 约束）。

> ⚠️ 旧定义（`schemas/review_schemas.py` 的文学性30/结构25/角色20/节奏15）已随 v8.5-clean 弃用，统一评审不再消费；实际运行以 `UnifiedFourDim` 为准。

### 1.3 统一评审输出（UnifiedReviewResult）

| 字段 | 类型 | 说明 |
|------|------|------|
| `final_score` | float | 综合分 0-100（LLM 五视角权衡） |
| `four_dim` | UnifiedFourDim | 四维分项（logic/writing/character/world） |
| `toxic_points` | list[dict] | `[{type, paragraph, severity, reason}]` |
| `severe_toxic` | bool | 是否有 severe 毒点（触发强制重写） |
| `shuangdian_count` / `shuangdian_points` | int / list | 爽点统计 |
| `water_paragraphs` | list[int] | 水段段落号 |
| `scene_transition_issues` | list[dict] | 跳跃/转场问题 |
| `human_like_score` / `attraction_score` / `immersion_score` | float | AI 味 / 吸引力 / 沉浸度（0-1 或 0-100） |
| `cross_chapter_score` | float | 跨章一致性 0-100 |
| `perspective_disagreements` | list[str] | 五视角分歧点（触发仲裁） |
| `failed` / `retried` / `consistency_fixed` | bool | 失败/重试/自洽修正标记 |

### 1.4 自洽校验（apply_consistency_check）

`evaluation/unified/parser.py` — 防 LLM 自相矛盾：

| 规则 | 条件 | 修正 |
|------|------|------|
| 毒点封顶 | `severe_toxic=True` | `final_score ≤ 70` |
| 无爽点封顶 | 无爽点 | `final_score ≤ 65` |
| 四维 rebase | `|four_dim.total() - final_score| > 15` | `final_score = four_dim.total()` |
| 仲裁后复核 | 仲裁修正分数后 | 重新执行一致性检查（v8.5-fix M2，防仲裁绕过硬约束） |

### 1.5 分歧仲裁（arbitration.py）

- 仅当存在 **severe 分歧** 时触发 1 次轻量 LLM 仲裁
- 输出 `[分数修正] final: 78.5 -> 74.0` + `[备注] PASS/REFINE/REWRITE`
- 仲裁分数钳制 0-100；解析失败返回原分
- 仲裁 **之后** 重新执行一致性检查（v8.5-fix M2）

---

## 二、VerdictEngine 融合与决议

### 2.1 融合逻辑（_fuse_unified）

`evaluation/verdict/engine.py`：

```python
final_score = clamp100(ur.final_score)
# 迭代宽松加分（缓解反复修复）
if attempt_info.loop_count > 0 or attempt_info.refine_attempts > 0:
    bonus = loop_count * 2.0 + refine_attempts * 1.0   # 封顶 4.0
    final_score = min(final_score + bonus, ur.final_score + 4.0)

quality_score = clamp100(ur.four_dim.total())
if quality_score <= 0:
    quality_score = final_score   # 轻量复查/四维缺失时兜底（v8.5-fix S3）
```

### 2.2 三级决议（_decide_unified_level）

判定顺序（优先级从高到低）：

| 优先级 | 条件 | 结果 |
|:------:|------|------|
| 1 | `rewrite_exhausted AND refine_exhausted` | **PASS**（防死循环兜底） |
| 2 | `ur.failed`（评审 API 失败） | **PASS**（v8.5-fix M3：评审失败≠质量差，避免故障触发整章重写） |
| 3 | `ur.severe_toxic AND not rewrite_exhausted` | **REWRITE** |
| 4 | `final_score >= VERDICT_PASS_THRESHOLD(80)` | **PASS** |
| 5 | `refine_exhausted`（润色用尽未达标） | **REWRITE**（换重写思路） |
| 6 | `rewrite_exhausted`（重写用尽未达标） | **REFINE**（换润色兜底） |
| 7 | `final_score >= VERDICT_REFINE_THRESHOLD(55)` | **REFINE** |
| 8 | 其他（final < 55） | **REWRITE** |

### 2.3 阈值参数（config/constants.py + quality_params.py 动态覆盖）

| 常量 | 默认值 | 说明 |
|------|:------:|------|
| `VERDICT_PASS_THRESHOLD` | 80.0 | 通过线（v9.1 收紧，原 73） |
| `VERDICT_REFINE_THRESHOLD` | 55.0 | 润色/重写分界 |
| `MAX_REWRITE_ATTEMPTS` | 5 | 最大重写次数 |
| `MAX_REFINE_ATTEMPTS` | 2 | 最大润色次数 |
| `VERDICT_ITERATION_BONUS_REWRITE` | 2.0 | 每次重写加分 |
| `VERDICT_ITERATION_BONUS_REFINE` | 1.0 | 每次润色加分 |
| `VERDICT_ITERATION_BONUS_MAX` | 4.0 | 加分封顶 |

运行时动态调参：`config/quality_params.py` 的 `verdict.pass_threshold` / `verdict.refine_threshold` / `verdict.iteration_bonus.*`，通过 CLI/API 实时调整。

### 2.4 路由

`evaluation/verdict/router.py` 的 `verdict_router`（纯路由，v8.5 已移除 state 修改逻辑）：

| VerdictLevel | 目标节点 | 说明 |
|:-----------:|---------|------|
| REWRITE | `chapter_planner` | 重写（rewrite 用尽+低分先走 `corrector_node` 注入意外事件） |
| REFINE | `chapter_refiner` | 润色（v9.1 两级：1=段落修复, 2=整章润色兜底） |
| PASS | `__exit_for_chapter__` | 通过 → database_writer → 子图退出 |

---

## 三、重试与润色上限

### 3.1 阈值映射

| final_score | 操作 | 上限 |
|------------|------|------|
| < 55 | 完全重写 → `chapter_planner` | `MAX_REWRITE_ATTEMPTS = 5` |
| 55 ≤ final < 80 | 润色 → `chapter_refiner` | `max_refine = 2` |
| ≥ 80 | 通过 → `__exit_for_chapter__` | 0 次 |

### 3.2 次数管理（AttemptInfo）

- `loop_count`：重写计数器，REWRITE 后 +1
- `refine_attempts`：润色计数器，REFINE 后 +1
- 双向用尽 → 强制 PASS（防死循环）；单向用尽 → 交叉换策略（见 2.2 优先级 5/6）
- 次数由 `AttemptInfo`（`evaluation/schemas.py`）追踪，写入 `attempt_info` 字段

---

## 四、轻量复查（quick_recheck）

`UnifiedReviewEngine.quick_recheck()` — 修订稿快速复核：

- 输入：修订稿 + 旧问题清单 + 旧分数
- 输出格式与主评审一致（`[评分] final=` 标签），解析复用 `parse_review_output`
- 复查失败 → 降级处理，不阻塞流程
- 协调器 `_try_quick_recheck` 包裹 try/except（v8.5-fix S2）

---

## 五、降级与容错

| 场景 | 处理 | 版本 |
|------|------|------|
| LLM 评审调用失败（`ur.failed`） | 降级 PASS（防重写死循环） | v8.5-fix M3 |
| 四维分项缺失（quality_score=0） | 回退用 `final_score` | v8.5-fix S3 |
| LLM 输出越界分数 | parser/仲裁/engine 三层钳制 | v8.5-fix S2 |
| 轻量复查失败 | try/except 降级默认值 | v8.5-fix S2 |
| 仲裁 LLM 失败 | 返回原分 | v8.2 |
| 双向次数用尽 | 强制 PASS | v8.2 |

---

## 六、评分数据结构

### 6.1 核心 Schema

| 文件 | 内容 |
|------|------|
| `evaluation/unified/schemas.py` | `UnifiedFourDim` / `UnifiedReviewResult`（统一评审产物） |
| `evaluation/schemas.py` | `VerdictResult` / `VerdictLevel` / `FeedbackBundle` / `AttemptInfo`（决议契约） |
| `schemas/review_schemas.py` | `FourDimScores`（旧四维，兼容保留，统一评审不再消费） |

```python
# evaluation/schemas.py — 统一评审决议（唯一输出契约）
class VerdictLevel(str, Enum):
    PASS = "pass"
    REFINE = "refine"
    REWRITE = "rewrite"

class VerdictResult(BaseModel):
    level: VerdictLevel
    passed: bool
    final_score: float = Field(ge=0.0, le=100.0)
    quality_score: float = Field(ge=0.0, le=100.0)
    programmatic_score: float = Field(ge=0.0, le=1.0)  # 兼容保留，恒 0.0
    cross_chapter_consistency: float
    debate_penalty: float                                  # 兼容保留，恒 0.0
    ai_style_score: float = Field(ge=0.0, le=1.0)
    llm_semantic_score / llm_human_like_score / llm_attraction_score: float
    llm_severe_toxic_detected / llm_analysis_failed: bool
    has_severe_toxic: bool
    is_short_text: bool
    feedback: FeedbackBundle
    attempt_info: AttemptInfo
```

### 6.2 状态字段映射

| 状态字段 | 类型 | 说明 | 来源 |
|---------|------|------|------|
| `verdict_result.level` | VerdictLevel | PASS/REFINE/REWRITE 三级决议 | VerdictEngine |
| `verdict_result.final_score` | float | 综合分 + 迭代加分 (0-100) | VerdictEngine |
| `verdict_result.quality_score` | float | 四维之和，<=0 时回退 final_score | VerdictEngine |
| `verdict_result.has_severe_toxic` | bool | LLM 判定 severe 毒点 | UnifiedReviewResult |
| `verdict_result.llm_analysis_failed` | bool | 统一评审是否失败 | UnifiedReviewResult |
| `crew_result.quality_score` | float | 质量总分 | coordinator |
| `best_version_text/quality` | str/float | 最佳版本保留（REWRITE 前保存，失败恢复） | coordinator |

---

## 七、相关代码文件

| 文件 | 职责 |
|------|------|
| `evaluation/unified/engine.py` | UnifiedReviewEngine — 单次调用五视角评审 + 轻量复查 |
| `evaluation/unified/parser.py` | 标签化双通道解析 + `_clamp` 钳制 + 自洽校验 |
| `evaluation/unified/prompts.py` | 五视角评审 prompt + 复查 prompt + 仲裁 prompt |
| `evaluation/unified/arbitration.py` | 分歧仲裁（仅 severe 分歧，1 次调用） |
| `evaluation/unified/schemas.py` | UnifiedFourDim / UnifiedReviewResult |
| `evaluation/verdict/engine.py` | VerdictEngine — 融合 + 迭代加分 + 三级决议 |
| `evaluation/verdict/router.py` | verdict_router 三分支路由（纯路由） |
| `evaluation/coordinator.py` | verdict_engine_node 图节点封装 + best_version 保存 + 轻量复查协调 |
| `evaluation/schemas.py` | VerdictResult / FeedbackBundle / AttemptInfo |
| `graph/crews/writing_crew.py` | WritingCrewLocalState（含 critic/guidance/best_version 字段声明） |
| `graph/crews/writing_nodes/routing.py` | _exit_for_chapter 出口 |
| `graph/crews/writing_nodes/reviewer.py` | chapter_refiner 润色节点 |
| `config/constants.py` | 阈值（VERDICT_PASS/REFINE_THRESHOLD、VERDICT_ITERATION_BONUS_*） |
| `config/quality_params.py` | 动态调参中心（verdict.* / unified.* 可运行时覆盖） |

---

## 八、常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| quality_score 为 0 | 四维分项缺失（复查/降级路径） | v8.5-fix：`quality_score <= 0` 回退 `final_score` |
| 评审失败触发整章重写 | `ur.failed` 未单独处理 | v8.5-fix：失败降级 PASS，仅记录 `llm_analysis_failed` |
| 分数越界导致 ValidationError | LLM 输出 >100 / >1.0 | parser/仲裁/engine 三层 `_clamp` 钳制 |
| 仲裁后毒点章节放行 | 仲裁高分绕过一致性硬约束 | v8.5-fix：仲裁后重新执行 `apply_consistency_check` |
| 反复重写死循环 | final < 55 且重写不改善 | 双向用尽强制 PASS；迭代加分缓解 |
| 次品堆分通过 | 迭代加分过多 | v9.1：加分封顶 4 分（原 8） |
| 最佳版本恢复失效 | 条件边内改 state 不写回 checkpoint | v8.5-fix：迁移到 `verdict_engine_node` 显式写入 |

---

**规则版本：** v3.0.0
**生效方式：** 智能生效
**最后更新：** 2026-08-17
