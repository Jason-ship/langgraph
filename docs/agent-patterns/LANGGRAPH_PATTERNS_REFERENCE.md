# LangGraph 模式参考 — NovelFactory 专业智能体

> 基于 LangGraph 教程深度索引（v5.4.0+）整理
> 来源：https://langchain-ai.github.io/langgraph/tutorials/
> 建立日期：2026-06-25

---

## 子文档索引

各智能体的独立模式参考文档：

| 智能体 | 文档 |
|--------|------|
| Setup（WorldBuilder / CharacterDesigner / OutlineWriter / VolumeDetailWriter） | [SETUP_AGENT_PATTERNS.md](SETUP_AGENT_PATTERNS.md) |
| Writing（ChapterWriter / ChapterReviewer / ChapterRefiner） | [WRITING_AGENT_PATTERNS.md](WRITING_AGENT_PATTERNS.md) |
| Review（KickoffReview / ChapterFinalReview） | [REVIEW_AGENT_PATTERNS.md](REVIEW_AGENT_PATTERNS.md) |
| Sync（FeishuSync / StateUpdate） | [SYNC_AGENT_PATTERNS.md](SYNC_AGENT_PATTERNS.md) |
| Media（Illustrator / TTSGenerator） | [MEDIA_AGENT_PATTERNS.md](MEDIA_AGENT_PATTERNS.md) |

---

## 目录

1. [Setup 智能体（WorldBuilder / CharacterDesigner / OutlineWriter / VolumeDetailWriter）](#1-setup-智能体)
2. [Writing 智能体（ChapterWriter / ChapterReviewer / ChapterRefiner）](#2-writing-智能体)
3. [Review 智能体（KickoffReview / ChapterFinalReview）](#3-review-智能体)
4. [Sync 智能体（FeishuSync）](#4-sync-智能体)
5. [Media 智能体（Illustrator / TTSGenerator）](#5-media-智能体)

---

## 1. Setup 智能体

### 涉及文件
- `setup_agents.py` — WorldBuilder, CharacterDesigner, OutlineWriter, VolumeDetailWriter
- 图结构：Setup Crew（Supervisor → 线性执行 4 个 Worker）

### 当前模式
- **Prompt Chaining**（线性链）：WorldBuilder → CharacterDesigner → OutlineWriter → VolumeDetailWriter
- **Orchestrator-Worker**：Setup Supervisor 编排 4 个 Worker

### 可借鉴的 LangGraph 模式

#### 5.1 Plan-and-Execute（规划执行）
- **Planner** 对应 OutlineWriter（生成卷级大纲）
- **Executor** 对应 VolumeDetailWriter（逐章执行）
- **Replanner** 当前缺失 → 建议增加 `Replanner` 节点，在 VolumeDetailWriter 完成后评估是否需要调整卷级大纲
- 优势：当 VolumeDetailWriter 发现某卷规划不合理时，可回退到 OutlineWriter 重新规划

#### 6.1 Basic Reflection（基础反思）
- WorldBuilder 输出后 → 可增加自我反思节点检查世界观自洽性
- CharacterDesigner 输出后 → 可增加自我反思节点检查角色成长弧线完整性
- 实现方式：在 Setup Supervisor 中增加反思边（worker → reflect → refine or pass）

#### 8.1 STORM（网络研究 / 角色扮演对话模拟）
- `InterviewState` 独立子图模式可借鉴用于生成角色设定
- 多个 Editor 角色（如"正派视角"、"反派视角"、"中立视角"）可并行生成不同角度的角色分析
- `Send()` API 实现并行面试（角色生成）

### 关键 API 参考
```python
# Plan-and-Execute: 联合类型区分继续规划 vs 结束
class Act(TypedDict):
    response: str
    plan: list[str]

# Send() 动态并行
from langgraph.types import Send
send = Send("node_name", {"arg": value})
```

---

## 2. Writing 智能体

### 涉及文件
- `writing_agents.py` — ChapterWriter, ChapterReviewer, ChapterRefiner
- 图结构：Writing Crew（Supervisor → 质量门控循环）

### 当前模式
- **Evaluator-Optimizer**（评估器-优化器）：ChapterWriter → ChapterReviewer → [score≥90: handoff] | [score≥60: ChapterRefiner → ChapterReviewer重审] | [score<60: ChapterWriter重写]
- **Supervisor**：Writing Crew Supervisor 控制循环路由

### 可借鉴的 LangGraph 模式

#### 5.1 Plan-and-Execute（规划执行）
- ChapterWriter 可借鉴 `Plan` 结构化输出，写作前先生成章节规划
- 当前已有 `<thinking>` 标签内的章节规划 → 可升级为结构化 `Plan` 输出（含步骤列表、字数分配、感官描写计划）
- 执行后可由 Replanner 检查是否按计划完成

#### 5.2 ReWOO（减少冗余 LLM 调用）
- `#E1 = Tool[args]` 变量替换模式可减少 ChapterWriter 的工具调用次数
- 当前 ChapterWriter 在写作中可能多次调用 Neo4j/Milvus 工具
- 优化：先规划所有工具调用 → 一次性执行 → 基于结果写作

#### 5.3 LLMCompiler（流式 DAG 任务执行）
- ChapterWriter 写作 + ChapterReviewer 评分可流式执行
- Joiner 模式可加速"写作 → 评分 → 润色"循环

#### 6.1 Basic Reflection（基础反思）
- 与当前 ChapterReviewer → ChapterRefiner 模式吻合
- **角色反转技巧**：将 AI 消息转为 Human 消息用于下一轮反思 → 已在 `chapter_draft` 传递中部分实现

#### 6.2 Reflexion（高级反思）
- 比 Basic Reflection 更深入的批判机制
- 评估**缺失和多余**信息（不仅评估质量问题）
- 可增加对"信息完整性"的检查：章节是否遗漏了关键情节点、是否包含多余描述

#### 6.3 Tree of Thoughts (ToT)
- **Expand**：ChapterWriter 生成多个章节草稿候选
- **Score**：ChapterReviewer 评估每个候选质量
- **Prune**：保留最高分候选，或并行推进多个候选
- 适用场景：关键转折章节（如第 10% / 50% / 90% 章节）
- `Send()` API 并行扩展多个候选

#### 8.4 Complex Data Extraction（复杂数据提取）
- `ValidatorNode` 模式可应用于 ChapterReviewer 的输出验证
- 重试策略与当前 `validate_json_output` 一致，但可增加 `field_validator` 字段级 Pydantic 验证
- 最大重试次数可配置（当前默认 3 次）

### 关键 API 参考
```python
# ToT: beam search + Send() 并行
def expand(state):
    candidates = [Send("generate", ...) for _ in range(num_candidates)]
    return candidates

def score_candidates(state):
    scored = [{"candidate": c, "score": score_fn(c)} for c in state.candidates]
    state.candidates = sorted(scored, key=lambda x: x["score"], reverse=True)[:beam_size]

# Reflexion: 评估缺失和多余
reflection_prompt = "分析以下章节：缺失了什么关键信息？包含了哪些多余内容？"
```

---

## 3. Review 智能体

### 涉及文件
- `review_agents.py` — KickoffReviewAgent, ChapterFinalReviewAgent
- 图结构：独立审核子图（质量门控后的终审）

### 当前模式
- **Agent**（简单工具使用 Agent）：审核 + 飞书通知

### 可借鉴的 LangGraph 模式

#### 6.1 Basic Reflection（基础反思）
- KickoffReview 已完成 4 维量化评分体系（世界观/角色/大纲/自洽性）
- 可增加**自反思循环**：Review Agent 对自己的审核结果进行复审

#### 7.1 Agent-based Evaluation（基于模拟用户的评估）
- **Simulated User**：让 LLM 扮演"读者"角色评审章节
- 当前的 ChapterReviewer 是专业评分者，缺少"普通读者"视角
- 建议增加：Simulated Reader Agent（模拟用户）评估可读性和吸引力

#### 6.2 Reflexion（高级反思）
- KickoffReview 评分标准已完善（Few-Shot示例 + 5档评分锚点）
- 可增加对"审核自身质量"的评估：审核意见是否具体、扣分点是否准确

#### 8.4 Complex Data Extraction（复杂数据提取）
- `_bind_validator_with_retries` 模式可通用化
- 当前 `validate_json_output` 已实现类似功能

### 关键 API 参考
```python
# 模拟用户评估
simulated_user_prompt = "你是一个普通读者，刚刚读完第X章的网络小说..."
simulated_user → evaluate readability and engagement

# 自反思审核
review_reflection_prompt = "请评价你对第X章的审核意见：扣分点是否准确？是否遗漏了重要问题？"
```

---

## 4. Sync 智能体

### 涉及文件
- `sync_agents.py` — FeishuSyncAgent, StateUpdate
- 图结构：Sync Crew（Agent → StateUpdate）

### 当前模式
- **Agent**（简单工具使用 Agent）：飞书文档同步 + 通知

### 可借鉴的 LangGraph 模式

#### 2.1 Customer Support（客服机器人）
- **Part 2: Add Confirmation**：`interrupt` 实现上传前人工确认
  - 当前直接上传到飞书，无确认步骤
  - 可引入 `interrupt_before=["feishu_sync_agent"]` 实现人工确认
- **Part 3: Conditional Interrupt**：仅首次同步需确认，后续自动同步
- **Part 4: Specialized Workflows**：多个专用助手（同步/通知/备份）

#### 1.2 Common Workflows - Routing
- 可根据同步结果路由到不同后处理：
  - 成功 → 发送完成通知
  - 部分成功 → 发送警告
  - 失败 → 重试或回退

### 关键 API 参考
```python
# interrupt 确认模式
interrupt_before=["feishu_sync"]
# 恢复执行
Command(resume={"confirmed": True})

# Conditional Interrupt
def should_interrupt(state):
    return state.get("is_first_sync", False)  # 仅首次同步中断
```

---

## 5. Media 智能体

### 涉及文件
- `media_agents.py` — IllustratorAgent, TTSGeneratorAgent
- 图结构：Media Crew（Supervisor → 并行执行 Illustrator + TTS）

### 当前模式
- **Parallelization**（并行化）：Illustrator + TTS Generator 并行执行
- **Agent**（简单 Agent）：目前无工具，直接调用 LLM + subprocess

### 可借鉴的 LangGraph 模式

#### 1.2 Common Workflows - Parallelization
- 当前已正确实现并行模式（两个独立 Worker）
- 可增加**聚合器**节点：`illustrate_and_tts_aggregator` 合并并行结果
- 聚合器可检测：插画是否成功、TTS 是否成功、是否需要重试

#### 3.3 Corrective RAG (CRAG) 的回退模式
- 当 `_generate_image_via_matrix` 失败时，可触发回退机制：
  - 主路径：Matrix MCP → LLM 生成插画
  - 回退路径：DALL-E / Stable Diffusion API → 替代插画
  - 二次回退：纯文字描述插画（无图）

#### 2.1 Customer Support - Error Handling
- 工具调用失败时的重试和降级策略
- 与当前 `_generate_image_via_matrix` 的回退逻辑一致

### 关键 API 参考
```python
# 并行 + 聚合器
parallel_workers = [illustrator_node, tts_node]
aggregator_node → {"illustration_url": ..., "audio_url": ...}

# CRAG 风格的回退
def decide_generation(state):
    if state.illustration_success:
        return "aggregate"
    else:
        return "fallback_illustrator"
```

---

## 通用改进建议

### 1. 中断机制（interrupt）
- 当前 `interrupt_before=[]` 全部禁用
- 建议按需启用：Setup 审核失败时中断、同步前中断确认

### 2. 子图组合（Subgraph Composition）
- 当前 4 个 Crew（Setup/Writing/Sync/Media）是独立编译的图
- 可参考 **Hierarchical Teams** 模式，在顶层图中使用 `node = crew_graph.compile()` 引用子图

### 3. 状态持久化（MemorySaver / Checkpointer）
- 当前已使用 `MemorySaver` 作为 checkpointer
- 可扩展为 SQLite/PostgreSQL 后端（参考 Customer Support 的 SQLite 工具）

### 4. 动态 Prompt（Dynamic Prompt）
- v6.0 已引入 `_build_*_dynamic_prompt` 函数
- 可进一步借鉴 **Self-Discover Agent** 模式，让 Agent 自主学习自身能力体系

### 5. Token 优化
- **ReWOO** 变量替换可减少 30-40% 的 LLM 调用
- **LLMCompiler** 流式规划 + 并行执行可进一步提升效率
- **Self-RAG** 的 4 维评分体系与当前评分系统高度一致

---

> 本文档基于 LangGraph 官方教程 (v5.4.0+) 整理，建议配合官方文档阅读。
> 官方地址：https://langchain-ai.github.io/langgraph/tutorials/
