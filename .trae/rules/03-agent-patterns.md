---
alwaysApply: false
description: "Agent开发规范，匹配agents/**、schemas/**。改Agent行为、调LLM参数、重试/熔断/缓存/配额/超时、结构化输出降级、Agent工厂、辩论Schema时触发。infra（retry/circuit_breaker/llm_cache/quota/timeout）、结构化输出、LLM配置（0.3/0.75/0.2）、WallTimeTracker、_ENV_OVERRIDES。"
---
# Agent 开发模式规范

**版本：** v1.2.0
**生效方式：** 智能生效
**优先级：** ⭐⭐⭐⭐⭐
**匹配模式：** `agents/**`, `schemas/**`, `evaluation/**`

---

## 一、基础设施层（agents/infra/）

`agents/infra/` 目录包含 13 个文件，为所有 Agent 提供通用基础设施：

| 文件 | 职责 |
|------|------|
| `retry.py` | 同步 LLM 调用包装器 — timeout + 错误类型感知重试 + 用量审计 |
| `async_retry.py` | 异步 LLM 调用包装器 — asyncio 版 timeout + 重试 + 缓存 |
| `_retry_common.py` | 同步/异步重试公共逻辑（状态分类、截断检测、用量记录） |
| `circuit_breaker.py` | 基于服务的熔断器（ARK/DeepSeek/硅基流动三态保护） |
| `llm_cache.py` | LLM 响应缓存 — 基于 Redis 的语义缓存，优雅降级 |
| ~~`rag_cache.py`~~ | ~~RAG 检索结果 LRU 缓存 - 线程级、TTL 驱动~~（已移除） |
| `quota.py` | Token/请求配额管理 — DeepSeek Billing API 限速检测 |
| `timeout.py` | LLM 调用超时守卫 — threading.Event 跨平台实现 |
| `serialization.py` | LLM 输出 JSON 提取与校验 — fail-closed 设计 |
| `logger.py` | 结构化日志 — 支持文件轮转/JSON 格式 |
| `helpers.py` | 状态访问助手 + AI 消息文本提取 |
| `stream.py` | Crew 流式输出 — 缓冲写入临时文件供用户可见 |
| `usage.py` | Token 用量追踪 — 进程级计数 + 成本估算 |

### 1.1 重试策略（retry.py + async_retry.py）

从 LangGraph `RetryPolicy` 迁移重试配置，与 `graph/checkpointer.py` 定义的常量对齐：

```python
from langgraph.types import RetryPolicy

DEFAULT_RETRY = RetryPolicy(
    max_attempts=3,
    initial_interval=1.0,
    max_interval=60.0,
    jitter=True,
)
```

**同步重试**（`retry.py` — `llm_call_with_retry`）：
- 使用 `threading.Event` + `with_timeout` 装饰器实现超时
- 适用于同步上下文（如 LangGraph 同步节点）

**异步重试**（`async_retry.py` — `async_llm_call_with_retry`）：
- 使用 `asyncio.wait_for` 替代 threading 超时
- 使用 `asyncio.sleep` 替代 `time.sleep`
- 内置 LLM 响应缓存（v5.4 新增），通过 `cache_prompt` 参数启用

**共用重试策略**：
| HTTP 状态 | 行为 | 说明 |
|-----------|------|------|
| 400/401/403 | 永不重试 | 请求格式/鉴权错误 |
| 429 | 立即重试 | 速率限制 |
| 500/502/503/504 | 指数退避重试 | 服务端临时故障 |
| TimeoutError/OSError | 指数退避重试 | 网络/连接故障 |

三种 RetryPolicy 映射（通过 `retry_policy` 参数选择）：
- `"default"` → `DEFAULT_RETRY`（3 次重试）
- `"writer"` → `WRITER_RETRY`（写作场景，更多重试）
- `"reviewer"` → `REVIEWER_RETRY`（评审场景）

### 1.2 熔断器（circuit_breaker.py）

基于服务的熔断器，保护外部 LLM API 调用。三种状态的自动恢复机制：

```python
# 关闭（CLOSED）— 正常调用
# 打开（OPEN）— 快速失败，cooldown 后自动切半开
# 半开（HALF-OPEN）— 探针请求，成功则关闭，失败则重开
```

**Provider 配置**：

| Provider | 最大失败数 | 冷却时间 | 半开最大探针数 |
|----------|-----------|---------|--------------|
| ark | 20 | 30s | 2 |
| deepseek | 20 | 30s | 2 |
| siliconflow | 10 | 60s | 1 |

**使用场景**：LLM API 调用前检查 `circuit_breaker_is_open("deepseek")`，已打开时快速返回 fallback，避免浪费配额和等待时间。成功调用后记录 `circuit_breaker_record_success`，失败时记录 `circuit_breaker_record_failure`。

### 1.3 LLM 缓存 + RAG 缓存

**LLM 缓存**（`llm_cache.py`）：
- 缓存键：`sha256(prompt[:2000]) + model + temperature`
- 默认 TTL：3600 秒（1 小时）
- Redis 不可用时优雅降级，不影响主流程
- `async_llm_call_with_retry` 通过 `cache_prompt` 参数集成

```python
cache_key = _build_cache_key(model, temperature, prompt)
# → "llm_cache:{model}:t{temperature:.2f}:{prompt_hash}"
```

**RAG 缓存**（~~`rag_cache.py`~~，已移除）：
- ~~线程级 LRU 缓存，键由 `dataset_id + sha256(query)[:16]` 构成~~
- ~~默认容量 500 条，TTL 3600 秒~~
- ~~提供 `get_rag_cache()` 线程局部单例~~

### 1.4 配额管理（quota.py）

Token 配额检查机制，防止超出 DeepSeek 预算：
- 定期从 DeepSeek Billing API 刷新配额快照
- 检查间隔默认 60 秒（可配置 `QUOTA_CHECK_INTERVAL_SECONDS`）
- 三级阈值：
  - **阻塞**（默认 ≤5%）：返回 `blocked=True`，阻止 LLM 调用
  - **警告**（默认 ≤20%）：返回 `reason="warn:quota_low:X.X%"`，仅日志
  - **正常**：无操作
- 配额 API 错误时优雅降级，不阻塞主流程

### 1.5 超时控制（timeout.py）

统一的 LLM 超时控制机制：
- 使用 `threading.Event` 实现跨平台兼容（非 `signal.SIGALRM`）
- 超时时返回默认值，不抛出异常
- 与重试策略配合：超时触发 `LLMTimeoutError`，归类为 `"backoff"` 重试

```python
def with_timeout(seconds: float, default: T) -> Callable:
    """Decorator: run a function with a timeout, return default on timeout."""
```

### 1.6 其他组件

| 模块 | 职责说明 |
|------|---------|
| `serialization.py` | `_extract_json_from_text()` 从 LLM 自由文本中提取 JSON；`validate_json_output()` 提供 fail-closed 校验 |
| `usage.py` | 进程级 Token 计数（替代 threading.local），支持 DeepSeek Flash 定价模型：¥0.5/1M 输入 + ¥2.0/1M 输出 |
| `logger.py` | `get_logger()` 工厂函数，支持文本/JSON 两种格式，可选 RotatingFileHandler（10MB × 5 备份） |
| `helpers.py` | `extract_ai_message_text()` 统一提取最后一条 AI 消息；`_extract_from_state()` 支持 `crew_result` 嵌套读取 |
| `stream.py` | `StreamWriter` 缓冲写入临时文件，每 5 次 write 批量刷盘；`get_crew_stream()` / `cleanup_crew_stream()` 管理 Crew 生命周期 |

---

## 二、结构化输出（agents/utils/structured.py）（v5.8 已移除）

> **⚠️ 已废弃**：`agents/utils/structured.py` 在 v5.8 已移除。`bind_structured` 和 `invoke_structured_or_freetext` 不再使用。当前 Agent 统一使用 `async_llm_call_with_retry` + 文本提取（`serialization.py` 的 `_extract_json_from_text` / `validate_json_output`）实现结构化输出。

v5.6 重构的统一结构化输出 API。借用了 TradingAgents 模式：

```python
def bind_structured(
    llm: object, schema: type[T], agent_name: str = ""
) -> object | None:
    """Wrap an LLM with structured output capability.

    自动展开 RunnableWithFallbacks，尝试 function_calling → json_mode 兜底。
    均不支持时返回 None。
    """
    inner_llm = _unwrap_llm(llm)
    for method in ("function_calling", "json_mode"):
        try:
            return inner_llm.with_structured_output(schema, method=method)
        except (NotImplementedError, AttributeError) as exc:
            logger.debug("%s: method=%s not supported (%s)", agent_name, method, exc)
        except Exception as exc:
            if "not support" in str(exc).lower():
                continue
            logger.warning(...)
    return None


def invoke_structured_or_freetext(
    structured_llm: object | None,
    plain_llm: object,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str = "",
) -> str:
    """Structured-first, free-text fallback.

    结构化成功 → render(result)。
    结构化失败/不可用 → plain_llm.invoke(prompt).content。
    """
    if structured_llm is not None:
        try:
            result = structured_llm.invoke(prompt)
            if result is None:
                raise ValueError("structured output returned no parsed result")
            return render(result)
        except Exception as exc:
            logger.warning("%s: 结构化失败 (%s); 降级自由文本", agent_name, exc)
    return plain_llm.invoke(prompt).content
```

**规范**：
- **异常安全**：`bind_structured` 捕获 `NotImplementedError` / `AttributeError`，provider 不支持的 method 自动跳过
- **自动降级**：`invoke_structured_or_freetext` 在结构化失败时回退到自由文本调用
- **一次性绑定**：在工厂函数级调用 `bind_structured` 一次，在节点函数中复用——避免每次 invoke 都重复绑定
- **RunnableWithFallbacks 展开**：DeepSeek 等 provider 的 `with_structured_output` 不穿透 fallback 层，需要展开后再绑定

---

## 三、Agent 工厂模式

### 3.1 Agent 文件清单

| 文件 | 职责 |
|------|------|
| `agents/writing_agents.py` | 写作/审查/润色 Agent |
| `agents/review_agents.py` | 审核 Agent（人工/自动） |
| `agents/setup_agents.py` | Setup Agent（项目初始化） |
| `agents/media_agents.py` | 媒体生成 Agent（图片/音频） |
| `agents/sync_agents.py` | 飞书同步 Agent |
| `agents/quality_panel_agents.py` | 旧版辩论评审 Agent（v6.3 废弃，仅保留兼容引用） |

### 3.2 多智能体并行模式

#### 3.2.1 线程级并行（media_crew）

`agents/media_agents.py` 中的 Illustrator Agent 和 TTS Generator Agent 通过 `ThreadPoolExecutor` 并发生成：

```python
# media_crew.py 中
with ThreadPoolExecutor(max_workers=2, thread_name_prefix="media_crew") as pool:
    future_ill = pool.submit(illustrator_agent, illustrator_input)
    future_tts = pool.submit(tts_agent, tts_input)
```

**约束**：
- Agent 必须是**同步 Runnable**（不支持 async）
- 每个 Agent 有独立的重试机制（`_media_tool_router`，最多 3 次）
- 单 Agent 失败自动降级，不影响另一 Agent

#### 3.2.2 逻辑并行（评审子Agent）

`evaluation/verdict/engine.py` 中的 `VerdictEngine.evaluate()` 统一编排 4 个评审维度，每个有独立 try/except：

```python
# 各子 Agent 地位平等，互不依赖
quality_result = _safe_call(run_quality_review, ...)    # 四维评分
ai_result = _safe_call(run_ai_style_review, ...)        # AI 味检测
old_result = _safe_call(run_old_reader_review, ...)     # 老书虫评审
debate_result = _safe_call(run_debate_review, ...)      # 辩论评审
```

#### 3.2.3 VerdictEngine 融合评审（替代 v5.5 QualityPanel）

评审已从 Agent-based quality_panel 重构为 `evaluation/` 模块中的结构化评分管线。核心 `VerdictEngine`（`evaluation/verdict/engine.py`）融合多源评分：

- **四维 LLM 评分** → `evaluation/llm/_shared.py`
- **程序化评分** → `evaluation/programmatic/runner.py`
- **LLM 老书虫语义分** → `evaluation/llm/old_reader_llm.py`
- **LLM AI味语义分** → `evaluation/llm/ai_style_llm.py`
- **辩论评分** → `evaluation/debate/engine.py`（InformedDebateEngine 编辑↔读者↔Critic 3 角色多轮辩论）
- **跨章一致性** → `evaluation/programmatic/cross_chapter_sensor.py`

详见 [04-quality-scoring.md](file:///Users/jason/Downloads/langgraph/.trae/rules/04-quality-scoring.md)

#### 3.2.4 并行 Agent 设计原则

| 原则 | 说明 | 违反后果 |
|------|------|---------|
| **容错降级** | 每个 Agent 有独立 try/except，失败返回默认值 | 单 Agent 故障阻塞整体流程 |
| **无共享可变状态** | Agent 间通过函数参数/返回值传递数据 | 线程安全问题 |
| **幂等设计** | 同一输入多次调用产生相同结果 | 重试导致数据不一致 |
| **超时保护** | 使用 `with_timeout` 防止 Agent 挂死 | 线程池耗尽 |
| **有限重试** | 最多 3 次重试后强制通过 | 无限循环 |

### 3.3 职责边界

| Agent | 输入 | 输出 | 使用场景 |
|-------|------|------|---------|
| `setup_agents` | seed_idea, genre | project_outline, world_setting | 项目初始化阶段 |
| `writing_agents` | chapter_outline, context | chapter_content | 章节写作 |
| `review_agents` | chapter_content | review_decision | 人工审核等待 |
| `quality_panel_agents` | chapter_content | quality_score, issues | 辩论式自动评审 |
| `media_agents` | chapter_content | media_urls | 媒体生成 |
| `sync_agents` | chapter_content | feishu_doc_url | 飞书同步 |

### 3.3 评审 Schema（evaluation/）

评分 Schema 定义在 `evaluation/schemas.py`，评审 Schema 定义在 `schemas/review_schemas.py`，辩论 Schema 定义在 `evaluation/debate/engine.py`。

---

## 四、LLM 配置模式

所有 LLM 参数集中在 `config/llm_params.py` 管理，禁止硬编码 temperature。

### 4.1 Tier 参数总览

| Tier | temperature | max_tokens | retry_policy | 用途 |
|------|:-----------:|:----------:|:------------:|------|
| supervisor | 0.3 | 65536 | default | 编排调度，低温度确保路由决策确定性 |
| worker | 0.7 | 65536 | writer | 创作写作，中等温度平衡创意与稳定 |
| reviewer | 0.2 | 65536 | reviewer | 结构评分，低温度确保评分一致性 |
| review | 0.2 | 65536 | default | 人工终审，低温度确保审核判断稳定 |
| writing | 0.75 | 65536 | writer | 叙事写作，最高温度追求文学创意 |

### 4.2 Agent 参数覆盖

部分 Agent 在 Tier 基础上有更精细的参数：

| Agent | Tier | temperature | timeout | 说明 |
|-------|:----:|:-----------:|:-------:|------|
| chapter_writer | worker | **0.75** | 600s | 较长超时较高创造性 |
| chapter_refiner | worker | **0.5** | 300s | 中低温度确保精确性 |
| chapter_planner | worker | **0.3** | 120s | 低温度确保结构化输出 |
| illustrator | worker | **0.8** | 120s | 高温度增加创意多样性 |
| four_dim_review | reviewer | **0.15** | 180s | 最低温度确保最严格评分 |
| editor_review | reviewer | **0.3** | 180s | 略高温度允许视角多样性 |
| reader_review | reviewer | **0.35** | 180s | 最高温度允许主观判断 |

### 4.3 运行时调优

```bash
# CLI 调优（运行时生效）
novel params set-tier worker --temperature 0.8
novel params set-agent worker chapter_writer --temperature 0.9

# API 查询
curl http://localhost:8123/health/params

# 环境变量覆盖（需重启）
export NOVELEACTORY_LLM_TEMPERATURE=0.5        # 全局
export NOVELEACTORY_LLM_WORKER_TEMPERATURE=0.8  # 仅 worker tier
export NOVELEACTORY_LLM_WORKER_CHAPTER_WRITER_TIMEOUT=900  # 单 Agent
```

### 4.4 新增 Agent 注册规范

```python
from novelfactory.config.llm_params import center, LLMParams

center.register_agent(
    "worker", "my_new_agent",
    LLMParams(temperature=0.6, timeout_seconds=180.0,
              description="用途说明"),
)
```

---

## 五、WallTimeTracker（v5.6）

全链路性能追踪器，融合 LLM 调用统计：

```python
from novelfactory.utils.wall_time_tracker import WallTimeTracker

tracker = WallTimeTracker()
tracker.start("context_builder", phase="writing")

# ... 节点执行 ...

tracker.end("context_builder", token_count=15000, llm_calls=3)  # v5.6: llm_calls
tracker.report()
```

**输出格式**：
```
node_name          duration  tokens  llm_calls  phase
context_builder    15.0s     15,000  3          writing
TOTAL              15.0s     15,000  3
```

**v5.6 新增**：`llm_calls` 参数追踪每次 `tracker.end()` 内发生的 LLM 调用次数，与 Token 用量联合分析效率比。

---

## 六、环境变量类型安全覆盖（v5.6）

借用了 TradingAgents `default_config.py` 的 `_ENV_OVERRIDES` 模式：

```python
_ENV_OVERRIDES = {
    "NOVELFACTORY_APP_VERSION": "APP_VERSION",
    "NOVELFACTORY_LOG_LEVEL": "LOG_LEVEL",
    "NOVELFACTORY_CHECKPOINT_TYPE": "CHECKPOINT_TYPE",
    "NOVELFACTORY_STORAGE_TYPE": "STORAGE_TYPE",
    "NOVELFACTORY_QUOTA_CHECK_BEFORE_CALL": "QUOTA_CHECK_BEFORE_CALL",
    "NOVELFACTORY_MAX_RETRIES": "MAX_RETRIES",
    "NOVELFACTORY_CHAPTER_MIN_WORD_COUNT": "CHAPTER_MIN_WORD_COUNT",
    "NOVELFACTORY_CHAPTER_TARGET_WORD_COUNT": "CHAPTER_TARGET_WORD_COUNT",
    "NOVELFACTORY_QUOTA_THRESHOLD": "QUOTA_THRESHOLD",
}
```

`_coerce_env` 的类型安全转换逻辑：

```python
_BOOL_TRUE = ("true", "1", "yes", "on")
_BOOL_FALSE = ("false", "0", "no", "off")

def _coerce_env(value: str, reference: object) -> object:
    """Coerce env-var string to the type of the existing default value."""
    if isinstance(reference, bool):
        normalized = value.strip().lower()
        if normalized in _BOOL_TRUE:
            return True
        if normalized in _BOOL_FALSE:
            return False
        raise ValueError(...)
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value
```

**使用方式**：
```bash
# 环境变量优先于 .env 文件优先于默认值
export NOVELFACTORY_CHECKPOINT_TYPE='postgres'
export NOVELFACTORY_MAX_RETRIES=5
```

`Settings.model_post_init()` 自动遍历 `_ENV_OVERRIDES`，为每个非空环境变量做类型安全转换后覆盖对应字段。

---

## 七、常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| quality_panel 辩论超限（旧版，v6.3 废弃） | 评分持续分歧 | 降级到默认通过（q=85, c=0.7），VerdictEngine 最多 3 轮辩论 |
| VerdictEngine 评分异常 | LLM 语义分全部降级 | 检查熔断器状态和配额余量 |
| `structured` 降级频繁 | provider 不支持 `with_structured_output` | 检查 `bind_structured` 日志 |
| 异步重试死锁 | 同步上下文调用异步重试 | 同步节点使用 `retry.py`，异步节点使用 `async_retry.py` |
| LLM 缓存未命中 | 缓存键不匹配 | 检查 `_build_cache_key` 使用的 model/temperature/prompt 是否一致 |
| 配额检查误阻塞 | 配额 API 返回异常数据 | 配额 API 错误时自动优雅降级，仅 HTTP 错误/超时时跳过检查 |
| 熔断器频繁打开 | Provider 服务不稳定 | 检查 `cooldown_seconds` 和 `max_failures` 配置，考虑增加冷却时间 |
| `extract_ai_message_text` 返回空 | messages 格式不符合预期 | 检查 result 中 messages 的结构，确认 `type=="ai"` 的消息存在 |
| RAG 缓存命中率低 | dataset_id 拼写不一致 | 确保所有调用点使用统一的 dataset_id 命名约定 |

---

**规则版本：** v1.2.0
**生效方式：** 智能生效
**最后更新：** 2026-07-04
