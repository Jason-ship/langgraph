# NovelFactory 项目全面体检报告

**体检日期**：2026-08-17
**体检范围**：Windows 服务器（100.64.0.6, D:\langgraph）运行态 + 本地源码静态审查
**体检方式**：SSH 连接 Docker 服务检查 + mypy/ruff 静态扫描 + P0 模块代码审查
**报告版本**：v1.0

---

## 一、执行摘要

| 维度 | 状态 | 说明 |
|------|------|------|
| 服务运行 | 🟢 正常 | 6 个容器全部 Up 且 healthy |
| 飞书链路 | 🟡 已修复 | tools_proxy 缺失 → 已启动恢复 |
| 本地-服务器一致性 | 🟢 已同步 | v8.4-r 5 处未提交修复已同步本地 |
| 静态检查 | 🟡 2 项环境问题 | mypy numpy/yaml 环境问题 |
| 代码质量 | 🔴 发现 9 严重 + 11 中等 | 详见第四节 |

**已完成的处置**：
1. 启动 `tools_proxy` 容器（飞书通知链路恢复，实测 /health 200）
2. 同步服务器 v8.4-r 未提交修复到本地源码（5 个文件）
3. 修复同步代码引入的重复 docstring + import 顺序问题
4. 服务器残留损坏文件 `C:\Users\16842\docker-compose.yaml` 已定位（实际使用 D:\langgraph 下的 compose）

---

## 二、服务运行状态

### 2.1 容器状态（Windows 100.64.0.6）

| 容器 | 状态 | 说明 |
|------|------|------|
| langgraph_api | Up (healthy) | v8.0.0，内存 408MiB/8GiB |
| langgraph_postgres | Up (healthy) | pgvector:pg16 |
| langgraph_redis | Up (healthy) | redis:6-alpine |
| langgraph_milvus | Up (healthy) | v2.4.17 |
| langgraph_neo4j | Up (healthy) | neo4j:5 |
| langgraph_tools_proxy | Up | **本次新启动** |

### 2.2 修复过程记录

1. **问题**：日志出现 `tools-proxy 不可用: http://172.28.0.1:5004`，飞书通知失败。
2. **根因**：`tools_proxy` 容器未运行；且 compose 默认目录（`C:\Users\16842`）存在损坏的 docker-compose.yaml，导致 compose 命令解析失败。
3. **修复**：切换到实际项目目录 `D:\langgraph`，Docker Desktop `credsStore: desktop` 在 SSH 会话中无法访问 Windows 凭据 → 通过 `DOCKER_CONFIG` 指向无凭据配置，先 `docker pull python:3.12-slim`，再 `docker compose up -d tools_proxy`，构建启动成功。

### 2.3 日志观察

- 当前正在执行「验证测试」项目写作流程（chapter 0/5），LLM 调用正常（deepseek-v4-flash）。
- `setup_quality_gate` score=87.0 通过；DB/Milvus/Neo4j/Redis 连接全部成功。
- 飞书在 tools_proxy 启动前创建目录失败（`飞书未配置，跳过目录创建`），启动后需观察恢复。

---

## 三、静态检查（本地）

### 3.1 ruff

- `src/novelfactory/channels/executor.py:16` — UP035 `AsyncIterator` 应导入自 `collections.abc`（1 项，可自动修复）。
- 其余全部通过；同步的 5 个 v8.4-r 文件已修复后通过。

### 3.2 mypy（2 项环境问题，非代码缺陷）

| 文件 | 错误 | 性质 |
|------|------|------|
| config/prompts/__init__.py:30 | Library stubs not installed for "yaml" | 缺 `types-PyYAML` |
| numpy `__init__.pyi:737` | Type statement only supported in 3.12+ | numpy 2.5.1 pyi 需 Python 3.12+ 语法，但 mypy 配置 `python_version = "3.10"` |

> 建议：安装 `types-PyYAML`；numpy pyi 语法问题可升级 mypy 的 python_version 目标或在 overrides 中 `follow_imports = "skip"` 处理。

### 3.3 测试

- `tests/unit/graph/test_routing_supervisor.py` + `test_graph_compile.py` + `test_lightweight_setup.py`：**34 passed, 1 skipped**（同步代码后验证通过）。

---

## 四、代码审查发现（P0/P1 模块）

> 由 3 个并行子代理审查 evaluation/、state/、agents/、config/、graph/ 全部模块，关键项已人工复核。行号为审查时快照，可能因后续修复漂移。

### 4.1 严重问题（9 项）

| # | 位置 | 问题 | 影响 |
|---|------|------|------|
| S1 | [coordinator.py](file:///Users/jason/Downloads/langgraph/src/novelfactory/evaluation/coordinator.py#L226-L232) | 降级分支写 `programmatic_score=50.0`，违反 [schemas.py](file:///Users/jason/Downloads/langgraph/src/novelfactory/evaluation/schemas.py#L597-L599) `Field(le=1.0)` 约束 | pydantic ValidationError → 评审失败降级路径二次抛错，**评审节点崩溃**（已复核） |
| S2 | [parser.py](file:///Users/jason/Downloads/langgraph/src/novelfactory/evaluation/unified/parser.py#L30-L34) / [arbitration.py](file:///Users/jason/Downloads/langgraph/src/novelfactory/evaluation/unified/arbitration.py#L53) | LLM 输出越界分数（>100 / >1.0）无钳制 → `VerdictResult` 构造抛 ValidationError | 主评审路径连锁崩溃；轻量复查路径无 try/except 保护 |
| S3 | [coordinator.py](file:///Users/jason/Downloads/langgraph/src/novelfactory/evaluation/coordinator.py#L193-L199) | 轻量复查 PASS 不含 `[四维-*]` → `quality_score=0` 写入章节 | 修复后章节被记录 0 分，污染质量数据 |
| S4 | [routing.py](file:///Users/jason/Downloads/langgraph/src/novelfactory/graph/routing.py#L104-L133) | `setup_aborted` 仅在 PHASE_SETUP 分支检查，checkpoint 恢复/其他入口绕过 | v8.4-r fail-closed 不完整，空设定仍可能进入写作 |
| S5 | [settings.py](file:///Users/jason/Downloads/langgraph/src/novelfactory/config/settings.py#L266-L267) / [_core.py](file:///Users/jason/Downloads/langgraph/src/novelfactory/integrations/feishu/_core.py#L31-L38) | `LARK_PROXY_URL` env 被 pydantic 丢弃（无字段）+ `lark_proxy_url` 属性恒真短路 | compose 意图 `tools_proxy:5004` 失效，实际打 `172.28.0.1:5004`，依赖宿主机发布端口才碰巧可用 |
| S6 | [writing_crew.py](file:///Users/jason/Downloads/langgraph/src/novelfactory/graph/crews/writing_crew.py#L121-L122) / [critic_pre.py](file:///Users/jason/Downloads/langgraph/src/novelfactory/graph/crews/writing_nodes/critic_pre.py#L78) | `critic_assessment` 未在子图 schema 声明 → 未知 channel 被丢弃 | **Critic 前置评估 FAIL 分支永久失效**（恒读 PASS），v7.3 功能未生效 |
| S7 | [routing.py](file:///Users/jason/Downloads/langgraph/src/novelfactory/graph/crews/writing_nodes/routing.py#L260) / [writing_crew.py](file:///Users/jason/Downloads/langgraph/src/novelfactory/graph/crews/writing_crew.py#L75-L125) | `chapter_needs_guidance` 未在子图声明 | 低分章节人工指导（HITL）永不触发 |
| S8 | [routing.py](file:///Users/jason/Downloads/langgraph/src/novelfactory/graph/routing.py#L86-L95) | 检查链进度推断基于跨章残留状态字段 | 第 2 章起 quality/foreshadowing 检查被跳过（已复核） |
| S9 | [settings.py](file:///Users/jason/Downloads/langgraph/src/novelfactory/config/settings.py#L349-L362) | 启动日志按字段名掩码，`DATABASE_URL`/`REDIS_URL` 内嵌密码不掩码 | **日志泄露 DB/Redis 明文密码** |

### 4.2 中等问题（11 项）

| # | 位置 | 问题 |
|---|------|------|
| M1 | evaluation/verdict/router.py:58-69 | `best_version_*` 在条件边内直接改 state（不写回 channel）且字段未声明 → 最佳版本恢复功能失效 |
| M2 | evaluation/unified/engine.py:57-72 | 仲裁在 consistency check 之后执行，高分仲裁可绕过硬约束放行毒点章节 |
| M3 | evaluation/verdict/engine.py:204-205 | LLM 评审失败降级为 REWRITE（与"降级为程序化评分"设计不符），一次 API 故障触发整章重写 |
| M4 | state/chapter_state.py:338 | `word_inventory` 合并对 LLM 畸形 JSON 无保护，缺 `name` 键抛 KeyError |
| M5 | config/settings.py + config/quota.py | 配额配置双源，settings 的 QUOTA_* 及 _ENV_OVERRIDES 条目无人读取（死配置） |
| M6 | config/llm_params.py:312-434 | retry_policy/timeout_seconds 注册值全部未消费（调用点均用默认 3 次/900s） |
| M7 | agents/infra/llm_cache.py | `cache_prompt` 参数无任何调用点（缓存未接线）；key 只哈希前 2000 字符有碰撞风险；缓存序列化必失败 |
| M8 | agents/infra/retry.py:136-148 | 429 分支最后一次失败时 `_record_provider_failures` 双计 → 熔断阈值提前触发 |
| M9 | config/llm.py:102-106 | DeepSeek 兜底链含已停用 ARK key / OPENAI_API_KEY，串用掩盖真实配置错误 |
| M10 | agents/registry.py:62-78 | `update` 重命名冲突时对象 name 已改但 dict key 未更新 |
| M11 | graph/nodes/setup_nodes.py:366-370 | `volume_detail_writer_node` 对 LLM 输出直接下标索引，缺键抛 KeyError（无 try/except） |

### 4.3 轻微问题（摘要）

- 四维评分定义三处分裂（review_schemas / unified/schemas / 规则文档不一致）
- `_fix_llm_json` 双括号正则笔误（setup_agents.py:292）
- `ch_range[0]` 空列表 IndexError（setup_agents.py:594）
- `LOG_FORMAT` 默认值为 Python 格式串但 logger 按 "json" 分支判断
- docker-compose 默认口令硬编码（novelpass2024 等）
- 若干死代码（lightweight_setup._retry_invoke、monitor_node.build_monitor_node、infra/helpers._extract_from_state 等）

---

## 五、配置一致性

| 项目 | 本地 | 服务器 | 一致性 |
|------|------|--------|--------|
| .env | ✅ | ✅ | 完全一致 |
| docker-compose.yml | ✅ | ✅ | 内容一致（仅行尾符差异） |
| 核心源码 (5 文件) | ✅ | ✅ | 已同步（v8.4-r 修复） |

> ⚠️ 服务器存在损坏残留文件 `C:\Users\16842\docker-compose.yaml`（仅 2 行残留内容），若在默认目录执行 compose 会解析失败。建议清理。

---

## 六、修复优先级建议

### 立即修复（崩溃级）
1. **S1** — coordinator.py:231 改为 `programmatic_score=0.5`（与 service.py 对齐）
2. **S2** — parser/arbitration 统一 clamp(0,100) / clamp(0,1)
3. **M11 / M3** — setup_nodes 下标防护 + LLM 失败降级逻辑

### 高优先级（功能失效）
4. **S6 / S7** — WritingCrewLocalState 声明 `critic_assessment` / `chapter_needs_guidance`（一行级修复）
5. **S5** — settings 增加 `LARK_PROXY_URL` 字段，`_core.py` 修正优先级
6. **S4** — route_from_supervisor 开头统一拦截 `setup_aborted`

### 中优先级（质量/安全）
7. **S9** — 启动日志对 URL 字段掩码 userinfo 密码
8. **S8** — 重构检查链进度推断
9. **S3** — 轻量复查 quality_score 兜底

### 清理项
10. 服务器残留损坏 compose 文件
11. mypy 环境（types-PyYAML + numpy pyi）
12. 死代码清理

---

*报告生成：TraeCode 体检流程 · 未修改任何业务逻辑（除已同步的 v8.4-r 修复与两处代码质量问题）*
