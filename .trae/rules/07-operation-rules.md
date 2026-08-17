---
alwaysApply: true
description: "操作纪律与规范，全局生效。触发：改代码、找文件、查数据库、构建/启动/重启API、看日志、提交代码、不用代理、配置镜像源。约束：本地直接编辑、Docker唯一运行时、源码唯一来源、先LS再操作、禁止sed批量改、禁止未确认建新文件、禁止提交密钥/.env、改代码前先读、绝对路径、PEP8+类型注解、事后回顾。docker compose -p langgraph统一部署，阿里云镜像源优先。"
---
# 操作纪律与规范

**版本：** v2.4.0
**生效方式：** 始终生效
**优先级：** ⭐⭐⭐⭐⭐

---

## 文件操作

- 所有项目文件在本地项目目录直接编辑
- Docker 是本项目的唯一部署运行时（Docker Desktop）
- 源码是唯一真实来源，Docker 构建时直接使用本地代码
- 每次结束任务前回顾是否所有任务都完成了
- 改代码前先读相关文件

## 路径与查找

- 路径使用绝对路径
- 找文件先 `LS` 目录内容再决定操作
- 遍历文件夹时先 `LS` 当前文件夹内容
- 不明确进程/格式问题时，去 GitHub 或 `research/` 目录查找源代码

## 数据库操作

- **禁止**从数据库直接导入或操作数据
- 只允许通过应用程序接口进行查询操作
- 所有数据修改必须经过正常的业务逻辑途径

## 部署命令

| 操作 | 命令 |
|------|------|
| 构建并启动 API | `docker compose -p langgraph up -d --build api` |
| 查看日志 | `docker compose -p langgraph logs -f api` |
| 查看所有服务日志 | `docker compose -p langgraph logs -f` |
| 重启服务 | `docker compose -p langgraph restart api` |
| 停止全部服务 | `docker compose -p langgraph down` |

## 代码规范

- 遵循 PEP 8 规范
- 使用类型注解
- 关键逻辑需要文档字符串
- 多智能体并行场景下，每个并行 Agent 必须有独立 try/except 保护
- 错误处理需要精确捕获异常类型
- `mypy && ruff check` 通过后方可提交（mypy 配置 `check_untyped_defs = true`）
- type: ignore 注释必须精确匹配错误码（如 `call-overload`、`union-attr`）
- 不将密钥提交到版本控制

## 多智能体并行操作规范

### 并行 Agent 开发注意事项

| 注意点 | 说明 | 示例 |
|--------|------|------|
| **容错保护** | 每个并行 Agent 必须有独立 try/except | 统一评审走 `async_llm_call_with_retry`，失败降级 PASS |
| **Reducer 声明** | 多节点写入同一字段必须使用 Annotated Reducer | `_last_value`, `add_messages`, `operator.add` |
| **递归上限** | 子图必须设置 `recursion_limit` | 根图 5000 / 子图 200 |
| **无 Checkpointer** | 子图编译不传 checkpointer | `build_writing_crew()` 编译时不传 |
| **同步 Agent** | ThreadPoolExecutor 只支持同步 Runnable | media_crew 中 illustrator/tts 为同步 |
| **线程安全** | 并行 Agent 间通过 state 通信，无共享内存 | 使用 `crew_result` 透传数据 |
| **有限重试** | 并行 Agent 重试不超过 3 次 | `_media_tool_router` 最多 3 次重试 |
| **字段声明** | 子图/根图状态字段必须先声明再写入 | v8.5-fix S6/S7：critic/guidance/best_version 声明后 HITL 才生效 |

### 新增并行模式检查清单

新增多智能体并行功能时，按以下清单逐项确认：

- [ ] 每个并行 Agent 是否有独立异常保护？
- [ ] 多节点写入同一字段是否有 Reducer 防冲突？
- [ ] 子图编译是否传了 checkpointer？（❌不应传）
- [ ] ThreadPoolExecutor 中的 Agent 是否为同步 Runnable？
- [ ] 并行 Agent 是否有超时保护（`with_timeout`）？
- [ ] 递归上限是否设置合理？
- [ ] 并行结果是否正确合并到 state？
- [ ] SSE progress 事件是否能区分不同 Agent？

## 依赖与代理

- 依赖包优先使用阿里云镜像源
- Docker 构建时使用阿里云 apt 源（Dockerfile 已配置）和清华 PyPI 镜像

## 禁止事项

- **禁止**使用 `sed` 批量修改源码
- **禁止**未确认直接创建新文件，优先修改现有文件
- **禁止**将 `.env`、`credentials.json` 等敏感文件提交到版本控制

## 语言与项目规范

- 语言：简体中文
- 项目名及文件名使用纯英文
- 所有项目均为企业级、生产级别代码质量
