# 贡献指南

感谢参与 NovelFactory LangGraph！本指南覆盖开发流程、提交规范与质量门槛。

## 环境准备

需要 Python 3.10+、Docker Desktop、`uv`（推荐）或 `pip`。

```bash
git clone https://github.com/Jason-ship/langgraph.git
cd langgraph

pip install -e ".[dev]"          # 安装项目 + 开发依赖
cp .env.example .env             # 按需填入密钥与端口
make up                          # 启动 PostgreSQL/Redis/Milvus/Neo4j
make dev                         # 启动原生 uvicorn，端口 8000，热重载
```

安装 pre-commit 钩子：

```bash
pip install pre-commit
pre-commit install
```

## 开发循环

1. 改动默认限定在 `src/novelfactory/` 目录。
2. 改完依次运行：

```bash
make format      # ruff format
make lint        # ruff check
make typecheck   # mypy strict
make test        # pytest
```

一次性跑全套：`make audit`（lint + typecheck + security + deps）。**提交前必须全绿。**

跑单个测试：`python -m pytest tests/unit/test_routing.py -v`

## 代码风格

- Python 3.10+，行宽 88，4 空格缩进（详见 `.editorconfig`）
- ruff 规则：E, F, W, I, N, UP（E501 忽略，formatter 强制行宽）
- mypy strict（第三方模块按 `pyproject.toml` 的 per-module 覆盖）
- 命名规范：
  - 模块/函数/变量：`snake_case`
  - 类：`PascalCase`
  - 常量：`UPPER_SNAKE_CASE`
  - 私有成员：前缀 `_`
  - 节点函数按角色前缀（如 `write_chapter_node`）
  - Crew 类以 `*Crew` 结尾，State 类以 `*State` 结尾
- 依赖必须有上下界（`>=, <`）
- 关键逻辑需要文档字符串
- 错误处理精确捕获异常类型，不用 bare `except:`

## Git 提交规范

```
<type>: <简短描述>
```

类型：`feat` / `fix` / `refactor` / `docs` / `test` / `style` / `chore` / `perf`

示例：
```
feat: 飞书通知增加字数统计
fix: 检查点断点重续修复
refactor: 评审路由简化为三路决策
```

PR 要求：标题同上格式，写清楚改了什么和原因，`make audit` 通过后提交。

## 多智能体并行开发规范

新增多智能体并行功能时，按以下清单逐项确认：

- [ ] 每个并行 Agent 是否有独立异常保护（try/except）
- [ ] 多节点写入同一字段是否有 Reducer 防冲突（`_last_value` / `add_messages`）
- [ ] 子图编译是否传了 checkpointer（不应传）
- [ ] ThreadPoolExecutor 中的 Agent 是否为同步 Runnable
- [ ] 并行 Agent 重试不超过 3 次
- [ ] 递归上限是否设置合理（根图 5000 / 子图 200）

## 操作纪律

### 必须

- 改代码前先读完整文件
- 路径使用绝对路径
- 找文件先 `ls` 目录内容再操作
- `make audit` 通过后方可提交
- 改完回头看是否所有任务都完成了

### 禁止

- 禁止 `sed` 批量修改源码
- 禁止将 `.env`、`credentials.json` 等密钥文件提交到版本控制
- 禁止从数据库直接导入或操作数据（只能走 API 或迁移脚本）
- 禁止 bare `except:`，必须精确捕获异常类型

## Docker 运维

```bash
docker compose -p langgraph up -d --build api   # 构建并启动 API
docker compose -p langgraph logs -f api          # 查看 API 日志
docker compose -p langgraph restart api          # 重启 API
docker compose -p langgraph down                 # 停止全部服务
```

## 项目文档

- [架构详情](docs/ARCHITECTURE.md) - 节点拓扑、路由表、状态字段
- [评分系统](docs/SCORING.md) - VerdictEngine 融合评分、题材阈值
- [设计原则](docs/PRINCIPLES.md) - 核心架构原则
- [Agent 模式](docs/agent-patterns/) - 各 Agent 的 LangGraph 模式参考
