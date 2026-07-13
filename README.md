# NovelFactory LangGraph

多智能体 AI 小说创作系统，基于 LangGraph 编排，DeepSeek V4 Flash 驱动写作。

从世界观设定到章节生成、质量评审、媒体制作、飞书同步，全流程自动化。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1.0-green.svg)](https://github.com/langchain-ai/langgraph)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://docs.docker.com/compose/)

## 创作流程

输入一句话创意，系统自动完成从设定到成稿的全流程：

```
用户输入创意
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  1. 设定阶段 (Setup Crew)                                    │
│     世界观构建 → 角色设计 → 大纲规划 → 卷章拆分 → 飞书建文档  │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  2. 写作阶段 (Writing Crew) — 逐章循环                       │
│                                                             │
│  上下文组装 → 章节写作 → VerdictEngine 评审                  │
│                                │                            │
│                    ┌───────────┼───────────┐                │
│                    ▼           ▼           ▼                │
│                 <55分       55-74分      ≥75分              │
│                 重写         润色        通过                │
│                    │           │           │                │
│                    └─────┬─────┘           │                │
│                          ▼                 ▼                │
│                    重新评审          状态提取 → 入库          │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  3. 媒体阶段 (Media Crew) — 插图 + 配音 并行生成              │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  4. 同步阶段 (Sync Crew) — 自动推送到飞书文档                 │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
                      下一章 / 完成
```

### 评审引擎如何工作

每章写完后，VerdictEngine 从四个维度打分，决定是否通过：

```
章节草稿
    │
    ├── 1. 程序化分析（毫秒级，纯代码）
    │      AI味检测(8维) + 老书虫爽点/毒点 + 跨章一致性
    │
    ├── 2. 知情辩论（LLM，注入程序化结果）
    │      编辑视角 ←→ 读者视角，最多3轮，差异<10分提前收敛
    │
    ├── 3. 四维评分（LLM，1次）
    │      文学性(30) + 结构(25) + 角色(20) + 节奏(15) = 80基础分
    │
    └── 4. 融合计算
           final = 质量×0.40 + 程序化×0.30 + 跨章×0.20 - 辩论惩罚×0.10

           ≥75 → 通过    55-74 → 润色    <55 → 重写
```

不同题材使用不同阈值：玄幻 85 分才通过，爽文 75 分即可，系统流 AI 味容忍度最高。详见 [评分系统](docs/SCORING.md)。

## 功能特性

- **全自动创作流水线** - 一句话创意 → 世界观/角色/大纲/章节/媒体/同步，全流程无需人工干预
- **多智能体协作** - 根图 18+ 节点，5 个 Crew 子图，各司其职
- **融合评审引擎** - VerdictEngine 四维评分 + 程序化 AI 味检测 + 编辑/读者多轮辩论
- **质量分级路由** - 不达标自动重写/润色，题材自适应阈值，防死循环兜底
- **跨章一致性** - 角色声音/文风/节奏/情节连贯/伏笔追踪，5维跨章检测
- **状态持久化** - PostgreSQL 检查点，支持中断恢复与时间旅行
- **语义检索** - Milvus 向量库章节搜索，Neo4j 角色关系图谱
- **飞书集成** - 全自动飞书文档同步、消息通知、审批回调
- **SSE 流式** - 实时推送写作进度、Token 用量、质量评分
- **人机交互** - interrupt 中断等待人工审核，支持章节级指导注入

## 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| 智能体框架 | LangGraph | >=1.1.0, <2 |
| LLM | DeepSeek V4 Flash (ARK -> DeepSeek -> SiliconFlow 三级降级) | - |
| API 服务 | FastAPI + Uvicorn | >=0.110.0 |
| 检查点存储 | PostgreSQL (pgvector) | pg16 |
| 向量检索 | Milvus | v2.4.17 |
| 图数据库 | Neo4j | 5 |
| 缓存/队列 | Redis | 6-alpine |
| 容器编排 | Docker Compose | - |
| Python 运行时 | Python | 3.12-slim |

---

## 部署指南

### 一、系统要求

#### 1.1 硬件要求

| 资源 | 最低 | 推荐 |
|------|------|------|
| CPU | 4 核 | 8 核+ |
| 内存 | 16 GB | 32 GB+ |
| 磁盘 | 20 GB 可用 | 50 GB+ SSD |
| 网络 | 可访问 Docker Hub / 镜像源 | - |

> 各服务内存占用：PostgreSQL 8G、API 8G、Milvus 4G、Neo4j 4G、Redis 512M、Nginx 256M、Tools Proxy 256M，合计约 25G。请确保 Docker 分配了足够资源。

#### 1.2 软件要求

| 软件 | 版本 | 说明 |
|------|------|------|
| Docker Desktop | 最新版 | Windows/macOS/Linux 均可 |
| Docker Compose | v2+ | Docker Desktop 已内置 |
| Git | 2.x+ | 克隆仓库 |
| Python | 3.10+ | 仅开发模式需要 |

### 二、Docker 环境配置

#### 2.1 安装 Docker Desktop

**Windows:**
1. 下载 [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
2. 安装时勾选 "Use WSL 2 instead of Hyper-V"
3. 安装完成后重启电脑
4. 启动 Docker Desktop，等待状态变为 "Running"

**macOS:**
1. 下载 [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
2. 选择对应芯片版本（Intel / Apple Silicon）
3. 拖拽到 Applications 目录
4. 启动 Docker Desktop

**Linux:**
```bash
curl -fsSL https://get.docker.com | sh
sudo systemctl enable docker
sudo systemctl start docker
```

#### 2.2 配置 Docker 资源（重要）

打开 Docker Desktop -> Settings -> Resources：

| 配置项 | 推荐值 | 说明 |
|--------|--------|------|
| CPUs | 8 | 至少 4 核 |
| Memory | 32 GB | 至少 16 GB |
| Swap | 4 GB | - |
| Disk image size | 80 GB | 镜像+数据卷 |

#### 2.3 配置 Docker 镜像加速（中国大陆用户必做）

打开 Docker Desktop -> Settings -> Docker Engine，添加国内镜像源：

```json
{
  "registry-mirrors": [
    "https://docker.1ms.run",
    "https://docker.xuanyuan.me",
    "https://docker.m.daocloud.io"
  ]
}
```

点击 "Apply & Restart"，等待 Docker 重启完成。

验证镜像源生效：
```bash
docker info | grep -A5 "Registry Mirrors"
```

### 三、获取代码

```bash
git clone https://github.com/Jason-ship/langgraph.git
cd langgraph
```

### 四、配置环境变量

#### 4.1 创建配置文件

```bash
cp .env.example .env
```

#### 4.2 编辑 .env 文件

使用文本编辑器打开 `.env`，按以下说明填写：

**必填项 - LLM API 密钥（三选一，推荐配置多个实现降级）：**

```bash
# 方式一：火山引擎方舟（推荐，国内速度快）
ARK_API_KEY=你的方舟API密钥

# 方式二：DeepSeek 直连
DEEPSEEK_API_KEY=sk-你的DeepSeek密钥

# 方式三：硅基流动（同时用于 Embedding）
SILICONFLOW_API_KEY=sk-你的硅基流动密钥
```

> API 密钥获取地址：
> - 火山引擎方舟：https://console.volcengine.com/ark
> - DeepSeek：https://platform.deepseek.com/api_keys
> - 硅基流动：https://cloud.siliconflow.cn/account/ak

**数据库密码（已有默认值，开箱即用，生产环境请修改）：**

```bash
POSTGRES_PASSWORD=novelpass2024        # PostgreSQL 默认密码
DB_PASSWORD=novelpass2024              # 与 POSTGRES_PASSWORD 相同
REDIS_PASSWORD=novelredis2024          # Redis 默认密码
NEO4J_PASSWORD=novelgraph2024          # Neo4j 默认密码
```

> 以上密码为 Docker 内网默认值，`.env.example` 已预填。如仅在本地使用，无需修改。
> 生产环境部署时，请在 `.env` 中修改为强密码。

**可选项 - 飞书集成（不填则跳过飞书同步）：**

```bash
LARK_APP_ID=cli_你的应用ID
LARK_APP_SECRET=你的应用密钥
FEISHU_USER_OPEN_ID=ou_你的用户ID
```

**可选项 - Embedding 配置（已有默认值）：**

```bash
EMBEDDING_API_KEY=sk-你的硅基流动密钥    # 与 SILICONFLOW_API_KEY 相同
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_DIMS=1024
```

#### 4.3 验证配置

确认 `.env` 文件中 LLM 密钥已填写（`xxxx` 占位符已替换）：

```bash
grep 'xxxx' .env
# 应该只有可选项为空时才有输出
```

### 五、构建与启动

#### 5.1 配置 Docker Compose 镜像源（已内置）

Dockerfile 已内置中国镜像源配置，无需额外操作：

| 组件 | 镜像源 | 说明 |
|------|--------|------|
| apt (Debian) | `mirrors.aliyun.com/debian` | Dockerfile 第 20 行 |
| PyPI (pip/uv) | `pypi.tuna.tsinghua.edu.cn` | Dockerfile 第 70 行 |
| npm (lark-cli) | `registry.npmmirror.com` | Dockerfile 第 50 行 |

#### 5.2 启动基础设施服务（首次部署推荐分步启动）

首次部署建议先启动数据库和缓存，确认正常后再启动 API：

```bash
# 步骤 1: 启动 PostgreSQL + Redis（约 30 秒）
docker compose -p langgraph up -d postgres redis

# 等待健康检查通过
docker compose -p langgraph ps
# 确认 postgres 和 redis 状态为 "healthy"

# 步骤 2: 启动 Milvus + Neo4j（约 60 秒）
docker compose -p langgraph up -d milvus neo4j

# 等待启动完成
docker compose -p langgraph ps
# 确认 milvus 和 neo4j 状态为 "running"
```

#### 5.3 构建 API 镜像（首次构建约 5-15 分钟）

```bash
# 构建 API 镜像并启动（包含 Python 依赖安装、lark-cli 安装）
docker compose -p langgraph up -d --build api
```

> **构建时间说明：**
> - 首次构建：5-15 分钟（取决于网络速度）
> - 后续构建：1-3 分钟（Docker 缓存生效）
> - 如果构建失败，请检查网络和镜像源配置

#### 5.4 启动 Nginx 反向代理

```bash
docker compose -p langgraph up -d nginx
```

#### 5.5 启动飞书工具代理（可选）

如果配置了飞书集成：

```bash
docker compose -p langgraph up -d tools_proxy
```

#### 5.6 一键启动全部服务（非首次部署）

```bash
# 启动全部服务
docker compose -p langgraph up -d --build api

# 或不重新构建（代码未变动时）
docker compose -p langgraph up -d
```

### 六、验证部署

#### 6.1 检查服务状态

```bash
docker compose -p langgraph ps
```

预期输出（7 个服务全部运行）：

```
NAME                      STATUS                   PORTS
langgraph_api             Up (healthy)             0.0.0.0:8123->8000/tcp
langgraph_postgres        Up (healthy)             0.0.0.0:5434->5432/tcp
langgraph_redis           Up (healthy)             0.0.0.0:6380->6379/tcp
langgraph_milvus          Up                       0.0.0.0:19530->19530/tcp
langgraph_neo4j           Up                       0.0.0.0:7474->7474/tcp, 0.0.0.0:7687->7687/tcp
langgraph_nginx           Up                       0.0.0.0:8081->80/tcp, 0.0.0.0:8443->443/tcp
langgraph_tools_proxy     Up                       0.0.0.0:5004->5004/tcp
```

#### 6.2 API 健康检查

```bash
# 通过 API 端口直接访问
curl http://localhost:8123/health

# 或通过 Nginx 代理访问
curl http://localhost:8081/health
```

预期返回：
```json
{"status": "healthy", "service": "novelfactory"}
```

#### 6.3 查看日志

```bash
# 查看 API 日志
docker compose -p langgraph logs -f api

# 查看所有服务日志
docker compose -p langgraph logs -f

# 查看最近 100 行 API 日志
docker compose -p langgraph logs --tail 100 api
```

### 七、服务端口说明

| 服务 | 宿主机端口 | 容器端口 | 说明 |
|------|-----------|---------|------|
| API (FastAPI) | 8123 | 8000 | 核心 API 服务，SSE 流式接口 |
| Nginx | 8081 / 8443 | 80 / 443 | HTTP/HTTPS 反向代理 |
| PostgreSQL | 5434 | 5432 | 数据库（端口避开本机 5432） |
| Redis | 6380 | 6379 | 缓存（端口避开本机 6379） |
| Milvus | 19530 / 9091 | 19530 / 9091 | 向量数据库 gRPC / HTTP |
| Neo4j | 7687 / 7474 | 7687 / 7474 | 图数据库 Bolt / HTTP |
| Tools Proxy | 5004 | 5004 | lark-cli HTTP 代理 |

> 如果端口冲突，修改 `.env` 中对应的 `*_PORT` 变量。

### 八、常见问题排查

#### 8.1 Docker 构建失败

**问题：拉取基础镜像超时**

```bash
# 检查 Docker 镜像源配置
docker info | grep -A5 "Registry Mirrors"

# 手动拉取测试
docker pull python:3.12-slim
```

**问题：pip install 超时**

Dockerfile 已配置清华 PyPI 镜像源。如果仍然超时，检查容器内网络：

```bash
# 进入容器测试网络
docker run --rm python:3.12-slim pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple requests
```

**问题：npm install lark-cli 失败**

Dockerfile 配置了 npmmirror.com 镜像源。如果失败，可手动指定：

```bash
# 构建时传入 npm 镜像源
docker compose -p langgraph build --build-arg NPM_CONFIG_REGISTRY=https://registry.npmmirror.com api
```

#### 8.2 服务启动失败

**问题：PostgreSQL 启动失败**

```bash
# 检查密码是否设置
grep POSTGRES_PASSWORD .env

# 检查端口冲突
lsof -i :5434    # macOS/Linux
netstat -ano | findstr 5434    # Windows
```

**问题：API 容器不断重启**

```bash
# 查看 API 启动日志
docker compose -p langgraph logs api | tail -50

# 常见原因：
# 1. .env 中 API 密钥未填写 -> 填写后重启
# 2. PostgreSQL 未就绪 -> 确保 postgres 状态为 healthy
# 3. 端口冲突 -> 修改 API_PORT
```

**问题：Milvus 启动慢或失败**

Milvus 首次启动需要 60-120 秒初始化 etcd 和存储。如果超过 3 分钟未启动：

```bash
# 检查 Milvus 日志
docker compose -p langgraph logs milvus

# 检查内存是否足够（Milvus 需要 4G+）
docker stats langgraph_milvus
```

#### 8.3 健康检查失败

```bash
# 手动运行健康检查脚本
docker compose -p langgraph exec api /usr/local/bin/healthcheck.sh

# 分别检查各服务
docker compose -p langgraph exec postgres pg_isready -U noveluser -d novelfactory
docker compose -p langgraph exec redis redis-cli -a 你的Redis密码 ping
```

### 九、日常运维命令

```bash
# 启动全部服务
docker compose -p langgraph up -d

# 停止全部服务（保留数据）
docker compose -p langgraph down

# 重启 API 服务
docker compose -p langgraph restart api

# 重新构建 API（代码更新后）
docker compose -p langgraph up -d --build api

# 查看 API 日志（实时）
docker compose -p langgraph logs -f api

# 查看所有服务状态
docker compose -p langgraph ps

# 健康检查
make health

# 连接 PostgreSQL
docker compose -p langgraph exec postgres psql -U noveluser -d novelfactory

# 完全清除（删除所有数据卷，谨慎操作！）
docker compose -p langgraph down -v
```

---

## 开发环境

### 安装开发依赖

```bash
pip install -e ".[dev]"

# 安装 pre-commit 钩子
pip install pre-commit
pre-commit install
```

### 启动开发服务器

开发模式下使用原生 uvicorn 热重载，仍需 Docker 提供数据库服务：

```bash
# 启动数据库服务
docker compose -p langgraph up -d postgres redis milvus neo4j

# 启动开发服务器（热重载，端口 8000）
make dev
```

### 代码质量

```bash
make format      # ruff 格式化
make lint        # ruff 代码检查
make typecheck   # mypy 类型检查
make test        # pytest 测试
make audit       # 完整审计（lint + typecheck + security + deps）
```

---

## 项目结构

```
langgraph/
├── src/novelfactory/          # 核心源码
│   ├── graph/                 # 图构建（根图、子图、路由、检查点）
│   ├── agents/                # Agent 定义（写作、评审、设定、媒体、同步）
│   ├── evaluation/            # 评审引擎（VerdictEngine、辩论、程序化评分）
│   ├── state/                 # 状态定义（NovelFactoryState、CrewState）
│   ├── server/                # FastAPI 服务（路由、SSE 流式、序列化）
│   ├── config/                # 配置层（LLM、数据库、常量、定价）
│   ├── store/                 # 持久层（PostgreSQL、Milvus、Neo4j、Redis）
│   ├── integrations/          # 外部集成（飞书 21 域工具集）
│   ├── pipeline/              # 创作管线（叙事编解码器、缩放管理器）
│   ├── middleware/            # 中间件链（摘要、大文件存储、Skill 注入）
│   ├── tools/                 # LangChain @tool（Neo4j/Milvus/飞书）
│   ├── schemas/               # Pydantic Schema
│   ├── skills/                # Skill 加载器
│   ├── cli/                   # CLI 命令行工具
│   └── api/                   # 时间旅行 API
├── deploy/                    # 部署配置（Nginx、初始化脚本）
├── server/tools-proxy/        # lark-cli HTTP 代理服务
├── tests/                     # 测试（单元测试、集成测试）
├── docs/                      # 项目文档
│   ├── ARCHITECTURE.md        # 系统架构详情
│   ├── SCORING.md             # 评分系统说明
│   ├── PRINCIPLES.md          # 设计原则
│   └── agent-patterns/        # Agent 模式参考文档
├── docker-compose.yml         # 服务编排
├── Dockerfile                 # 容器镜像（含镜像源配置）
├── pyproject.toml             # Python 项目配置
├── Makefile                   # 构建任务
└── langgraph.json             # LangGraph 配置
```

## 配置说明

所有配置通过环境变量注入，完整列表参见 `.env.example`。

### LLM 三级降级

系统按以下顺序尝试 LLM 调用，任一可用即跳过后续：

```
ARK (火山引擎方舟) -> DeepSeek (直连) -> SiliconFlow (硅基流动)
```

| 环境变量 | 说明 | 获取地址 |
|----------|------|---------|
| `ARK_API_KEY` | 火山引擎方舟 API Key | https://console.volcengine.com/ark |
| `DEEPSEEK_API_KEY` | DeepSeek API Key | https://platform.deepseek.com/api_keys |
| `SILICONFLOW_API_KEY` | 硅基流动 API Key（Embedding + 降级 LLM） | https://cloud.siliconflow.cn/account/ak |

### 评分阈值

不同题材使用不同的质量阈值，详见 [docs/SCORING.md](docs/SCORING.md)。

## 文档

- [架构详情](docs/ARCHITECTURE.md) - 节点拓扑、路由表、子图结构、状态字段
- [评分系统](docs/SCORING.md) - VerdictEngine 融合评分、题材阈值、三路决策
- [设计原则](docs/PRINCIPLES.md) - 核心架构原则与版本变更记录
- [Agent 模式](docs/agent-patterns/) - 各 Agent 的 LangGraph 模式参考
- [贡献指南](CONTRIBUTING.md) - 开发流程、提交规范、质量门槛

## License

[MIT](LICENSE)
