---
alwaysApply: false
description: "部署与运维规范，匹配Dockerfile、docker-compose.yml、deploy/**、Makefile。重建/rebuild、构建、启动/重启API、看日志、恢复线程、Docker构建失败、镜像拉不下、端口冲突、健康检查时触发。docker compose -p langgraph、端口（8123/8081/8443/5434/6380/19530）、Dockerfile（lark-cli/PM2）、断点重续、REST端点、DaoCloud镜像源、调试清单。"
---
# 部署与运维规范

**版本：** v3.1.0
**生效方式：** 智能生效
**优先级：** ⭐⭐⭐⭐
**匹配模式：** `Dockerfile`, `docker-compose.yml`, `deploy/**`, `Makefile`

---

## 一、本地部署

### 1.1 运行时环境

| 项目 | 值 |
|------|-----|
| 操作系统 | macOS / Linux / Windows（Docker Desktop） |
| 部署方式 | Docker Compose（项目根 `docker-compose.yml`） |
| 项目名称 | `langgraph`（`docker compose -p langgraph`） |
| 源码路径 | 项目根目录 |
| 容器工作路径 | `/app`（Docker build context） |
| lark-cli 版本 | v1.0.57（容器内 `/usr/local/bin/lark-cli`） |
| API 直连端口 | `localhost:8123` |
| Nginx HTTP 端口 | `localhost:8081` |
| Nginx HTTPS 端口 | `localhost:8443` |

### 1.2 启动命令

```bash
# 启动全部服务
docker compose -p langgraph up -d

# 选择性启动（先核心依赖）
docker compose -p langgraph up -d postgres redis

# 构建并启动 API（代码变更后）
docker compose -p langgraph up -d --build api

# 仅重启 API 容器（不重新构建）
docker compose -p langgraph restart api

# 查看 API 日志
docker compose -p langgraph logs -f api

# 查看所有容器日志
docker compose -p langgraph logs -f

# 停止全部服务
docker compose -p langgraph down

# 停止并删除数据卷（慎用，会丢失数据库数据）
docker compose -p langgraph down -v
```

### 1.3 容器间网络

所有容器通过 Docker bridge 网络 `langgraph_langgraph_net` 通信，容器内使用服务名互联：

```python
# 容器内 → 容器内连接方式
conn = psycopg.connect("postgresql://noveluser:pass@postgres:5432/novelfactory")
redis.Redis(host="redis", port=6379)
MilvusClient(uri="http://milvus:19530")
GraphDatabase.driver("bolt://neo4j:7687", auth=("neo4j", "pass"))
```

```python
# 容器内 → 宿主机（仅限 Minimax 等外部 API）
TOOLS_PROXY = "http://172.28.0.1:5004"  # Docker 默认网关
```

### 1.4 服务端口映射

| 服务 | 宿主机端口 | 容器内端口 | 说明 |
|------|:---------:|:----------:|------|
| FastAPI Server | 8123 | 8000 | LangGraph API（Nginx 反代） |
| Nginx HTTP | 8081 | 80 | 反向代理 HTTP |
| Nginx HTTPS | 8443 | 443 | 反向代理 HTTPS |
| PostgreSQL | 5434 | 5432 | 检查点/Store (pgvector:pg16) |
| Redis | 6380 | 6379 | 缓存/队列 |
| Milvus | 19530 | 19530 | 向量检索 |
| Neo4j Browser | 7474 | 7474 | 图数据库 Web 管理界面 |
| Neo4j Bolt | 7687 | 7687 | 图数据库连接协议 |

---

## 二、Docker 容器

### 2.1 容器列表

| 容器名 | 镜像 | 端口映射 | 说明 |
|--------|------|----------|------|
| langgraph_api | 自建 Dockerfile | 8123:8000 | FastAPI + LangGraph + PM2 |
| langgraph_nginx | nginx:latest | 8081:80, 8443:443 | 反向代理 |
| langgraph_postgres | pgvector/pgvector:pg16 | 5434:5432 | 检查点/Store（含 pgvector 插件） |
| langgraph_redis | redis:6-alpine | 6380:6379 | 缓存/队列 |
| langgraph_milvus | milvusdb/milvus:v2.4.17 | 19530:19530 | 向量检索 |
| langgraph_neo4j | neo4j:5 | 7474:7474, 7687:7687 | 图数据库 |

### 2.2 容器健康检查

```yaml
# docker-compose.yml 片段
services:
  api:
    healthcheck:
      test: ["CMD", "/usr/local/bin/healthcheck.sh"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

---

## 三、健康检查与验证

### 3.1 API 健康检查

```bash
# API 健康检查
curl -s http://localhost:8123/health

# 就绪检查
curl -s http://localhost:8123/ready

# 部署信息
curl -s http://localhost:8123/info

# Prometheus 指标
curl -s http://localhost:8123/metrics
```

### 3.2 通过 Makefile

```bash
make help    # 查看所有快捷命令
make health  # 健康检查
make logs    # 查看日志
make restart # 重启服务
```

---

## 四、断点重续 SOP

### 4.1 检查线程状态

```bash
curl -s http://localhost:8123/threads/{thread_id}/state
```

关键字段：
- `values.current_phase` — 当前阶段（setup/writing/volume_parallel/media/sync/done）
- `values.current_chapter` — 当前章节号
- `values.completed_chapters` — 已完成章节列表
- `next` — 接下来要执行的节点列表（空表示线程已完成或空闲）
- `interrupts` — 中断信息列表（非空表示线程在等待人工介入）

### 4.2 恢复方式

```bash
# 场景 1：线程有中断（interrupts 非空）— 携带恢复数据
curl -s -X POST http://localhost:8123/threads/{thread_id}/runs \
  -H 'Content-Type: application/json' \
  -d '{"input":{"resume":"continue"}}'

# 场景 2：线程无中断但 next 非空（正常继续）
curl -s -X POST http://localhost:8123/threads/{thread_id}/runs \
  -H 'Content-Type: application/json' -d '{}'

# 场景 3：新线程（next 为空）— 携带完整 seed
curl -s -X POST http://localhost:8123/threads/{thread_id}/runs \
  -H 'Content-Type: application/json' \
  -d '{
    "input": {
      "seed_idea": "小说创意描述...",
      "project_name": "项目名",
      "target_chapters": 1000,
      "genre": "都市/异能/系统/爽文"
    },
    "assistant_id": "novelfactory",
    "stream": true
  }'
```

### 4.3 可恢复状态判定

| state.next | current_phase | 含义 | 操作 |
|-----------|---------------|------|------|
| `["writing_crew"]` | writing | 写作中（单个 Agent 执行） | 空 run 继续 |
| `["sync_crew"]` | sync | 同步中 | 空 run 继续 |
| `["media_crew"]` | media | 媒体生成中（并行 Agent 执行） | 空 run 继续 |
| `["volume_dispatch"]` | volume_parallel | ~~并行分发中（v6.3 已移除）~~ | 空 run 继续 |
| `["wait_for_review"]` | writing | 等待人工审核 | 提交 resume 数据 |
| `[]` | done | 已完成 | 需创建新线程 + 新 seed |
| `[]` | — | 空线程（刚创建） | 需提交带 seed 的 run |

> 并行场景下的断点恢复：media_crew 在 ThreadPoolExecutor 并行执行中如果中断，恢复后从 `_parallel_media_node` 重新执行，插图+配音两个 Agent 同时重试。

---

## 五、REST API 端点参考

> 详细 API 实现见 `05-api-server.md`。

### 5.1 基础信息

| 项目 | 值 |
|------|-----|
| API 基础 URL（宿主机） | `http://localhost:8123` |
| API 基础 URL（容器内） | `http://localhost:8000` |
| 认证方式 | 无（内网部署，无需鉴权） |
| Content-Type | `application/json` |

### 5.2 健康检查

```bash
curl -s http://localhost:8123/health
curl -s http://localhost:8123/ready
```

### 5.3 线程管理

```bash
# 创建线程
curl -s -X POST http://localhost:8123/threads \
  -H 'Content-Type: application/json' \
  -d '{"metadata":{"project_name":"我的小说"}}'

# 获取线程状态
curl -s http://localhost:8123/threads/{thread_id}/state

# 删除线程
curl -s -X DELETE http://localhost:8123/threads/{thread_id}
```

### 5.4 运行写作

```bash
# 启动新小说（流式）
curl -s -X POST http://localhost:8123/threads/{thread_id}/runs \
  -H 'Content-Type: application/json' \
  -d '{"input":{"seed_idea":"...","project_name":"...","target_chapters":1000,"genre":"..."},"assistant_id":"novelfactory","stream":true}'

# 断点重续
curl -s -X POST http://localhost:8123/threads/{thread_id}/runs \
  -H 'Content-Type: application/json' -d '{}'
```

### 5.5 SSE 事件格式

| 事件类型 | 说明 |
|---------|------|
| `values` | 完整 State dict |
| `updates` | 增量更新 |
| `messages` | AI 消息 Token 流 |
| `progress` | StreamStateTracker 阶段性摘要 |
| `interrupt` | 人机交互中断 |
| `metadata` | 运行元数据 |
| `end` | 运行正常结束 |
| `error` | 运行出错 |

---

## 六、构建文件清单

### 6.1 必须存在的构建文件

| 文件 | 用途 |
|------|------|
| `Dockerfile` | 容器镜像定义 |
| `docker-compose.yml` | 服务编排 |
| `pyproject.toml` | 依赖声明 |
| `src/` | 源码目录 |
| `deploy/scripts/` | 部署脚本（含 healthcheck.sh） |
| `deploy/nginx/` | Nginx 配置 |
| `.env` | 环境变量（本地管理，非 Git 跟踪） |

---

## 七、Docker 镜像源策略

### 7.1 优先镜像源

当 Docker Hub 直接拉取失败时，按以下优先级尝试备用镜像源：

```bash
# DaoCloud 镜像源（推荐）
docker pull docker.m.daocloud.io/library/python:3.12-slim
docker tag docker.m.daocloud.io/library/python:3.12-slim python:3.12-slim
```

### 7.2 其他镜像的备用源

```bash
# PostgreSQL（pgvector）
docker pull docker.m.daocloud.io/pgvector/pgvector:pg16
docker tag docker.m.daocloud.io/pgvector/pgvector:pg16 pgvector/pgvector:pg16

# Redis
docker pull docker.m.daocloud.io/library/redis:6-alpine
docker tag docker.m.daocloud.io/library/redis:6-alpine redis:6-alpine

# Nginx
docker pull docker.m.daocloud.io/library/nginx:latest
docker tag docker.m.daocloud.io/library/nginx:latest nginx:latest

# Milvus
docker pull docker.m.daocloud.io/milvusdb/milvus:v2.4.17
docker tag docker.m.daocloud.io/milvusdb/milvus:v2.4.17 milvusdb/milvus:v2.4.17

# Neo4j
docker pull docker.m.daocloud.io/library/neo4j:5
docker tag docker.m.daocloud.io/library/neo4j:5 neo4j:5
```

---

## 八、常见问题排查

### 8.1 Docker 构建与部署

| 问题 | 原因 | 解决 |
|------|------|------|
| **Docker build 失败: COPY file not found** | 文件缺失 | 确认所有构建文件存在（见第六节） |
| **Docker Hub 连接超时** | 网络无法直连 registry-1.docker.io | 使用 DaoCloud 镜像源（见第七节） |
| **容器启动后健康检查失败** | 依赖服务未就绪或配置错误 | 检查 postgres/redis 先于 API 启动；确认 `.env` 文件存在 |
| **容器启动后立即退出** | 启动脚本错误或依赖缺失 | `docker compose -p langgraph logs api` 查看日志 |
| **端口冲突** | 宿主机端口已被占用 | `netstat -ano | findstr :8123` 检查占用 |
| **磁盘空间不足** | 容器日志或镜像堆积 | `docker system prune -af` 清理 |

### 8.2 调试检查清单

| 步骤 | 操作 | 预期结果 |
|:----:|------|----------|
| **1. 容器状态** | `docker compose -p langgraph ps` | 所有容器状态均为 `Up` |
| **2. 端口监听** | `netstat -ano | findstr :8123` | 端口被 Docker 监听 |
| **3. 健康检查** | `curl -s http://localhost:8123/health` | `{"status":"ok"}` |
| **4. 就绪检查** | `curl -s http://localhost:8123/ready` | `{"status":"ready","graph_compiled":true}` |
| **5. 查看日志** | `docker compose -p langgraph logs --tail 50 api` | 无 `ERROR` 级别日志 |

**快速诊断命令**：
```bash
echo "=== Docker PS ===" && docker compose -p langgraph ps && echo "=== Health ===" && curl -s http://localhost:8123/health
```

---

**规则版本：** v3.2.0
**生效方式：** 智能生效
**最后更新：** 2026-07-04
