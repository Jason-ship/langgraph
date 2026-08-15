# NovelFactory API 接口总览（基于源代码整理）

> 来源：`src/novelfactory/server/app.py` + `src/novelfactory/server/routes/*` + `src/novelfactory/api/time_travel.py` + `src/novelfactory/server/stub_router.py`
> 基准：容器内 uvicorn 监听 `8000`，宿主机直连 `http://localhost:8123`，经 Nginx 代理 `http://localhost:8081`（`/api/` 前缀由 Nginx 剥离，直连 API 不需要前缀）
> 标准请求：`Content-Type: application/json`；运行类端点支持 `stream: true/false`

## 一、app 直接端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/ui/{assistant_id}` | UI 组件（stub） |
| GET | `/favicon.ico` | 站点图标 |
| GET | `/metrics` | Prometheus 指标（与 health router 重复，实际以先注册的 health 为准） |

## 二、健康与运维（routes/health.py）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查（状态+版本+tools_proxy） |
| GET | `/ready` | 就绪检查（数据库连接） |
| GET | `/info` | 系统信息（版本、配置摘要） |
| GET | `/debug/config` | 调试配置信息 |
| GET | `/params` | LLM 参数总览 |
| GET | `/metrics` | Prometheus 指标 |

## 三、线程与创作（routes/threads.py、runs.py、branches.py、compact.py、regenerate.py、token_usage.py）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/threads` | 创建线程（一部小说） |
| POST | `/threads/search` | 搜索线程 |
| GET | `/threads/{thread_id}` | 获取线程详情 |
| PATCH | `/threads/{thread_id}` | 更新线程 |
| DELETE | `/threads/{thread_id}` | 删除线程 |
| POST | `/threads/{thread_id}/copy` | 复制线程 |
| GET | `/threads/{thread_id}/state` | 获取当前状态 |
| POST | `/threads/{thread_id}/state` | 更新状态 |
| PATCH | `/threads/{thread_id}/state` | 局部更新状态 |
| POST | `/threads/{thread_id}/state/checkpoint` | 保存检查点 |
| GET | `/threads/{thread_id}/state/{checkpoint}` | 获取指定检查点状态（时间旅行） |
| POST | `/threads/{thread_id}/history` | 查看运行历史 |
| POST | `/threads/{thread_id}/branches` | 创建分支 |
| POST | `/threads/{thread_id}/compact` | 上下文压缩 |
| GET | `/threads/{thread_id}/token-usage` | Token 用量 |
| POST | `/threads/{thread_id}/runs/regenerate/prepare` | 重写准备 |
| POST | `/runs/stream` | 流式运行（无 thread 自动创建） |
| POST | `/runs/wait` | 同步运行（无 thread 自动创建） |
| POST | `/runs/batch` | 批量运行（stub） |
| GET | `/threads/{thread_id}/runs` | 列出线程运行记录 |
| POST | `/threads/{thread_id}/runs` | 创建运行（默认流式 SSE） |
| POST | `/threads/{thread_id}/runs/stream` | 流式运行（别名） |
| POST | `/threads/{thread_id}/runs/wait` | 同步运行（等待完成，后台任务用） |
| GET | `/threads/{thread_id}/runs/{run_id}` | 运行详情 |
| DELETE | `/threads/{thread_id}/runs/{run_id}` | 删除运行 |
| POST | `/threads/{thread_id}/runs/{run_id}/cancel` | 取消运行 |
| GET | `/threads/{thread_id}/runs/{run_id}/join` | 阻塞至运行完成 |
| GET | `/threads/{thread_id}/runs/{run_id}/stream` | 已完成的运行 SSE 回放 |

## 四、评价反馈（routes/feedback.py、quality_feedback.py）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/threads/{thread_id}/runs/{run_id}/feedback` | 提交评价 |
| GET | `/threads/{thread_id}/runs/{run_id}/feedback` | 获取评价列表 |
| GET | `/threads/{thread_id}/runs/{run_id}/feedback/stats` | 评价统计 |
| DELETE | `/threads/{thread_id}/runs/{run_id}/feedback/{feedback_id}` | 删除评价 |
| POST | `/feishu/quality-feedback` | 质量反馈（飞书侧） |
| POST | `/feishu/quality-params` | 质量参数调整（飞书侧） |

## 五、飞书集成（routes/feishu_callback.py）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/feishu/callback` | 飞书事件回调（含验证） |
| POST | `/feishu/resume/{thread_id}` | 飞书人工审批后恢复运行 |

## 六、记忆（routes/memory.py，prefix=/memory）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/memory` | 获取记忆 |
| PUT | `/memory` | 写入记忆 |
| DELETE | `/memory` | 清空记忆 |
| GET | `/memory/facts` | 事实列表 |
| POST | `/memory/facts` | 新增事实 |
| DELETE | `/memory/facts/{fact_id}` | 删除事实 |
| PATCH | `/memory/facts/{fact_id}` | 更新事实 |
| GET | `/memory/export` | 导出记忆 |
| POST | `/memory/import` | 导入记忆 |
| GET | `/memory/config` | 记忆配置 |
| GET | `/memory/status` | 记忆状态 |

## 七、渠道连接（routes/channel_connections.py，prefix=/channels）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/channels` | 渠道列表 |
| GET | `/channels/{provider}/connect` | 获取连接地址（OAuth） |
| POST | `/channels/{provider}/configure` | 配置渠道运行时 |
| POST | `/channels/{provider}/disconnect` | 断开渠道 |
| POST | `/channels/{provider}/restart` | 重启渠道 |

## 八、控制台（routes/console.py，prefix=/console）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/console/stats` | 统计概览 |
| GET | `/console/runs` | 运行列表 |
| GET | `/console/runs/recent` | 最近运行 |

## 九、Assistants（routes/assistants.py）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/assistants` | 助手列表 |
| POST | `/assistants/search` | 搜索助手 |
| GET | `/assistants/{assistant_id}` | 助手详情 |
| POST | `/assistants` | 创建助手 |
| PATCH | `/assistants/{assistant_id}` | 更新助手 |
| DELETE | `/assistants/{assistant_id}` | 删除助手 |
| GET | `/assistants/{assistant_id}/graph` | 图结构 |
| GET | `/assistants/{assistant_id}/schemas` | 输入输出 Schema |
| GET | `/assistants/{assistant_id}/subgraphs` | 子图列表 |
| GET | `/assistants/{assistant_id}/subgraphs/{namespace:path}` | 指定子图结构 |
| POST | `/assistants/{assistant_id}/versions` | 创建版本 |
| POST | `/assistants/{assistant_id}/latest` | 设为最新版本 |

## 十、Agent 管理（routes/agents.py）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/agents` | Agent 列表 |
| GET | `/agents/check` | Agent 健康检查 |
| GET | `/agents/{name}` | Agent 详情 |
| POST | `/agents` | 注册 Agent |
| PUT | `/agents/{name}` | 更新 Agent |
| DELETE | `/agents/{name}` | 删除 Agent |

## 十一、Skills（routes/skills.py）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/skills` | 技能列表 |
| GET | `/skills/{name}` | 技能详情 |
| PUT | `/skills/{name}` | 更新技能 |
| POST | `/skills/install` | 安装技能 |

## 十二、Cron 定时任务（routes/crons.py）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/runs/crons` | 创建全局定时运行 |
| POST | `/threads/{thread_id}/runs/crons` | 创建线程定时运行 |
| DELETE | `/runs/crons/{cron_id}` | 删除定时任务 |
| POST | `/runs/crons/search` | 搜索定时任务 |

## 十三、存储（routes/store.py）

| 方法 | 路径 | 说明 |
|------|------|------|
| PUT | `/store/items` | 写入存储项 |
| GET | `/store/items` | 读取存储项 |
| DELETE | `/store/items` | 删除存储项 |
| POST | `/store/items/search` | 语义搜索存储项 |
| POST | `/store/namespaces` | 列命名空间 |

## 十四、其他业务端点

| 方法 | 路径 | 来源 | 说明 |
|------|------|------|------|
| POST | `/input-polish` | input_polish.py | 输入润色 |
| POST | `/suggestions` | suggestions.py | 输入建议 |
| GET | `/suggestions/config` | suggestions.py | 建议配置 |
| GET | `/features` | features.py | 功能开关列表 |
| POST | `/api/rollback` | api/time_travel.py | 状态回滚（时间旅行） |
| GET | `/api/checkpoints` | api/time_travel.py | 检查点列表 |

## 十五、Stub 兼容端点（stub_router.py，最后挂载仅兜底，同名真实端点优先）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/v1/auth/me` | 认证信息（兼容） |
| GET | `/v1/auth/setup-status` | 初始化状态（兼容） |
| POST | `/v1/auth/logout` | 登出（兼容） |
| GET | `/models` | 模型列表 |
| GET | `/threads/{thread_id}/messages/page` | 分页消息 |
| GET | `/threads/{thread_id}/messages` | 消息列表 |
| GET | `/threads/{thread_id}/runs/{run_id}/workspace-changes` | 工作区变更 |
| GET | `/threads/{thread_id}/runs/{run_id}/events` | 运行事件 |
| GET | `/threads/{thread_id}/runs/{run_id}/messages` | 运行消息 |
| PUT | `/threads/{thread_id}/runs/{run_id}/feedback` | 提交评价（兼容） |
| DELETE | `/threads/{thread_id}/runs/{run_id}/feedback` | 删除评价（兼容） |
| POST | `/threads/{thread_id}/uploads` | 上传文件 |
| GET | `/threads/{thread_id}/uploads/limits` | 上传限额 |
| GET | `/threads/{thread_id}/uploads/list` | 上传文件列表 |
| DELETE | `/threads/{thread_id}/uploads/{filename:path}` | 删除上传文件 |
| GET | `/threads/{thread_id}/artifacts/{path:path}` | 产物文件 |
| GET | `/scheduled-tasks` | 定时任务列表 |
| POST | `/scheduled-tasks` | 创建定时任务 |
| GET | `/scheduled-tasks/{task_id}` | 任务详情 |
| PATCH | `/scheduled-tasks/{task_id}` | 更新任务 |
| DELETE | `/scheduled-tasks/{task_id}` | 删除任务 |
| POST | `/scheduled-tasks/{task_id}/pause` | 暂停任务 |
| POST | `/scheduled-tasks/{task_id}/resume` | 恢复任务 |
| POST | `/scheduled-tasks/{task_id}/trigger` | 立即触发 |
| GET | `/scheduled-tasks/{task_id}/runs` | 任务运行历史 |
| GET | `/threads/{thread_id}/scheduled-tasks` | 线程定时任务 |
| GET | `/channels/providers` | 渠道提供方 |
| GET | `/channels/connections` | 连接列表 |
| POST | `/channels/{provider}/connect` | 连接（兼容） |
| POST | `/channels/{provider}/runtime-config` | 运行时配置（兼容） |
| DELETE | `/channels/{provider}/runtime-config` | 删除运行时配置（兼容） |
| DELETE | `/channels/connections/{connection_id}` | 删除连接（兼容） |
| GET | `/mcp/config` | MCP 配置 |
| PUT | `/mcp/config` | 更新 MCP 配置 |
| GET | `/threads/{thread_id}/goal` | 目标（兼容） |
| PUT | `/threads/{thread_id}/goal` | 更新目标（兼容） |
| DELETE | `/threads/{thread_id}/goal` | 删除目标（兼容） |

## 统计

- 实际实现端点（一至十四）：约 100 个
- Stub 兜底端点（十五）：53 个（其中 `/agents*`、`/skills*`、`/threads/{id}/compact`、`/threads/{id}/branches`、`/threads/{id}/token-usage`、`/threads/{id}/runs/regenerate/prepare`、`/channels/{provider}/connect` 与真实端点路径重叠，真实实现优先）
- 交互式文档：`http://localhost:8123/docs`
