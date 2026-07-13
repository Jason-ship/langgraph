# NovelFactory — 架构原则

## 核心原则

1. **单一权威产出** — `VerdictResult` 是评分的唯一来源。消除旧版三源问题（reviewer/quality_panel/analysis 三套评分）。
2. **纯代码做判断，LLM 做裁判** — 程序化传感器（AI味/老书虫/跨章）只产出客观数据不做判断，LLM 评分 + 辩论负责语义判断。
3. **子图无检查点** — 子图编译不传 checkpointer，由根图统一管理持久化。
4. **向下兼容** — 废弃代码（如 volume_dispatch）保留文件但标记 DEPRECATED，确保旧检查点可恢复。
5. **防御性路由** — 未知 phase → 日志警告 + 路由到 END，不做无声终止。

## v6.3 关键变更

| 变更 | 旧 (v5.x) | 新 (v6.3) |
|------|-----------|-----------|
| 评分节点 | `chapter_reviewer` → `_score_router` (12分支) | `verdict_engine` → `verdict_router` (3路) |
| 评分引擎 | quality_panel 辩论子图 (多轮) | VerdictEngine (1轮知情辩论 + 1次四维评分) |
| 并行分发 | Send Map-Reduce | 移除，改为线性逐章创作 |
| Agent 防循环 | `loop_guard.py` (post_model_hook) | 移除，`len<500` 兜底 |
| 评分融合 | 无统一源，三源问题 | `VerdictResult` 唯一产出 |
| 新模块 | — | `evaluation/` |
| 路由 | route_from_supervisor 含 volume_parallel/review | 仅 setup/writing/media/sync/done |
