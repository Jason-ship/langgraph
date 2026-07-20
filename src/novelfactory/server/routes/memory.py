"""Memory API — 全局记忆管理端点。

Migrated from DeerFlow app/gateway/routers/memory.py.

提供用户上下文、历史上下文、事实（facts）的 CRUD 管理。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/memory", tags=["memory"])


class ContextSection(BaseModel):
    """上下文区块。"""

    summary: str = Field(default="", description="摘要内容")
    updatedAt: str = Field(default="", description="更新时间")


class UserContext(BaseModel):
    """用户上下文。"""

    workContext: ContextSection = Field(default_factory=ContextSection)
    personalContext: ContextSection = Field(default_factory=ContextSection)
    topOfMind: ContextSection = Field(default_factory=ContextSection)


class HistoryContext(BaseModel):
    """历史上下文。"""

    recentMonths: ContextSection = Field(default_factory=ContextSection)
    earlierContext: ContextSection = Field(default_factory=ContextSection)
    longTermBackground: ContextSection = Field(default_factory=ContextSection)


class Fact(BaseModel):
    """记忆事实。"""

    id: str = Field(..., description="唯一标识")
    content: str = Field(..., description="事实内容")
    category: str = Field(default="context", description="分类")
    confidence: float = Field(default=0.5, description="置信度(0-1)")
    createdAt: str = Field(default="", description="创建时间")
    source: str = Field(default="unknown", description="来源线程 ID")


class MemoryResponse(BaseModel):
    """记忆数据响应。"""

    version: str = Field(default="1.0")
    lastUpdated: str = Field(default="")
    user: UserContext = Field(default_factory=UserContext)
    history: HistoryContext = Field(default_factory=HistoryContext)
    facts: list[Fact] = Field(default_factory=list)


class MemoryUpdateRequest(BaseModel):
    """记忆更新请求。"""

    user: UserContext | None = None
    history: HistoryContext | None = None


class FactCreateRequest(BaseModel):
    """事实创建请求。"""

    content: str = Field(..., min_length=1, max_length=2000, description="事实内容")
    category: str = Field(default="context", description="分类")
    confidence: float = Field(default=0.5, ge=0, le=1, description="置信度")


# ── 内存存储（简化版，可使用 Redis 或文件持久化） ──

_memory_store: dict[str, MemoryResponse] = {}
_fact_store: list[Fact] = []
_fact_id_counter: int = 0


@router.get("", response_model=MemoryResponse)
async def get_memory():
    """获取当前记忆数据。"""
    return _memory_store.get("default", MemoryResponse(lastUpdated=datetime.now(UTC).isoformat()))


@router.put("", response_model=MemoryResponse)
async def update_memory(body: MemoryUpdateRequest):
    """更新记忆数据。"""
    existing = _memory_store.get("default", MemoryResponse())
    now = datetime.now(UTC).isoformat()

    if body.user:
        existing.user = body.user
    if body.history:
        existing.history = body.history

    existing.lastUpdated = now
    _memory_store["default"] = existing
    return existing


@router.get("/facts", response_model=list[Fact])
async def list_facts(category: str | None = None):
    """获取事实列表。

    Args:
        category: 可选分类过滤。
    """
    if category:
        return [f for f in _fact_store if f.category == category]
    return list(_fact_store)


@router.post("/facts", response_model=Fact)
async def create_fact(body: FactCreateRequest):
    """创建新事实。"""
    global _fact_id_counter
    _fact_id_counter += 1

    fact = Fact(
        id=f"fact_{_fact_id_counter}",
        content=body.content,
        category=body.category,
        confidence=body.confidence,
        createdAt=datetime.now(UTC).isoformat(),
    )
    _fact_store.append(fact)
    return fact


@router.delete("/facts/{fact_id}")
async def delete_fact(fact_id: str):
    """删除事实。"""
    global _fact_store
    _fact_store = [f for f in _fact_store if f.id != fact_id]
    return {"status": "deleted", "fact_id": fact_id}


@router.patch("/facts/{fact_id}", response_model=MemoryResponse)
async def update_fact(fact_id: str, body: FactCreateRequest | None = None):
    """部分更新事实（content/category/confidence 可选）。"""
    global _fact_store
    for i, f in enumerate(_fact_store):
        if f.id == fact_id:
            if body:
                _fact_store[i] = Fact(
                    id=f.id,
                    content=body.content if body.content else f.content,
                    category=body.category if body.category else f.category,
                    confidence=body.confidence if body.confidence else f.confidence,
                    createdAt=f.createdAt,
                    source=f.source,
                )
            break
    return _memory_store.get("default", MemoryResponse(lastUpdated=datetime.now(UTC).isoformat()))


@router.delete("", response_model=MemoryResponse)
async def clear_memory():
    """清除所有记忆数据。"""
    _memory_store.pop("default", None)
    _fact_store.clear()
    return MemoryResponse(lastUpdated=datetime.now(UTC).isoformat())


@router.get("/export", response_model=MemoryResponse)
async def export_memory():
    """导出记忆为 JSON。"""
    return _memory_store.get("default", MemoryResponse(lastUpdated=datetime.now(UTC).isoformat()))


@router.post("/import", response_model=MemoryResponse)
async def import_memory():
    """导入并覆盖记忆数据（stub）。"""
    return _memory_store.get("default", MemoryResponse(lastUpdated=datetime.now(UTC).isoformat()))


@router.get("/config")
async def memory_config():
    """记忆系统配置。"""
    return {
        "enabled": True,
        "mode": "tool",
        "injection_enabled": True,
        "shutdown_flush_timeout_seconds": 30.0,
        "manager_class": "novelfactory.store.memory.MemoryManager",
        "backend_config": {},
    }


@router.get("/status")
async def memory_status():
    """记忆系统状态。"""
    return {
        "enabled": True,
        "fact_count": len(_fact_store),
        "memory_configured": "default" in _memory_store,
        "config": {
            "enabled": True,
            "mode": "tool",
            "injection_enabled": True,
            "shutdown_flush_timeout_seconds": 30.0,
            "manager_class": "novelfactory.store.memory.MemoryManager",
            "backend_config": {},
        },
        "memory": _memory_store.get("default", MemoryResponse(lastUpdated=datetime.now(UTC).isoformat())),
    }