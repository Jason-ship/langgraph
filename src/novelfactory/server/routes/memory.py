"""Memory API — 全局记忆管理端点。

Migrated from DeerFlow app/gateway/routers/memory.py.

提供用户上下文、历史上下文、事实（facts）的 CRUD 管理。
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter
from pydantic import BaseModel, Field

from novelfactory.store.memory_store import MemoryStore, get_memory_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/memory", tags=["memory"])


class ContextSection(BaseModel):
    """上下文区块。"""

    summary: str = Field(default="", description="摘要内容")
    updatedAt: str = Field(default="", description="更新时间")  # noqa: N815 — API 协议字段


class UserContext(BaseModel):
    """用户上下文。"""

    workContext: ContextSection = Field(default_factory=ContextSection)  # noqa: N815 — API 协议字段
    personalContext: ContextSection = Field(default_factory=ContextSection)  # noqa: N815 — API 协议字段
    topOfMind: ContextSection = Field(default_factory=ContextSection)  # noqa: N815 — API 协议字段


class HistoryContext(BaseModel):
    """历史上下文。"""

    recentMonths: ContextSection = Field(default_factory=ContextSection)  # noqa: N815 — API 协议字段
    earlierContext: ContextSection = Field(default_factory=ContextSection)  # noqa: N815 — API 协议字段
    longTermBackground: ContextSection = Field(default_factory=ContextSection)  # noqa: N815 — API 协议字段


class Fact(BaseModel):
    """记忆事实。"""

    id: str = Field(..., description="唯一标识")
    content: str = Field(..., description="事实内容")
    category: str = Field(default="context", description="分类")
    confidence: float = Field(default=0.5, description="置信度(0-1)")
    createdAt: str = Field(default="", description="创建时间")  # noqa: N815 — API 协议字段
    source: str = Field(default="unknown", description="来源线程 ID")


class MemoryResponse(BaseModel):
    """记忆数据响应。"""

    version: str = Field(default="1.0")
    lastUpdated: str = Field(default="")  # noqa: N815 — API 协议字段
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


# ── 存储（v8.4 落库修复：PostgreSQL 持久化，无 DB 时回退内存） ──

_memory_store: MemoryStore | None = None


def _get_store() -> MemoryStore:
    """获取共享 MemoryStore。"""
    global _memory_store
    if _memory_store is None:
        _memory_store = get_memory_store()
    return _memory_store


def _to_response(data: dict | None) -> MemoryResponse:
    if data is None:
        return MemoryResponse(lastUpdated=datetime.now(UTC).isoformat())
    # data 来自 store（JSON round-trip），补齐缺失字段
    try:
        return MemoryResponse.model_validate(data)
    except Exception:
        return MemoryResponse(lastUpdated=data.get("lastUpdated", datetime.now(UTC).isoformat()))


@router.get("", response_model=MemoryResponse)
async def get_memory():
    """获取当前记忆数据。"""
    return _to_response(_get_store().load_memory())


@router.put("", response_model=MemoryResponse)
async def update_memory(body: MemoryUpdateRequest):
    """更新记忆数据。"""
    store = _get_store()
    existing = _to_response(store.load_memory())
    now = datetime.now(UTC).isoformat()

    if body.user:
        existing.user = body.user
    if body.history:
        existing.history = body.history

    existing.lastUpdated = now
    store.save_memory("default", existing.model_dump())
    return existing


@router.get("/facts", response_model=list[Fact])
async def list_facts(category: str | None = None):
    """获取事实列表。

    Args:
        category: 可选分类过滤。
    """
    facts = _get_store().list_facts(category)
    return [Fact.model_validate(f) for f in facts]


@router.post("/facts", response_model=Fact)
async def create_fact(body: FactCreateRequest):
    """创建新事实。"""
    fact = Fact(
        id=f"fact_{uuid.uuid4().hex[:12]}",
        content=body.content,
        category=body.category,
        confidence=body.confidence,
        createdAt=datetime.now(UTC).isoformat(),
    )
    _get_store().create_fact(fact.model_dump())
    return fact


@router.delete("/facts/{fact_id}")
async def delete_fact(fact_id: str):
    """删除事实。"""
    _get_store().delete_fact(fact_id)
    return {"status": "deleted", "fact_id": fact_id}


@router.patch("/facts/{fact_id}", response_model=MemoryResponse)
async def update_fact(fact_id: str, body: FactCreateRequest | None = None):
    """部分更新事实（content/category/confidence 可选）。"""
    store = _get_store()
    if body:
        store.update_fact(fact_id, body.model_dump())
    else:
        # 无 body 时视为仅更新内容为空——保持原事实不变
        current = None
        for f in store.list_facts():
            if f["id"] == fact_id:
                current = f
                break
        if current:
            store.update_fact(fact_id, {})
    return _to_response(store.load_memory())


@router.delete("", response_model=MemoryResponse)
async def clear_memory():
    """清除所有记忆数据。"""
    _get_store().clear()
    return MemoryResponse(lastUpdated=datetime.now(UTC).isoformat())


@router.get("/export", response_model=MemoryResponse)
async def export_memory():
    """导出记忆为 JSON。"""
    return _to_response(_get_store().load_memory())


@router.post("/import", response_model=MemoryResponse)
async def import_memory():
    """导入并覆盖记忆数据（stub）。"""
    return _to_response(_get_store().load_memory())


@router.get("/config")
async def memory_config():
    """记忆系统配置。"""
    store = _get_store()
    return {
        "enabled": True,
        "mode": "tool",
        "injection_enabled": True,
        "shutdown_flush_timeout_seconds": 30.0,
        "manager_class": "novelfactory.store.memory_store.MemoryStore",
        "backend_config": {
            "persistent": store.persistent,
            "backend": "postgresql" if store.persistent else "in-memory",
        },
    }


@router.get("/status")
async def memory_status():
    """记忆系统状态。"""
    store = _get_store()
    facts = store.list_facts()
    memory = _to_response(store.load_memory())
    return {
        "enabled": True,
        "fact_count": len(facts),
        "memory_configured": store.load_memory() is not None,
        "persistent": store.persistent,
        "config": {
            "enabled": True,
            "mode": "tool",
            "injection_enabled": True,
            "shutdown_flush_timeout_seconds": 30.0,
            "manager_class": "novelfactory.store.memory_store.MemoryStore",
            "backend_config": {
                "persistent": store.persistent,
                "backend": "postgresql" if store.persistent else "in-memory",
            },
        },
        "memory": memory,
    }
