# ==============================================================================
# SDK: /store  — 使用 AsyncPostgresStore 持久化存储
# ==============================================================================
# v5.1.1: 原 _store_data in-memory dict 已迁移到 AsyncPostgresStore,
#         通过 deps.get_store() 获取 graph 绑定的 store 实例，重启后数据不丢失。

from __future__ import annotations

from fastapi import APIRouter, Query

from novelfactory.server.deps import get_store
from novelfactory.server.models import StoreItem, StoreSearchRequest

router = APIRouter()


def _ns_to_tuple(ns: list[str]) -> tuple[str, ...]:
    """Convert namespace from list to tuple (PostgresStore API requirement)."""
    return tuple(ns)


@router.put("/store/items", tags=["store"])
async def store_put_item(item: StoreItem) -> dict:
    """Store or update an item (persisted to Postgres)."""
    store = await get_store()
    await store.aput(_ns_to_tuple(item.namespace), item.key, item.value)
    return {"status": "ok"}


@router.get("/store/items", tags=["store"])
async def store_get_item(namespace: list[str] = Query(), key: str = Query()) -> dict:
    """Get a single item from persistent store."""
    store = await get_store()
    item = await store.aget(_ns_to_tuple(namespace), key)
    value = item.value if item else {}
    return {"namespace": namespace, "key": key, "value": value}


@router.delete("/store/items", tags=["store"])
async def store_delete_item(namespace: list[str] = Query(), key: str = Query()) -> dict:
    """Delete an item from persistent store."""
    store = await get_store()
    await store.adelete(_ns_to_tuple(namespace), key)
    return {"deleted": f"{'/'.join(namespace)}:{key}"}


@router.post("/store/items/search", tags=["store"])
async def store_search_items(req: StoreSearchRequest) -> list:
    """Search store items with namespace prefix (persisted)."""
    store = await get_store()
    items = await store.asearch(
        _ns_to_tuple(req.namespace_prefix), limit=req.limit, offset=req.offset
    )
    return [
        {"namespace": item.namespace, "key": item.key, "value": item.value}
        for item in items
    ]


@router.post("/store/namespaces", tags=["store"])
async def store_list_namespaces() -> list:
    """List all namespaces from persistent store (v5.1.1: 使用官方 alist_namespaces API)."""
    store = await get_store()
    # 优先使用官方 alist_namespaces API
    try:
        ns_list = await store.alist_namespaces()
    except AttributeError:
        # 回退到 asearch 方式 (兼容旧版 store 实现)
        all_items = await store.asearch((), limit=10000)
        namespaces = set()
        for item in all_items:
            namespaces.add(tuple(item.namespace))
        ns_list = [list(ns) for ns in sorted(namespaces)]
    return [list(ns) if isinstance(ns, tuple) else ns for ns in ns_list]
