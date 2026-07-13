# ── Custom JSON encoder for langchain BaseMessage ──────────────────────────
#
# 注意：不再做全局 json.dumps monkey-patch（影响整个进程），改为通过
# FastAPI CustomJSONResponse（app.py）在 HTTP 响应层面注入 _MessageJSONEncoder。
# 其他地方如需序列化 AIMessage，显式使用 _message_json_dumps() 或 json.dumps(..., cls=_MessageJSONEncoder)。

from __future__ import annotations

import json
from typing import Any


class _MessageJSONEncoder(json.JSONEncoder):
    """Handle AIMessage/BaseMessage serialization for JSON responses."""

    def default(self, obj: Any) -> Any:
        if hasattr(obj, "model_dump"):
            try:
                return obj.model_dump()
            except Exception:
                pass
        if hasattr(obj, "dict"):
            try:
                return obj.dict()
            except Exception:
                pass
        if hasattr(obj, "content"):
            return {"content": obj.content, "type": getattr(obj, "type", "unknown")}
        return super().default(obj)


def _message_json_dumps(*args: Any, **kwargs: Any) -> str:
    """json.dumps wrapper with _MessageJSONEncoder as default cls."""
    kwargs.setdefault("cls", _MessageJSONEncoder)
    return json.dumps(*args, **kwargs)
