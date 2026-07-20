"""Local persistence for runtime IM channel configuration.

Migrated from DeerFlow app/channels/runtime_config_store.py.
"""

from __future__ import annotations

import json
import logging
import tempfile
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RUNTIME_CHANNEL_DISABLED_FLAG = "_runtime_disabled"


class ChannelRuntimeConfigStore:
    """JSON-backed store for channel credentials entered from the UI.

    This mirrors ChannelStore: local/private deployments get
    durable runtime configuration without needing config.yaml edits.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = Path(".novelfactory") / "channels" / "runtime-config.json"
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict[str, Any]] = self._load()
        self._lock = threading.Lock()

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt channel runtime config store at %s, starting fresh", self._path)
                return {}
            if isinstance(raw, dict):
                return {str(name): dict(value) for name, value in raw.items() if isinstance(value, dict)}
        return {}

    def _save(self) -> None:
        fd = tempfile.NamedTemporaryFile(
            mode="w",
            dir=self._path.parent,
            suffix=".tmp",
            delete=False,
        )
        try:
            json.dump(self._data, fd, indent=2, ensure_ascii=False)
            fd.close()
            Path(fd.name).replace(self._path)
        except BaseException:
            fd.close()
            Path(fd.name).unlink(missing_ok=True)
            raise

    def load_all(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {name: dict(config) for name, config in self._data.items()}

    def get_provider_config(self, provider: str) -> dict[str, Any] | None:
        with self._lock:
            config = self._data.get(provider)
            return dict(config) if isinstance(config, dict) else None

    def set_provider_config(self, provider: str, config: dict[str, Any]) -> None:
        with self._lock:
            self._data[provider] = dict(config)
            self._save()

    def set_provider_disconnected(self, provider: str) -> None:
        with self._lock:
            self._data[provider] = {
                "enabled": False,
                RUNTIME_CHANNEL_DISABLED_FLAG: True,
            }
            self._save()

    def remove_provider_config(self, provider: str) -> bool:
        with self._lock:
            if provider not in self._data:
                return False
            del self._data[provider]
            self._save()
            return True