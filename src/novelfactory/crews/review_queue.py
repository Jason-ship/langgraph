"""Unified Review Queue - centralized review management.

All review items (kickoff, outline, chapter, milestone) are managed
through this single queue. Uses file locking for thread-safety.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock

# Cross-platform file I/O helpers
_IS_WINDOWS = os.name == "nt"

if _IS_WINDOWS:
    # Windows: no fcntl. File I/O is protected by the caller's _FILE_LOCK.
    # Note: threading.Lock is NOT reentrant — we must NOT acquire it nested.
    # The caller already holds _FILE_LOCK, so _read_file/_write_file must NOT
    # acquire any lock (they are private helpers only called from within
    # UnifiedReviewQueue methods that already hold _FILE_LOCK).
    # The "threading lock is sufficient" comment above was misleading — we
    # rely on the caller's _FILE_LOCK, not a nested lock.

    def _read_file(path: str) -> dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _write_file(path: str, data: dict) -> None:
        temp = path + ".tmp"
        with open(temp, "w", encoding="utf-8") as tf:
            json.dump(data, tf, ensure_ascii=False, indent=2, default=_json_default)
        os.replace(temp, path)
else:
    # Unix: fcntl for real file-level locking (safe across processes)
    import fcntl

    def _read_file(path: str) -> dict:
        with open(path, encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                return json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _write_file(path: str, data: dict) -> None:
        temp = path + ".tmp"
        with open(temp, "w", encoding="utf-8") as tf:
            fcntl.flock(tf.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(data, tf, ensure_ascii=False, indent=2, default=_json_default)
            finally:
                fcntl.flock(tf.fileno(), fcntl.LOCK_UN)
        os.replace(temp, path)


def _json_default(obj) -> str:
    """Handle non-JSON-serializable types: Enum → .value, datetime → isoformat."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class ReviewStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


@dataclass
class ReviewItem:
    """A single review queue entry."""

    thread_id: str
    review_type: str  # "kickoff" | "outline" | "chapter" | "milestone"
    project_name: str
    current_chapter: int
    content_summary: str
    feishu_doc_url: str | None = None
    status: ReviewStatus = ReviewStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    decided_at: datetime | None = None
    decision: str | None = None  # "approve" | "reject" | "modify"
    comment: str | None = None
    modifications: dict | None = None


class UnifiedReviewQueue:
    """Thread-safe, file-backed review queue.

    All operations are synchronous (safe for use in interrupt callbacks).

    File layout:
        ~/.novelfactory/unified_review_queue.json
        {
          "pending": [...],
          "completed": [...]
        }
    """

    _FILE_LOCK = Lock()

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path: str = db_path or str(
            Path.home() / ".novelfactory" / "unified_review_queue.json"
        )
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        if not os.path.exists(self._db_path):
            self._write_sync({"pending": [], "completed": []})

    # ── File I/O ──────────────────────────────────────────────────────────────

    def _read_sync(self) -> dict:
        """Read the queue file (platform-safe)."""
        return _read_file(self._db_path)

    def _write_sync(self, data: dict) -> None:
        """Write the queue file atomically (platform-safe)."""
        _write_file(self._db_path, data)

    # ── Queue Operations ──────────────────────────────────────────────────────

    def add(self, item: ReviewItem) -> ReviewItem:
        """Add a new review item to the pending queue."""
        with self._FILE_LOCK:
            queue = self._read_sync()
            queue["pending"].append(asdict(item))
            self._write_sync(queue)
        return item

    def decide(
        self,
        thread_id: str,
        decision: str,
        comment: str | None = None,
        modifications: dict | None = None,
    ) -> bool:
        """Record a review decision and move the item to completed."""
        status_map = {
            "approve": ReviewStatus.APPROVED,
            "reject": ReviewStatus.REJECTED,
            "modify": ReviewStatus.MODIFIED,
        }
        new_status = status_map.get(decision, ReviewStatus.PENDING)

        with self._FILE_LOCK:
            queue = self._read_sync()
            for i, raw_item in enumerate(queue.get("pending", [])):
                if raw_item["thread_id"] == thread_id:
                    raw_item["status"] = new_status.value
                    raw_item["decision"] = decision
                    raw_item["comment"] = comment
                    raw_item["modifications"] = modifications
                    raw_item["decided_at"] = datetime.now().isoformat()

                    # Move from pending → completed
                    queue["completed"].append(raw_item)
                    queue["pending"].pop(i)
                    self._write_sync(queue)
                    return True
            return False

    def get_decision_sync(self, thread_id: str) -> dict | None:
        """Poll for a decision on a specific thread (used by interrupt recovery).

        Checks both pending and completed queues, since decide() atomically
        moves items from pending → completed. Returns the decision dict if
        found and decision is set, else None.
        """
        with self._FILE_LOCK:
            queue = self._read_sync()
            # Check pending first, then completed (decide() moves items atomically)
            for bucket in ("pending", "completed"):
                for item in queue.get(bucket, []):
                    if item["thread_id"] == thread_id and item.get("decision"):
                        return {
                            "decision": item["decision"],
                            "comment": item.get("comment"),
                            "modifications": item.get("modifications"),
                        }
        return None

    @staticmethod
    def _from_dict(raw: dict) -> ReviewItem:
        """Deserialize a dict into a ReviewItem, converting string dates to datetime."""
        created_at_str = raw.get("created_at")
        if isinstance(created_at_str, str):
            raw["created_at"] = datetime.fromisoformat(created_at_str)
        decided_at_str = raw.get("decided_at")
        if isinstance(decided_at_str, str):
            raw["decided_at"] = datetime.fromisoformat(decided_at_str)
        # status field may be a string; convert to ReviewStatus enum
        status_val = raw.get("status")
        if isinstance(status_val, str):
            try:
                raw["status"] = ReviewStatus(status_val)
            except ValueError:
                raw["status"] = ReviewStatus.PENDING
        return ReviewItem(**raw)

    def get_pending(self, thread_id: str | None = None) -> list[ReviewItem]:
        """Return all pending review items, optionally filtered by thread_id."""
        with self._FILE_LOCK:
            queue = self._read_sync()
            items = [
                self._from_dict(raw)
                for raw in queue.get("pending", [])
                if thread_id is None or raw["thread_id"] == thread_id
            ]
        return items

    def get_completed(self, limit: int = 50) -> list[ReviewItem]:
        """Return the most recent completed review items."""
        with self._FILE_LOCK:
            queue = self._read_sync()
            completed = queue.get("completed", [])[-limit:]
            return [self._from_dict(raw) for raw in completed]

    def clear_completed(self, before_dt: datetime | None = None) -> int:
        """Remove completed items older than before_dt. Returns count removed."""
        if before_dt is None:
            before_dt = datetime.now()

        with self._FILE_LOCK:
            queue = self._read_sync()
            kept, removed = [], []
            for raw in queue.get("completed", []):
                decided_at_str = raw.get("decided_at")
                if not decided_at_str:
                    # orphaned item — remove it
                    removed.append(raw)
                    continue
                dt = datetime.fromisoformat(decided_at_str)
                (kept if dt >= before_dt else removed).append(raw)
            queue["completed"] = kept
            self._write_sync(queue)
            return len(removed)


# ── Module-level singleton ─────────────────────────────────────────────────────

review_queue = UnifiedReviewQueue()
