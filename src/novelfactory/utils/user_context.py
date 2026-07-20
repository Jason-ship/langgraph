"""Request-scoped user context for task-local authorization.

Migrated from DeerFlow runtime/user_context.py.

Uses ContextVar (task-local under asyncio) to manage the current
authenticated user across async boundaries.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Protocol, runtime_checkable


@runtime_checkable
class CurrentUser(Protocol):
    """Structural type for the current authenticated user.

    Any object with an ``.id: str`` attribute satisfies this protocol.
    """

    id: str


_current_user: ContextVar[CurrentUser | None] = ContextVar("novelfactory_current_user", default=None)


def set_current_user(user: CurrentUser) -> Token[CurrentUser | None]:
    """Set the current user for this async task.

    Returns a reset token for use in a ``finally`` block.
    """
    return _current_user.set(user)


def reset_current_user(token: Token[CurrentUser | None]) -> None:
    """Restore the context to the state captured by token."""
    _current_user.reset(token)


def get_current_user() -> CurrentUser | None:
    """Return the current user, or None if unset."""
    return _current_user.get()


def require_current_user() -> CurrentUser:
    """Return the current user, or raise RuntimeError."""
    user = _current_user.get()
    if user is None:
        raise RuntimeError("Accessed without user context")
    return user


DEFAULT_USER_ID: str = "default"


def get_effective_user_id() -> str:
    """Return the current user's id, or DEFAULT_USER_ID if unset."""
    user = _current_user.get()
    if user is None:
        return DEFAULT_USER_ID
    return str(user.id)


__all__ = [
    "CurrentUser",
    "set_current_user",
    "reset_current_user",
    "get_current_user",
    "require_current_user",
    "get_effective_user_id",
    "DEFAULT_USER_ID",
]