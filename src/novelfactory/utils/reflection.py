"""Dynamic class/variable resolution from string paths.

Migrated from DeerFlow reflection/resolvers.py.

Provides a safe way to load classes, functions, or variables from
module path strings (e.g. "package.module:ClassName").
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, TypeVar

T = TypeVar("T")


def resolve_variable(
    variable_path: str,
    expected_type: type[T] | tuple[type, ...] | None = None,
) -> T:
    """Resolve a variable from a path.

    Args:
        variable_path: Path to the variable (e.g. "package.module:ClassName").
        expected_type: Optional type to validate against.

    Returns:
        The resolved variable.

    Raises:
        ImportError: If the module path is invalid.
        ValueError: If the resolved variable doesn't match expected_type.
    """
    try:
        module_path, variable_name = variable_path.rsplit(":", 1)
    except ValueError as err:
        raise ImportError(
            f"{variable_path} doesn't look like a variable path. "
            "Example: package.module:ClassName"
        ) from err

    try:
        module = import_module(module_path)
    except ImportError as err:
        raise ImportError(f"Could not import module {module_path}: {err}") from err

    try:
        variable = getattr(module, variable_name)
    except AttributeError as err:
        raise ImportError(f"Module {module_path} has no attribute {variable_name}") from err

    if expected_type is not None and not isinstance(variable, expected_type):
        raise ValueError(
            f"Variable {variable_path} is of type {type(variable).__name__}, "
            f"expected {expected_type}"
        )

    return variable


def resolve_class(class_path: str) -> type:
    """Resolve a class from a path string.

    Shortcut for resolve_variable that returns a type.

    Args:
        class_path: Path to the class (e.g. "package.module:ClassName").

    Returns:
        The resolved class.
    """
    result = resolve_variable(class_path)
    if not isinstance(result, type):
        raise ValueError(f"{class_path} is not a class (got {type(result).__name__})")
    return result


__all__ = [
    "resolve_variable",
    "resolve_class",
]