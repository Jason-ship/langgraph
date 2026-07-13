"""NodeSpec dynamic registration for the NovelFactory graph.

Borrowed from TradingAgents pattern (analyst_execution.py):
  - PhaseCheckNodeSpec: frozen dataclass describing a phase check node
  - PHASE_CHECK_SPECS: central registry of all phase check nodes
  - build_check_chain(): constructs the dynamic routing chain from specs

This eliminates the hardcoded add_node() + add_edge() calls in new_builder.py.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from novelfactory.graph.nodes.phase_checks import (
    foreshadowing_check_node,
    quality_check_node,
    volume_check_node,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PhaseCheckNodeSpec:
    """Specification for a single phase check node.

    Following TradingAgents' AnalystNodeSpec pattern:
      - ``key``: unique node identifier in the graph
      - ``node_fn``: the callable node function
      - ``only_for_genres``: if set, ONLY these genres trigger this check
        (exclusive filter—genres NOT in this tuple skip the check)
      - ``skip_genres``: genres that explicitly skip this check
      - ``description``: human-readable description for logging
    """

    key: str
    node_fn: Callable[..., dict[str, Any]]
    only_for_genres: tuple[str, ...] = ()
    skip_genres: tuple[str, ...] = ()
    description: str = ""


# ── Phase Check Node Registry ─────────────────────────────────────────────────

PHASE_CHECK_SPECS: tuple[PhaseCheckNodeSpec, ...] = (
    PhaseCheckNodeSpec(
        key="volume_check",
        node_fn=volume_check_node,
        only_for_genres=(),
        skip_genres=("短篇",),
        description="卷结构检查：评估分卷合理性、章节分配",
    ),
    PhaseCheckNodeSpec(
        key="quality_check",
        node_fn=quality_check_node,
        only_for_genres=(),
        skip_genres=(),
        description="质量检查：累积质量统计、趋势分析",
    ),
    PhaseCheckNodeSpec(
        key="foreshadowing_check",
        node_fn=foreshadowing_check_node,
        only_for_genres=("悬疑灵异", "仙侠", "玄幻"),
        skip_genres=("短篇",),
        description="伏笔检查：回收状态、新设伏笔合理性",
    ),
)


def build_check_chain(
    specs: tuple[PhaseCheckNodeSpec, ...],
    genre: str,
) -> list[str]:
    """Build the ordered list of phase check node keys for a given genre.

    Filters out nodes whose ``skip_genres`` includes the target genre,
    and ensures ``only_for_genres`` nodes are restricted to matching genres.

    Results are cached per-genre since ``specs`` is the immutable
    ``PHASE_CHECK_SPECS`` constant and genre doesn't change mid-project.
    This avoids redundant filtering on every supervisor routing tick.

    Args:
        specs: The phase check spec registry (PHASE_CHECK_SPECS).
        genre: The target novel genre.

    Returns:
        Ordered list of node keys to execute.
    """
    cached = _check_chain_cache.get(genre)
    if cached is not None:
        return cached

    chain: list[str] = []
    for spec in specs:
        if genre in spec.skip_genres:
            logger.debug("Skipping %s for genre=%s", spec.key, genre)
            continue
        if spec.only_for_genres and genre not in spec.only_for_genres:
            continue
        chain.append(spec.key)
    logger.info("Phase check chain for genre=%s: %s", genre, chain)

    _check_chain_cache[genre] = chain
    return chain


_check_chain_cache: dict[str, list[str]] = {}
