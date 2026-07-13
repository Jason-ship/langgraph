"""writing_nodes subpackage — extracted node functions from writing_crew.py.

Active node files:
    writer.py       — _chapter_writer_node
    reviewer.py     — _chapter_refiner_node (v6.3: only refiner remains)
    routing.py      — _exit_for_chapter
    subgraph_integration.py — context_builder_node_fn, state_extractor_node_fn, database_writer_node_fn
    helpers.py      — _sanitize_human_guidance, _make_record

v6.3: ``_score_router`` and ``_chapter_reviewer_node`` have been replaced by
      ``verdict_router`` (evaluation.verdict.router) and ``verdict_engine_node``
      (evaluation.coordinator).  Only the active nodes remain in this package.
"""

from __future__ import annotations

from novelfactory.graph.crews.writing_nodes.helpers import (
    _make_record,
    _sanitize_human_guidance,
)
from novelfactory.graph.crews.writing_nodes.reviewer import (
    _chapter_refiner_node,
)
from novelfactory.graph.crews.writing_nodes.routing import (
    _exit_for_chapter,
)
from novelfactory.graph.crews.writing_nodes.subgraph_integration import (
    context_builder_node_fn,
    database_writer_node_fn,
    state_extractor_node_fn,
)
from novelfactory.graph.crews.writing_nodes.writer import (
    _chapter_writer_node,
)

__all__ = [
    "_chapter_writer_node",
    "_chapter_refiner_node",
    "_exit_for_chapter",
    "context_builder_node_fn",
    "state_extractor_node_fn",
    "database_writer_node_fn",
    "_sanitize_human_guidance",
    "_make_record",
]
