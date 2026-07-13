"""State package — schema definitions and custom reducers."""

from novelfactory.state.chapter_state import ChapterStateTracker
from novelfactory.state.crew_state import (
    BaseCrewState,
    ProjectContext,
)
from novelfactory.state.novel_state import (
    NovelFactoryState,
    QuotaInfo,
    compress_completed_chapters,
)
from novelfactory.state.reducers import (
    _add_chapters_compressed,
    _add_usage,
    _chapter_key,
    _last_value,
)

__all__ = [
    "NovelFactoryState",
    "QuotaInfo",
    "compress_completed_chapters",
    "BaseCrewState",
    "ProjectContext",
    "ChapterStateTracker",
    # Reducers
    "_last_value",
    "_add_usage",
    "_chapter_key",
    "_add_chapters_compressed",
]
