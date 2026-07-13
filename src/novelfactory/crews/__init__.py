"""NovelFactory Crews package.

Exposes the UnifiedReviewQueue singleton and crew supervisor factory.
"""

from novelfactory.crews.review_queue import (
    ReviewItem,
    ReviewStatus,
    UnifiedReviewQueue,
    review_queue,
)
from novelfactory.crews.supervisor import create_crew_supervisor, crew_handoff

__all__ = [
    "review_queue",
    "UnifiedReviewQueue",
    "ReviewItem",
    "ReviewStatus",
    "create_crew_supervisor",
    "crew_handoff",
]
