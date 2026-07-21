"""NovelFactory agents package."""

from novelfactory.agents.infra import *  # noqa: F403 — backward compat for agent_infra
from novelfactory.agents.registry import AgentDefinition, AgentRegistry
from novelfactory.agents.media_agents import (
    IllustratorOutput,
    TTSGeneratorOutput,
    create_illustrator_agent,
    create_tts_generator_agent,
)
from novelfactory.agents.review_agents import (
    ChapterReviewOutput,
    KickoffReviewOutput,
    create_chapter_final_review_agent,
    create_kickoff_review_agent,
    create_review_agent,
)
from novelfactory.agents.setup_agents import (
    CharacterDesignerOutput,
    OutlineWriterOutput,
    WorldBuilderOutput,
    create_character_designer_agent,
    create_outline_writer_agent,
    create_world_builder_agent,
)
from novelfactory.agents.sync_agents import (
    FeishuSyncOutput,
    StateUpdateOutput,
    create_feishu_sync_agent,
    update_project_state,
)
from novelfactory.agents.writing_agents import (
    ChapterRefinerOutput,
    ChapterReviewerOutput,
    ChapterWriterOutput,
    create_chapter_refiner_agent,
    create_chapter_reviewer_agent,
    create_chapter_writer_agent,
)

__all__ = [
    # Agent Registry
    "AgentRegistry",
    "AgentDefinition",
    # Setup
    "create_world_builder_agent",
    "create_character_designer_agent",
    "create_outline_writer_agent",
    "WorldBuilderOutput",
    "CharacterDesignerOutput",
    "OutlineWriterOutput",
    # Writing
    "create_chapter_writer_agent",
    "create_chapter_reviewer_agent",
    "create_chapter_refiner_agent",
    "ChapterWriterOutput",
    "ChapterReviewerOutput",
    "ChapterRefinerOutput",
    # Review
    "create_review_agent",
    "create_kickoff_review_agent",
    "create_chapter_final_review_agent",
    "KickoffReviewOutput",
    "ChapterReviewOutput",
    # Media
    "create_illustrator_agent",
    "create_tts_generator_agent",
    "IllustratorOutput",
    "TTSGeneratorOutput",
    # Sync
    "create_feishu_sync_agent",
    "update_project_state",
    "FeishuSyncOutput",
    "StateUpdateOutput",
]
