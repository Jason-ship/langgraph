-- ═══════════════════════════════════════════════════════════════════════════
-- NovelFactory v4.2 — PostgreSQL Initialization Script
-- ═══════════════════════════════════════════════════════════════════════════
-- 
-- Automatic provisioning: all app tables + LangGraph checkpoint schema.
-- Schema synced with: database_writer.py, novel_state_tracker.py, novel_scale.py
-- ═══════════════════════════════════════════════════════════════════════════

BEGIN;

-- ── Extensions ─────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
GRANT USAGE ON SCHEMA public TO PUBLIC;

-- ── LangGraph Checkpoint Schema ────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS langgraph AUTHORIZATION noveluser;
GRANT ALL PRIVILEGES ON SCHEMA langgraph TO PUBLIC;


-- ═══════════════════════════════════════════════════════════════════════════
--  NovelFactory Application Tables (v5.4)
-- ═══════════════════════════════════════════════════════════════════════════

-- ── novel_projects ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS novel_projects (
    project_name        TEXT PRIMARY KEY,
    genre               TEXT DEFAULT '',
    chapter_count       INT DEFAULT 0,
    world_setting       TEXT DEFAULT '',
    character_setting   TEXT DEFAULT '',
    story_outline       TEXT DEFAULT '',
    chapter_outlines    TEXT DEFAULT '',
    created_at          TIMESTAMP DEFAULT NOW(),
    updated_at          TIMESTAMP DEFAULT NOW()
);

-- ── writing_guides (v5.4: 写作指南知识库) ────────────────────────────────
CREATE TABLE IF NOT EXISTS writing_guides (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title               TEXT NOT NULL,
    content             TEXT NOT NULL DEFAULT '',
    source              TEXT NOT NULL DEFAULT '',
    source_url          TEXT DEFAULT '',
    tags                TEXT[] DEFAULT '{}',
    genre               TEXT DEFAULT '',
    quality_score       REAL DEFAULT 0.0 CHECK (quality_score >= 0 AND quality_score <= 1),
    chapter_ref         TEXT DEFAULT '',
    guide_type          TEXT NOT NULL DEFAULT 'technique'
        CHECK (guide_type IN ('technique', 'analysis', 'template', 'case')),
    created_at          TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_writing_guides_genre ON writing_guides(genre);
CREATE INDEX IF NOT EXISTS idx_writing_guides_guide_type ON writing_guides(guide_type);
CREATE INDEX IF NOT EXISTS idx_writing_guides_quality ON writing_guides(quality_score DESC);

-- ── novel_chapters ─────────────────────────────────────────────────────────
-- v4.2: UNIQUE(project_name, chapter_number) required for ON CONFLICT upsert
CREATE TABLE IF NOT EXISTS novel_chapters (
    id                  SERIAL PRIMARY KEY,
    project_name        TEXT NOT NULL,
    chapter_number      INT NOT NULL,
    title               TEXT DEFAULT '',
    word_count          INT DEFAULT 0,
    quality_score       REAL DEFAULT 0.0,
    summary             TEXT DEFAULT '',
    full_text_hash      TEXT DEFAULT '',
    created_at          TIMESTAMP DEFAULT NOW(),
    updated_at          TIMESTAMP DEFAULT NOW(),
    UNIQUE(project_name, chapter_number)
);

-- ── novel_character_states ─────────────────────────────────────────────────
-- v4.2: 扩展为结构化字段 (location, mood, power_level, status,
--       relationships, knowledge, items) + raw_state JSONB 兼容旧数据
CREATE TABLE IF NOT EXISTS novel_character_states (
    id                  SERIAL PRIMARY KEY,
    project_name        TEXT NOT NULL,
    chapter_number      INT NOT NULL,
    character_name      TEXT NOT NULL,
    location            TEXT DEFAULT '',
    mood                TEXT DEFAULT '',
    power_level         TEXT DEFAULT '',
    status              TEXT DEFAULT '健在',
    relationships       JSONB DEFAULT '{}',
    knowledge           JSONB DEFAULT '[]',
    items               JSONB DEFAULT '[]',
    raw_state           JSONB DEFAULT '{}',
    created_at          TIMESTAMP DEFAULT NOW(),
    UNIQUE(project_name, chapter_number, character_name)
);

-- ── novel_plot_threads ────────────────────────────────────────────────────
-- v4.2: UNIQUE(project_name, thread_name) required for ON CONFLICT upsert
CREATE TABLE IF NOT EXISTS novel_plot_threads (
    id                  SERIAL PRIMARY KEY,
    project_name        TEXT NOT NULL,
    thread_name         TEXT NOT NULL,
    status              TEXT DEFAULT 'open',
    created_chapter     INT DEFAULT 0,
    description         TEXT DEFAULT '',
    related_characters  JSONB DEFAULT '[]',
    created_at          TIMESTAMP DEFAULT NOW(),
    updated_at          TIMESTAMP DEFAULT NOW(),
    UNIQUE(project_name, thread_name)
);
CREATE INDEX IF NOT EXISTS idx_plot_threads_project_status
    ON novel_plot_threads(project_name, status);

-- ── novel_foreshadowing ────────────────────────────────────────────────────
-- v4.2: 重构 — 使用 name 作为主标识, added category/priority/related_characters
--       UNIQUE(project_name, name) required for ON CONFLICT upsert
CREATE TABLE IF NOT EXISTS novel_foreshadowing (
    id                      SERIAL PRIMARY KEY,
    project_name            TEXT NOT NULL,
    name                    TEXT NOT NULL,
    planted_chapter         INT DEFAULT 0,
    planned_resolve_chapter INT DEFAULT 0,
    actual_resolve_chapter  INT DEFAULT 0,
    priority                INT DEFAULT 5,
    category                TEXT DEFAULT 'plot',
    related_characters      JSONB DEFAULT '[]',
    status                  TEXT DEFAULT 'planted',
    notes                   TEXT DEFAULT '',
    details                 TEXT DEFAULT '',
    description             TEXT DEFAULT '',
    created_at              TIMESTAMP DEFAULT NOW(),
    updated_at              TIMESTAMP DEFAULT NOW(),
    UNIQUE(project_name, name)
);

-- ── novel_audit_reports ────────────────────────────────────────────────────
-- v4.2: 重构 — chapter_start/chapter_end 替代 chapter_number, 
--       findings_json 替代 audit_result, overall_score 替代旧审计字段
CREATE TABLE IF NOT EXISTS novel_audit_reports (
    id                  SERIAL PRIMARY KEY,
    project_name        TEXT NOT NULL,
    chapter_start       INT DEFAULT 0,
    chapter_end         INT DEFAULT 0,
    findings_json       JSONB DEFAULT '[]',
    overall_score       REAL DEFAULT 0.0,
    summary             TEXT DEFAULT '',
    created_at          TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_audit_reports_project_chapter
    ON novel_audit_reports(project_name, chapter_start);

-- ── novel_pacing_snapshots ────────────────────────────────────────────────
-- v4.2: 扩展为结构化节奏指标 (event_density, dialogue_ratio, action_ratio,
--       description_ratio, pacing_label, intensity)
--       UNIQUE(project_name, chapter_number) required for ON CONFLICT upsert
CREATE TABLE IF NOT EXISTS novel_pacing_snapshots (
    id                  SERIAL PRIMARY KEY,
    project_name        TEXT NOT NULL,
    chapter_number      INT NOT NULL,
    intensity           REAL DEFAULT 5.0,
    event_density       REAL DEFAULT 0.0,
    dialogue_ratio      REAL DEFAULT 0.0,
    action_ratio        REAL DEFAULT 0.0,
    description_ratio   REAL DEFAULT 0.0,
    pacing_label        TEXT DEFAULT 'balanced',
    created_at          TIMESTAMP DEFAULT NOW(),
    UNIQUE(project_name, chapter_number)
);

-- ── novel_cost_records ────────────────────────────────────────────────────
-- v4.2: 列重命名 tokens_input→input_tokens, tokens_output→output_tokens
--       新增 model, phase 字段
CREATE TABLE IF NOT EXISTS novel_cost_records (
    id                  SERIAL PRIMARY KEY,
    project_name        TEXT NOT NULL,
    chapter_number      INT DEFAULT 0,
    model               TEXT DEFAULT 'deepseek-chat',
    input_tokens        INT DEFAULT 0,
    output_tokens       INT DEFAULT 0,
    cost_rmb            REAL DEFAULT 0.0,
    phase               TEXT DEFAULT 'writing',
    created_at          TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_cost_records_project_chapter
    ON novel_cost_records(project_name, chapter_number);

-- ── novel_quality_trends ───────────────────────────────────────────────────
-- v4.2: UNIQUE(project_name, chapter_number) required for ON CONFLICT upsert
CREATE TABLE IF NOT EXISTS novel_quality_trends (
    id                  SERIAL PRIMARY KEY,
    project_name        TEXT NOT NULL,
    chapter_number      INT NOT NULL,
    quality_score       REAL DEFAULT 0.0,
    word_count          INTEGER DEFAULT 0,
    rewrite_count       INTEGER DEFAULT 0,
    audit_score         REAL DEFAULT 100.0,
    review_comments     TEXT DEFAULT '',
    created_at          TIMESTAMP DEFAULT NOW(),
    UNIQUE(project_name, chapter_number)
);

-- ── novel_volumes ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS novel_volumes (
    id                  SERIAL PRIMARY KEY,
    project_name        TEXT NOT NULL,
    volume_number       INT NOT NULL,
    title               TEXT DEFAULT '',
    theme               TEXT DEFAULT '',
    summary             TEXT DEFAULT '',
    start_chapter       INT DEFAULT 0,
    end_chapter         INT DEFAULT 0,
    status              TEXT DEFAULT 'active',
    created_at          TIMESTAMP DEFAULT NOW(),
    updated_at          TIMESTAMP DEFAULT NOW(),
    UNIQUE(project_name, volume_number)
);

-- ── novel_chapter_outlines ────────────────────────────────────────────────
-- v4.2: 扩展为结构化大纲字段 (goal, key_beats, pov_character,
--       characters_involved, word_count_target, foreshadowing_plant/resolve)
CREATE TABLE IF NOT EXISTS novel_chapter_outlines (
    id                      SERIAL PRIMARY KEY,
    project_name            TEXT NOT NULL,
    volume_number           INT DEFAULT 0,
    chapter_number          INT NOT NULL,
    title                   TEXT DEFAULT '',
    goal                    TEXT DEFAULT '',
    key_beats               JSONB DEFAULT '[]',
    pov_character           TEXT DEFAULT '',
    characters_involved     JSONB DEFAULT '[]',
    foreshadowing_plant     JSONB DEFAULT '[]',
    foreshadowing_resolve   JSONB DEFAULT '[]',
    word_count_target       INT DEFAULT 3000,
    status                  TEXT DEFAULT 'pending',
    created_at              TIMESTAMP DEFAULT NOW(),
    updated_at              TIMESTAMP DEFAULT NOW(),
    UNIQUE(project_name, chapter_number)
);

-- ── novel_key_events ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS novel_key_events (
    id                      SERIAL PRIMARY KEY,
    project_name            TEXT NOT NULL,
    chapter_number          INT DEFAULT 0,
    event_text              TEXT DEFAULT '',
    event_type              TEXT DEFAULT '',
    description             TEXT DEFAULT '',
    characters              JSONB DEFAULT '[]',
    characters_involved     JSONB DEFAULT '[]',
    importance              INT DEFAULT 5,
    created_at              TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_key_events_project_importance
    ON novel_key_events(project_name, importance DESC);

-- ── novel_character_arcs ──────────────────────────────────────────────────
-- v4.2: 扩展 arc_type/stages_json/current_stage/arc_completeness 字段
CREATE TABLE IF NOT EXISTS novel_character_arcs (
    id                  SERIAL PRIMARY KEY,
    project_name        TEXT NOT NULL,
    character_name      TEXT NOT NULL,
    chapter_number      INT DEFAULT 0,
    arc_type            TEXT DEFAULT '',
    arc_stage           TEXT DEFAULT '',
    stages_json         JSONB DEFAULT '[]',
    current_stage       TEXT DEFAULT '',
    arc_completeness    REAL DEFAULT 0.0,
    status              TEXT DEFAULT 'active',
    notes               TEXT DEFAULT '',
    created_at          TIMESTAMP DEFAULT NOW(),
    updated_at          TIMESTAMP DEFAULT NOW(),
    UNIQUE(project_name, character_name, chapter_number)
);

-- ── novel_checkpoint_log ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS novel_checkpoint_log (
    id                  SERIAL PRIMARY KEY,
    project_name        TEXT NOT NULL,
    chapter_number      INT DEFAULT 0,
    status              TEXT DEFAULT '',
    details             TEXT DEFAULT '',
    created_at          TIMESTAMP DEFAULT NOW()
);

-- ═══════════════════════════════════════════════════════════════════════════
--  Permissions + Observability
-- ═══════════════════════════════════════════════════════════════════════════

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO PUBLIC;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO PUBLIC;

-- ── Observability Views ────────────────────────────────────────────────────

CREATE OR REPLACE VIEW pg_connection_stats AS
SELECT
    datname                                             AS database_name,
    numbackends                                         AS active_connections,
    xact_commit                                         AS transactions_committed,
    xact_rollback                                       AS transactions_rolled_back,
    blks_hit                                            AS buffer_hits,
    blks_read                                           AS buffer_reads,
    ROUND(100.0 * blks_hit / NULLIF(blks_hit + blks_read, 0), 2)
                                                        AS cache_hit_ratio_pct
FROM pg_stat_database
WHERE datname = current_database();

CREATE OR REPLACE VIEW pg_long_running_queries AS
SELECT
    pid,
    usename                                              AS username,
    application_name,
    client_addr,
    backend_start,
    xact_start,
    query_start,
    state,
    wait_event_type,
    wait_event,
    REPLACE(LEFT(query, 200), E'\n', ' ')               AS query_preview,
    ROUND(EXTRACT(EPOCH FROM (now() - query_start)), 2)  AS duration_seconds
FROM pg_stat_activity
WHERE state != 'idle'
  AND pid != pg_backend_pid()
ORDER BY duration_seconds DESC;

GRANT SELECT ON pg_connection_stats TO PUBLIC;
GRANT SELECT ON pg_long_running_queries TO PUBLIC;

COMMIT;

DO $$
BEGIN
    RAISE NOTICE 'NovelFactory v4.2 PostgreSQL init complete — pgvector=%',
        (SELECT extversion FROM pg_extension WHERE extname = 'vector');
END $$;