-- ─────────────────────────────────────────────────────────────────────────────
-- Migration: Add missing columns to novel_quality_trends
-- 
-- The code (novel_phase3.py) expects these columns but init-db.sql didn't have them.
-- This migration is safe to run multiple times (IF NOT EXISTS).
-- ─────────────────────────────────────────────────────────────────────────────

DO $$
BEGIN
    -- word_count — tracks chapter word count at time of quality check
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'novel_quality_trends' AND column_name = 'word_count'
    ) THEN
        ALTER TABLE novel_quality_trends
        ADD COLUMN word_count INTEGER NOT NULL DEFAULT 0;
    END IF;

    -- rewrite_count — tracks how many times this chapter has been rewritten
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'novel_quality_trends' AND column_name = 'rewrite_count'
    ) THEN
        ALTER TABLE novel_quality_trends
        ADD COLUMN rewrite_count INTEGER NOT NULL DEFAULT 0;
    END IF;

    -- audit_score — score from post-hoc audit (distinct from quality_score)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'novel_quality_trends' AND column_name = 'audit_score'
    ) THEN
        ALTER TABLE novel_quality_trends
        ADD COLUMN audit_score REAL NOT NULL DEFAULT 100.0;
    END IF;
END $$;
