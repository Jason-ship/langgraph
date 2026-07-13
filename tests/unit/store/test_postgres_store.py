"""Tests for store/postgres_store.py — persistence layer."""

from __future__ import annotations


class TestPostgresStoreImports:
    """Verify store modules import correctly."""

    def test_checkpointer_imports(self):
        """Verify checkpointer module imports."""
        import novelfactory.graph.checkpointer  # noqa: F401

        assert True

    def test_checkpointer_has_factory(self):
        """Verify checkpointer has create_checkpointer function."""
        from novelfactory.graph.checkpointer import create_checkpointer

        assert callable(create_checkpointer)

    def test_checkpointer_has_store_factory(self):
        """Verify checkpointer has create_store function."""
        from novelfactory.graph.checkpointer import create_store

        assert callable(create_store)

    def test_checkpointer_has_cleanup(self):
        """Verify cleanup functions exist."""
        from novelfactory.graph.checkpointer import (
            cleanup_thread_full,
        )

        assert callable(cleanup_thread_full)


class TestPostgresStoreConfig:
    """Test store configuration helpers."""

    def test_db_url_from_env_defaults(self):
        """Verify default DSN builder handles missing env vars."""
        import os

        # Save and clear relevant env vars
        saved = {
            k: os.environ.pop(k, None)
            for k in [
                "DATABASE_URL",
                "DB_HOST",
                "DB_PORT",
                "DB_NAME",
                "DB_USER",
                "DB_PASSWORD",
            ]
        }

        try:
            from novelfactory.graph.checkpointer import _db_url_from_env

            # Without env vars, should return localhost default
            url = _db_url_from_env()
            assert url is None or "localhost" in url or "noveluser" in url
        finally:
            # Restore env vars
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_checkpoint_type_default(self):
        """Verify default checkpoint type is postgres."""
        from novelfactory.graph.checkpointer import _checkpoint_type

        cp_type = _checkpoint_type()
        assert cp_type == "postgres"

    def test_store_type_default(self):
        """Verify store type matches settings default."""
        from novelfactory.config.settings import settings
        from novelfactory.graph.checkpointer import _store_type

        st_type = _store_type()
        assert st_type == settings.STORAGE_TYPE.lower()
