"""Platform-wide async database utility.

Provides a lazy-initialized SQLAlchemy async engine and session factory,
plus health-check and convenience helpers.  Everything reads the connection
string from :class:`config.settings.Settings` unless an explicit URL is
supplied at call time.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

# ---------------------------------------------------------------------------
# Module-level singletons (lazy)
# ---------------------------------------------------------------------------

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _resolve_url(db_url: str | None = None) -> str:
    """Return the database URL from the argument, env-var, or settings."""

    if db_url:
        return db_url

    # Prefer env-var (12-factor) so that external tooling can inject a URL
    # without needing to import our settings object.
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url

    # Fall back to the project settings singleton.
    from config.settings import settings

    return settings.database_url


def get_engine(db_url: str | None = None) -> AsyncEngine:
    """Get (or lazily create) the shared async SQLAlchemy engine.

    Parameters
    ----------
    db_url:
        Override database URL.  When *None* the URL is resolved from the
        environment or the project settings.

    Returns
    -------
    AsyncEngine
        The singleton engine instance.
    """

    global _engine

    if _engine is None:
        url = _resolve_url(db_url)
        _engine = create_async_engine(
            url,
            echo=False,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
    return _engine


async def get_session() -> AsyncSession:
    """Return a new async session bound to the platform engine.

    The caller is responsible for closing the session.  Typical usage::

        async with get_session() as session:
            result = await session.execute(...)

    If you are in a FastAPI dependency you may prefer :func:`session_dependency`
    which yields the session inside a context-manager.
    """

    global _session_factory

    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )

    return _session_factory()


@asynccontextmanager
async def session_dependency() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI-compatible async session context-manager.

    Usage in a FastAPI route::

        from fastapi import Depends

        @router.get("/health")
        async def health(db: AsyncSession = Depends(session_dependency)):
            ...
    """

    session = await get_session()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def check_db_health() -> bool:
    """Check whether the database is reachable.

    Returns
    -------
    bool
        ``True`` when a ``SELECT 1`` query succeeds, ``False`` otherwise.
    """

    try:
        session = await get_session()
        try:
            await session.execute(text("SELECT 1"))
            return True
        finally:
            await session.close()
    except Exception:
        return False


def reset_engine() -> None:
    """Tear down the global engine (useful in test teardowns)."""

    global _engine, _session_factory
    _engine = None
    _session_factory = None
