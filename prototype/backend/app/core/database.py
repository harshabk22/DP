"""Database engine, session factories, and dialect detection."""

import sqlite3
from typing import Any, AsyncGenerator, Generator
from pathlib import Path
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

_backend_root = Path(__file__).resolve().parent.parent.parent


def _resolve_sqlite_url(url: str) -> str:
    """Resolve relative SQLite database paths to backend root directory."""
    if "sqlite:///:memory:" in url or "sqlite+aiosqlite:///:memory:" in url:
        return url
    if url.startswith("sqlite:///./"):
        return f"sqlite:///{_backend_root / url[12:]}"
    if url.startswith("sqlite+aiosqlite:///./"):
        return f"sqlite+aiosqlite:///{_backend_root / url[22:]}"
    return url


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
    """Enforce foreign key constraints for SQLite connections."""
    try:
        # Check if native sqlite3 connection or aiosqlite wrapper
        if isinstance(dbapi_connection, sqlite3.Connection):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
        elif hasattr(dbapi_connection, "cursor"):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            if hasattr(cursor, "close"):
                cursor.close()
    except Exception:
        pass


def is_postgres(session_or_engine: Any) -> bool:
    """Determine whether a given session, engine, connection, or dialect is PostgreSQL."""
    if isinstance(session_or_engine, str):
        return "postgres" in session_or_engine.lower()
    if hasattr(session_or_engine, "bind") and session_or_engine.bind is not None:
        bind = session_or_engine.bind
        if hasattr(bind, "dialect"):
            return bind.dialect.name == "postgresql"
    if hasattr(session_or_engine, "get_bind"):
        try:
            bind = session_or_engine.get_bind()
            if hasattr(bind, "dialect"):
                return bind.dialect.name == "postgresql"
        except Exception:
            pass
    if hasattr(session_or_engine, "dialect"):
        return session_or_engine.dialect.name == "postgresql"
    if hasattr(session_or_engine, "sync_engine"):
        return session_or_engine.sync_engine.dialect.name == "postgresql"
    return False


# Sync engine configuration
sync_url = _resolve_sqlite_url(settings.SYNC_DATABASE_URL)
sync_connect_args: dict[str, Any] = {}
if "sqlite" in sync_url:
    sync_connect_args["check_same_thread"] = False

sync_engine: Engine = create_engine(
    sync_url,
    echo=False,
    future=True,
    connect_args=sync_connect_args,
)

SessionLocal = sessionmaker(
    bind=sync_engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    class_=Session,
)

# Async engine configuration
async_url = _resolve_sqlite_url(settings.DATABASE_URL)
async_connect_args: dict[str, Any] = {}
if "sqlite" in async_url:
    async_connect_args["check_same_thread"] = False

async_engine: AsyncEngine = create_async_engine(
    async_url,
    echo=False,
    future=True,
    connect_args=async_connect_args,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    class_=AsyncSession,
)


def get_db() -> Generator[Session, None, None]:
    """Dependency that yields a synchronous database session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency that yields an asynchronous database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
