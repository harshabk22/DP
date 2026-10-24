"""Database triggers enforcing Invariant I5 (append-only corrections)."""

from contextlib import contextmanager
from typing import Any, Generator
from sqlalchemy import Connection, Engine, event, text
from sqlalchemy.orm import Session

from app.core.database import is_postgres
from app.models.question import Correction

SQLITE_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS _messmate_flags (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """,
    "DROP TRIGGER IF EXISTS trg_corrections_no_update",
    """
    CREATE TRIGGER trg_corrections_no_update
    BEFORE UPDATE ON corrections
    BEGIN
        SELECT RAISE(ABORT, 'Invariant I5 violated: corrections table is append-only. Updates are prohibited.');
    END
    """,
    "DROP TRIGGER IF EXISTS trg_corrections_no_delete",
    """
    CREATE TRIGGER trg_corrections_no_delete
    BEFORE DELETE ON corrections
    WHEN (SELECT value FROM _messmate_flags WHERE key = 'hard_delete') IS NOT 'on'
    BEGIN
        SELECT RAISE(ABORT, 'Invariant I5 violated: corrections table is append-only. Deletes are prohibited.');
    END
    """,
]

POSTGRES_STATEMENTS = [
    """
    CREATE OR REPLACE FUNCTION prevent_corrections_mutation()
    RETURNS TRIGGER AS $$
    DECLARE
        bypass_flag TEXT;
    BEGIN
        bypass_flag := current_setting('messmate.hard_delete', true);
        IF bypass_flag = 'on' AND TG_OP = 'DELETE' THEN
            RETURN OLD;
        END IF;

        RAISE EXCEPTION 'Invariant I5 violated: corrections table is strictly append-only. Operation % blocked.', TG_OP
            USING ERRCODE = 'integrity_constraint_violation';
    END;
    $$ LANGUAGE plpgsql SECURITY DEFINER;
    """,
    "DROP TRIGGER IF EXISTS trg_corrections_immutable ON corrections",
    "DROP TRIGGER IF EXISTS trg_corrections_immutable_rows ON corrections",
    """
    CREATE TRIGGER trg_corrections_immutable_rows
    BEFORE UPDATE OR DELETE ON corrections
    FOR EACH ROW
    EXECUTE FUNCTION prevent_corrections_mutation()
    """,
    "DROP TRIGGER IF EXISTS trg_corrections_block_truncate ON corrections",
    """
    CREATE TRIGGER trg_corrections_block_truncate
    BEFORE TRUNCATE ON corrections
    FOR EACH STATEMENT
    EXECUTE FUNCTION prevent_corrections_mutation()
    """,
]


def _execute_statements(conn: Connection, statements: list[str]) -> None:
    """Execute each DDL statement individually."""
    for stmt in statements:
        cleaned = stmt.strip()
        if cleaned:
            conn.execute(text(cleaned))


def register_i5_triggers(target_or_bind: Any, connection: Any = None, **kw: Any) -> None:
    """Register Invariant I5 triggers on corrections table for SQLite or PostgreSQL.

    Can be called manually with an Engine, Connection, or Session, or used as
    an SQLAlchemy `after_create` DDL event listener.
    """
    if connection is not None:
        conn = connection
        if is_postgres(conn):
            _execute_statements(conn, POSTGRES_STATEMENTS)
        else:
            _execute_statements(conn, SQLITE_STATEMENTS)
        return

    bind = target_or_bind
    if isinstance(bind, Session):
        conn = bind.connection()
        if is_postgres(conn):
            _execute_statements(conn, POSTGRES_STATEMENTS)
        else:
            _execute_statements(conn, SQLITE_STATEMENTS)
    elif isinstance(bind, Connection):
        if is_postgres(bind):
            _execute_statements(bind, POSTGRES_STATEMENTS)
        else:
            _execute_statements(bind, SQLITE_STATEMENTS)
    elif isinstance(bind, Engine):
        with bind.begin() as conn:
            if is_postgres(conn):
                _execute_statements(conn, POSTGRES_STATEMENTS)
            else:
                _execute_statements(conn, SQLITE_STATEMENTS)


def set_hard_delete_bypass(bind: Any, enabled: bool = True) -> None:
    """Toggle the hard delete bypass flag for Invariant I5 (used for complete user erasure)."""
    val = "on" if enabled else "off"

    if isinstance(bind, Session):
        conn = bind.connection()
        _apply_bypass_on_conn(conn, val)
    elif isinstance(bind, Connection):
        _apply_bypass_on_conn(bind, val)
    elif isinstance(bind, Engine):
        with bind.begin() as conn:
            _apply_bypass_on_conn(conn, val)


def _apply_bypass_on_conn(conn: Connection, val: str) -> None:
    if is_postgres(conn):
        conn.execute(text(f"SET LOCAL messmate.hard_delete = '{val}';"))
    else:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS _messmate_flags (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """))
        conn.execute(
            text("""
                INSERT INTO _messmate_flags (key, value) VALUES ('hard_delete', :val)
                ON CONFLICT(key) DO UPDATE SET value = :val
            """),
            {"val": val},
        )


def reset_hard_delete_bypass(bind: Any) -> None:
    """Reset the hard delete bypass flag."""
    set_hard_delete_bypass(bind, enabled=False)


@contextmanager
def hard_delete_session(bind: Any) -> Generator[None, None, None]:
    """Context manager to safely execute hard deletions under Invariant I5."""
    set_hard_delete_bypass(bind, enabled=True)
    try:
        yield
    finally:
        reset_hard_delete_bypass(bind)


# Attach after_create hook to Correction table
event.listen(Correction.__table__, "after_create", register_i5_triggers)
