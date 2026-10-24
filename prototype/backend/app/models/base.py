"""Declarative base and dialect-compatible SQLAlchemy types."""

import json
import uuid
from typing import Any, Optional
from sqlalchemy import JSON, CHAR, BigInteger, Integer, String
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import TypeDecorator


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy declarative models."""
    pass


class GUID(TypeDecorator):
    """Platform-independent GUID type.

    Uses PostgreSQL's native UUID type, otherwise uses CHAR(36),
    storing as stringified hex values and returning uuid.UUID instances.
    """

    impl = CHAR(36)
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(postgresql.UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value: Optional[Any], dialect: Any) -> Optional[Any]:
        if value is None:
            return None
        if not isinstance(value, uuid.UUID):
            try:
                value = uuid.UUID(str(value))
            except (ValueError, TypeError):
                return str(value)
        if dialect.name == "postgresql":
            return value
        return str(value)

    def process_result_value(self, value: Optional[Any], dialect: Any) -> Optional[uuid.UUID]:
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))


# Dialect-compatible JSON/JSONB type
JSONB_compat = JSON().with_variant(postgresql.JSONB, "postgresql")

# Dialect-compatible BigInteger (INTEGER on SQLite to enable auto-increment rowid, BIGINT on PG)
BigInt_compat = BigInteger().with_variant(Integer, "sqlite")


class Array_compat(TypeDecorator):
    """Platform-independent Array type.

    Uses PostgreSQL's native ARRAY on Postgres, and JSON-serialized list on SQLite.
    """

    impl = JSON
    cache_ok = True

    def __init__(self, item_type: Any = String, *args: Any, **kwargs: Any) -> None:
        self.item_type = item_type
        super().__init__(*args, **kwargs)

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            item_t = self.item_type() if isinstance(self.item_type, type) else self.item_type
            return dialect.type_descriptor(postgresql.ARRAY(item_t))
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value: Optional[Any], dialect: Any) -> Optional[Any]:
        if value is None:
            return None
        return list(value)

    def process_result_value(self, value: Optional[Any], dialect: Any) -> Optional[list[Any]]:
        if value is None:
            return None
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return list(value)
