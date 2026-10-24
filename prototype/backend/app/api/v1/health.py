"""Health check endpoint reporting operational status, DB connectivity, and version."""

from typing import Any, Dict
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.database import get_db

router = APIRouter(tags=["health"])


@router.get("/health", summary="Health check endpoint")
def health_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Return backend service operational health status, DB connectivity, and version."""
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        db_status = f"unhealthy: {exc}"

    status = "ok" if db_status == "connected" else "degraded"
    return {
        "status": status,
        "database": db_status,
        "version": "2.0.0",
    }
