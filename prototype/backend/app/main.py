"""FastAPI application entrypoint for MessMate v2.0."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sqlalchemy import text
from app.api.v1.router import api_router, api_v1_router
from app.core.config import settings
from app.core.database import is_postgres, sync_engine
from app.core.errors import register_error_handlers
from app.models.base import Base
from app.models.triggers import register_i5_triggers


def _sync_sqlite_columns() -> None:
    """Ensure newly added columns in SQLite tables are migrated automatically on startup."""
    if is_postgres(sync_engine):
        return
    with sync_engine.connect() as conn:
        for table_name, table in Base.metadata.tables.items():
            try:
                res = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
                existing = {row[1] for row in res}
                if existing:
                    for col in table.columns:
                        if col.name not in existing:
                            col_type = col.type.compile(sync_engine.dialect)
                            default_clause = " DEFAULT 0" if str(col.type).lower().startswith("bool") else ""
                            conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col.name} {col_type}{default_clause}"))
                            conn.commit()
            except Exception:
                pass


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager ensuring database tables, columns, and triggers exist."""
    Base.metadata.create_all(bind=sync_engine)
    _sync_sqlite_columns()
    register_i5_triggers(sync_engine)

    # Pre-warm neural models (Meta DINOv2 ViT-S/14 & FastEmbed BGE) for sub-30ms inference
    try:
        from app.services.vision.reid import get_dinov2_model
        from app.services.resolution.embedding import get_neural_embedding_model
        get_dinov2_model()
        get_neural_embedding_model()
    except Exception:
        pass

    yield


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title="MessMate API",
        version="2.0.0",
        description="MessMate Open-World Dietary Intelligence Platform Backend",
        lifespan=lifespan,
    )

    # Cross-Origin Resource Sharing (CORS) for physical phones, emulator, and web clients
    # Allows physical phone IP (e.g. 10.5.138.91), USB ADB reverse (127.0.0.1/localhost), and LANs
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(1[6-9]|2[0-9]|3[0-1])\.\d+\.\d+)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register RFC 7807 problem details exception handlers
    register_error_handlers(app)

    # Mount API v1 router under canonical /v1 and legacy /api/v1 and /api prefixes
    app.include_router(api_v1_router)
    app.include_router(api_v1_router, prefix="/api")
    app.include_router(api_router, prefix="/api")

    # Root health probe
    @app.get("/health", tags=["health"])
    def root_health() -> dict[str, str]:
        return {"status": "ok", "version": "2.0.0"}

    # Mount media storage directory relative to backend root or current dir
    backend_dir = Path(__file__).resolve().parent.parent
    media_dir = backend_dir / "media" if (backend_dir / "media").exists() else Path("media")
    media_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/media", StaticFiles(directory=str(media_dir)), name="media")

    # Mount UI static web assets relative to backend root or current dir
    static_dir = backend_dir / "static" if (backend_dir / "static").exists() else Path("static")
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", include_in_schema=False)
        def serve_ui() -> FileResponse:
            index_path = static_dir / "index.html"
            return FileResponse(index_path)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=not settings.is_production,
    )
