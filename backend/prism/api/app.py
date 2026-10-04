from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from prism import __version__
from prism.api.routers import analyze, health, pipeline, signals, stress
from prism.bootstrap import Container, build_container


def create_app(container: Container | None = None) -> FastAPI:
    """Run with: uvicorn prism.api.app:create_app --factory"""
    container = container or build_container()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if container.scheduler:
            container.scheduler.start()
        yield
        if container.scheduler:
            await container.scheduler.stop()

    app = FastAPI(
        title="PRISM Risk Engine API",
        version=__version__,
        description="Unstructured text in, structured financial risk signals and stress tests out.",
        lifespan=lifespan,
    )
    app.state.container = container
    origins = [o.strip() for o in container.settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

    for module in (health, signals, analyze, pipeline, stress):
        app.include_router(module.router)
    return app
