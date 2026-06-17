"""Application FastAPI : sert l'API REST et la SPA Svelte buildée.

Lancement :
    cd photo_comparer
    .venv/bin/python -m uvicorn web.server.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import router

app = FastAPI(title="Photo Comparator Web")

# Accès LAN sans auth : CORS ouvert (utile aussi pour le dev Vite sur un autre port).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


if _DIST.is_dir():
    # Sert les assets buildés + fallback SPA sur index.html.
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(_DIST / "index.html")

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str):
        candidate = _DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_DIST / "index.html")
