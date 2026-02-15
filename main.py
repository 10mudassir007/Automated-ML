"""Application entrypoint.

Run locally:
    uvicorn main:app --reload

This keeps the FastAPI app definition under `api/router.py`.
"""

from api.router import app

__all__ = ["app"]
