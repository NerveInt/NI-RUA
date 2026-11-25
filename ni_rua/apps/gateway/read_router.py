"""Shared read-only task-review and evidence-file router for Gateway and Library."""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from aiohttp import web

from ...core.storage.store import file_digest
from ..library.experiments import catalog_id, contained
from ..library.task_review import TaskReviewService

GATEWAY_EVIDENCE_FILES = ("summary.json", "evaluation.json")


class RunReadRouter:
    """Shared read-only task-review and evidence-file endpoints."""

    def __init__(self, task_source_root=None, *, config=None, task_reviews=None):
        self.task_reviews = (
            task_reviews
            if task_reviews is not None
            else TaskReviewService(task_source_root, config=config)
        )

    async def task_review_data(self, entry):
        return await asyncio.to_thread(self.task_reviews.get, entry)

    async def task_review_response(self, entry):
        return web.json_response(await self.task_review_data(entry))

    async def evidence_file_response(
        self,
        path: Path,
        *,
        content_type: str = "application/json",
        expected_sha256: str | None = None,
    ):
        sha = await asyncio.to_thread(file_digest, path)
        if expected_sha256 and sha != expected_sha256:
            raise web.HTTPInternalServerError(text="evidence_digest_mismatch")
        return web.FileResponse(
            path,
            headers={"Content-Type": content_type, "X-Evidence-SHA256": sha},
        )

    def mount_gateway_routes(self, app, *, root: Path, task_supplier, status_supplier):
        review_id = catalog_id(".")

        async def task_review(request):
            entry = {
                "path": root,
                "metadata": {},
                "item": {
                    "id": review_id,
                    "task": task_supplier(),
                    "status": status_supplier(),
                },
            }
            return await self.task_review_response(entry)

        async def review_file(request):
            if request.match_info["id"] != review_id:
                raise web.HTTPNotFound(text="unindexed_evidence_file")
            # These are the only downloads linked by TaskReviewService. Never expose
            # manifest credentials, private files, task source, or arbitrary paths.
            asset = request.match_info["asset"]
            for name in GATEWAY_EVIDENCE_FILES:
                if asset == hashlib.sha256(name.encode()).hexdigest()[:32]:
                    path = root / name
                    if path.is_symlink() or not path.is_file():
                        raise web.HTTPNotFound(text="unindexed_evidence_file")
                    contained(root, name)
                    return await self.evidence_file_response(path, content_type="application/json")
            raise web.HTTPNotFound(text="unindexed_evidence_file")

        app.router.add_get("/v1/task-review", task_review)
        app.router.add_get("/v1/experiments/{id}/files/{asset}", review_file)
        return review_id


__all__ = ["GATEWAY_EVIDENCE_FILES", "RunReadRouter"]
