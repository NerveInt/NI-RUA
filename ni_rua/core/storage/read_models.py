"""Versioned, paginated projections shared by live gateways and archives."""
import asyncio
import json

from aiohttp import web


def record_updates(store, query):
    if query.get("mode") == "history":
        return store.review_page(int(query["before"]) if query.get("before") else None,
                                 int(query.get("limit", 200)))
    limit = int(query.get("limit", 500))
    result = {"schema_version": "1", "event_cursor": store.event_cursor()}
    for name, kind, mutable in (("observations", "observation", False),
                                ("items", "codex_item", True), ("requests", "native_request", True)):
        page = (store.state_page if mutable else store.record_page)(kind, int(query.get(name, 0)), limit)
        if name == "observations":
            for entry in page["entries"]:
                entry["value"] = store.observation_view(entry["value"])
        result[name] = page
    return result


async def artifact_response(store, reference, *, image=False, allowed_scopes=("actor", "observer", "internal")):
    """File-backed media keep HEAD/Range streaming; only reconstructed RGB has a body."""
    try:
        info, path = await asyncio.to_thread(store.artifact_file, reference, allowed_scopes)
    except FileNotFoundError:
        if not image:
            raise
        info, content = await asyncio.to_thread(store.resolve_image, reference, allowed_scopes)
        return web.Response(body=content, headers={"Content-Type": info["mime_type"],
            "X-Artifact-SHA256": info["sha256"], "X-Image-Reconstructed": "video",
            "X-Source-SHA256": info["reconstruction"]["source_sha256"]})
    return web.FileResponse(path, headers={"Content-Type": info["mime_type"], "X-Artifact-SHA256": info["sha256"]})


def observation_perception(store, observation_id):
    """Operator-only saved estimates for one observation; no inference or actor projection.

    Old archives may contain only attachment diagnostics. Query the exact identity
    across the whole event archive rather than a bounded first page.
    """
    state = store.read_state("perception", observation_id)
    if state is not None:
        return state
    with store._open("observation_perception"):
        row = store.db.execute(
            "SELECT body FROM events WHERE json_extract(body, '$.type')=? "
            "AND json_extract(body, '$.payload.observation_id')=? ORDER BY seq DESC LIMIT 1",
            ("geometry.obs3d_attached", observation_id)).fetchone()
    if row is None:
        raise KeyError("perception_not_found")
    payload = json.loads(row[0]).get("payload") or {}
    return dict(observation_id=observation_id, camera_id=payload.get("camera_id"),
                overlay_ref=(payload.get("object_table") or {}).get("overlay_ref"), diagnostics=payload)


def recorded_remote_status(root):
    """Recorded service snapshot only: never a current health assertion."""
    path = root / "private" / "remote_services.json"
    if not path.is_symlink() and path.is_file():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                return {"enabled": True, **data, "archive": True}
        except (OSError, ValueError):
            pass
    return {"enabled": False}
