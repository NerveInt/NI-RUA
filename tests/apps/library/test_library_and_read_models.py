"""Consolidated tests:

- test_library: Synthetic HTTP/WS checks for the authenticated public-origin entry.
- test_library_loading: Real HTTP checks for compact library views and private revalidation.
- test_library_native_terminal: Native replay transport selection; fixture bytes never represent a real run.
- test_read_models: CPU read contracts: incremental projections, sealed archives and streaming integrity.
- test_runtime_config: Unified configuration reaches runtime boundaries without GPU or model calls.
- test_task_review: Read-only scoring provenance, native/normalized separation and observer API.
"""

from pathlib import Path
import tempfile
import unittest
from aiohttp import ClientSession, WSServerHandshakeError, web
from ni_rua.apps.library.experiments import allocate_run, atomic_json, catalog_id
from ni_rua.apps.library.service import Library
import copy
import gzip
import json
import subprocess
from unittest.mock import patch
import zlib
from aiohttp import ClientSession, web
from ni_rua.apps.library.experiments import atomic_json, catalog_id, register_run
import base64
from ni_rua.core.storage.store import Store
import asyncio
import hashlib
import sqlite3
import shutil
from types import SimpleNamespace
from aiohttp.test_utils import TestClient, TestServer
from ni_rua.core.storage.read_models import record_updates
from ni_rua.apps.library.experiments import allocate_run, catalog_id
from ni_rua.core.storage.store import Store, open_read_store
import contextlib
import io
import os
from unittest.mock import MagicMock, patch
from ni_rua.cortex.agents.codex.session import CodexSession
from ni_rua.core.config import Config
from ni_rua.core.storage.images import image_settings, store_image_settings
from ni_rua.robot.model.geometry import GeometryTools
from ni_rua.apps.library.service import Library, main as library_main
from ni_rua.robot.backends.robodojo.robodojo import RobotBackend
from ni_rua.apps.reference.reference_api import ResponsesClient
from ni_rua.perception.scene.scene_service import SceneService, _compute
from ni_rua.apps.library.task_review import TaskReviewService
from ni_rua.apps.gateway.workbench import Workbench
from support.paths import REPO_ROOT, isolate_home


def setUpModule():
    isolate_home()   # a fresh Workbench opens the grasp memory store at its '~' default


NI_RUA_TEST_TIER = "component"

# --- test_library.py ---
class PublicOriginCheck(unittest.IsolatedAsyncioTestCase):
    async def test_page_and_assets_revalidate_after_deployment(self):
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as assets:
            static = Path(assets)
            (static / "library.html").write_text('<script src="/assets/library.js?v=new"></script>')
            (static / "index.html").write_text('<script src="/app.js?v=new"></script>')
            for name in ("library.js", "app.js"):
                (static / name).write_text("window.synthetic_fixture = true;")
            run = allocate_run(directory, category="synthetic", task="cache-check")
            identifier = catalog_id(run.relative_to(Path(directory).resolve()).as_posix())
            with patch("ni_rua.apps.library.service.STATIC", static):
                runner = web.AppRunner(Library(directory).app(), shutdown_timeout=.2)
                await runner.setup()
                try:
                    site = web.TCPSite(runner, "127.0.0.1", 0)
                    await site.start()
                    local = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
                    async with ClientSession() as client:
                        for url in ("/", f"/runs/{identifier}/"):
                            async with client.get(local + url) as response:
                                self.assertEqual(response.status, 200)
                                self.assertEqual(response.headers["Cache-Control"], "no-store")
                                self.assertIn("?v=new", await response.text())
                        for url in ("/assets/library.js?v=new", "/app.js?v=new"):
                            async with client.get(local + url) as response:
                                self.assertEqual(response.status, 200)
                                self.assertEqual(response.headers["Cache-Control"], "no-cache, max-age=0, must-revalidate")
                                etag = response.headers["ETag"]
                            async with client.get(local + url, headers={"If-None-Match": etag}) as response:
                                self.assertEqual(response.status, 304)
                                self.assertIn("must-revalidate", response.headers["Cache-Control"])
                finally:
                    await runner.cleanup()

    async def test_exact_public_origin_validation_http_and_websocket(self):
        origin = "https://library.example.org"
        with tempfile.TemporaryDirectory() as directory:
            for invalid in ("", "http://library.example.org", "https://*.example.org", origin + "/",
                            origin + "/path", origin + "?", origin + "#", origin + ":bad",
                            origin + ":65536", "https://user:password@library.example.org",
                            "https://@library.example.org", origin + "\n", "\t" + origin,
                            "https://library.example.org\\evil", origin + "\x7f"):
                with self.subTest(invalid=repr(invalid)), self.assertRaises(ValueError):
                    Library(directory, public_origin=invalid)
            run = allocate_run(directory, category="synthetic", task="origin-check")
            (run / "terminal.bin").write_bytes(b"synthetic public-origin terminal")
            atomic_json(run / "terminal.json", dict(session_id="synthetic-public-pty", exit_code=0))
            identifier = catalog_id(run.relative_to(Path(directory).resolve()).as_posix())
            async with ClientSession() as client:
                for configured in (None, origin):
                    library = Library(directory, public_origin=configured)
                    runner = web.AppRunner(library.app(), shutdown_timeout=.2)
                    await runner.setup()
                    try:
                        site = web.TCPSite(runner, "127.0.0.1", 0)
                        await site.start()
                        local = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
                        for headers in ({}, {"Origin": local}, {"Origin": origin},
                                        {"Origin": "https://other.example.org"},
                                        {"Origin": origin, "Host": "library.example.org"}):
                            expected = 200 if not headers or headers.get("Origin") == local or headers == {"Origin": configured} else 403
                            async with client.get(local + "/v1/experiments", headers=headers) as response:
                                self.assertEqual(response.status, expected, (configured, headers))
                        for ws_origin in (local, origin, "https://other.example.org"):
                            url = local + f"/runs/{identifier}/v1/terminal"
                            if ws_origin == local or ws_origin == configured:
                                async with client.ws_connect(url, headers={"Origin": ws_origin}) as ws:
                                    self.assertEqual((await ws.receive_json(timeout=3))["type"], "ready")
                                    self.assertEqual((await ws.receive_json(timeout=3))["type"], "output")
                            else:
                                with self.assertRaises(WSServerHandshakeError) as rejected:
                                    await client.ws_connect(url, headers={"Origin": ws_origin})
                                self.assertEqual(rejected.exception.status, 403)
                    finally:
                        await runner.cleanup()


# --- test_library_loading.py ---
class LoadingCheck(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.run = self.root / "benchmark" / "fixture" / "runs" / "first"
        register_run(self.run, category="benchmark", task="pick", run_id="first")
        self.summary = self.run / "summary.json"
        atomic_json(self.summary, dict(normalized_score=.5, task_success=False, scored_episode=True, valid_episode=True))
        atomic_json(self.run.parent.parent / "manifest.json", dict(method="fixture",
            protocol_cases=[dict(task="pick", variant="standard", eval_seed=0, layout_id=0)],
            cases=[dict(run_dir=str(self.run), task="pick", seed=0, layout_id=0, cwd="/fixture")]))
        self.identifier = catalog_id(self.run.relative_to(self.root).as_posix())
        self.library = Library(self.root, scan_interval=3600)
        self.runner = web.AppRunner(self.library.app(), shutdown_timeout=.2)
        await self.runner.setup()
        self.addAsyncCleanup(self.runner.cleanup)
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        self.client = ClientSession()
        self.addAsyncCleanup(self.client.close)
        self.group_id = self.library.group_snapshot["groups"][0]["id"]

    async def get(self, path, **kwargs):
        async with self.client.get(self.url + path, **kwargs) as response:
            self.assertEqual(response.status, 200, await response.text())
            return await response.json(), response.headers["ETag"]

    def javascript(self, script, value):
        result = subprocess.run(["node", "--input-type=module", "-e", script],
            input=json.dumps(value), text=True, capture_output=True, check=True,
            cwd=REPO_ROOT)
        return json.loads(result.stdout)

    async def test_summary_retains_exact_formal_scores_denominators_and_lineage(self):
        summaries = [
            {"normalized_score": 0, "task_success": False}, {"normalized_score": 1, "task_success": True},
            {"score": .4}, {"normalized_score": None, "score": .2}, {"normalized_score": False},
            {"normalized_score": "0.5"}, {"normalized_score": 1.1}, {},
            *({"normalized_score": .8, "score_status": status} for status in ("unavailable", "unscored", "invalid")),
            {"normalized_score": .8, "scored_episode": False}, {"normalized_score": .8, "valid_episode": False},
            {"normalized_score": .8, "primary_outcome": "unscored_infrastructure"},
            {"normalized_score": .8, "infrastructure_error": True},
            {"normalized_score": .8, "infrastructure_error": True, "scored_episode": True},
        ]
        snapshot = copy.deepcopy(self.library.group_snapshot)
        group = snapshot["groups"][0]
        group.pop("revision"); group.pop("details_loaded")
        template = copy.deepcopy(group["cases"][0])
        group["cases"] = [{**template, "case_key": str(index), "run": {**template["run"],
            "summary": {**summary, "cleanup_status": "omitted", "termination_reason": "omitted"}}}
            for index, summary in enumerate(summaries)]
        group["cases"].append({**template, "case_key": "missing", "run": None})
        group["case_count"] = len(group["cases"])
        with patch.object(self.library.experiment_groups, "snapshot", return_value=snapshot):
            await self.library.refresh()
        full, full_tag = await self.get("/v1/experiment-groups")
        summary, summary_tag = await self.get("/v1/experiment-groups?view=summary")
        explicit, _ = await self.get("/v1/experiment-groups?view=full")
        detail, _ = await self.get(f"/v1/experiment-groups/{self.group_id}")
        self.assertEqual(detail, full["groups"][0])
        self.assertEqual(explicit["groups"], full["groups"])
        self.assertNotEqual(full_tag, summary_tag)
        compact = summary["groups"][0]
        self.assertFalse(compact["details_loaded"])
        self.assertTrue(detail["details_loaded"])
        self.assertEqual({key: value for key, value in compact.items() if key not in {"cases", "details_loaded"}},
                         {key: value for key, value in detail.items() if key not in {"cases", "details_loaded"}})
        for row in compact["cases"]:
            self.assertEqual(set(row), {"case_key", "run"})
            if row["run"]:
                self.assertEqual(set(row["run"]), {"summary"})
                self.assertNotIn("cleanup_status", row["run"]["summary"])
                self.assertNotIn("termination_reason", row["run"]["summary"])
        scores = self.javascript("""
            import { formalScore } from './web/experiment-metrics.js';
            let input = ''; for await (const chunk of process.stdin) input += chunk;
            console.log(JSON.stringify(JSON.parse(input).map(group => group.cases.map(row =>
                [row.case_key, row.run ? formalScore(row.run.summary) : null, row.run?.summary?.task_success ?? null]))));
        """, [detail, compact])
        self.assertEqual(scores[0], scores[1])
        self.assertEqual([row[1] for row in scores[1]], [0, 1, .4, .2, None, None, None, None,
                                                        None, None, None, None, None, None, None, .8, None])
        for path, status in (("/v1/experiment-groups?view=unknown", 400),
                             ("/v1/experiment-groups/no-such-group", 404)):
            async with self.client.get(self.url + path, headers={"If-None-Match": "*"}) as response:
                self.assertEqual(response.status, status)

    async def test_merged_execution_links_resolve_to_the_same_group(self):
        snapshot = copy.deepcopy(self.library.group_snapshot)
        snapshot["groups"][0]["id_aliases"] = ["group-old-recovery"]
        snapshot["groups"][0]["execution_groups"] = [{"id": self.group_id}, {"id": "group-old-recovery"}]
        with patch.object(self.library.experiment_groups, "snapshot", return_value=snapshot):
            await self.library.refresh()
        current, _ = await self.get(f"/v1/experiment-groups/{self.group_id}")
        previous, previous_tag = await self.get("/v1/experiment-groups/group-old-recovery")
        self.assertEqual(previous, current)
        self.assertEqual(len(current["execution_groups"]), 2)
        async with self.client.get(self.url + "/v1/experiment-groups/group-old-recovery",
                                   headers={"If-None-Match": previous_tag}) as response:
            self.assertEqual(response.status, 304)
        listing, _ = await self.get("/v1/experiment-groups?view=summary")
        self.assertEqual([g["id"] for g in listing["groups"]], [self.group_id])
        self.assertNotIn("execution_groups", listing["groups"][0])
        self.assertEqual(listing["groups"][0]["id_aliases"], ["group-old-recovery"])

    async def test_revalidation_across_refresh_score_changes_and_query_variants(self):
        paths = ["/v1/experiments", "/v1/experiment-groups", "/v1/experiment-groups?view=summary",
                 f"/v1/experiment-groups/{self.group_id}", f"/v1/experiments/{self.identifier}",
                 f"/v1/experiments/{self.identifier}/media"]
        before = {path: await self.get(path) for path in paths}
        await self.library.refresh()
        self.assertNotEqual(before[paths[0]][0]["observed_at"], self.library.refreshed_at)
        for path, (_, etag) in before.items():
            for method in ("GET", "HEAD"):
                async with self.client.request(method, self.url + path, headers={"If-None-Match": etag}) as response:
                    self.assertEqual(response.status, 304, path)
                    self.assertEqual(await response.read(), b"")
                    self.assertEqual(response.headers["ETag"], etag)
                    self.assertEqual(response.headers["Cache-Control"], "private, no-cache")
                    self.assertEqual(response.headers["Vary"], "Accept-Encoding")
            async with self.client.head(self.url + path) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(await response.read(), b"")
                self.assertEqual(response.headers["ETag"], etag)
        _, variant_tag = await self.get("/v1/experiments?limit=50", headers={"If-None-Match": before[paths[0]][1]})
        self.assertNotEqual(variant_tag, before[paths[0]][1])
        for header in ('"no-match", ' + before[paths[0]][1], before[paths[0]][1].removeprefix("W/"), "*"):
            async with self.client.get(self.url + paths[0], headers={"If-None-Match": header}) as response:
                self.assertEqual(response.status, 304)
        for header in ("garbage", before[paths[0]][1].strip('W/"'), '"no-match"', '"*"', 'W/"*"'):
            await self.get(paths[0], headers={"If-None-Match": header})
        atomic_json(self.summary, dict(normalized_score=1, task_success=True, scored_episode=True, valid_episode=True))
        await self.library.refresh()
        for path in paths[:-1]:
            _, tag = await self.get(path, headers={"If-None-Match": before[path][1]})
            self.assertNotEqual(tag, before[path][1])
        # A score-only change does not change the media inventory.
        async with self.client.get(self.url + paths[-1], headers={"If-None-Match": before[paths[-1]][1]}) as response:
            self.assertEqual(response.status, 304)
        _, group_tag = await self.get(paths[3])
        register_run(self.root / "unrelated", category="synthetic", task="other", run_id="other")
        await self.library.refresh()
        async with self.client.get(self.url + paths[3], headers={"If-None-Match": group_tag}) as response:
            self.assertEqual(response.status, 304)

    async def test_media_preserves_camera_aliases_without_reviews_or_full_reports(self):
        for name in ("head.mp4", "recording-a.mp4", "recording-b.mp4", "recording-c.mp4"):
            (self.run / name).write_bytes(b"synthetic local media")
        atomic_json(self.run / "report.json", dict(secret="never emitted", videos=[
            dict(camera_id="head_camera", video_path="/old/recording-a.mp4", diagnostics="never emitted"),
            dict(camera_id="left_wrist_camera", video_path="recording-b.mp4", private="never emitted"),
            dict(camera_id="cam_right_wrist", video_path="recording-c.mp4"), None]))
        atomic_json(self.run / "array-report.json", ["not a camera report"])
        path = f"/v1/experiments/{self.identifier}/media"
        with patch.object(self.library.task_reviews, "get", side_effect=AssertionError("review must not be built")), \
                patch.object(self.library, "detail", side_effect=AssertionError("full detail must not be built")):
            media, etag = await self.get(path)
        self.assertEqual(set(media), {"id", "view_url", "replay_available", "media", "reports"})
        self.assertEqual(len(media["reports"]), 1)
        self.assertEqual(set(media["reports"][0]), {"content"})
        self.assertEqual(set(media["reports"][0]["content"]), {"videos"})
        self.assertNotIn("never emitted", json.dumps(media))
        for descriptor in media["reports"][0]["content"]["videos"]:
            self.assertLessEqual(set(descriptor), {"camera_id", "video_path", "url"})
        cameras = self.javascript("""
            import { cameraVideos } from './web/experiment-media.js';
            globalThis.location = {href: 'http://localhost/', origin: 'http://localhost'};
            let input = ''; for await (const chunk of process.stdin) input += chunk;
            const data = JSON.parse(input); console.log(JSON.stringify(cameraVideos(data.media, data.reports)));
        """, media)
        self.assertEqual({key: value["name"] for key, value in cameras.items()},
                         {"head": "recording-a.mp4", "left": "recording-b.mp4", "right": "recording-c.mp4"})
        (self.run / "extra.mp4").write_bytes(b"new local media")
        _, changed_tag = await self.get(path, headers={"If-None-Match": etag})
        self.assertNotEqual(changed_tag, etag)
        with patch.object(self.library.task_reviews, "get", return_value={"fixture": "full detail retained"}):
            detail, _ = await self.get(f"/v1/experiments/{self.identifier}")
        self.assertEqual(detail["task_review"], {"fixture": "full detail retained"})
        self.assertIn("never emitted", json.dumps(detail["reports"]))

    async def test_conditional_requests_still_validate_origin_host_and_run_paths(self):
        paths = ["/v1/experiments", "/v1/experiment-groups?view=summary",
                 f"/v1/experiment-groups/{self.group_id}", f"/v1/experiments/{self.identifier}",
                 f"/v1/experiments/{self.identifier}/media"]
        for path in paths:
            _, etag = await self.get(path)
            for denied in ({"Origin": "https://other.example"}, {"Host": "other.example"}):
                async with self.client.get(self.url + path, headers={"If-None-Match": etag, **denied}) as response:
                    self.assertEqual(response.status, 403)
        path = paths[-1]
        _, etag = await self.get(path)
        original = self.run.with_name("moved")
        self.run.rename(original)
        self.run.symlink_to(original, target_is_directory=True)
        async with self.client.get(self.url + path, headers={"If-None-Match": etag}) as response:
            self.assertEqual(response.status, 400)
            self.assertIn("symlink_run_directory_denied", await response.text())

    async def test_json_compression_negotiation_and_uncompressed_video_ranges(self):
        path = "/v1/experiment-groups?view=summary"
        tag = None
        async with ClientSession(auto_decompress=False) as client:
            for accepted, expected in (("gzip", "gzip"), ("deflate", "deflate"),
                                       ("gzip;q=0, deflate;q=0.5", "deflate"),
                                       ("deflate;q=0.3, gzip;q=0.8", "gzip"),
                                       ("gzip;q=0, deflate;q=0", None), ("identity", None),
                                       ("gzip;q=0.5, identity;q=1", None), ("br", None), ("*", "gzip")):
                async with client.get(self.url + path, headers={"Accept-Encoding": accepted}) as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.headers.get("Content-Encoding"), expected)
                    body = await response.read()
                    decoded = gzip.decompress(body) if expected == "gzip" else zlib.decompress(body) if expected == "deflate" else body
                    self.assertFalse(json.loads(decoded)["groups"][0]["details_loaded"])
                    if tag is not None: self.assertEqual(response.headers["ETag"], tag)
                    tag = response.headers["ETag"]
            async with client.get(self.url + path, headers={"Accept-Encoding": "*;q=0"}) as response:
                self.assertEqual(response.status, 406)
            (self.run / "head.mp4").write_bytes(b"synthetic-video-bytes")
            media, _ = await self.get(f"/v1/experiments/{self.identifier}/media")
            async with client.get(self.url + media["media"][0]["url"], headers={"Accept-Encoding": "gzip", "Range": "bytes=0-8"}) as response:
                self.assertEqual(response.status, 206)
                self.assertNotIn("Content-Encoding", response.headers)
                self.assertEqual(await response.read(), b"synthetic")


# --- test_library_native_terminal.py ---
class NativeTerminalTransport(unittest.IsolatedAsyncioTestCase):
    async def test_native_restore_preserves_original_and_reconnect_cursor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            run = allocate_run(root, category='synthetic', task='native-display')
            original = b'Error: account/read failed during TUI bootstrap: workspace routing discovery failed'
            (run / 'terminal.bin').write_bytes(original)
            atomic_json(run / 'terminal.json', dict(thread_id='fixture-thread', session_id='old', exit_code=1))
            identity = dict(run_id=run.name, episode_id='fixture-episode', env_instance_id='fixture-env', env_generation=1)
            store = Store(run, identity)
            try:
                store.set_state('runtime', 'codex', dict(thread_id='fixture-thread'))
            finally:
                store.close()
            identifier = catalog_id(run.relative_to(root))
            restored = b'\x1b[32mSynthetic native capture fixture\x1b[0m\r\n'
            cache = root / 'cache.bin'; cache.write_bytes(restored)
            result = dict(path=cache, status=dict(session_id='recovered-fixture', cursor=len(restored), cols=140, rows=40,
                          bootstrap_failed=False, exit_code=0, mode='read_only', replay=dict(kind='recovered_native_tui')))
            class Replay:
                value = None
                calls = 0
                fail = False
                def available(self, evidence, thread): return thread == 'fixture-thread'
                def peek(self, *args): return self.value
                def get(self, *args):
                    self.calls += 1
                    if self.fail: raise RuntimeError('private diagnostics must not reach browser')
                    self.value = result
                    return result
            library = Library(root); replay = Replay(); library.native_terminals = replay
            runner = web.AppRunner(library.app(), shutdown_timeout=.2)
            await runner.setup()
            try:
                site = web.TCPSite(runner, '127.0.0.1', 0); await site.start()
                origin = f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
                endpoint = origin + f'/runs/{identifier}/v1/'
                async with ClientSession() as client:
                    async with client.get(endpoint + 'run') as response:
                        self.assertEqual((await response.json())['terminal']['replay']['kind'], 'native_resume_pending')
                    async with client.ws_connect(endpoint + 'terminal') as socket:
                        self.assertEqual((await socket.receive_json())['type'], 'restoring')
                        ready = await socket.receive_json(); self.assertEqual(ready['status']['session_id'], 'recovered-fixture')
                        data = await socket.receive_json(); self.assertEqual(base64.b64decode(data['data']), restored)
                        self.assertEqual((await socket.receive_json())['type'], 'status')
                        await socket.send_json(dict(type='input', data='unauthorized input'))
                        self.assertEqual((await socket.receive_json())['message'], 'replay_is_read_only')
                    self.assertEqual(replay.calls, 1)
                    async with client.get(endpoint + 'run') as response:
                        self.assertEqual((await response.json())['terminal']['replay']['kind'], 'recovered_native_tui')
                    async with client.ws_connect(endpoint + f'terminal?after={len(restored)}&session_id=recovered-fixture') as socket:
                        self.assertEqual((await socket.receive_json())['type'], 'ready')
                        self.assertEqual((await socket.receive_json())['type'], 'status')
                    async with client.ws_connect(endpoint + 'terminal?after=7&session_id=recovered-fixture') as socket:
                        await socket.receive_json()
                        data = await socket.receive_json(); self.assertEqual(base64.b64decode(data['data']), restored[7:])
                    async with client.get(endpoint + 'terminal-log') as response:
                        self.assertEqual(response.status, 200); self.assertEqual(await response.read(), original)
                    self.assertEqual((run / 'terminal.bin').read_bytes(), original)
                    replay.value = None; replay.fail = True
                    async with client.ws_connect(endpoint + 'terminal') as socket:
                        self.assertEqual((await socket.receive_json())['type'], 'restoring')
                        error = await socket.receive_json(); self.assertEqual(error['type'], 'error'); self.assertTrue(error['terminal'])
                        self.assertNotIn('private diagnostics', str(error))
            finally:
                await runner.cleanup()


# --- test_read_models.py ---
class ReadModelsTest(unittest.TestCase):
    def test_sealed_wal_reads_include_commits_without_source_sidecars(self):
        from ni_rua.apps.evaluation.audit import audit
        from ni_rua.apps.evaluation.report import summarize
        from ni_rua.apps.library.task_review import _recorded_instruction
        directory = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(directory)
        writer = Store(root / 'writer', dict(run_id='wal-archive', episode_id='episode',
            env_instance_id='env', env_generation=0))
        self.addCleanup(writer.close)
        writer.db.execute('PRAGMA wal_autocheckpoint=0')
        reference = writer.put_artifact(b'wal-evidence', 'trace', 'application/octet-stream')
        writer.event('fixture.wal_committed', {})
        writer.save('observation', 'obs', {'task_instruction': 'Use the WAL instruction'})
        writer.save('tool_call', 'call', {'params': {'tool': 'robot_observe'}})
        archive = root / 'archive'
        archive.mkdir()
        for name in ('events.sqlite', 'events.sqlite-wal'):
            shutil.copyfile(writer.root / name, archive / name)
        shutil.copytree(writer.blobs, archive / 'blobs')
        (archive / 'manifest.json').write_text(json.dumps({'run_id': 'wal-archive'}))
        (archive / 'summary.json').write_text(json.dumps({'task_success': None}))

        def fingerprint():
            return {str(p.relative_to(archive)): (p.read_bytes(), p.stat().st_mtime_ns)
                    for p in archive.rglob('*') if p.is_file()}

        before = fingerprint()
        entry = {'path': archive.resolve(), 'metadata': {}, 'item': {'live': False}}
        library = Library(root)
        with library.store(entry) as reader:
            self.assertEqual(reader.events()[0]['type'], 'fixture.wal_committed')
            self.assertEqual(reader.resolve_artifact(reference)[1], b'wal-evidence')
            temporary_database = Path(reader.db.execute('PRAGMA database_list').fetchone()[2])
            self.assertNotEqual(temporary_database.parent, archive)
            with self.assertRaisesRegex(ValueError, 'store_is_read_only'):
                reader.save('observation', 'forbidden', {})
        self.assertFalse(temporary_database.parent.exists())
        self.assertEqual(_recorded_instruction(archive), 'Use the WAL instruction')
        self.assertEqual(summarize(archive)['actor_tools'], {'robot_observe': 1})
        self.assertTrue(audit(archive)['passed'])
        self.assertEqual(fingerprint(), before)
        self.assertFalse((archive / 'events.sqlite-shm').exists())

    def test_sealed_wal_snapshot_rejects_concurrent_change_and_symlink(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(directory)
        writer = Store(root / 'writer', dict(run_id='wal-archive', episode_id='episode',
            env_instance_id='env', env_generation=0))
        self.addCleanup(writer.close)
        writer.event('fixture.committed', {})
        archive = root / 'archive'
        archive.mkdir()
        for name in ('events.sqlite', 'events.sqlite-wal'):
            shutil.copyfile(writer.root / name, archive / name)
        copy = shutil.copyfile

        def changing_copy(source, destination):
            copy(source, destination)
            if Path(source).name.endswith('-wal'):
                with Path(source).open('ab') as stream:
                    stream.write(b'concurrent-change')

        with patch('ni_rua.core.storage.store.shutil.copyfile', side_effect=changing_copy):
            with self.assertRaisesRegex(ValueError, 'archived_database_changed_during_snapshot'):
                with open_read_store(archive):
                    self.fail('unstable evidence must not be read')
        self.assertFalse((archive / 'events.sqlite-shm').exists())
        (archive / 'events.sqlite-wal').unlink()
        (archive / 'events.sqlite-wal').symlink_to(writer.root / 'events.sqlite-wal')
        with self.assertRaisesRegex(ValueError, 'database_symlink_denied'):
            with open_read_store(archive):
                self.fail('symlink evidence must not be read')

    def test_actor_turn_queries_filter_evaluator_and_other_turns_in_sql(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        store = Store(directory, dict(run_id='fixture', episode_id='episode', env_instance_id='env', env_generation=0))
        self.addCleanup(store.close)
        correlation = dict(thread_id='actor-thread', turn_id='actor-turn')
        first = store.event('motion.cost_preview', {'visible': 1}, correlation)
        store.event('observer.evaluated', {'privileged_marker': True}, correlation)
        for kind in ('action.batch_preview', 'program.call_started'):
            store.event(kind, {'privileged_marker': True}, {**correlation, 'turn_id': 'other'})
            store.event(kind, {'privileged_marker': True}, {**correlation, 'thread_id': 'other'})
            with patch.dict(store.identity, episode_id='other'):
                store.event(kind, {'privileged_marker': True}, correlation)
        request = store.event('program.call_started', {'visible': 'request'}, correlation)
        last = store.event('action.batch_preview', {'visible': 2}, correlation)
        decode = json.loads

        def allowed_decode(body):
            self.assertNotIn('privileged_marker', body, 'excluded event reached actor-side Python decoding')
            return decode(body)

        with patch('ni_rua.core.storage.store.json.loads', side_effect=allowed_decode):
            events = store.turn_preview_events(thread_id='actor-thread', turn_id='actor-turn', episode_id='episode')
        self.assertEqual([value['server_seq'] for value in events],
                         [first['server_seq'], request['server_seq'], last['server_seq']])

        item = dict(thread_id='actor-thread', turn_id='actor-turn', type='agentMessage',
                    status='completed', phase='final_answer', text='first final')
        for index, updates in enumerate(({}, {'phase': 'commentary'}, {'type': 'userMessage'},
                {'status': 'running'}, {'thread_id': 'other'}, {'turn_id': 'other'}, {'text': 42},
                {'text': '   '}, {'text': 'last final'})):
            store.set_state('codex_item', str(index), {**item, **updates})
        self.assertEqual(store.completed_final_texts('actor-thread', 'actor-turn'), ['first final', 'last final'])

    def test_record_sequence_keeps_numeric_ids(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        store = Store(directory, dict(run_id='fixture', episode_id='episode', env_instance_id='env', env_generation=0))
        self.addCleanup(store.close)
        receipt = {'status': 'unknown', 'executed_native_steps': 1}
        store.save('action_receipt', 'a2', receipt)
        store.save('action_request', 'a999', receipt)
        for identifier in ('a10', 'a300tail', 'pa999', 'a', 'a²'):
            store.save('action_receipt', identifier, receipt)
        self.assertEqual(store.record_sequence('action_receipt', 'a'), 10)
        self.assertEqual(store.record_sequence('action_request', 'a'), 999)
        self.assertEqual(store.record_sequence('unified_action_plan', 'p'), 0)
        for identifier in ('a2', 'a2', 'a10', None):
            store.event('action.accepted', {}, {'action_id': identifier})
        store.event('action.completed', {}, {'action_id': 'excluded'})
        self.assertEqual(store.accepted_action_count(), 2)

    def test_scene_job_read_preserves_sealed_archive_without_service_startup(self):
        from ni_rua.perception.scene.scene_service import read_scene_job
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = Store(root, {})
            with store.db:
                store.db.execute('CREATE TABLE scene_jobs (id TEXT PRIMARY KEY, body TEXT NOT NULL)')
                store.db.execute('INSERT INTO scene_jobs VALUES (?,?)', ('job', json.dumps({'status': 'failed'})))
            store.close()
            before = (root / 'events.sqlite').read_bytes()
            store = Store(root, {}, read_only=True, immutable=True)
            try:
                with patch('ni_rua.perception.scene.scene_service.SceneService.__init__', side_effect=AssertionError('read cannot start service')):
                    self.assertEqual(read_scene_job(store, 'job'), {'status': 'failed'})
                    with self.assertRaises(KeyError):
                        read_scene_job(store, 'missing')
            finally:
                store.close()
            self.assertEqual((root / 'events.sqlite').read_bytes(), before)
            self.assertFalse((root / 'events.sqlite-wal').exists())
            self.assertFalse((root / 'events.sqlite-shm').exists())

    def test_event_type_queries_keep_global_sequence_and_bound_results(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        store = Store(directory, dict(run_id='fixture', episode_id='episode', env_instance_id='env', env_generation=0))
        self.addCleanup(store.close)
        for kind in ('observer.evaluated', 'action.completed', 'observer.evaluated'):
            store.event(kind, {})
        self.assertEqual([value['server_seq'] for value in store.events_by_type(['observer.evaluated'])], [1, 3])
        self.assertEqual(store.events_by_type(['observer.evaluated'], after=1, newest_first=True, limit=1)[0]['server_seq'], 3)
        with self.assertRaisesRegex(ValueError, 'invalid_event_query'):
            store.events_by_type('observer.evaluated')

    def test_mutable_paging_preserves_updates_and_stable_order(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        store = Store(directory, {})
        self.addCleanup(store.close)
        for index in range(4):
            store.set_state('codex_item', str(index), {'id': str(index), 'text': 'initial'})
        first = store.state_page('codex_item', limit=2)
        store.set_state('codex_item', '0', {'id': '0', 'text': 'updated'})
        second = store.state_page('codex_item', first['cursor'], limit=2)
        third = store.state_page('codex_item', second['cursor'], limit=2)
        self.assertEqual([row['id'] for row in first['entries'] + second['entries'] + third['entries']], ['0', '1', '2', '3', '0'])
        self.assertEqual(first['entries'][0]['position'], third['entries'][0]['position'])
        store.set_state('codex_item', '0', {'id': '0', 'text': 'updated'})
        self.assertEqual(store.state_page('codex_item', third['cursor'])['entries'], [])
        self.assertTrue(store.state_page('codex_item', 9999)['reset'])
        old = store.review_page(before=3, limit=1)
        self.assertEqual(old['items'][0]['id'], '1')
        self.assertTrue(old['has_more'])

    def test_legacy_readonly_projection_does_not_upgrade_or_create_wal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with sqlite3.connect(root / 'events.sqlite') as db:
                db.executescript("CREATE TABLE state(kind TEXT,id TEXT,body TEXT); CREATE TABLE records(kind TEXT,id TEXT,body TEXT); CREATE TABLE events(seq INTEGER,body TEXT);")
                db.execute("INSERT INTO state VALUES ('codex_item','old','{\"id\":\"old\"}')")
            before = (root / 'events.sqlite').read_bytes()
            store = Store(root, {}, read_only=True, immutable=True)
            try:
                page = record_updates(store, {})
                self.assertFalse(page['items']['incremental'])
                self.assertEqual(page['items']['entries'][0]['id'], 'old')
            finally:
                store.close()
            self.assertEqual(before, (root / 'events.sqlite').read_bytes())
            self.assertEqual([p.name for p in root.iterdir()], ['events.sqlite'])

    def test_streamed_digest_cache_detects_same_size_mutation_and_scope(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        store = Store(directory, {})
        self.addCleanup(store.close)
        ref = store.put_artifact(b'original-video', 'video', 'video/mp4', 'observer')
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('whole-file allocation')), patch('hashlib.file_digest', wraps=hashlib.file_digest) as digest:
            _, path = store.artifact_file(ref, ('observer',))
            store.artifact_file(ref, ('observer',))
            self.assertEqual(digest.call_count, 1)
            with self.assertRaisesRegex(ValueError, 'scope_denied'):
                store.artifact_file(ref)
            path.write_bytes(b'corruptd-video')
            with self.assertRaisesRegex(ValueError, 'digest_mismatch'):
                store.artifact_file(ref, ('observer',))


class LibrarySnapshotTest(unittest.IsolatedAsyncioTestCase):
    async def test_failed_background_scan_preserves_snapshot_and_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allocate_run(root, category='synthetic', task='test', run_id='snapshot')
            library = Library(root, scan_interval=.01)
            async with TestClient(TestServer(library.app())) as client:
                original_refresh = library.refresh
                snapshot, observed_at = library.catalog, library.refreshed_at
                recovered = asyncio.Event()
                during_failure = {}
                attempts = 0

                async def refresh():
                    nonlocal attempts
                    attempts += 1
                    if attempts == 1:
                        raise OSError('temporary scan failure')
                    if attempts == 2:
                        during_failure.update(catalog=library.catalog, observed_at=library.refreshed_at,
                                              listing=await (await client.get('/v1/experiments')).json())
                    await original_refresh()
                    recovered.set()

                with patch.object(library, 'refresh', side_effect=refresh):
                    await asyncio.wait_for(recovered.wait(), 2)
                self.assertGreaterEqual(attempts, 2)
                self.assertIs(during_failure['catalog'], snapshot)
                self.assertEqual(during_failure['observed_at'], observed_at)
                self.assertEqual(during_failure['listing']['total'], 1)
                self.assertEqual(during_failure['listing']['errors'][-1]['error'], 'catalog_refresh_failed')
                self.assertIsNone(library.refresh_error)
                self.assertEqual((await (await client.get('/v1/experiments')).json())['errors'], [])

    async def test_gateway_and_scene_use_streaming_media_with_their_original_scopes(self):
        from ni_rua.apps.gateway.workbench import Workbench
        from ni_rua.perception.scene.scene_service import SceneService
        with tempfile.TemporaryDirectory() as directory:
            for kind in ('gateway', 'scene'):
                with self.subTest(kind=kind):
                    root = Path(directory) / kind
                    owner = (Workbench(root, robot=SimpleNamespace(close=lambda: None)) if kind == 'gateway'
                             else SceneService(root))
                    try:
                        internal = owner.store.put_artifact(b'0123456789', 'video', 'video/mp4', 'internal')
                        observer = owner.store.put_artifact(b'observer-video', 'video', 'video/mp4', 'observer')
                        async with TestClient(TestServer(owner.app())) as client:
                            with patch.object(Path, 'read_bytes', side_effect=AssertionError('whole-file allocation')):
                                url = '/v1/artifacts/' + internal.removeprefix('artifact:')
                                self.assertEqual((await client.head(url)).status, 200)
                                response = await client.get(url, headers={'Range': 'bytes=2-4'})
                                self.assertEqual(response.status, 206)
                                self.assertEqual(await response.read(), b'234')
                                response = await client.get('/v1/artifacts/' + observer.removeprefix('artifact:'))
                                self.assertEqual(response.status, 200 if kind == 'gateway' else 422)
                            if kind == 'gateway':
                                response = await client.get('/v1/records')
                                self.assertEqual(response.status, 200)
                                self.assertEqual((await response.json())['schema_version'], '1')
                    finally:
                        if kind == 'gateway':
                            await owner.close()

    async def test_requests_share_background_snapshot_and_stream_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            run = allocate_run(root, category='synthetic', task='test', run_id='snapshot')
            store = Store(run, {})
            ref = store.put_artifact(b'0123456789', 'video', 'video/mp4', 'observer')
            store.close()
            library = Library(root, scan_interval=3600)
            async with TestClient(TestServer(library.app())) as client:
                identifier = catalog_id(run.relative_to(root).as_posix())
                with patch.object(library, 'refresh', side_effect=AssertionError('request must not scan')):
                    for _ in range(3):
                        response = await client.get('/v1/experiments')
                        self.assertEqual(response.status, 200)
                        self.assertEqual((await response.json())['total'], 1)
                    url = f'/runs/{identifier}/v1/artifacts/{ref.removeprefix("artifact:")}'
                    with patch.object(Path, 'read_bytes', side_effect=AssertionError('whole-file allocation')):
                        response = await client.head(url)
                        self.assertEqual(response.status, 200)
                        response = await client.get(url, headers={'Range': 'bytes=2-4'})
                        self.assertEqual(response.status, 206)
                        self.assertEqual(await response.read(), b'234')
                    # Catalog caching must not authorize a directory replaced after publication.
                    held = run.with_name('held-snapshot')
                    run.rename(held)
                    run.symlink_to(held, target_is_directory=True)
                    response = await client.get(f'/runs/{identifier}/v1/records')
                    self.assertEqual(response.status, 400)
                    self.assertIn('symlink', (await response.json())['error'])


# --- test_runtime_config.py ---
class RuntimeConfigTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()

    def store(self):
        store = Store(self.root / "run", dict(run_id="config", episode_id="episode",
                      env_instance_id="environment", env_generation=1))
        self.addCleanup(store.close)
        return store

    def test_explicit_config_reaches_all_runtime_consumers_without_environment_reload(self):
        config = Config.from_dict({
            "run": {"task": "stack_bowls", "gpu": 2, "seed": 8, "layout_id": 3},
            "features": {"tracking_guard": True, "joint_replan": True, "observation_profile": "rgb_proprio_v1"},
            "backend": {"root": str(self.root / "official"), "python": str(self.root / "python")},
            "model": {"name": "fixture-model", "effort": "high", "codex_home": str(self.root / "codex")},
            "images": {"transport_format": "jpeg", "reuse_exact_images": False},
            "library": {"root": str(self.root / "library"), "scan_interval": 3},
        })
        store = self.store()
        with patch.object(Config, "load", side_effect=AssertionError("unexpected environment reload")):
            robot = RobotBackend(config=config)
            session = CodexSession(store.root, store, lambda _: {}, config=config)
            library = Library(config=config)
            review = TaskReviewService(config=config)
        self.assertEqual((robot.gpu_id, robot.seed, robot.layout_id), (2, 8, 3))
        self.assertEqual((robot.root, robot.python), (str(self.root / "official"), str(self.root / "python")))
        self.assertTrue(robot.tracking_guard)
        self.assertTrue(robot.joint_replan)
        self.assertEqual(robot.observation_profile, "rgb_proprio_v1")
        self.assertEqual(robot.image_settings, session.image_settings)
        self.assertFalse(session.image_settings["reuse_exact_images"])
        self.assertEqual((session.model, session.effort), ("fixture-model", "high"))
        self.assertEqual(library.scan_interval, 3)
        self.assertEqual(library.catalog.root, self.root / "library")
        self.assertEqual(library.task_reviews.root, review.root)
        self.assertEqual(review.root, self.root / "official")

    def test_explicit_constructor_arguments_override_config_even_when_false_or_zero(self):
        config = Config.from_dict({"run": {"gpu": 2, "seed": 8},
                                   "features": {"tracking_guard": True, "joint_replan": True},
                                   "model": {"name": "configured-model"}})
        robot = RobotBackend(config=config, gpu_id=0, seed=0, tracking_guard=False, joint_replan=False)
        self.assertEqual((robot.gpu_id, robot.seed), (0, 0))
        self.assertFalse(robot.tracking_guard)
        self.assertFalse(robot.joint_replan)
        store = self.store()
        session = CodexSession(store.root, store, lambda _: {}, model="explicit-model", config=config)
        self.assertEqual(session.model, "explicit-model")

    def test_saved_image_policy_wins_over_new_config_and_legacy_missing_reuse_stays_disabled(self):
        store = self.store()
        old = image_settings({"transport_format": "jpeg", "reuse_exact_images": False})
        old.pop("reuse_exact_images")
        store.set_state("runtime", "image_settings", old)
        config = Config.from_dict({"images": {"transport_format": "webp", "reuse_exact_images": True}})
        resolved = store_image_settings(store, config.images.to_dict())
        self.assertEqual(resolved["transport_format"], "jpeg")
        self.assertFalse(resolved["reuse_exact_images"])
        session = CodexSession(store.root, store, lambda _: {}, config=config)
        self.assertEqual(session.image_settings, resolved)

    def test_backend_explicit_none_flags_remain_invalid(self):
        config = Config.from_dict({})
        for flag in ("tracking_guard", "joint_replan"):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "invalid_" + flag):
                RobotBackend(config=config, **{flag: None})

    def test_visual_helper_codex_home_overrides_ambient_home(self):
        configured = self.root / "configured"
        configured.mkdir()
        (configured / "config.toml").write_text('model_provider="fixture"\n[model_providers.fixture]\nbase_url="https://configured.invalid/v1"\nenv_key="CONFIG_FIXTURE_TOKEN"\n')
        (configured / "auth.json").write_text('{"OPENAI_API_KEY":"synthetic-fixture"}')
        with patch.dict(os.environ, {"CODEX_HOME": str(self.root / "absent")}), \
                patch.object(ResponsesClient, "__init__", return_value=None) as constructor:
            ResponsesClient.from_codex_config(self.root / "helper", codex_home=configured, model="gpt-6.1-sol")
        constructor.assert_called_once_with("https://configured.invalid/v1", "synthetic-fixture",
                                             self.root / "helper", model="gpt-6.1-sol", effort="high")

    def test_geometry_service_and_child_use_frozen_images_without_environment_reload(self):
        settings = Config.from_dict({"images": {"transport_format": "jpeg", "jpeg_quality": 61}}).images.to_dict()
        request = dict(run_id="geometry", episode_id="episode", env_instance_id="environment", env_generation=1)
        connection = MagicMock()
        guard = MagicMock()
        guard.prctl.return_value = 0
        with patch.object(Config, "load", side_effect=AssertionError("unexpected environment reload")):
            service = SceneService(self.root / "scene", image_settings=settings)
            self.addCleanup(service.store.close)
            geometry = GeometryTools(service.scene, settings=service.image_settings)
            self.assertEqual(geometry.image_settings, settings)
            with patch("ni_rua.perception.scene.scene_service.ctypes.CDLL", return_value=guard), \
                    patch.object(GeometryTools, "execute", return_value={"status": "fixture"}):
                _compute(str(service.root), request, connection, os.getppid(), service.image_settings)
        connection.send.assert_called_once_with({"result": {"status": "fixture"}})

    def test_image_settings_reads_unified_file_and_legacy_environment(self):
        path = self.root / "config.json"
        path.write_text(json.dumps({"images": {"webp_quality": 71}}))
        with patch.dict(os.environ, {"NI_RUA_CONFIG": str(path),
                                     "NI_RUA_IMAGE_SETTINGS": '{"webp_quality":30,"reuse_exact_images":false}'}):
            settings = image_settings()
        self.assertEqual(settings["webp_quality"], 71)
        self.assertFalse(settings["reuse_exact_images"])

    def test_library_print_config_applies_cli_overrides_without_starting_server(self):
        path = self.root / "config.json"
        path.write_text(json.dumps({"library": {"port": 8790, "scan_interval": 2,
                                                "root": str(self.root / "catalog")}}))
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch("ni_rua.apps.library.service.serve", side_effect=AssertionError("server started")):
            library_main(["--config", str(path), "--port", "8791", "--print-config"])
        resolved = json.loads(output.getvalue())
        self.assertEqual(resolved["library"]["port"], 8791)
        self.assertEqual(resolved["library"]["scan_interval"], 2)
        self.assertEqual(resolved["library"]["root"], str(self.root / "catalog"))


# --- test_task_review.py ---
TASK_SOURCE = '''
raise RuntimeError("task review must never import task code")
class Fixture:
    def get_score(self):
        self.reward_manager.score([["first"], ["second"], ["last"]], [15, 40, 100], score_mode="transition")
    def run_reward(self):
        self.reward_manager.check(["complete", "arms returned"])
    def gen_instruction(self, env_idx):
        templates = ["Official fixture instruction."]
        return templates
'''
VERSION = {"commit": "commit-one", "diff_sha256": "diff-one"}


def install_fixture(root, task="fixture", source=TASK_SOURCE):
    directory = root / "task/RoboDojo"
    (directory / "tasks").mkdir(parents=True, exist_ok=True)
    (directory / "config").mkdir(parents=True, exist_ok=True)
    (directory / "tasks" / (task + ".py")).write_text(source)
    (directory / "config" / (task + ".yml")).write_text("Rigid: []\n")


def fixture_entry(root, *, task="fixture", summary=None, evaluation=None):
    run = allocate_run(root, category="synthetic", task=task, run_id="task-review")
    manifest = dict(task=task, capabilities=dict(versions=dict(robodojo_commit=VERSION["commit"],
                                                             robodojo_local_diff_sha256=VERSION["diff_sha256"])))
    atomic_json(run / "manifest.json", manifest)
    if summary is not None:
        atomic_json(run / "summary.json", summary)
    if evaluation is not None:
        atomic_json(run / "evaluation.json", dict(result=evaluation))
    metadata = json.loads((run / "experiment.json").read_text())
    return dict(path=run, metadata=metadata, item=dict(task=task, id=catalog_id(run.relative_to(root)),
                                                     status="completed", summary=summary or {}))


class TaskReviewChecks(unittest.TestCase):
    def setUp(self):
        self.base = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.source = self.base / "official"
        install_fixture(self.source)
        self.version = patch("ni_rua.apps.library.task_review.repository_version", return_value=VERSION)
        self.version.start()
        self.addCleanup(self.version.stop)
        self.service = TaskReviewService(self.source)

    def entry(self, **kwargs):
        return fixture_entry(self.base / "runs", **kwargs)

    def test_scored_success_keeps_native_scale_and_unknown_breakdown(self):
        entry = self.entry(summary=dict(scored_episode=True, primary_outcome="scored_success", task_success=True,
                                        normalized_score=1, native_process_score=100, native_reward=1),
                           evaluation=dict(task="fixture", task_success=True, normalized_score=1, native_process_score=100))
        review = self.service.get(entry)
        self.assertEqual(review["description"]["text"], "Official fixture instruction.")
        self.assertEqual([item["score"] for item in review["scoring"]["criteria"]], [15, 40, 100])
        self.assertEqual(review["scoring"]["rule_source"]["status"], "matched")
        self.assertEqual(review["result"]["status"], "scored")
        self.assertEqual(review["result"]["normalized_score"], 1)
        self.assertEqual(review["result"]["native_process_score"], 100)
        self.assertTrue(all(item["recorded_result"] is None and item["evidence_status"] == "not_recorded"
                            for item in review["scoring"]["criteria"]))
        self.assertEqual(review["result"]["breakdown_status"], "not_recorded")
        self.assertTrue(review["read_only"])
        self.assertEqual(review["audience"], "observer_only")

    def test_infrastructure_exclusion_cannot_be_overwritten_by_evaluator_zero(self):
        entry = self.entry(summary=dict(scored_episode=False, primary_outcome="unscored_infrastructure", normalized_score=None,
                                        task_success=None, infrastructure_error="transport failed",
                                        partial_native_result=dict(normalized_score=0, native_process_score=0)),
                           evaluation=dict(task="fixture", task_success=False, normalized_score=0, native_process_score=0, native_reward=0))
        review = self.service.get(entry)
        self.assertEqual(review["result"]["status"], "unscored")
        self.assertIsNone(review["result"]["normalized_score"])
        self.assertIsNone(review["result"]["task_success"])
        self.assertEqual(review["result"]["partial_native_result"]["normalized_score"], 0)
        self.assertEqual(review["result"]["evaluation_normalized_score"], 0)
        self.assertIn("transport failed", " ".join(review["result"]["explanation"]))

    def test_legacy_infrastructure_error_also_remains_unscored(self):
        entry = self.entry(summary=dict(infrastructure_error="worker unavailable", normalized_score=0, task_success=False),
                           evaluation=dict(task="fixture", native_process_score=0, normalized_score=0))
        result = self.service.get(entry)["result"]
        self.assertEqual(result["status"], "unscored")
        self.assertIsNone(result["normalized_score"])

    def test_scored_failure_explains_division_and_evaluator_only_stays_partial(self):
        entry = self.entry(summary=dict(scored_episode=True, primary_outcome="scored_failure", task_success=False,
                                        normalized_score=.4, native_process_score=40, native_reward=0))
        result = self.service.get(entry)["result"]
        self.assertEqual(result["status"], "scored")
        self.assertIn("40 ÷ 100 = 0.4", " ".join(result["explanation"]))
        (entry["path"] / "summary.json").unlink()
        entry["item"]["summary"] = {}
        atomic_json(entry["path"] / "evaluation.json", dict(result=dict(task="fixture", task_success=False, native_process_score=40, normalized_score=.4)))
        result = self.service.get(entry)["result"]
        self.assertEqual(result["status"], "partial")
        self.assertIsNone(result["scored_episode"])

    def test_version_mismatch_and_missing_versions_never_claim_historical_rules(self):
        entry = self.entry()
        manifest = json.loads((entry["path"] / "manifest.json").read_text())
        manifest["capabilities"]["versions"]["robodojo_commit"] = "old"
        atomic_json(entry["path"] / "manifest.json", manifest)
        provenance = self.service.get(entry)["scoring"]["rule_source"]
        self.assertEqual(provenance["status"], "current_source_only")
        self.assertEqual(provenance["recorded"]["commit"], "old")
        atomic_json(entry["path"] / "manifest.json", dict(task="fixture"))
        self.assertEqual(self.service.get(entry)["scoring"]["rule_source"]["status"], "current_source_only")

    def test_recorded_instruction_beats_dynamic_source_template_and_cache_invalidates(self):
        entry = self.entry()
        database = entry["path"] / "events.sqlite"
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE records (kind TEXT, body TEXT)")
            connection.execute("INSERT INTO records VALUES (?,?)", ("observation", json.dumps(dict(task_instruction="Put actual pens into the left basket."))))
        review = self.service.get(entry)
        self.assertEqual(review["description"], dict(text="Put actual pens into the left basket.", source_kind="recorded_observation", status="recorded"))
        review["scoring"]["criteria"].clear()
        self.assertEqual(len(self.service.get(entry)["scoring"]["criteria"]), 3)
        atomic_json(entry["path"] / "summary.json", dict(scored_episode=True, task_success=False, normalized_score=.15, native_process_score=15))
        self.assertEqual(self.service.get(entry)["result"]["normalized_score"], .15)

    def test_variable_gradients_keep_variants_separate(self):
        source = '''
class Task:
    def get_score(self):
        score_lists = {4: [5, 15, 30, 100], 5: [5, 15, 25, 40, 100]}
        self.rm.score_single_env(env_idx, stages, score_lists[n], score_mode="transition")
    def gen_instruction(self, env_idx):
        templates = [f"Arrange {count} digits."]
        return templates
'''
        install_fixture(self.source, "arrange_largest_number", source)
        entry = self.entry(task="arrange_largest_number")
        review = self.service.get(entry)
        self.assertEqual(review["description"]["text"], "Arrange {count} digits.")
        criteria = review["scoring"]["criteria"]
        self.assertEqual([item["score"] for item in criteria if item["variant"] == "4 个数字"], [5, 15, 30, 100])
        self.assertEqual([item["score"] for item in criteria if item["variant"] == "5 个数字"], [5, 15, 25, 40, 100])
        self.assertFalse(review["scoring"]["description_reviewed"])

    def test_binary_and_unknown_tasks_do_not_invent_process_points(self):
        install_fixture(self.source, "binary", "class Binary:\n def run_reward(self):\n  self.rm.check(['complete'])\n")
        entry = self.entry(task="binary", summary=dict(scored_episode=True, task_success=False, normalized_score=0, native_process_score=None))
        review = self.service.get(entry)
        self.assertEqual(review["scoring"]["mode"], "binary_success")
        self.assertTrue(all(item["score"] is None for item in review["scoring"]["criteria"]))
        atomic_json(entry["path"] / "manifest.json", dict(task="unknown_task"))
        review = self.service.get(entry)
        self.assertEqual(review["scoring"]["rule_source"]["status"], "unavailable")
        self.assertEqual(review["scoring"]["criteria"], [])

    def test_foreign_task_evaluation_is_not_assigned_to_this_run(self):
        entry = self.entry(evaluation=dict(task="different", task_success=True, normalized_score=1, native_process_score=100))
        result = self.service.get(entry)["result"]
        self.assertIsNone(result["normalized_score"])
        self.assertIsNone(result["native_process_score"])
        self.assertIn("不一致", result["explanation"][0])


class TaskReviewHTTPChecks(unittest.IsolatedAsyncioTestCase):
    async def test_standalone_current_run_review_refresh_and_bounded_evidence_downloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            install_fixture(base / "official")
            root = base / "standalone"
            root.mkdir()
            atomic_json(root / "manifest.json", dict(task="fixture"))
            # Exercise real HTTP routing without starting a robot, session or writable Store.
            workbench = Workbench.__new__(Workbench)
            workbench.root, workbench.task, workbench.status = root, "fixture", "running"
            workbench.identity = dict(run_id=root.name)
            workbench.config = Config().with_overrides({"library": {"task_source_root": str(base / "official")}})
            with patch("ni_rua.apps.library.task_review.repository_version", return_value=VERSION), \
                    patch("ni_rua.apps.library.experiments.Catalog.scan", side_effect=AssertionError("standalone must not scan")):
                async with TestClient(TestServer(workbench.app())) as client:
                    response = await client.get("/v1/task-review")
                    self.assertEqual(response.status, 200)
                    self.assertEqual((await response.json())["result"]["status"], "pending")
                    atomic_json(root / "summary.json", dict(scored_episode=True, task_success=False, normalized_score=.4, native_process_score=40))
                    atomic_json(root / "evaluation.json", dict(result=dict(task="fixture", native_process_score=40, normalized_score=.4)))
                    before = {path.name: path.read_bytes() for path in root.iterdir()}
                    workbench.status = "replay"
                    review = await (await client.get("/v1/task-review")).json()
                    self.assertEqual(review["result"]["status"], "scored")
                    self.assertEqual(review["result"]["normalized_score"], .4)
                    self.assertEqual(review["audience"], "observer_only")
                    urls = [source["url"] for source in review["sources"] if source.get("url")]
                    self.assertEqual(len(urls), 2)
                    for url in urls:
                        response = await client.get(url)
                        self.assertEqual(response.status, 200)
                        raw = await response.read()
                        self.assertEqual(response.headers["X-Evidence-SHA256"], hashlib.sha256(raw).hexdigest())
                        self.assertIn(raw, before.values())
                        self.assertEqual((await client.head(url)).status, 200)
                        response = await client.get(url, headers={"Range": "bytes=0-15"})
                        self.assertEqual(response.status, 206)
                        self.assertEqual(await response.read(), raw[:16])
                        self.assertEqual((await client.post(url, json={})).status, 405)
                    self.assertEqual((await client.head("/v1/task-review")).status, 200)
                    self.assertEqual((await client.post("/v1/task-review", json={})).status, 405)
                    self.assertEqual((await client.get("/v1/task-review", headers={"X-Ni-Rua-Run-ID": "foreign"})).status, 404)
                    prefix = f"/v1/experiments/{catalog_id('.')}/files/"
                    for name in ("manifest.json", "private.json", "../evaluation.json"):
                        self.assertEqual((await client.get(prefix + hashlib.sha256(name.encode()).hexdigest()[:32])).status, 404)
                    self.assertEqual((await client.get(urls[0].replace(catalog_id('.'), catalog_id('foreign')))).status, 404)
                    self.assertEqual(before, {path.name: path.read_bytes() for path in root.iterdir()})
                    (root / "evaluation.json").unlink()
                    (root / "evaluation.json").symlink_to(root / "manifest.json")
                    self.assertEqual((await client.get(prefix + hashlib.sha256(b"evaluation.json").hexdigest()[:32])).status, 404)

    async def test_detail_and_readonly_run_api_share_review_and_safe_evidence_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            install_fixture(base / "official")
            entry = fixture_entry(base / "library", summary=dict(scored_episode=False, primary_outcome="unscored_infrastructure", normalized_score=None),
                                  evaluation=dict(task="fixture", native_process_score=0, native_reward=0, normalized_score=0))
            with patch("ni_rua.apps.library.task_review.repository_version", return_value=VERSION):
                library = Library(base / "library", task_source_root=base / "official")
                runner = web.AppRunner(library.app(), shutdown_timeout=.2)
                await runner.setup()
                try:
                    site = web.TCPSite(runner, "127.0.0.1", 0)
                    await site.start()
                    origin = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
                    identifier = entry["item"]["id"]
                    route = f"/runs/{identifier}/v1/task-review"
                    async with ClientSession() as client:
                        async with client.get(origin + route) as response:
                            self.assertEqual(response.status, 200)
                            review = await response.json()
                        async with client.get(origin + f"/v1/experiments/{identifier}") as response:
                            detail = await response.json()
                        self.assertEqual(detail["task_review"], review)
                        self.assertEqual(review["result"]["status"], "unscored")
                        for source in review["sources"]:
                            if source.get("url"):
                                self.assertTrue(source["url"].startswith(f"/v1/experiments/{identifier}/files/"))
                                async with client.get(origin + source["url"]) as response:
                                    self.assertEqual(response.status, 200)
                        async with client.post(origin + route, json={}) as response:
                            self.assertEqual(response.status, 403)
                        async with client.head(origin + route) as response:
                            self.assertEqual(response.status, 200)
                        # Empty run observations/latest returns 404 (not 400 TypeError)
                        async with client.get(origin + f"/runs/{identifier}/v1/observations/latest") as response:
                            self.assertEqual(response.status, 404)
                        # Server-side digest mismatch on imported evidence returns 500 (not client 400)
                        summary_asset = f"/v1/experiments/{identifier}/files/{hashlib.sha256(b'summary.json').hexdigest()[:32]}"
                        (entry["path"] / "import-inventory.json").write_text(
                            json.dumps({"files": {"summary.json": "0" * 64}})
                        )
                        async with client.get(origin + summary_asset) as response:
                            self.assertEqual(response.status, 500)
                        # Open SSE stream does not block runner.cleanup() thanks to on_shutdown cancellation
                        sse_resp = await client.get(origin + "/v1/experiments/events")
                        self.assertEqual(sse_resp.status, 200)
                finally:
                    await runner.cleanup()



if __name__ == "__main__":
    unittest.main()
