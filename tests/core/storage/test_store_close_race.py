"""Store lifetime under threads: closing a Store while another thread writes must never crash the process.

Regression for a SIGSEGV: a background model-listing thread (``ModelListingMixin._start_model_catalogue``) recorded
``agent.model_catalogue`` while the run's Store was being closed. The race is made on purpose and runs in a child
process, so a crash is an exit code of the child (-11) and not the end of the test run.
"""
import contextlib
import io
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from ni_rua.core.config import Config
from ni_rua.core.storage.store import Store, StoreClosed, open_read_store
from ni_rua.cortex.agents.model_catalogue import CatalogueModel, ModelCatalogue
from ni_rua.cortex.agents.model_listing import ModelListingMixin
from ni_rua.cortex.agents.codex.session import CodexSession
from support.paths import REPO_ROOT

NI_RUA_TEST_TIER = "component"
IDENTITY = dict(run_id="run", episode_id="ep", env_instance_id="env", env_generation=1)

# A worker thread (first a writer in Store.event, then a reader in Store.events) is stopped inside its SQLite statement
# (progress handler) until the closer thread has asked for the Store (it reached Store.lock) or, on code that closes
# without the lock, until close() returned or 2 s passed. Then the worker finishes its statement. A Store that closes
# under the worker crashes right there.
CHILD = r"""
import faulthandler, sqlite3, sys, tempfile, threading
faulthandler.enable()
from ni_rua.core.storage.store import Store

class WatchedLock:
    # Store.lock with one extra signal: the closer thread is about to wait for it.
    def __init__(self, lock, asked):
        self.lock, self.asked = lock, asked
    def acquire(self, *args, **kwargs):
        if threading.current_thread().name == "closer":
            self.asked.set()
        return self.lock.acquire(*args, **kwargs)
    def release(self):
        self.lock.release()
    def __enter__(self):
        self.acquire()
        return self
    def __exit__(self, *exc):
        self.release()

def park(directory, access):
    # Run access(store) on a worker thread, park it in its first SQLite statement, close from another thread.
    store = Store(directory, dict(run_id="run", episode_id="ep", env_instance_id="env", env_generation=1))
    store.event("seed", {})
    inside, asked, closed, result = threading.Event(), threading.Event(), threading.Event(), {}
    store.lock = WatchedLock(store.lock, asked)

    def progress():
        if threading.current_thread().name == "worker" and not inside.is_set():
            inside.set()
            asked.wait(2.0)
            result["closed_during_access"] = closed.is_set()
        return 0

    def work():
        try:
            result["value"] = access(store)
        except Exception as error:
            result["error"] = repr(error)

    def close():
        store.close()
        closed.set()

    store.db.set_progress_handler(progress, 1)
    worker = threading.Thread(target=work, name="worker")
    closer = threading.Thread(target=close, name="closer")
    worker.start()
    assert inside.wait(10), "the worker never reached its statement"
    closer.start()
    worker.join(30); closer.join(30)
    assert not worker.is_alive() and not closer.is_alive(), "deadlock"
    assert result.get("closed_during_access") is False, result
    assert "error" not in result, result
    assert closed.is_set() and store.closed
    return store, result["value"]

with tempfile.TemporaryDirectory() as directory:
    store, events = park(directory, lambda store: store.events())
    assert [event["type"] for event in events] == ["seed"], events
with tempfile.TemporaryDirectory() as directory:
    store, event = park(directory, lambda store: store.event("probe", {"n": 1}))
    assert event["server_seq"] == 2, event
    for late in (lambda: store.event("late", {}), lambda: store.set_state("k", "i", {}), lambda: store.events(),
                 lambda: store.raw_agent("in", {}, channel="native")):
        try:
            late()
        except sqlite3.ProgrammingError as error:
            assert str(error) == "store_closed", error
        else:
            raise AssertionError("a closed Store accepted an access")
    assert store.refused_after_close == {"event": 1, "set_state": 1, "events": 1, "raw_agent": 1}
    store.close()   # idempotent
print("survived")
"""


class ListingStub(ModelListingMixin):
    """The listing mixin alone: ``_read_catalogue`` blocks until ``release`` is set."""
    backend_kind = "stub"
    wire_api = "anthropic_messages"
    model_catalogue_sources = ("explicit",)

    def __init__(self, store, model="model-a"):
        self.store, self.config = store, Config.from_dict({"model": {"name": model}})
        self.reading, self.release = threading.Event(), threading.Event()

    def _actor_upstream(self):
        return SimpleNamespace(source="explicit")

    def _read_catalogue(self, base_url, headers, wire_api):
        self.reading.set()
        self.release.wait(10)
        return ModelCatalogue("ok", None, (CatalogueModel("model-a", "Model A", image_input=False),)), {}


class StoreCloseRaceTests(unittest.TestCase):
    def test_close_during_a_write_in_another_thread_does_not_crash_the_process(self):
        child = subprocess.run([sys.executable, "-c", CHILD], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=120)
        self.assertEqual(child.returncode, 0, f"child exit {child.returncode} (-11 = SIGSEGV)\n{child.stderr[-3000:]}")
        self.assertEqual(child.stdout.strip(), "survived")

    def test_closed_store_refuses_every_access_and_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory, IDENTITY)
            store.event("before", {})
            store.put_artifact(b"kept", "k", "text/plain")
            self.assertEqual(store.refused_after_close, {})
            store.close()
            store.close()
            files = sorted(str(path.relative_to(directory)) for path in Path(directory).rglob("*"))
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                for access in (lambda: store.event("after", {}), lambda: store.save("k", "i", {}),
                               lambda: store.read_state("k", "i"), lambda: store.event_cursor(),
                               lambda: store.put_artifact(b"late", "k", "text/plain"),
                               lambda: store.put_artifact(b"late", "k", "text/plain"),
                               lambda: store.raw_codex("in", {})):
                    with self.assertRaisesRegex(StoreClosed, "store_closed"):
                        access()
            # Refusals are counted per method, and announced once (run id and method only).
            self.assertEqual(store.refused_after_close, dict(event=1, save=1, read_state=1, event_cursor=1,
                                                             put_artifact=2, raw_codex=1))
            self.assertEqual(stderr.getvalue().count("\n"), 1)
            self.assertIn("run run: the Store was used after it was closed (event)", stderr.getvalue())
            # A closed run directory is never changed: no late blob, no raw log.
            self.assertEqual(sorted(str(path.relative_to(directory)) for path in Path(directory).rglob("*")), files)
            self.assertEqual(len(list((Path(directory) / "blobs").iterdir())), 1)
            with open_read_store(directory) as archive:
                self.assertEqual([event["type"] for event in archive.events()], ["before"])
            self.assertFalse((Path(directory) / "codex-raw.jsonl").exists())


class ListingLifetimeTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(directory)
        self.store = Store(self.root, IDENTITY)
        self.addCleanup(self.store.close)

    def finish(self, session, thread):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            session.release.set()
            thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertTrue(session._catalogue_done.is_set())
        return stderr.getvalue()

    def recorded(self):
        with open_read_store(self.root) as archive:
            return [event["type"] for event in archive.events()]

    def test_listing_that_ends_after_the_session_closed_is_dropped_and_reported(self):
        session = ListingStub(self.store)
        thread = session._start_model_catalogue("http://127.0.0.1:9", {})
        self.assertTrue(session.reading.wait(10))
        session._close_model_catalogue()
        self.store.close()
        stderr = self.finish(session, thread)
        self.assertTrue(session._catalogue_dropped)
        self.assertIsNone(session._catalogue)
        self.assertIsNone(session.model_options())
        self.assertIn("ended after the session was closed; its result was dropped and not recorded", stderr)
        self.assertNotIn("image input", stderr)
        self.assertEqual(self.store.refused_after_close, {})   # the detached thread never reached the Store
        self.assertEqual(self.recorded(), [])
        self.assertIsNone(session._start_model_catalogue("http://127.0.0.1:9", {}))

    def test_store_closed_without_closing_the_session_drops_the_record_honestly(self):
        session = ListingStub(self.store)
        thread = session._start_model_catalogue("http://127.0.0.1:9", {})
        self.assertTrue(session.reading.wait(10))
        self.store.close()
        stderr = self.finish(session, thread)
        self.assertTrue(session._catalogue_dropped)
        self.assertIn("ended after the run's Store was closed; its record was dropped (the list is kept", stderr)
        self.assertIn("without image input", stderr)   # the operator warning does not depend on the record
        self.assertEqual(session.model_options(), [dict(value="model-a", name="Model A", image_input=False)])
        self.assertEqual(self.store.refused_after_close, {"event": 1})
        self.assertEqual(self.recorded(), [])

    def test_session_closed_and_started_again_keeps_a_listing_that_is_still_being_read(self):
        # CodexSession.start / start_forked restart one session object: close(), then closing = False.
        session = ListingStub(self.store)
        thread = session._start_model_catalogue("http://127.0.0.1:9", {})
        self.assertTrue(session.reading.wait(10))
        self.assertTrue(session._close_model_catalogue())    # a read was pending
        session._reopen_model_catalogue()
        stderr = self.finish(session, thread)
        self.assertFalse(session._catalogue_dropped)
        self.assertNotIn("dropped", stderr)
        self.assertEqual(session.model_options(), [dict(value="model-a", name="Model A", image_input=False)])
        self.assertEqual(self.store.events_by_type(["agent.model_catalogue"])[0]["payload"]["status"], "ok")
        self.assertFalse(session._close_model_catalogue())   # nothing pending any more

    def test_codex_restart_reopens_the_listing_where_it_reopens_the_session(self):
        import inspect
        from ni_rua.apps.evaluation.checkpoint import ForkedCodexSession
        for restart in (CodexSession.start, ForkedCodexSession.start_forked):
            lines = [line.strip() for line in inspect.getsource(restart).splitlines()]
            reopened = lines.index("self.closing = False")
            self.assertTrue(lines[reopened + 1].startswith("self._reopen_model_catalogue()"), restart.__qualname__)
            self.assertLess(lines.index("self.close()"), reopened)

    def test_listing_that_ends_while_the_session_is_open_is_kept_and_recorded(self):
        session = ListingStub(self.store)
        thread = session._start_model_catalogue("http://127.0.0.1:9", {})
        stderr = self.finish(session, thread)
        self.assertFalse(session._catalogue_dropped)
        self.assertEqual(session.model_options(), [dict(value="model-a", name="Model A", image_input=False)])
        self.assertIn("without image input", stderr)
        self.assertEqual(self.store.events_by_type(["agent.model_catalogue"])[0]["payload"]["status"], "ok")
        self.assertEqual(len(self.store.events_by_type(["agent.model_image_input_unsupported"])), 1)
        session._close_model_catalogue()
        self.assertFalse(session._catalogue_dropped)

    def test_nothing_is_read_after_close(self):
        session = ListingStub(self.store)
        session._close_model_catalogue()
        self.assertIsNone(session._start_model_catalogue("http://127.0.0.1:9", {}))
        self.assertIsNone(session._catalogue_done)

    def test_codex_and_mcp_session_close_detach_the_listing(self):
        from support.agent_fixtures import StubSession
        home = self.root / "codex-home"
        home.mkdir()
        (home / "config.toml").write_text('model_provider = "fixture"\n[model_providers.fixture]\nname = "fixture"\n'
                                          'base_url = "http://127.0.0.1:1"\n')
        codex = CodexSession(self.root, self.store, lambda value: {},
                             config=Config.from_dict({"model": {"codex_home": str(home)}}))
        mcp = StubSession(self.root, self.store, None)
        for session in (codex, mcp):
            self.assertFalse(session._catalogue_closed)
            session.close()
            self.assertTrue(session._catalogue_closed)
        # A close with nothing pending and no thread left states nothing extra.
        self.assertEqual(codex.close_report, dict(threads_alive=[], model_listing_pending=False))
        self.assertFalse(mcp.close_report["model_listing_pending"])
        self.assertEqual(self.store.events_by_type(["codex.session_close_incomplete"]), [])

    def test_close_reports_record_reader_threads_and_a_listing_that_outlive_the_session(self):
        from support.agent_fixtures import StubSession
        release = threading.Event()
        self.addCleanup(release.set)
        threads = [threading.Thread(target=release.wait, args=(30,), daemon=True) for _ in range(2)]
        for thread in threads:
            thread.start()

        class Client:   # the two SDK threads CodexSession.close looks at, alive after the SDK's bounded join
            _proc, _reader_thread, _stderr_thread = None, threads[0], threads[1]

            def close(self):
                pass

        codex = CodexSession(self.root, self.store, lambda value: {}, config=Config())
        codex.client = Client()
        codex._catalogue_done = threading.Event()   # a listing read that has not ended
        codex.close()
        self.assertEqual(codex.close_report, dict(threads_alive=["codex-reader", "codex-stderr"],
                                                  model_listing_pending=True))
        event, = self.store.events_by_type(["codex.session_close_incomplete"])
        self.assertEqual((event["payload"], event["source"]), (codex.close_report, "agent_backend"))

        class Backend(StubSession):
            def _stop_backend(self):
                self.close_report["backend"] = dict(threads_alive=["reader-a"])

        mcp = Backend(self.root, self.store, None)
        mcp._catalogue_done = threading.Event()
        report = mcp.close()
        self.assertTrue(report["model_listing_pending"])
        payload = self.store.events_by_type(["agent.session_closed"])[0]["payload"]
        self.assertEqual((payload["model_listing_pending"], payload["threads_alive"]), (True, ["reader-a"]))
        clean = StubSession(self.root, self.store, None)
        clean.close()
        payload = self.store.events_by_type(["agent.session_closed"])[1]["payload"]
        self.assertEqual({"model_listing_pending", "threads_alive"} & set(payload), set())

    def test_store_init_rollback_observation_perception_close_guard_source_seq_and_atomic_cleanup(self):
        import sqlite3
        from unittest.mock import patch
        from ni_rua.core.storage.read_models import observation_perception
        from ni_rua.core.storage.store import StoreClosed, open_read_store

        # DEF-15 #2: observation_perception enforces StoreClosed and increments refused_after_close on event fallback too.
        closed_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        closed_store = Store(closed_dir, dict(run_id="r", episode_id="e", env_instance_id="i", env_generation=1))
        closed_store.close()
        with patch.object(closed_store, "read_state", return_value=None):
            with self.assertRaises(StoreClosed):
                observation_perception(closed_store, "obs-1")
        self.assertEqual(closed_store.refused_after_close.get("observation_perception"), 1)

        # DEF-15 #3: Store.event does not advance source_seq when canonical() rejects NaN payload.
        seq_before = self.store.source_seq
        with self.assertRaises(ValueError):
            self.store.event("test.nan", {"bad": float("nan")})
        self.assertEqual(self.store.source_seq, seq_before)
        ok_evt = self.store.event("test.ok", {"ok": 1})
        self.assertEqual(ok_evt["source_seq"], seq_before + 1)

        # DEF-16 #1: Store.put_artifact unlinks temporary blob file when os.replace fails.
        with patch("ni_rua.core.storage.store.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.store.put_artifact(b"transient-bytes-xyz", "test", "application/octet-stream")
        leftovers = [p.name for p in self.store.blobs.iterdir() if p.name.startswith("tmp")]
        self.assertEqual(leftovers, [])

        # DEF-15 #1: Store.__init__ closes sqlite connection if post-connect schema setup raises.
        init_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        closed_conns = []
        real_connect = sqlite3.connect

        class TrackingConn:
            def __init__(self, inner):
                self._inner = inner

            def execute(self, *args, **kwargs):
                raise sqlite3.OperationalError("simulated pragma failure")

            def close(self):
                closed_conns.append(True)
                self._inner.close()

        with patch("ni_rua.core.storage.store.sqlite3.connect", side_effect=lambda *a, **kw: TrackingConn(real_connect(*a, **kw))):
            with self.assertRaises(sqlite3.OperationalError):
                Store(init_dir, dict(run_id="r", episode_id="e", env_instance_id="i", env_generation=1))
        self.assertEqual(closed_conns, [True])

        # P2-1: open_read_store falls back cleanly if WAL disappears concurrently during checkpoint.
        snap_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        snap_store = Store(snap_dir, dict(run_id="r", episode_id="e", env_instance_id="i", env_generation=1))
        snap_store.save("observation", "obs-snap", {"observation_id": "obs-snap"})
        snap_store.close()
        wal_path = snap_dir / "events.sqlite-wal"
        real_exists = Path.exists
        first_wal_check = {"count": 0}

        def raced_exists(p):
            if p == wal_path and first_wal_check["count"] == 0:
                first_wal_check["count"] += 1
                return True
            return real_exists(p)

        with patch.object(Path, "exists", raced_exists):
            with open_read_store(snap_dir) as reader:
                self.assertEqual(reader.read("observation", "obs-snap")["observation_id"], "obs-snap")


if __name__ == "__main__":
    unittest.main()
