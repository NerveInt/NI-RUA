"""SQLite WAL events and immutable SHA-256 artifacts. No message broker."""

import hashlib
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import shutil
import tempfile

from ..contracts import SCHEMA_VERSION, canonical, digest, new_id, utc_now


def _file_identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=512)
def _file_digest(path, identity):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if _file_identity(path) != identity:
        raise ValueError("artifact_digest_mismatch")
    return actual


def file_digest(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("artifact_path_outside_store")
    return _file_digest(path, _file_identity(path))


def verified_file(path, sha256, byte_size=None):
    """Stream integrity checks, cached only while the complete file identity holds."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError("artifact_path_outside_store")
    identity = _file_identity(path)
    if byte_size is not None and identity[2] != byte_size:
        raise ValueError("artifact_digest_mismatch")
    if _file_digest(path, identity) != sha256:
        raise ValueError("artifact_digest_mismatch")
    return path


@contextmanager
def open_read_store(root, identity=None, *, live=False):
    """Read sealed WAL evidence without letting SQLite create source sidecars.

    A WAL contains committed records absent from the main database, so simply
    opening the main file as immutable would silently omit evidence. Recover a
    stable copy in a private temporary directory; artifacts stay at their source.
    Live readers retain SQLite's normal concurrent-reader protocol.
    """
    root = Path(root).resolve()
    database, wal = root / "events.sqlite", root / "events.sqlite-wal"
    for path in (database, wal):
        if path.is_symlink():
            raise ValueError("database_symlink_denied")
    if live or not wal.exists():
        store = Store(root, identity or {}, read_only=True, immutable=not live)
        try:
            yield store
        finally:
            store.close()
        return
    try:
        before = {path: _file_identity(path) for path in (database, wal)}
        temp_ctx = tempfile.TemporaryDirectory(prefix="ni-rua-evidence-")
    except FileNotFoundError:
        if not wal.exists() and database.exists():
            store = Store(root, identity or {}, read_only=True, immutable=True)
            try:
                yield store
            finally:
                store.close()
            return
        raise
    with temp_ctx as directory:
        copy_root = Path(directory)
        try:
            for path in before:
                shutil.copyfile(path, copy_root / path.name)
            after = {path: _file_identity(path) for path in (database, wal)}
        except FileNotFoundError:
            if not wal.exists() and database.exists():
                store = Store(root, identity or {}, read_only=True, immutable=True)
                try:
                    yield store
                finally:
                    store.close()
                return
            raise
        if before != after:
            raise ValueError("archived_database_changed_during_snapshot")
        store = Store(copy_root, identity or {}, read_only=True)
        store.root, store.blobs = root, root / "blobs"
        try:
            yield store
        finally:
            store.close()


class StoreClosed(sqlite3.ProgrammingError):
    """``Store`` access after ``Store.close()`` (message ``store_closed``). A ``sqlite3.ProgrammingError``, as
    SQLite itself reports use of a closed connection, so existing handlers keep working."""


class _OpenStore:
    """Context manager of ``Store._open()``: holds ``Store.lock`` and refuses a closed Store."""
    __slots__ = ("store", "method")

    def __init__(self, store, method):
        self.store, self.method = store, method

    def __enter__(self):
        store = self.store
        store.lock.acquire()
        if store.closed:
            store.lock.release()
            store._refuse(self.method)
        return store

    def __exit__(self, *exc):
        self.store.lock.release()
        return False


class Store:
    """One run's SQLite event log, records, state and artifact index.

    Thread safety and lifetime: one ``sqlite3`` connection (``db``, ``check_same_thread=False``) is shared by every
    thread of the process, and every access holds ``lock`` (an ``RLock``). ``close()`` takes the same lock, so it
    waits for an access in flight and never closes the connection under another thread (closing it during a
    statement crashes the interpreter: SIGSEGV). After ``close()`` every method that uses the connection or
    appends to a raw log or writes a blob raises ``StoreClosed`` and leaves the run directory untouched; a late
    writer (for example a background thread that outlived its owner) gets that error instead of touching a released
    connection. Refusals are counted, never silent: ``refused_after_close`` is ``{method name: number of refused
    calls}`` (empty while nothing was refused), and the first refusal prints one stderr line with the run id and the
    method. Both are operator diagnostics of this process; they are not written to the closed run and never reach
    model context. Code that uses ``db`` directly must hold ``lock`` and must not run after ``close()`` (SQLite then
    raises ``sqlite3.ProgrammingError``; such a use is not counted).
    """
    # Backend-neutral transcript rows live in the historical "codex_item" state kind
    # (design 5.1); new code goes through the helpers below so a rename is one constant.
    TRANSCRIPT_KIND = "codex_item"
    RAW_AGENT_CHANNELS = ("native", "mcp", "control")
    # Seconds a statement waits for a database lock held by another connection before it raises ("database is
    # locked"): sqlite3's default timeout, named because the relay's settle bound is derived from it.
    LOCK_WAIT_S = 5.0

    def __init__(self, root, identity, read_only=False, immutable=False):
        self.root = Path(root).resolve()
        self.read_only = read_only
        if not read_only:
            self.root.mkdir(parents=True, exist_ok=True)
        self.blobs = self.root / "blobs"
        if not read_only:
            self.blobs.mkdir(exist_ok=True)
        self.identity = identity
        self.lock = threading.RLock()
        self.closed = False
        self.refused_after_close = {}
        database = self.root / "events.sqlite"
        if database.is_symlink():
            raise ValueError("database_symlink_denied")
        if immutable and (not read_only or (self.root / "events.sqlite-wal").exists()):
            raise ValueError("immutable_requires_sealed_database")
        uri = database.as_uri() + "?mode=ro" + ("&immutable=1" if immutable else "")
        self.db = (sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=self.LOCK_WAIT_S) if read_only
                   else sqlite3.connect(database, check_same_thread=False, timeout=self.LOCK_WAIT_S))
        try:
            if read_only:
                self.db.execute("PRAGMA query_only=ON")
            else:
                self.db.execute("PRAGMA journal_mode=WAL")
                has_updates = self.db.execute("SELECT 1 FROM sqlite_master WHERE name='state_updates'").fetchone()
                self.db.executescript("""
                CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS artifacts (id TEXT PRIMARY KEY, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS records (kind TEXT, id TEXT, digest TEXT, body TEXT, PRIMARY KEY(kind,id));
                CREATE TABLE IF NOT EXISTS ingested (event_id TEXT PRIMARY KEY, digest TEXT NOT NULL, seq INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS state (kind TEXT, id TEXT, body TEXT, PRIMARY KEY(kind,id));
                CREATE TABLE IF NOT EXISTS state_updates (seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL, id TEXT NOT NULL, UNIQUE(kind,id));
                """)
                if not has_updates:
                    with self.db:
                        self.db.execute("INSERT OR IGNORE INTO state_updates(kind,id) SELECT kind,id FROM state ORDER BY rowid")
        except BaseException:
            self.db.close()
            raise
        self.source_instance_id = new_id("gateway")
        self.source_seq = 0

    def _require_writer(self):
        if self.read_only:
            raise ValueError("store_is_read_only")

    def put_artifact(self, data, kind, mime_type, access_scope="actor"):
        self._require_writer()
        if self.closed:   # before the blob write: a closed run directory is never changed (_open re-checks under lock)
            self._refuse("put_artifact")
        sha = hashlib.sha256(data).hexdigest()
        path = self.blobs / sha
        if not path.exists():
            temporary = self.blobs / new_id("tmp")
            try:
                temporary.write_bytes(data)
                os.replace(temporary, path)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        identifier = sha + "-" + hashlib.sha256(canonical([kind, mime_type, access_scope]).encode()).hexdigest()[:16]
        value = dict(artifact_id=identifier, kind=kind, mime_type=mime_type, byte_size=len(data), sha256=sha,
                     storage_key=sha, access_scope=access_scope)
        with self._open("put_artifact"), self.db:
            self.db.execute("INSERT OR IGNORE INTO artifacts VALUES (?,?)", (identifier, canonical(value)))
        return f"artifact:{identifier}"

    def artifact(self, reference):
        if not isinstance(reference, str) or not reference:
            raise ValueError("invalid_artifact_reference")
        identifier = reference.removeprefix("artifact:")
        with self._open("artifact"):
            row = self.db.execute("SELECT body FROM artifacts WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise KeyError(identifier)
        value = json.loads(row[0])
        return value, self.blobs / value["storage_key"]

    def artifacts(self):
        with self._open("artifacts"):
            rows = self.db.execute("SELECT body FROM artifacts ORDER BY rowid").fetchall()
        return [json.loads(row[0]) for row in rows]

    def resolve_artifact(self, reference, allowed_scopes=("actor", "internal")):
        value, path = self._artifact_path(reference, allowed_scopes)
        data = path.read_bytes()
        if len(data) != value["byte_size"] or hashlib.sha256(data).hexdigest() != value["sha256"]:
            raise ValueError("artifact_digest_mismatch")
        return value, data

    def artifact_file(self, reference, allowed_scopes=("actor", "internal")):
        """Validate a file without allocating its bytes (also used for HEAD/Range)."""
        value, path = self._artifact_path(reference, allowed_scopes)
        return value, verified_file(path, value["sha256"], value["byte_size"])

    def _artifact_path(self, reference, allowed_scopes):
        value, path = self.artifact(reference)
        if value["access_scope"] not in allowed_scopes:
            raise ValueError("artifact_access_scope_denied")
        if path.parent != self.blobs or path.name != value["sha256"]:
            raise ValueError("invalid_artifact_storage_key")
        if path.is_symlink() or self.blobs.is_symlink() or path.resolve().parent != self.blobs.resolve():
            raise ValueError("artifact_path_outside_store")
        return value, path

    def set_state(self, kind, identifier, value):
        """Mutable projections; immutable records and events remain the evidence."""
        self._require_writer()
        with self._open("set_state"), self.db:
            content = canonical(value)
            old = self.db.execute("SELECT body FROM state WHERE kind=? AND id=?", (kind, identifier)).fetchone()
            if old and old[0] == content:
                return value
            self.db.execute("INSERT INTO state VALUES (?,?,?) ON CONFLICT(kind,id) DO UPDATE SET body=excluded.body",
                            (kind, identifier, content))
            self.db.execute("INSERT OR REPLACE INTO state_updates(kind,id) VALUES (?,?)", (kind, identifier))
        return value

    def resolve_image(self, reference, allowed_scopes=("actor", "internal")):
        """Explicitly permit a labelled video reconstruction when RGB was pruned."""
        try:
            return self.resolve_artifact(reference, allowed_scopes)
        except FileNotFoundError:
            source, _ = self.artifact(reference)
            mapping = self.read_state("image_archive", source["artifact_id"])
            if mapping is None:
                raise
            from .image_archive import resolve_frame
            return resolve_frame(self, source, mapping)

    def observation_view(self, observation):
        if observation is None:
            return None
        cameras = []
        for camera in observation["cameras"]:
            info, path = self.artifact(camera["rgb_ref"])
            mapping = None if path.exists() else self.read_state("image_archive", info["artifact_id"])
            cameras.append({**camera, **({"image_reconstruction": mapping} if mapping else {})})
        return {**observation, "cameras": cameras}

    def read_state(self, kind, identifier, default=None):
        with self._open("read_state"):
            row = self.db.execute("SELECT body FROM state WHERE kind=? AND id=?", (kind, identifier)).fetchone()
        return json.loads(row[0]) if row else default

    def states(self, kind):
        with self._open("states"):
            rows = self.db.execute("SELECT body FROM state WHERE kind=? ORDER BY rowid", (kind,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def completed_final_texts(self, thread_id, turn_id):
        """Only completed final prose from the requested native actor turn."""
        with self._open("completed_final_texts"):
            rows = self.db.execute(
                "SELECT json_extract(body,'$.text') FROM state WHERE kind='codex_item' "
                "AND json_extract(body,'$.thread_id')=? AND json_extract(body,'$.turn_id')=? "
                "AND json_extract(body,'$.type')='agentMessage' "
                "AND json_extract(body,'$.status')='completed' "
                "AND json_extract(body,'$.phase')='final_answer' ORDER BY rowid",
                (thread_id, turn_id)).fetchall()
        return [text for text, in rows if isinstance(text, str) and text.strip()]

    def save(self, kind, identifier, value):
        self._require_writer()
        content = canonical(value)
        sha = digest(value)
        with self._open("save"), self.db:
            previous = self.db.execute("SELECT digest FROM records WHERE kind=? AND id=?", (kind, identifier)).fetchone()
            if previous and previous[0] != sha:
                raise ValueError("immutable_id_content_conflict")
            self.db.execute("INSERT OR IGNORE INTO records VALUES (?,?,?,?)", (kind, identifier, sha, content))
        return value

    def read(self, kind, identifier):
        with self._open("read"):
            row = self.db.execute("SELECT body FROM records WHERE kind=? AND id=?", (kind, identifier)).fetchone()
        if row is None:
            raise KeyError(identifier)
        return json.loads(row[0])

    def records(self, kind):
        with self._open("records"):
            rows = self.db.execute("SELECT body FROM records WHERE kind=? ORDER BY rowid", (kind,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def record_items(self, kind):
        with self._open("record_items"):
            rows = self.db.execute("SELECT id,body FROM records WHERE kind=? ORDER BY rowid", (kind,)).fetchall()
        return {identifier: json.loads(body) for identifier, body in rows}

    def record_sequence(self, kind, prefix):
        """Last numeric suffix for a record namespace; nonnumeric IDs do not count."""
        with self._open("record_sequence"):
            rows = self.db.execute(
                "SELECT substr(id,?) FROM records WHERE kind=? AND substr(id,1,?)=?",
                (len(prefix) + 1, kind, len(prefix), prefix)).fetchall()
        return max((int(suffix) for suffix, in rows if suffix.isascii() and suffix.isdigit()), default=0)

    def record_page(self, kind, after=0, limit=500):
        return self._page(kind, after, limit, mutable=False)

    def state_page(self, kind, after=0, limit=500):
        return self._page(kind, after, limit, mutable=True)

    def review_page(self, before=None, limit=200):
        """Bounded historical native-item window, with its recorded event links."""
        if before is not None and (type(before) is not int or before < 1):
            raise ValueError("invalid_review_cursor")
        if type(limit) is not int or not 1 <= limit <= 500:
            raise ValueError("invalid_review_limit")
        with self._open("review_page"):
            rows = self.db.execute("SELECT rowid,id,body FROM state WHERE kind='codex_item' AND rowid<? ORDER BY rowid DESC LIMIT ?",
                (before or 9223372036854775807, limit + 1)).fetchall()
            entries = [dict(position=row[0], id=row[1], value=json.loads(row[2])) for row in reversed(rows[:limit])]
            ids = [entry["id"] for entry in entries]
            events = []
            if ids:
                placeholders = ','.join('?' for _ in ids)
                events = self.db.execute(f"SELECT body FROM events WHERE json_extract(body,'$.correlation.item_id') IN ({placeholders}) OR json_extract(body,'$.correlation.tool_call_id') IN ({placeholders}) ORDER BY seq LIMIT 5001", (*ids, *ids)).fetchall()
            selected_events = [json.loads(row[0]) for row in events[:5000]]
            generations = {(event.get('episode_id'), event.get('env_generation')) for event in selected_events}
            videos = []
            if generations:
                videos = self.db.execute("""SELECT body FROM events
                    WHERE json_extract(body,'$.type')='observer.video_exported'
                    AND EXISTS (SELECT 1 FROM json_each(?) AS generation
                        WHERE json_extract(generation.value,'$[0]')=json_extract(events.body,'$.episode_id')
                        AND json_extract(generation.value,'$[1]')=json_extract(events.body,'$.env_generation'))
                    ORDER BY seq DESC LIMIT 32""", (json.dumps(list(generations)),)).fetchall()
        return dict(items=entries, events=selected_events,
                    videos=[json.loads(row[0]) for row in reversed(videos)],
                    before=entries[0]["position"] if entries else None, has_more=len(rows) > limit,
                    events_truncated=len(events) > 5000, context="recorded_item_window")

    def _page(self, kind, after, limit, *, mutable):
        if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("invalid_record_cursor_or_limit")
        with self._open("_page"):
            incremental = not mutable or bool(self.db.execute("SELECT 1 FROM sqlite_master WHERE name='state_updates'").fetchone())
            if mutable and incremental:
                query = "SELECT u.seq,s.id,s.body,s.rowid FROM state_updates u JOIN state s ON s.kind=u.kind AND s.id=u.id WHERE u.kind=?"
                order = "u.seq"
                maximum = self.db.execute("SELECT COALESCE(MAX(seq),0) FROM state_updates WHERE kind=?", (kind,)).fetchone()[0]
            else:
                table = "state" if mutable else "records"
                query = f"SELECT rowid,id,body,rowid FROM {table} WHERE kind=?"
                order = "rowid"
                maximum = self.db.execute(f"SELECT COALESCE(MAX(rowid),0) FROM {table} WHERE kind=?", (kind,)).fetchone()[0]
            reset = after > maximum or (not incremental and after == 0)
            after = 0 if reset else after
            rows = self.db.execute(query + f" AND {order}>? ORDER BY {order} LIMIT ?", (kind, after, limit + 1)).fetchall()
        page = rows[:limit]
        return dict(entries=[dict(id=row[1], value=json.loads(row[2]), position=row[3]) for row in page],
                    cursor=page[-1][0] if page else after, has_more=len(rows) > limit,
                    incremental=incremental, reset=reset)

    def event(self, event_type, payload, correlation=None, world_kind="observed", source="execution_gateway", **timing):
        self._require_writer()
        if set(timing) - {"occurred_at", "clock_domain", "sim_step", "step_domain", "sim_time_s"}:
            raise ValueError("invalid_event_timing_field")
        with self._open("event"):
            next_source_seq = self.source_seq + 1
            event = dict(schema_version=SCHEMA_VERSION, event_id=new_id("evt"), **self.identity,
                         world_kind=world_kind, source=source, source_instance_id=self.source_instance_id,
                         source_seq=next_source_seq, type=event_type, occurred_at=utc_now(), received_at=utc_now(),
                         clock_domain=f'{self.identity["env_instance_id"]}/generation-{self.identity["env_generation"]}',
                         sim_step=None, step_domain=None, sim_time_s=None, correlation=correlation or {}, payload=payload)
            event.update(timing)
            initial_body = canonical(event)
            with self.db:
                cursor = self.db.execute("INSERT INTO events(body) VALUES (?)", (initial_body,))
                event["server_seq"] = cursor.lastrowid
                self.db.execute("UPDATE events SET body=? WHERE seq=?", (canonical(event), cursor.lastrowid))
            self.source_seq = next_source_seq
        return event

    def ingest_event(self, value):
        """Trusted independent outbox: retain source identity, assign global cursor once."""
        self._require_writer()
        body = dict(value)
        body.pop("server_seq", None)
        source_digest = digest(body)
        with self._open("ingest_event"), self.db:
            prior = self.db.execute("SELECT digest,seq FROM ingested WHERE event_id=?", (body["event_id"],)).fetchone()
            if prior:
                if prior[0] != source_digest: raise ValueError("event_id_content_conflict")
                return prior[1]
            body["received_at"] = utc_now()
            cursor = self.db.execute("INSERT INTO events(body) VALUES (?)", (canonical(body),))
            body["server_seq"] = cursor.lastrowid
            self.db.execute("UPDATE events SET body=? WHERE seq=?", (canonical(body), cursor.lastrowid))
            self.db.execute("INSERT INTO ingested VALUES (?,?,?)", (body["event_id"], source_digest, cursor.lastrowid))
            return cursor.lastrowid

    def event_cursor(self):
        with self._open("event_cursor"):
            return self.db.execute("SELECT COALESCE(MAX(seq),0) FROM events").fetchone()[0]

    def accepted_action_count(self):
        with self._open("accepted_action_count"):
            return self.db.execute(
                "SELECT COUNT(DISTINCT json_extract(body,'$.correlation.action_id')) FROM events "
                "WHERE json_extract(body,'$.type')='action.accepted'").fetchone()[0]

    def turn_preview_events(self, *, thread_id, turn_id, episode_id):
        """Select public previews and program requests, never decode evaluator events."""
        with self._open("turn_preview_events"):
            rows = self.db.execute("""SELECT body FROM events
                WHERE json_extract(body,'$.type') IN
                    ('motion.cost_preview', 'action.batch_preview', 'program.call_started')
                AND json_extract(body,'$.episode_id')=?
                AND json_extract(body,'$.correlation.thread_id')=?
                AND json_extract(body,'$.correlation.turn_id')=? ORDER BY seq""",
                (episode_id, thread_id, turn_id)).fetchall()
        return [json.loads(body) for body, in rows]

    def events(self, after=0, limit=500):
        with self._open("events"):
            rows = self.db.execute("SELECT body FROM events WHERE seq>? ORDER BY seq LIMIT ?", (after, limit)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def events_by_type(self, types, *, after=0, limit=500, newest_first=False):
        if isinstance(types, str):
            raise ValueError("invalid_event_query")
        try:
            types = tuple(types)
        except TypeError as error:
            raise ValueError("invalid_event_query") from error
        if (not types or any(not isinstance(value, str) or not value for value in types)
                or type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 1000
                or type(newest_first) is not bool):
            raise ValueError("invalid_event_query")
        placeholders = ','.join('?' for _ in types)
        with self._open("events_by_type"):
            rows = self.db.execute(
                f"SELECT body FROM events WHERE seq>? AND json_extract(body,'$.type') IN ({placeholders}) ORDER BY seq {'DESC' if newest_first else 'ASC'} LIMIT ?",
                (after, *types, limit)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def raw_codex(self, direction, message):
        self._require_writer()
        with self._open("raw_codex"), (self.root / "codex-raw.jsonl").open("a") as stream:
            stream.write(canonical(dict(received_at=utc_now(), direction=direction, message=message)) + "\n")

    def raw_agent(self, direction, message, *, channel):
        """Append one agent-backend wire record to agent-raw.jsonl.

        channel: "native" (SDK messages / SSE events / HTTP calls), "mcp" (MCP JSON-RPC)
        or "control" (host->agent commands such as prompt submission and interrupt).
        """
        if channel not in self.RAW_AGENT_CHANNELS:
            raise ValueError("invalid_agent_raw_channel")
        self._require_writer()
        with self._open("raw_agent"), (self.root / "agent-raw.jsonl").open("a") as stream:
            stream.write(canonical(dict(received_at=utc_now(), direction=direction, channel=channel, message=message)) + "\n")

    def transcript_item(self, identifier, default=None):
        return self.read_state(self.TRANSCRIPT_KIND, identifier, default)

    def set_transcript_item(self, identifier, value):
        return self.set_state(self.TRANSCRIPT_KIND, identifier, value)

    def recorded_task_instruction(self):
        with self._open("recorded_task_instruction"):
            row = self.db.execute("SELECT json_extract(body,'$.task_instruction') FROM records WHERE kind='observation' "
                                  "AND length(json_extract(body,'$.task_instruction'))>0 ORDER BY rowid DESC LIMIT 1").fetchone()
        return row[0] if row else None

    def _open(self, method):
        """``with self._open("<method name>"):`` holds ``lock`` for one access; once the Store is closed it raises
        ``StoreClosed`` and counts the refusal under ``method`` (``_refuse``)."""
        return _OpenStore(self, method)

    def _refuse(self, method):
        """Count one access refused after ``close()`` in ``refused_after_close[method]``, print one stderr line on
        the first refusal of this Store (run id and method name only), and raise ``StoreClosed``."""
        import sys
        with self.lock:
            first = not self.refused_after_close
            self.refused_after_close[method] = self.refused_after_close.get(method, 0) + 1
        if first:
            print(f"ni_rua: warning: run {self.identity.get('run_id')}: the Store was used after it was closed "
                  f"({method}); this and later accesses are refused and counted, not written.",
                  file=sys.stderr, flush=True)
        raise StoreClosed("store_closed")

    def close(self):
        """Close the connection; idempotent and thread-safe. Waits (on ``lock``) for an access in flight in another
        thread, then every later access raises ``StoreClosed`` (see the class doc)."""
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.db.close()
