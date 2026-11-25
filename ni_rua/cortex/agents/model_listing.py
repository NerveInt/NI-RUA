"""Provider model listing of an agent session: operator suggestions for the console's model control.

One code path for every backend (design 16 §3.3 / §6, WP-C): ``ModelListingMixin`` reads the upstream's model listing
with the host credentials once at session start on a daemon thread and offers it through
``AgentSession.model_options``; the MCP backends (``McpSessionMixin``) inherit it, and ``CodexModelListingMixin`` adapts
it to Codex (``agent.codex_upstream`` profile or the ``model.codex_home`` provider). Nothing here is model context: the
listing, the ``agent.model_catalogue`` event and ``upstream_profile_block`` are operator / observer data, and no
credential, header or URL is ever recorded.

Where the listing comes from, by the actor upstream's source (``model_catalogue_sources`` names the sources read):

* ``profile`` (a NI-RUA provider profile, ``ni_rua/cortex/agents/profiles.py``): through a ``CatalogueCache`` keyed by
  (profile id, wire), at the wire's ``models_url`` when the profile names one (else ``model_catalogue.models_url``).
  The cache is ``catalogue_cache`` when a caller set it, else the process-wide ``shared_catalogue_cache`` of the
  profile file's directory (``<profiles_dir>/.cache``), so the gateway's console listing and every session of the
  process share one in-flight read, the 600 s freshness window and the disk copy.
* every other source a backend reads (``claude_settings``, ``opencode_config``, ``explicit``, Codex ``codex_home``):
  one direct ``model_catalogue.fetch_upstream_catalogue`` read, exactly as before profiles existed.

A failed read never fails a start: it is a ``ModelCatalogue`` with ``status="unavailable"`` and a short reason.

Lifetime: the session owns the read. ``AgentSession.close`` calls ``_close_model_catalogue``, after which the thread
never touches the session's Store: a listing that ends later is dropped (not kept, not recorded) and reported once on
the gateway's stderr and in ``_catalogue_dropped``. A session object that is started again (``CodexSession.start``)
calls ``_reopen_model_catalogue`` and keeps a read that is still running. For a ``profile`` source the shared
``CatalogueCache`` still keeps a dropped answer for the console and for successor sessions (it belongs to the process,
not to the run); a direct read (every other source) is not cached, so a dropped answer is gone.
"""
import os
import threading
from pathlib import Path

__all__ = ["ModelListingMixin", "CodexModelListingMixin", "shared_catalogue_cache"]

_SHARED_CACHES = {}
_SHARED_LOCK = threading.Lock()


def shared_catalogue_cache(directory):
    """The process-wide ``model_catalogue.CatalogueCache`` of the cache directory ``directory`` (normally
    ``<profiles_dir>/.cache``, ``model_catalogue.CACHE_DIR_NAME``), created on first use with the cache defaults (600 s
    freshness, 5 s per-page timeout, 10 pages). Every caller passing the same directory (after ``~`` expansion and
    resolution) gets the same instance, so a session and the gateway console never read one listing twice within
    the freshness window. Thread-safe."""
    from .model_catalogue import CatalogueCache
    key = str(Path(directory).expanduser().resolve())
    with _SHARED_LOCK:
        cache = _SHARED_CACHES.get(key)
        if cache is None:
            cache = _SHARED_CACHES[key] = CatalogueCache(Path(key))
        return cache


def _mtime_ns(path):
    """``st_mtime_ns`` of ``path`` (the profile version token of the cache, as ``ProfileInfo.mtime_ns``), or None."""
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


class ModelListingMixin:
    """Start-time provider model listing and the console suggestions built from it (see the module doc).

    The session provides: ``config`` (``Config``), ``store`` (run Store), ``backend_kind``, ``wire_api`` (the relay
    wire, set before ``_start_model_catalogue``) and ``_actor_upstream()`` (the actor ``UpstreamConfig``, None = a
    backend's own default source). A backend that resolves a ``source="profile"`` upstream at start sets ``_profile``
    (the loaded ``profiles.Profile``), so the listing, the forwarded variables and the provenance use the very file
    version the credential came from.

    Class attributes: ``model_catalogue_sources`` (source names whose listing is read; () = never),
    ``model_catalogue_timeout_s`` (per-page timeout in s of a direct read; profile reads use the cache's own timeout,
    also 5 s by default), ``catalogue_cache`` (``CatalogueCache`` for profile listings; None = ``shared_catalogue_cache``
    of ``<profile file directory>/.cache``). State: ``_catalogue`` (the ``ModelCatalogue`` once read, also a failed
    read), ``_catalogue_done`` (Event set when the read ended; None when no read was started),
    ``_catalogue_closed`` (True from ``_close_model_catalogue`` until ``_reopen_model_catalogue``: no result is kept
    or recorded meanwhile), ``_catalogue_dropped`` (True when a read ended meanwhile and its result was dropped, or
    when the run's Store was already closed and only the record was dropped).
    """

    model_catalogue_sources = ()
    model_catalogue_timeout_s = 5.0
    catalogue_cache = None
    _catalogue = None
    _catalogue_done = None
    _catalogue_closed = False
    _catalogue_dropped = False
    _profile = None

    # ---- backend hooks ----------------------------------------------------------------------------------------------
    def _actor_upstream(self):
        """Backend hook: the ``UpstreamConfig`` of the actor (``agent.<backend>.upstream``), or None."""
        raise NotImplementedError

    def _catalogue_source(self):
        """Name of the credential source the listing would come from: the actor upstream's ``source``; None when the
        backend has no upstream section (a backend overrides this for its own default source)."""
        upstream = self._actor_upstream()
        return None if upstream is None else upstream.source

    # ---- provider profile -------------------------------------------------------------------------------------------
    def _upstream_profile(self):
        """The ``profiles.Profile`` of the ``source="profile"`` actor upstream: ``_profile`` as resolved at start,
        else loaded from ``upstream.settings`` now (and kept). Raises ``ValueError("profile_<code>:...")`` when the
        file cannot be loaded; never reads the key."""
        if self._profile is None:
            from .profiles import load_profile
            self._profile = load_profile(self._actor_upstream().settings)
        return self._profile

    def upstream_profile_block(self):
        """Manifest ``agent_backend.upstream_profile`` (design 16 §7.2) for a ``source="profile"`` actor upstream, else
        None: ``{"id": profile id, "name": display name, "wire_api": the wire this session runs on, "wires": every wire
        the profile serves (sorted)}``. Names only: never the key, a header or a credential variable value. Before the
        profile was resolved, ``id`` is ``upstream.provider`` and ``name`` / ``wires`` are None."""
        upstream = self._actor_upstream()
        if upstream is None or upstream.source != "profile":
            return None
        profile = self._profile
        return dict(id=profile.id if profile is not None else upstream.provider,
                    name=profile.name if profile is not None else None,
                    wire_api=getattr(self, "wire_api", None) or upstream.wire_api,
                    wires=sorted(profile.wires) if profile is not None else None)

    # ---- listing ----------------------------------------------------------------------------------------------------
    def _read_catalogue(self, base_url, headers, wire_api):
        """``(ModelCatalogue, extra record fields)`` of one listing read (blocking; see the module doc). ``base_url`` /
        ``headers``: the relay upstream and host credential headers. Profile reads return the cache's view as a
        ``ModelCatalogue`` (a failed read whose earlier list the cache still holds counts as ``unavailable`` here: the
        session never runs on a stale list) and ``extra`` = ``{"source": "profile", "profile": id, "origin":
        network|shared|memory|disk, "fetched_at": UTC or None, "stale": bool}``; direct reads give ``{}``."""
        from .model_catalogue import CACHE_DIR_NAME, ModelCatalogue, fetch_upstream_catalogue
        if self._catalogue_source() != "profile":
            return fetch_upstream_catalogue(base_url, headers, wire_api=wire_api,
                                            timeout_s=self.model_catalogue_timeout_s), {}
        profile = self._upstream_profile()
        cache = self.catalogue_cache or shared_catalogue_cache(profile.path.parent / CACHE_DIR_NAME)
        wire = profile.wires.get(wire_api)
        view = cache.read(profile.id, wire_api, credential=lambda: (base_url, headers),
                          profile_mtime=_mtime_ns(profile.path), listing_url=wire.models_url if wire else None)
        catalogue = ModelCatalogue(view.status, view.reason, () if view.stale else tuple(view.models), view.pages,
                                   view.auth)
        return catalogue, dict(source="profile", profile=profile.id, origin=view.origin, fetched_at=view.fetched_at,
                               stale=view.stale)

    def _start_model_catalogue(self, base_url, headers):
        """Read the provider's model listing on a daemon thread (``_read_catalogue`` with the relay upstream
        ``base_url`` and host credential ``headers``), only when ``_catalogue_source()`` is one of
        ``model_catalogue_sources`` and no read was started before (once per session). Never fails the start: the
        result (also a failure) is kept in ``_catalogue`` and recorded once as ``agent.model_catalogue`` (status,
        reason, the usable models with their limits and image flag, plus the profile fields of ``_read_catalogue``;
        observer data, never model context); ``_catalogue_done`` (Event) is set when the read ended. Nothing is
        started after ``_close_model_catalogue``; a read that ends after it is dropped (``_keep_catalogue``).
        Returns the thread, or None when nothing is read."""
        from .model_catalogue import ModelCatalogue
        if self._catalogue_source() not in self.model_catalogue_sources or self._catalogue_done is not None:
            return None
        with self._catalogue_guard():
            if self._catalogue_closed:
                return None
            done = self._catalogue_done = threading.Event()
        wire_api = self.wire_api

        def read():
            catalogue, extra = None, {}
            try:
                try:
                    catalogue, extra = self._read_catalogue(base_url, headers, wire_api)
                except Exception as error:  # noqa: BLE001 - a listing failure is a status, never a start failure
                    catalogue = ModelCatalogue("unavailable", "listing_failed:" + type(error).__name__)
                self._keep_catalogue(catalogue, extra, wire_api, done)
            finally:
                done.set()
        thread = threading.Thread(target=read, name=f"{self.backend_kind}-model-catalogue", daemon=True)
        thread.start()
        return thread

    def _catalogue_guard(self):
        """The session's lock (``threading.Lock``, created on first use) that serialises keeping / recording a
        listing result against ``_close_model_catalogue``."""
        return self.__dict__.setdefault("_catalogue_lock", threading.Lock())

    def _keep_catalogue(self, catalogue, extra, wire_api, done):
        """End of the listing thread: keep ``catalogue`` in ``_catalogue``, set ``done`` and record it
        (``agent.model_catalogue`` with ``extra``, and ``agent.model_image_input_unsupported`` when
        ``_image_input_unsupported``), under ``_catalogue_guard`` so a concurrent ``_close_model_catalogue`` waits
        for these Store writes instead of racing them. The stderr lines are printed after the guard is released.
        While the session is closed the result is dropped: ``_catalogue`` and the Store stay untouched,
        ``_catalogue_dropped`` becomes True and one stderr line says so. When the session is open but the run's
        Store is already closed, the list is kept for the console and only its record is dropped (same flag, its
        own stderr line)."""
        import sys
        from ...core.storage.store import StoreClosed
        lines, model = [], None
        with self._catalogue_guard():
            if self._catalogue_closed:
                self._catalogue_dropped = True
                lines.append(f"ni-rua: the provider model listing ({catalogue.status}) ended after the session was "
                             "closed; its result was dropped and not recorded.")
            else:
                self._catalogue = catalogue
                done.set()
                model = self._image_input_unsupported(catalogue)
                try:
                    self._record_listing_event("agent.model_catalogue",
                                               lambda: dict(catalogue.record(wire_api), **extra))
                    if model is not None:
                        self._record_listing_event("agent.model_image_input_unsupported",
                                                   lambda: dict(model=model, source="provider_model_catalogue"))
                except StoreClosed:
                    self._catalogue_dropped = True
                    lines.append(f"ni-rua: the provider model listing ({catalogue.status}) ended after the run's "
                                 "Store was closed; its record was dropped (the list is kept for the console).")
        if model is not None:
            lines.append(f"ni-rua: warning: the provider lists model {model} without image input (image_input: "
                         "false); observation images will likely be rejected. Switch models in the console if "
                         "needed.")
        for line in lines:
            print(line, file=sys.stderr, flush=True)

    def _record_listing_event(self, event_type, build):
        """One observer event of the listing thread (``source="agent_backend"``; never model context). ``build``:
        zero-argument callable returning the payload dict. Raises ``StoreClosed`` when the run's Store is closed;
        any other failure (building or writing) is reported on stderr (event type and error type only) and never
        fails the session."""
        import sys
        from ...core.storage.store import StoreClosed
        try:
            self.store.event(event_type, build(), source="agent_backend")
        except StoreClosed:
            raise
        except Exception as error:  # noqa: BLE001 - observer record only; reported, never a session failure
            print(f"ni-rua: warning: {event_type} could not be recorded: {type(error).__name__}",
                  file=sys.stderr, flush=True)

    def _close_model_catalogue(self):
        """Detach the listing read from the session (``AgentSession.close`` calls this first): no read starts any
        more, and one still running never touches the Store or ``_catalogue`` afterwards (its result is dropped, see
        ``_keep_catalogue``). Waits only for the Store writes of a result that is being recorded at this moment
        (at most two events), never for the network read or a stderr line. Idempotent; never raises. Returns True
        when a read was still running (the close report records it as ``model_listing_pending``), else False."""
        with self._catalogue_guard():
            self._catalogue_closed = True
            done = self._catalogue_done
            return done is not None and not done.is_set()

    def _reopen_model_catalogue(self):
        """Undo ``_close_model_catalogue`` for a session object that is started again after ``close()``
        (``CodexSession.start`` / ``start_forked``): a read still running is kept and recorded when it ends, as
        for a session that was never closed. A result that ended while the session was closed stays dropped (the
        read is once per session object). Idempotent; never raises."""
        with self._catalogue_guard():
            self._catalogue_closed = False

    def _await_model_catalogue(self, timeout_s):
        """The ``ModelCatalogue`` once the start-time read ended, waiting at most ``timeout_s`` seconds; None when
        no read was started, it is still running after the wait, or it failed (``status="unavailable"``)."""
        done = self._catalogue_done
        if done is None or not done.wait(max(0.0, timeout_s)):
            return None
        catalogue = self._catalogue
        return None if catalogue is None or catalogue.status == "unavailable" else catalogue

    def _catalogue_entry(self, model):
        """The ``CatalogueModel`` the provider listing reports for ``model`` (exact id), or None."""
        catalogue = self._catalogue
        if catalogue is None or catalogue.status == "unavailable":
            return None
        return next((item for item in catalogue.models if item.id == model), None)

    def _image_input_unsupported(self, catalogue):
        """The configured model id (``config.model.name``) when the provider lists it with ``image_input: false``
        (the observations are images; such a provider answers them with e.g. HTTP 400 ``IMAGE_INPUT_UNSUPPORTED``),
        else None. ``_keep_catalogue`` then records the observer event ``agent.model_image_input_unsupported`` and
        prints one gateway stderr warning. Never refuses or switches the model; the console shows the same flag on
        its model control."""
        model = self.config.model.name
        entry = next((item for item in catalogue.models if item.id == model), None)
        return model if entry is not None and entry.image_input is False else None

    def model_options(self):
        """Console suggestions (``AgentSession.model_options``): the provider-listed models usable on the session's
        wire (``model_catalogue.usable_models``: ``protocols`` include the wire's protocol when listed, ``tools``
        not false), as ``{"value", "name", "image_input"}``; None until the listing was read, when it failed, or for
        an upstream source that is not read (the console then falls back to its own list, at least the current
        model)."""
        catalogue = self._catalogue
        if catalogue is None or catalogue.status == "unavailable":
            return None
        return [dict(value=model.id, name=model.name, image_input=model.image_input)
                for model in catalogue.usable(self.wire_api)] or None


class CodexModelListingMixin(ModelListingMixin):
    """``ModelListingMixin`` for ``CodexSession`` (Codex has no ``agent.codex.upstream`` section of its own).

    Sources (``_catalogue_source``): ``profile`` when ``agent.codex_upstream`` names a provider profile; otherwise the
    ``model.codex_home`` provider: ``codex_home`` for an API-key provider (``GET <base_url>/models`` with the same
    credential the relay forwards with), ``codex_chatgpt_login`` for a ChatGPT login (account control; its backend has
    no OpenAI-shaped listing, so nothing is read and the suggestions are the pinned native catalogue
    ``cortex/session/codex_models.json`` that this Codex itself runs with: entries with ``visibility: "list"``, by
    ``priority``). Listed models that the native catalogue knows get its display name, image input and limits for
    every field the listing does not report (``model_catalogue.merge_model``).

    ``CodexSession`` calls ``_start_model_catalogue(upstream_url, upstream_auth)`` in ``_prepare`` once the upstream
    and its host credential headers are resolved (before they are replaced by the private relay).
    """

    model_catalogue_sources = ("codex_home", "profile")
    wire_api = "openai_responses"
    _listing_source = None

    def _actor_upstream(self):
        """``agent.codex_upstream`` (a ``source="profile"`` ``UpstreamConfig``) or None (= ``model.codex_home``)."""
        from .factory import actor_upstream
        return actor_upstream(self.config, "codex")

    def _catalogue_source(self):
        """``profile`` | ``codex_home`` | ``codex_chatgpt_login`` (see the class doc); decided once per session (the
        console polls ``model_options``) and kept in ``_listing_source``."""
        if self._listing_source is None:
            upstream = self._actor_upstream()
            if upstream is not None:
                self._listing_source = upstream.source
            else:
                self._listing_source = "codex_chatgpt_login" if self._codex_chatgpt_login() else "codex_home"
        return self._listing_source

    def _codex_chatgpt_login(self):
        """True when ``model.codex_home`` serves its provider through a ChatGPT login (the account-control path of
        ``CodexSession._prepare``); False for an API-key provider or an unreadable home."""
        import tomllib
        from .codex.session import _chatgpt_login, _model_provider
        home = Path(self.config.model.codex_home).expanduser()
        try:
            _name, provider, _url = _model_provider(tomllib.loads((home / "config.toml").read_text()), home)
            return bool(provider.get("requires_openai_auth")) and _chatgpt_login(home)
        except (OSError, ValueError, AttributeError):
            return False

    def model_options(self):
        """``ModelListingMixin.model_options`` for Codex: the listed models usable on ``openai_responses`` merged with
        the pinned native catalogue's metadata; for a ChatGPT login the native catalogue's listed models. None when
        nothing is known (not read yet, or the read failed)."""
        from .model_catalogue import _codex_models, merge_model, native_codex_model
        if self._listing_source == "codex_chatgpt_login":
            entries = sorted((entry for entry in _codex_models().values() if entry.get("visibility") == "list"),
                             key=lambda entry: (entry.get("priority") if type(entry.get("priority")) is int
                                                else 1 << 30, entry["slug"]))
            models = [native_codex_model(entry["slug"]) for entry in entries]
        else:
            catalogue = self._catalogue
            if catalogue is None or catalogue.status == "unavailable":
                return None
            models = [merge_model(model, native_codex_model(model.id)) for model in catalogue.usable(self.wire_api)]
        return [dict(value=model.id, name=model.name, image_input=model.image_input) for model in models] or None
