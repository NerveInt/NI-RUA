"""Real ``Workbench`` startups on ``agent.backend = "external"`` for the ``test_agent_external_*`` suites.

``ExternalRun`` starts the real Workbench (real ``ExternalSession``, real ``NiRuaMcpServer`` on an ephemeral loopback
port, real Harness and Store) over the deterministic in-process robots of ``support.embodiment_golden``: the ARX-X5
RoboDojo backend (``x5``) or the SO-101 backend on ``configs/so101.example.json`` (``so101``). No model, network, GPU,
agent binary or real arm. ``host_instructions(embodiment, ...)`` runs the same startup with ``FakeAgentSession`` (a
host-mode backend) and returns the startup instructions it received, scrubbed by ``ActorPresentation.present_text``
exactly as a host session delivers them, with the run directory replaced by ``RUN_MARKER``; ``host_startup`` also
returns the actor catalogue and the MCP ``tools/list`` a host MCP backend publishes for it.

Use in a test::

    with ExternalRun("x5", prompt=None) as run:
        client = run.client()                 # FakeMcpClient after initialize + initialized
        briefing = client.call_tool("run_begin", {})
"""
import asyncio
import tempfile
import threading
from copy import deepcopy
from pathlib import Path
from unittest import mock

from support import embodiment_golden as golden
from support.fake_mcp_client import FakeMcpClient

RUN_MARKER = "<run>"


def _config(embodiment, root, overrides):
    layer = None if embodiment == "x5" else "configs/so101_gateway.json"
    # The shipped configuration as the gateway loads it (self_briefing stays at its default true: it acts only in a
    # benchmark run), without a PTY and with grasp memory in the temporary root.
    base = {"features": {"terminal": False, "grasp_memory_dir": str(root / "grasp-memory")}}
    if embodiment == "so101":
        base["backend"] = {"deployment": str(golden.so101_deployment(root / "deployment"))}
    return golden.load_config(layer, overrides=golden._merge(base, overrides or {}))


def _robot(embodiment, root, config):
    if embodiment == "x5":
        return golden._x5_robot_class()(root / "robot", config)
    return golden._so101_robot_class()(config=config, run_dir=root / "run" / "private" / "robot")


EXTERNAL = {"agent": {"backend": "external", "external": {"mcp_port": 0}}}


class ExternalRun:
    """One external-backend Workbench run on its own event loop thread (``with`` block = the run's lifetime).

    ``embodiment``: ``"x5"`` | ``"so101"``; ``prompt``: what the gateway passes to ``Workbench.start`` (None or
    ``DEFAULT_PROMPT`` = no ``--prompt``); ``overrides``: config document merged over the shipped configuration and
    ``EXTERNAL``. Attributes after ``__enter__``: ``workbench``, ``session``, ``harness``, ``store``, ``root``
    (temporary directory), ``run_dir``, ``url`` (MCP endpoint), ``started_with`` (``{"tools", "instructions"}`` the
    session got). ``call(coroutine_function)`` runs an async Workbench method on the run's loop and returns its result.
    """

    def __init__(self, embodiment="x5", *, prompt=None, overrides=None, root=None, resume=False):
        """``root``: an existing directory to run in (kept on exit; ``root / "run"`` is the run directory) instead of
        a fresh temporary one; ``resume``: resume the run in ``root / "run"`` (``--resume``)."""
        self.embodiment, self.prompt, self.overrides = embodiment, prompt, overrides
        self.given_root, self.resume = root, resume
        self.started_with = {}

    def __enter__(self):
        golden._tests_on_path()
        from ni_rua.apps.gateway import workbench as workbench_module
        from ni_rua.apps.gateway.workbench import Workbench
        self._directory = None if self.given_root is not None else tempfile.TemporaryDirectory(prefix="ni-rua-external-")
        self.root = Path(self.given_root if self.given_root is not None else self._directory.name).resolve()
        self._cwd = golden.repository_cwd()
        self._cwd.__enter__()
        config = _config(self.embodiment, self.root, golden._merge(EXTERNAL, self.overrides or {}))
        robot = _robot(self.embodiment, self.root, config)
        original = workbench_module.create_session
        captured = self.started_with

        def create(*args, **kwargs):
            session = original(*args, **kwargs)
            start = session.start

            def recording_start(tools, instructions, *rest, **options):
                captured.update(tools=deepcopy(tools), instructions=instructions)
                return start(tools, instructions, *rest, **options)
            session.start = recording_start
            return session

        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, name="external-run-loop", daemon=True)
        self._thread.start()
        with mock.patch.object(workbench_module, "create_session", side_effect=create):
            options = dict(resume=True) if self.resume else {}
            self.workbench = Workbench(self.root / "run", robot=robot, visual_review_model=None, config=config,
                                       **options)
        self.run_dir = self.workbench.root
        self.session, self.harness, self.store = self.workbench.session, self.workbench.harness, self.workbench.store
        try:
            self.call(lambda: self.workbench.start(self.prompt))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        self.url = self.session.bridge.url
        return self

    def call(self, factory, timeout=120):
        async def run():
            return await factory()
        return asyncio.run_coroutine_threadsafe(run(), self.loop).result(timeout)

    def control(self, value):
        """``Workbench.execute_control(value)`` on the run's loop (no HTTP layer)."""
        return self.call(lambda: self.workbench.execute_control(value))

    def client(self, *, handshake=True, name="fake-mcp-client", version="0.0.0"):
        """A ``FakeMcpClient`` on the run's endpoint, without a bearer (external mode), after ``initialize`` +
        ``notifications/initialized`` unless ``handshake=False``; ``name`` / ``version`` are its ``clientInfo``."""
        client = FakeMcpClient(self.url, protocol_version="2025-11-25", client_name=name, client_version=version)
        if handshake:
            client.handshake()
        return client

    def __exit__(self, *exc):
        try:
            if getattr(self, "workbench", None) is not None and not self.workbench.closed:
                self.call(self.workbench.close)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self._thread.join(10)
            self.loop.close()
            self._cwd.__exit__(None, None, None)
            if self._directory is not None:
                self._directory.cleanup()
        return False


def host_startup(embodiment, overrides=None):
    """What a host-mode backend gets for the same embodiment and configuration (sandbox off, as every external run):
    the real Workbench startup with ``FakeAgentSession``. Returns ``{"instructions": the startup instructions after
    ``ActorPresentation.present_text`` (as every host session delivers them) with the run directory replaced by
    ``RUN_MARKER``, "tools": the actor catalogue passed to ``session.start``, "catalog": the MCP ``tools/list`` a host
    MCP backend publishes for them (``mcp_tool_catalog`` with the session's parallel hints and its presentation)}``."""
    golden._tests_on_path()
    from ni_rua.apps.gateway import workbench as workbench_module
    from ni_rua.apps.gateway.workbench import DEFAULT_PROMPT, Workbench
    from ni_rua.core.presentation.actor_presentation import ActorPresentation
    from ni_rua.cortex.agents.mcp_server import mcp_tool_catalog
    from support.fake_agent import FakeAgentSession
    created = []

    def create(config, run_dir, store, tool_handler, *, price_card=None, task_planning=None):
        session = FakeAgentSession(run_dir, store, tool_handler, price_card=price_card)
        start = session.start

        def recording_start(tools, instructions, *rest, **options):
            created.append(dict(session=session, tools=deepcopy(tools), instructions=instructions))
            return start(tools, instructions, *rest, **options)
        session.start = recording_start
        return session

    with golden.repository_cwd(), tempfile.TemporaryDirectory(prefix="ni-rua-external-host-") as directory:
        root = Path(directory).resolve()
        config = _config(embodiment, root, overrides)
        robot = _robot(embodiment, root, config)
        with mock.patch.object(workbench_module, "create_session", side_effect=create):
            workbench = Workbench(root / "run", robot=robot, visual_review_model=None, config=config)

        async def run():
            try:
                await workbench.start(DEFAULT_PROMPT)
                started, = created
                presentation = ActorPresentation(workbench.store)
                return dict(instructions=presentation.present_text(started["instructions"]).replace(
                                str(root / "run"), RUN_MARKER),
                            tools=started["tools"],
                            catalog=mcp_tool_catalog(started["tools"], presentation,
                                                     parallel_hints=started["session"].tool_parallel_hints))
            finally:
                await workbench.close()
        return asyncio.run(run())


def host_instructions(embodiment, overrides=None):
    """``host_startup(embodiment, overrides)["instructions"]``."""
    return host_startup(embodiment, overrides)["instructions"]
