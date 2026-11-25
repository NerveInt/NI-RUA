"""Shared canonical paths inside the NI-RUA checkout, invariant to test subzone nesting depth, and the two switches
that keep a test process off the state of the machine and account running the suite."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

TESTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TESTS_ROOT.parent
FIXTURES_ROOT = TESTS_ROOT / "fixtures"
CONFIGS_ROOT = REPO_ROOT / "configs"
WEB_ROOT = REPO_ROOT / "web"
# Never created: the configuration layer a test process reads instead of the checkout's configs/ni-rua.local.json.
ABSENT_LOCAL_CONFIG = CONFIGS_ROOT / "ni-rua.local.json.absent"


def ignore_local_config():
    """Make ``Config.load()`` in this process skip the checkout's git-ignored ``configs/ni-rua.local.json``.

    Called on import of every test zone package (``tests/<zone>/__init__.py``): the operator's local overrides
    (model, features, queue, agent upstreams) exist only on their machine, so a test that reaches ``Config.load()``
    (directly, or through ``Harness``, ``Workbench``, a session or the CLI without an explicit config) would
    otherwise depend on that file. A test of the local layer still patches ``ni_rua.core.config.LOCAL_CONFIG_PATH``
    to its own file. Subprocesses a test starts are not affected.
    """
    from ni_rua.core import config
    config.LOCAL_CONFIG_PATH = ABSENT_LOCAL_CONFIG


def isolate_home():
    """For ``setUpModule``: HOME is a fresh temporary directory until the module's tests are done (then restored and
    removed), so every ``~`` default the runtime expands is private to the test run, not the account's own state:
    ``features.grasp_memory_dir`` (the persistent grasp memory store every fresh ``Workbench`` opens),
    ``model.codex_home``, ``run.experiment_root``. Returns the temporary home."""
    home = unittest.enterModuleContext(tempfile.TemporaryDirectory(prefix="home-"))
    unittest.enterModuleContext(mock.patch.dict(os.environ, {"HOME": home}))
    return home
