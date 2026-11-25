"""Install a built wheel outside the checkout and verify runtime entry points."""
from pathlib import Path
import subprocess
import sys
import tempfile


CHECK = r'''
import hashlib
import json
from pathlib import Path
import runpy
import sys

installation = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(installation))
import ni_rua
from ni_rua.apps.library.static_assets import static_directory
from ni_rua.core.config import Config, ConfigError
from ni_rua.cortex.agents.factory import backend_requirements
from ni_rua.robot.calibration.evidence.config import evidence_settings
from ni_rua.robot.backends.so101.deployment import FIELDS
from ni_rua.robot.backends.factory import ENVIRONMENT_BACKENDS
from ni_rua.robot.embodiments.libero_franka_panda import libero_franka_panda_profile
from ni_rua.robot.model.robot_information import robot_information
from ni_rua.apps.evaluation.report import build_report
from ni_rua.apps.reference.reference_api import REFERENCE_ROOT
from ni_rua.cortex.context.public_task_rules import CATALOG_PATH, public_task_rules
from ni_rua.core.storage.store import Store
from ni_rua.cortex.session.actor_source import snapshot_actor_source
from ni_rua.cortex.agents.codex.sandbox import CodexSandbox

assert evidence_settings("video_evidence")["window_size_px"] == 21
assert FIELDS["calibration"]["touch_slide_distance_m"]["default"] == 0
assert "libero" in ENVIRONMENT_BACKENDS
panda_info = robot_information("libero-franka-panda")
assert panda_info["available"] is True and panda_info["reference"]["identity"]["simulation_only"] is True
assert libero_franka_panda_profile().robot_model_id == "libero-franka-panda"

package = Path(ni_rua.__file__).resolve().parent
assert package.is_relative_to(installation), package
snapshot = snapshot_actor_source(Path("published-harness"))
assert (snapshot / "ni_rua/runtime/tools/definitions.py").is_file()
assert not (snapshot / "ni_rua/robot/backends/robodojo/robot_worker.py").exists()
assert (static_directory() / "index.html").is_file()
assert (static_directory() / "app.js").is_file()
assert (package / "runtime/policy/policy_workers/pi05_arx_manifest.json").is_file()
assert (package / "runtime/policy/policy_workers/pi05_arx_client.py").is_file()
assert (package / "core/default_config.json").is_file()
assert (package / "core/so101_standalone.json").is_file()
assert (package / "core/so101.example.json").is_file()
assert (package / "core/so101_board.json").is_file()
catalog = json.loads(CATALOG_PATH.read_text())
assert catalog['version'] == 2 and len(catalog['tasks']) == 54
for task, entry in catalog['tasks'].items():
    rules = public_task_rules(task)
    assert rules['text'] == '\n\n'.join(entry['public_rules'])
    assert '| Score | Condition |' in rules['text']
config = Config.load(environ={})
assert config.queue.max_parallel == 60
assert config.queue.eligible_gpu_indices is None
assert config.queue.memory_per_case_mib == 10240
assert config.queue.gpu_headroom_mib == 0
assert (config.model.name, config.model.effort, config.model.verbosity) == ("gpt-6.1-sol", "high", "high")
assert config.model.codex_home == str(Path("~/.codex").expanduser())
assert config.model.price_card is None
assert config.model.native_memories is True
assert config.features.sandbox_enabled is False
assert config.features.external_perception is True
assert config.features.persistent_object_tracking is False
assert config.features.multiview_scene3d is False
assert config.features.held_perception is False
assert config.features.grip_non_empty_min_aperture_m == 0.0015
assert config.features.grip_rigid_min_aperture_m == 0.002
assert config.features.grip_wide_min_aperture_m == 0.003
# The external agent backend (agent.backend="external") passes the launch validation from the installed wheel's own
# defaults (mcp_port 0 = ephemeral, never probed); the same call refuses a one-turn run, so the check is not vacuous.
external = config.with_overrides({"agent": {"backend": "external", "external": {"mcp_port": 0}}})
assert external.agent.backend == "external" and backend_requirements(external.validate_launch()) is None
try:
    backend_requirements(external.with_overrides({"run": {"exit_after_turn": True}}))
except ConfigError as error:
    assert str(error).startswith("external_backend_is_interactive_only"), error
else:
    raise AssertionError("external backend accepted exit_after_turn")
archived = Config.from_dict({"version": 1})
assert (archived.model.name, archived.model.effort) == ("gpt-6.1-sol", "high")
assert archived.model.codex_home == str(Path("~/.codex").expanduser())
assert archived.model.price_card is None
assert archived.model.native_memories is True
operator_config = Path("operator.json")
operator_config.write_text(json.dumps({"model": {"name": "operator-model"}}))
operator = Config.load(environ={"NI_RUA_CONFIG": str(operator_config)})
assert (operator.model.name, operator.model.effort) == ("operator-model", "high")
assert operator.model.codex_home == config.model.codex_home
assert operator.model.price_card is None
feature_config = Path("feature-only.json")
feature_config.write_text(json.dumps({"features": {"enable_wait": False, "sandbox_enabled": True}}))
feature_only = Config.load(feature_config, environ={"NI_RUA_CONFIG": "missing.json"})
assert feature_only.model == config.model
assert feature_only.features.enable_wait is False
assert feature_only.features.sandbox_enabled is True
legacy_config = Path("legacy.json")
legacy_config.write_text(json.dumps({"model": {"name": "gpt-6-astra", "effort": "xhigh", "codex_home": "archived-auth", "price_card": None}}))
legacy = Config.load(legacy_config, environ={"NI_RUA_CONFIG": str(operator_config)})
assert (legacy.model.name, legacy.model.effort) == ("gpt-6-astra", "xhigh")
assert legacy.model.codex_home == str((legacy_config.parent / "archived-auth").resolve())
assert legacy.model.price_card is None
provenance = json.loads((REFERENCE_ROOT / "third_party/roboprobe/api-provenance.json").read_text())
for name, record in provenance["files"].items():
    assert hashlib.sha256((REFERENCE_ROOT / provenance["vendored_root"] / name).read_bytes()).hexdigest() == record["sha256"]
store = Store("smoke-records", {})
reference = store.put_artifact(b"package integrity", "test", "text/plain")
assert store.resolve_artifact(reference)[1] == b"package integrity"
store.close()
# Validate bundled preset resolution (--config so101 / so101_standalone / ni-rua) outside the git checkout.
so101_cfg = Config.load("so101", environ={})
assert so101_cfg.backend.environment == "so101_real"
assert so101_cfg.backend.robot_model_id == "so101"
assert so101_cfg.features.external_perception is False
assert so101_cfg.backend.deployment == str(Path("~/.config/ni-rua/so101.local.json").expanduser().resolve())
assert Config.load("so101_standalone", environ={}) == so101_cfg
assert Config.load("ni-rua", environ={}) == config
# Validate ni-rua-so101 init-config can read bundled so101.example.json and emit a valid deployment outside checkout.
from ni_rua.apps.so101_cli import main as so101_cli_main
from ni_rua.robot.backends.so101.deployment import load_deployment
generated_so101 = Path("generated-so101.local.json")
assert so101_cli_main(["init-config", "--non-interactive", "--port", "/dev/ttyACM0", "--output", str(generated_so101)]) == 0
loaded_so101 = load_deployment(generated_so101)
assert loaded_so101["port"] == "/dev/ttyACM0"
for module in ("ni_rua.apps.gateway", "ni_rua.apps.so101_cli", "ni_rua.apps.evaluation.queue", "ni_rua.apps.evaluation.report", "ni_rua.apps.evaluation.audit"):
    sys.argv = [module, "--help"]
    try:
        runpy.run_module(module, run_name="__main__")
    except SystemExit as result:
        assert result.code in (None, 0), (module, result.code)
for subcommand in ("cameras", "commission", "charuco-board", "init-config"):
    try:
        so101_cli_main([subcommand, "--help"])
    except SystemExit as result:
        assert result.code in (None, 0), (subcommand, result.code)
assert (installation / "bin/ni-rua-so101").is_file()
print("Installed wheel: config defaults, bundled presets, SO-101 CLI, archived config, imports, CLI, Web, policy, pinned reference and Store checks passed")
'''


if __name__ == "__main__":
    wheel = Path(sys.argv[1]).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="ni-rua-installed-") as directory:
        installation = Path(directory) / "site-packages"
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(installation), str(wheel)], check=True)
        subprocess.run([sys.executable, "-I", "-c", CHECK, str(installation)], cwd=directory, check=True)
