"""Generate ``agent-plugin`` MCP entries and ``manifest.generated.json`` from central contracts.

Usage (from the repository root)::

    mamba run -n ni-rua python scripts/gen_agent_plugin_codex_mcp.py            # write generated agent-plugin files
    mamba run -n ni-rua python scripts/gen_agent_plugin_codex_mcp.py --check    # exit 1 when any file is stale
"""
import argparse
import functools
import itertools
import json
from pathlib import Path
import sys
import tomllib
from types import SimpleNamespace

REPOSITORY = Path(__file__).resolve().parents[1]
# Import `ni_rua` from this checkout so worktrees regenerate from their own registry and contracts.
sys.path.insert(0, str(REPOSITORY))
PLUGIN_DIR = REPOSITORY / "agent-plugin"
TARGET = PLUGIN_DIR / ".codex-mcp.json"

from ni_rua.cortex.agents import plugin_contract

MCP_URL = plugin_contract.default_mcp_url()
SERVER_NAME = plugin_contract.SERVER_NAME
OUTPUT_TOKEN_LIMIT = plugin_contract.OUTPUT_TOKEN_LIMIT
LARGEST_KNOWN_RESULT_CHARS = plugin_contract.LARGEST_KNOWN_RESULT_CHARS
TIMEOUTS = dict(plugin_contract.CODEX_TIMEOUTS)


def package_version() -> str:
    """Return the single package version from ``pyproject.toml``."""
    pyproject = tomllib.loads((REPOSITORY / "pyproject.toml").read_text())
    return str(pyproject["project"]["version"])


def _profiles():
    """The embodiments of the repository: the default ARX-X5 and the SO-101 of the shipped example deployment."""
    from ni_rua.robot.model.embodiments import ARX_X5_PROFILE

    yield ARX_X5_PROFILE
    from ni_rua.robot.backends.so101 import deployment

    document = json.loads((REPOSITORY / "configs" / "so101.example.json").read_text())
    yield deployment.embodiment_profile(deployment.validate_deployment(document, base=REPOSITORY / "configs"))


def _feature_overrides():
    """Feature combinations changing registered actor tool names across backends."""
    perception = ("vggt_height", "object_table", "scene3d", "object_table_observe", "scene3d_observe")
    adding = ("pose_skill", "robotic_view", "stereo_guide")
    for backend in ("codex", "external"):
        for perception_on in (False, True):
            for values in itertools.product((False, True), repeat=len(adding)):
                features = dict(zip(adding, values))
                features.update({name: perception_on for name in perception})
                yield {"agent": {"backend": backend}, "features": features}


@functools.lru_cache(maxsize=None)
def _registry_actor_tools():
    """Union of actor-role tool names the central registry can register across profiles/features."""
    from ni_rua.core.config import Config
    from ni_rua.core.tool_errors import ToolInputError
    from ni_rua.runtime.tools.builtins import create_registry
    from ni_rua.runtime.tools.plugins import configured_tool_specs

    names = set()
    for profile in _profiles():
        for overrides in _feature_overrides():
            try:
                config = Config().with_overrides(overrides)
            except ValueError as error:
                raise RuntimeError(f"feature or backend override rejected: {overrides}") from error
            runtime = SimpleNamespace(config=config, embodiment=profile)
            registry = create_registry(configured_tool_specs(config), runtime=runtime)
            for name in registry.names:
                try:
                    registry.spec(name, role="actor")
                except ToolInputError:
                    continue
                names.add(name)
    return frozenset(names)


def registry_actor_tools():
    return set(_registry_actor_tools())


def actor_tool_names():
    return sorted(registry_actor_tools())


def render_files():
    """Return ``{relative_path: content}`` for all generated files under ``agent-plugin/``."""
    return plugin_contract.render_files(version=package_version(), tool_names=actor_tool_names())


def build_document():
    return json.loads(render())


def render():
    return render_files()[".codex-mcp.json"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Do not write; exit 1 when generated files differ")
    parser.add_argument("--output", type=Path, default=None, help="Optional single-file override for .codex-mcp.json")
    args = parser.parse_args(argv)
    files = render_files()
    if args.output is not None:
        targets = [(args.output, files[".codex-mcp.json"])]
    else:
        targets = [(PLUGIN_DIR / rel, content) for rel, content in files.items()]
    if args.check:
        stale = [path for path, expected in targets if not path.exists() or path.read_text() != expected]
        if stale:
            joined = ", ".join(str(p) for p in stale)
            print(f"{joined} stale; run scripts/gen_agent_plugin_codex_mcp.py", file=sys.stderr)
            return 1
        return 0
    for path, content in targets:
        path.write_text(content)
    print(f"wrote {', '.join(str(p) for p, _ in targets)} ({len(actor_tool_names())} tools)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
