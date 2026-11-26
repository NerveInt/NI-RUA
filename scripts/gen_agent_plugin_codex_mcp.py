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
def _registry_actor_catalog():
    """收集中央工具注册表在全部具身与特性组合下的 Actor 工具名与静态便携 MCP 定义。"""
    from ni_rua.core.config import Config
    from ni_rua.core.schema_guide import strip_guided_schema_descriptions
    from ni_rua.core.tool_errors import ToolInputError
    from ni_rua.cortex.agents.mcp_server import (
        DEFAULT_RESULT_SIZE_HINT_CHARS,
        RESULT_SIZE_META_KEY,
        portable_input_schema,
    )
    from ni_rua.runtime.tools.builtins import create_registry
    from ni_rua.runtime.tools.plugins import configured_tool_specs

    # 优先采集 SO-101 具身定义作为默认冷启动目录，再补齐其他具身/特性独有工具
    profiles = list(_profiles())
    ordered_profiles = profiles[1:] + profiles[:1] if len(profiles) > 1 else profiles
    tools = {}
    for profile in ordered_profiles:
        for overrides in _feature_overrides():
            try:
                config = Config().with_overrides(overrides)
            except ValueError as error:
                raise RuntimeError(f"feature or backend override rejected: {overrides}") from error
            runtime = SimpleNamespace(config=config, embodiment=profile, disabled_tools=frozenset())
            registry = create_registry(configured_tool_specs(config), runtime=runtime)
            for name in registry.names:
                try:
                    spec = registry.spec(name, role="actor")
                except ToolInputError:
                    continue
                if name in tools:
                    continue
                defn = spec.definition
                desc = str(defn["description"]).strip()
                first_line = desc.splitlines()[0] if desc else f"NI-RUA tool {name}"
                stripped = strip_guided_schema_descriptions(defn["inputSchema"], desc)
                tools[name] = {
                    "name": name,
                    "description": first_line,
                    "inputSchema": portable_input_schema(stripped),
                    "annotations": {"readOnlyHint": spec.parallel is True},
                    "_meta": {RESULT_SIZE_META_KEY: DEFAULT_RESULT_SIZE_HINT_CHARS},
                }
    return tuple(tools[name] for name in sorted(tools))


def registry_actor_tools():
    return {item["name"] for item in _registry_actor_catalog()}


def actor_tool_names():
    return sorted(registry_actor_tools())


def actor_tool_definitions():
    return list(_registry_actor_catalog())


def render_files():
    """Return ``{relative_path: content}`` for all generated files under ``agent-plugin/``."""
    return plugin_contract.render_files(
        version=package_version(),
        tool_names=actor_tool_names(),
        tool_definitions=actor_tool_definitions(),
    )


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
