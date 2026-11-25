"""Pinned third-party RoboProbe source loader shared across core/cortex/apps without upward imports."""

from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_DIR = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = (
    ROOT
    if (ROOT / "third_party/roboprobe/api-provenance.json").is_file()
    else _PACKAGE_DIR / "reference_sources"
)
UPSTREAM_COMMIT = "9ffacc54f372beef6479d61711bf4945e31ae4ce"


@lru_cache(maxsize=1)
def upstream():
    """Import unchanged vendored modules under their original package names."""
    provenance = json.loads((REFERENCE_ROOT / "third_party/roboprobe/api-provenance.json").read_text())
    source = REFERENCE_ROOT / provenance["vendored_root"]
    if provenance["commit"] != UPSTREAM_COMMIT:
        raise ValueError("reference_source_commit_mismatch")
    for name, info in provenance["files"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError("reference_source_digest_mismatch:" + name)
    if "XPolicyLab" in sys.modules:
        raise RuntimeError("reference_requires_its_own_process_namespace")
    package = types.ModuleType("XPolicyLab")
    package.__path__ = [str(source)]
    sys.modules["XPolicyLab"] = package
    prefix = "XPolicyLab.policy.RoboDojo_Agent_L3_Inspect"
    return types.SimpleNamespace(
        joint=importlib.import_module(prefix + ".policy"),
        eef=importlib.import_module(prefix + "_EEF.policy"),
        docs=importlib.import_module(prefix + "_EEF.docs"),
        responses=importlib.import_module("XPolicyLab.utils.openai_responses"),
        provenance=provenance,
    )


def joint_policy():
    """Return the vendored RoboProbe joint policy module (error classes, retry policy, response parser)."""
    return upstream().joint
