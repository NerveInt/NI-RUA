"""Build ``opencode_wire_bodies.json``: real opencode 2.2.7 generation requests and client attempt counts for the relay
wire tests (P3-4, re-captured for 2.2.7 by OC2-E1).

Source: the evidence directory written by ``capture_opencode.py`` (the pinned binary launched through
``opencode.config.build_launch`` against loopback fake models; see its docstring). Kept per relay wire:

* ``cases.<wire>``: from the ``basic`` scenario (text, MCP image results, an ``isError`` result, two parallel calls,
  a native ``shell`` call, a reasoning block, a second user turn): the request path, the client request headers of
  the first generation request (``Host`` and ``Content-Length`` dropped) and the first and last generation bodies,
  verbatim except that the capture's run directory (both its ``/var/...`` and ``/private/var/...`` spellings) becomes
  ``/run``.
* ``client_attempts.<wire>``: the ``attempts`` measurement: for every answer form served verbatim on every request of
  one session, ``form``, ``status``, ``headers``, ``body``, the number of requests OpenCode made (``attempts``),
  ``stopped`` (``idle`` = OpenCode ended the turn; ``cap`` = still retrying, interrupted), the idle ``outcome``, the
  last assistant ``error_type``, ``synthetic_messages`` and ``span_s`` (s from the first to the last attempt).

The bodies keep the OpenCode-specific fields the egress walker must leave alone (``prompt_cache_key``, ``include``,
``store``, ``reasoning``, ``instructions``, ``max_output_tokens`` / ``max_tokens``, ``thinking``, ``output_config``,
``cache_control``) and the 4x3 PNG the capture served. Usage, from the worktree root::

    PYTHONPATH=tests mamba run -n ni-rua python tests/fixtures/opencode/build_opencode_bodies.py \\
        --evidence <capture evidence directory> [--opencode <2.2.7 binary>]

``--opencode`` first runs the whole capture into ``--evidence`` (which must not exist yet).
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().with_name("opencode_wire_bodies.json")
CAPTURE = Path(__file__).resolve().with_name("capture_opencode.py")
WIRES = ("anthropic_messages", "openai_responses")
DROPPED_HEADERS = ("host", "content-length")
LOCAL_MARKERS = ("/var/folders", "/Users/", "oc2e1-", "ni-rua-agent-harness")


def sanitise(node, prefixes):
    if isinstance(node, dict):
        return {k: sanitise(v, prefixes) for k, v in node.items()}
    if isinstance(node, list):
        return [sanitise(v, prefixes) for v in node]
    if isinstance(node, str):
        for prefix in prefixes:
            node = node.replace(prefix, "/run")
        return node
    return node


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--evidence", required=True, help="capture_opencode.py output directory")
    parser.add_argument("--opencode", help="run the capture first with this opencode 2.2.7 binary")
    args = parser.parse_args()
    evidence = Path(args.evidence)
    if args.opencode:
        subprocess.run([sys.executable, str(CAPTURE), "--opencode", args.opencode, "--evidence", str(evidence)],
                       check=True)
    capture = json.loads((evidence / "capture.json").read_text())
    cases, attempts = {}, {}
    for wire in WIRES:
        base = evidence / f"basic-{wire}"
        result = json.loads((base / "result.json").read_text())
        requests = json.loads((base / "requests.json").read_text())
        # Longest first: the realpath spelling contains the plain one.
        prefixes = sorted({result["run_dir_real"], result["run_dir"]}, key=len, reverse=True)
        headers = {k: v for k, v in requests[0]["headers"].items() if k.lower() not in DROPPED_HEADERS}
        cases[wire] = dict(path=requests[0]["path"], headers=headers,
                           first_body=sanitise(requests[0]["body"], prefixes),
                           last_body=sanitise(requests[-1]["body"], prefixes))
        attempts[wire] = json.loads((evidence / f"attempts-{wire}.json").read_text())["forms"]
    about = "%s Captured %s with opencode %s; see build_opencode_bodies.py." % (
        __doc__.split("\n\n")[0].replace("\n", " "), capture["captured_at"][:10], capture["opencode_version"])
    text = json.dumps(dict(about=about, opencode_version=capture["opencode_version"], cases=cases,
                           client_attempts=attempts), indent=1, ensure_ascii=True) + "\n"
    for marker in LOCAL_MARKERS:
        assert marker not in text, marker
    OUT.write_text(text)
    print("wrote", OUT, len(text.encode()), "bytes")


if __name__ == "__main__":
    main()
