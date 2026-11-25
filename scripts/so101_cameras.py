"""List the video devices present on this host and the OpenCV index each gets; opens no camera (read-only).

Use it to fill in ``cameras.<name>.unique_id`` of an SO-101 deployment (``docs/development/so101-integration.md``
§3.3), and, with ``--deployment``, to see exactly what the robot worker would bind each configured camera to (or why
it would refuse) without starting the worker. Device listing and resolution are ``ni_rua.robot.backends.so101.
camera_devices`` (the code the worker runs): macOS reads the AVFoundation device list (the list OpenCV opens from)
and derives each OpenCV index from the OpenCV version of this interpreter; Linux reads ``/sys/class/video4linux``
and the ``/dev/v4l/by-id`` / ``by-path`` links.

Run it with the robot worker's interpreter, since its OpenCV version decides the macOS index rule::

    /opt/homebrew/Caskroom/miniforge/base/envs/so101/bin/python scripts/so101_cameras.py \\
        --deployment configs/so101.local.json

Two cameras of the same model report the same name. To tell them apart, unplug one (or cover its lens and look at a
console preview): the unique ID that disappears (or the camera whose preview goes dark) is that camera. On macOS a
USB camera's unique ID encodes the USB port it is plugged into, so keep each camera on its own port.

Exit status: 0, or 1 when ``--deployment`` is given and a configured camera would be refused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ni_rua.robot.backends.so101.camera_devices import (  # noqa: E402
    INDEX_RULES, bind_cameras, list_video_devices, resolve_camera)


class _Configured:
    """A deployment camera that only binds (``bind_cameras`` input); nothing is opened."""

    def __init__(self, name, config):
        self.name, self.config = name, config

    def bind(self, inventory):
        return resolve_camera(self.name, self.config, inventory)


def resolve_deployment(path, inventory):
    """The worker's binding record (``camera_devices.bind_cameras``) for the cameras of the deployment at ``path``."""
    from ni_rua.robot.backends.so101.deployment import load_deployment
    cameras = {name: _Configured(name, config) for name, config in load_deployment(path)["cameras"].items()}
    return bind_cameras(cameras, list_devices=lambda: inventory)


def render(inventory, record):
    lines = [f"platform {inventory['platform']}, OpenCV {inventory['opencv_version']}, index rule: "
             + (INDEX_RULES.get(inventory["index_rule"]) or f"UNVERIFIED ({inventory['index_rule_error']})")]
    if inventory["error"]:
        lines.append(f"device list unavailable: {inventory['error']}")
    lines.append(f"{'opencv':<8}{'transport':<11}{'unique_id':<44}name / model")
    for device in inventory["devices"]:
        where = device["path"] or ("?" if device["index"] is None else str(device["index"]))
        ids = device["unique_ids"] or ["-"]
        lines.append(f"{where:<8}{device['transport'] or '-':<11}{ids[0]:<44}{device['name']} / {device['model'] or '-'}")
        lines.extend(f"{'':<19}{extra}" for extra in ids[1:])
    if not inventory["devices"]:
        lines.append("(no video device)")
    for name, binding in (record or {}).get("cameras", {}).items():
        if "refused" in binding:
            lines.append(f"{name}: REFUSED {binding['refused']}")
            continue
        device = binding["device"] or {}
        lines.append(f"{name}: {binding['mode']} binding -> OpenCV {binding['opened']!r} = {device.get('name')!r} "
                     f"{device.get('unique_ids')} transport={device.get('transport')}"
                     + (f" (device unknown: {binding['device_unknown_reason']})" if binding["device_unknown_reason"] else "")
                     + (" [index binding shifts when a camera is unplugged: set unique_id]" if binding["mode"] == "index" else ""))
    lines.extend(f"REFUSED {message}" for message in (record or {}).get("refused", [])
                 if message.startswith("so101_camera_duplicate_device"))
    lines.append('bind a camera: "cameras": {"<wrist|fixed>": {"unique_id": "<that camera\'s unique_id above>", ...}}')
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--deployment", help="SO-101 deployment JSON; also resolve its cameras as the worker would")
    parser.add_argument("--json", action="store_true", help="print the device list (and resolutions) as JSON")
    arguments = parser.parse_args(argv)
    inventory = list_video_devices()
    record = resolve_deployment(arguments.deployment, inventory) if arguments.deployment else None
    if arguments.json:
        print(json.dumps({"devices": inventory, "binding": record}, indent=2, ensure_ascii=False))
    else:
        print(render(inventory, record))
    return 1 if record and record["refused"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
