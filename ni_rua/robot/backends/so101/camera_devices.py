"""Bind each SO-101 camera to one physical video device by a stable identity.

OpenCV opens a camera by position (``cv2.VideoCapture(index)``), and that position is not an identity: when one USB
camera is unplugged the others move up, and an index-only deployment opens whatever device now holds the index
(2026-11-16: with the wrist camera disconnected, ``fixed`` index 1 opened the MacBook's built-in camera). A deployment
camera can therefore name its device with ``cameras.<name>.unique_id`` and/or ``device_name``. This module lists the
present video devices without opening any of them, resolves that identity to exactly one device and the target
OpenCV opens for it, and refuses otherwise. It never falls back to ``index_or_path``.

Device list and index rule per platform:

* macOS: ``[AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo]`` followed by ``devicesWithMediaType:
  AVMediaTypeMuxed``, read through the Objective-C runtime (ctypes; no capture session, no camera opened). That is the
  list OpenCV's AVFoundation backend builds in ``modules/videoio/src/cap_avfoundation_mac.mm``; since OpenCV 4.7.0 it
  sorts the list by ``uniqueID`` (``NSString compare:``) and opens ``devices[index]`` (checked in the 4.7.0 to 4.13.0
  and 5.0.0 sources; 4.6.0 and older use the unsorted list). So the OpenCV index of a device is its position in the
  list sorted by unique ID. ``unique_id`` is ``AVCaptureDevice.uniqueID``, the string ``system_profiler
  SPCameraDataType`` prints as ``spcamera_unique-id``: location based for a USB camera (``0x1340001b170209``: USB
  port + vendor + product id, so it changes when the camera moves to another port), a UUID for a built-in camera. The
  ``system_profiler`` list order is not the OpenCV order. The camera is opened with ``cv2.CAP_AVFOUNDATION``.
* Linux: one entry per ``/sys/class/video4linux/videoN`` node (its ``name`` file) with the ``/dev/v4l/by-id`` and
  ``/dev/v4l/by-path`` links that point at it. ``unique_id`` is one of those links (full path); the camera opens the
  node ``/dev/videoN`` with ``cv2.CAP_V4L2``. A UVC camera has a second (metadata) node with the same name, so a
  ``device_name`` alone usually matches two nodes and is refused as ambiguous; bind by the ``...-video-index0`` link.
* Other platforms: no device list. Identity binding is refused; index binding opens ``index_or_path`` as before.

Every record here is operator/audit information (``<run_dir>/camera_binding.json``, ``robot-worker.log``, the S0
self-check); none of it is published to the actor.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import sys

#: First OpenCV release whose AVFoundation backend sorts the device list by uniqueID (cap_avfoundation_mac.mm).
AVFOUNDATION_SORTED_SINCE = (4, 7, 0)
INDEX_RULES = {
    "avfoundation_sorted_unique_id": "OpenCV index = position in the AVFoundation video+muxed device list sorted by "
                                     "uniqueID (OpenCV >= 4.7.0, cap_avfoundation_mac.mm)",
    "v4l2_node": "OpenCV V4L2 opens the node /dev/videoN; identity bindings open that node path",
}
HELPER = "scripts/so101_cameras.py lists the present devices and their unique IDs"


class CameraBindingError(RuntimeError):
    """A configured camera identity resolves to no device, several devices, or an unverifiable index."""


# ------------------------------------------------------------------------------------------------ enumeration ---
def _avfoundation_devices():
    """AVFoundation video+muxed capture devices in OpenCV's concatenation order (unsorted).

    Reads ``uniqueID``, ``localizedName``, ``modelID`` and ``transportType`` through the Objective-C runtime. Creates
    no capture session and opens no device. Returns ``[{"unique_id", "name", "model", "transport"}]``.
    """
    import ctypes
    import ctypes.util
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
    avfoundation = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/AVFoundation.framework/AVFoundation")
    objc.objc_getClass.restype, objc.objc_getClass.argtypes = ctypes.c_void_p, [ctypes.c_char_p]
    objc.sel_registerName.restype, objc.sel_registerName.argtypes = ctypes.c_void_p, [ctypes.c_char_p]
    objc.objc_autoreleasePoolPush.restype = ctypes.c_void_p
    objc.objc_autoreleasePoolPop.argtypes = [ctypes.c_void_p]
    address = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value

    def send(restype, receiver, selector, *args, argtypes=()):
        function = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)(address)
        return function(receiver, objc.sel_registerName(selector.encode()), *args)

    def text(string):
        raw = send(ctypes.c_char_p, string, "UTF8String") if string else None
        return None if raw is None else raw.decode("utf-8", "replace")

    pool = objc.objc_autoreleasePoolPush()
    try:
        device_class, devices = objc.objc_getClass(b"AVCaptureDevice"), []
        for media in ("AVMediaTypeVideo", "AVMediaTypeMuxed"):
            media_type = ctypes.c_void_p.in_dll(avfoundation, media).value
            array = send(ctypes.c_void_p, device_class, "devicesWithMediaType:", media_type, argtypes=(ctypes.c_void_p,))
            for position in range(send(ctypes.c_ulong, array, "count") if array else 0):
                device = send(ctypes.c_void_p, array, "objectAtIndex:", position, argtypes=(ctypes.c_ulong,))
                transport = send(ctypes.c_uint32, device, "transportType")
                devices.append({"unique_id": text(send(ctypes.c_void_p, device, "uniqueID")),
                                "name": text(send(ctypes.c_void_p, device, "localizedName")),
                                "model": text(send(ctypes.c_void_p, device, "modelID")),
                                "transport": (transport.to_bytes(4, "big").decode("latin-1").strip() or None)
                                if transport else None})
        return devices
    finally:
        objc.objc_autoreleasePoolPop(pool)


def _opencv_version():
    try:
        import cv2
    except ImportError:
        return None
    return cv2.__version__


def _version_tuple(text):
    return tuple(int(part) for part in re.findall(r"\d+", text or "")[:3])


def _v4l_devices(v4l_root, v4l_links):
    links = {}
    for kind in ("by-id", "by-path"):
        folder = Path(v4l_links) / kind
        if folder.is_dir():
            for link in sorted(folder.iterdir()):
                links.setdefault(Path(os.path.realpath(link)).name, []).append(f"/dev/v4l/{kind}/{link.name}")
    devices = []
    root = Path(v4l_root)
    for entry in sorted(root.glob("video*"), key=lambda p: (len(p.name), p.name)) if root.is_dir() else ():
        number = entry.name.removeprefix("video")
        if not number.isdigit():
            continue
        try:
            name = (entry / "name").read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            name = None
        devices.append({"index": int(number), "path": f"/dev/{entry.name}", "name": name, "model": None,
                        "transport": None, "unique_ids": links.get(entry.name, [])})
    return devices


def list_video_devices(*, platform=None, avfoundation=None, opencv_version=None,
                       v4l_root="/sys/class/video4linux", v4l_links="/dev/v4l"):
    """Present video devices and the rule that gives each its OpenCV index; opens no device.

    ``platform`` defaults to ``sys.platform``; ``avfoundation`` (macOS device reader) and ``opencv_version`` default
    to the real ones, ``v4l_root`` / ``v4l_links`` to the Linux sysfs and ``/dev/v4l`` directories (tests pass
    fakes). Returns ``{"platform", "opencv_version", "index_rule", "index_rule_error", "error", "devices"}`` where each
    device is ``{"index", "path", "name", "model", "transport", "unique_ids"}``: ``index`` is the OpenCV index (macOS,
    ``None`` while ``index_rule_error`` is set) or the node number N of ``/dev/videoN`` (Linux); ``unique_ids`` holds
    the identities ``cameras.<name>.unique_id`` may name; ``transport`` is the macOS transport FourCC (``usb``,
    ``bltn`` built-in, ...), else ``None``. ``error`` is set, with no devices, when the list cannot be read.
    """
    platform = platform or sys.platform
    report = {"platform": platform, "opencv_version": None, "index_rule": None, "index_rule_error": None,
              "error": None, "devices": []}
    try:
        if platform == "darwin":
            version = opencv_version if opencv_version is not None else _opencv_version()
            report["opencv_version"] = version
            raw = list((avfoundation or _avfoundation_devices)())
            ordered = sorted(raw, key=lambda item: item["unique_id"] or "")
            if version is None:
                report["index_rule_error"] = "opencv_not_importable"
            elif _version_tuple(version) < AVFOUNDATION_SORTED_SINCE:
                report["index_rule_error"] = f"opencv_{version}_indexes_the_unsorted_avfoundation_list"
            elif not all(isinstance(item["unique_id"], str) and item["unique_id"].isascii() for item in raw):
                # Python's code-point order equals NSString compare: for ASCII IDs only.
                report["index_rule_error"] = "unique_id_missing_or_not_ascii"
            else:
                report["index_rule"] = "avfoundation_sorted_unique_id"
            verified = report["index_rule"] is not None
            report["devices"] = [{"index": position if verified else None, "path": None, "name": item["name"],
                                  "model": item["model"], "transport": item["transport"],
                                  "unique_ids": [item["unique_id"]] if item["unique_id"] else []}
                                 for position, item in enumerate(ordered)]
        elif platform.startswith("linux"):
            report["opencv_version"] = opencv_version if opencv_version is not None else _opencv_version()
            report["index_rule"] = "v4l2_node"
            report["devices"] = _v4l_devices(v4l_root, v4l_links)
        else:
            report["error"] = f"no_device_list_on_platform_{platform}"
    except Exception as error:  # an unreadable list must not hide behind an empty one
        report.update(error=f"device_list_failed: {type(error).__name__}: {error}", devices=[])
    return report


# ------------------------------------------------------------------------------------------------- resolution ---
def _describe(device):
    ids = ", ".join(device["unique_ids"]) or "no unique id"
    where = device["path"] or f"index {device['index']}"
    return f"[{where}] {device['name']!r} ({ids})"


def present_text(inventory):
    """One-line operator summary of the present devices (for refusal messages)."""
    if inventory.get("error"):
        return f"device list unavailable ({inventory['error']})"
    return "present: " + ("; ".join(_describe(device) for device in inventory["devices"]) or "no video device")


def _api(platform):
    """OpenCV backend an identity binding opens with: the one whose device order the index rule describes."""
    if platform == "darwin":
        return "avfoundation"
    return "v4l2" if platform.startswith("linux") else "any"


def _device_at(target, inventory):
    """Device an index-only binding opens, or ``(None, reason)`` when it cannot be named."""
    if inventory.get("error"):
        return None, inventory["error"]
    if inventory.get("index_rule_error"):
        return None, inventory["index_rule_error"]
    if isinstance(target, int):
        matches = [device for device in inventory["devices"] if device["index"] == target]
    else:
        real = os.path.realpath(target)
        matches = [device for device in inventory["devices"]
                   if device["path"] in (target, real) or target in device["unique_ids"]]
    return (matches[0], None) if len(matches) == 1 else (None, f"no_present_device_at_{target}")


def resolve_camera(camera_id, config, inventory):
    """Resolve one deployment camera to ``{"mode", "opened", "api", "device", ...}``; opens nothing.

    ``mode`` is ``"unique_id"`` / ``"device_name"`` when the config names an identity (``unique_id`` exact match
    against the device's ``unique_ids``; ``device_name`` case-insensitive substring of its name or model; both must
    hold when both are set) and ``"index"`` otherwise. ``opened`` is what ``cv2.VideoCapture`` receives (macOS index,
    Linux node path, or the configured ``index_or_path`` in index mode) and ``api`` the OpenCV backend name.
    Raises ``CameraBindingError`` when a configured identity matches no device (``so101_camera_not_found``), more
    than one (``so101_camera_ambiguous``), or its index cannot be derived (``so101_camera_index_rule_unverified``,
    ``so101_camera_device_list_unavailable``). Index mode never raises: ``device`` is the device at that index, or
    ``None`` with ``device_unknown_reason``.
    """
    unique_id = (config.get("unique_id") or "").strip() or None
    device_name = (config.get("device_name") or "").strip() or None
    requested = {"unique_id": unique_id, "device_name": device_name, "index_or_path": config.get("index_or_path")}
    base = {"camera": camera_id, "requested": requested, "api": _api(inventory["platform"])}
    if unique_id is None and device_name is None:
        target = config["index_or_path"]
        device, reason = _device_at(target, inventory)
        return {**base, "mode": "index", "opened": target, "device": device, "device_unknown_reason": reason,
                "api": "avfoundation" if inventory["platform"] == "darwin" and isinstance(target, int) else "any"}
    identity = ",".join(f"{key}={value}" for key, value in (("unique_id", unique_id), ("device_name", device_name))
                        if value)
    if inventory.get("error"):
        raise CameraBindingError(f"so101_camera_device_list_unavailable:{camera_id}:{identity}: {inventory['error']}")
    if inventory.get("index_rule_error"):
        raise CameraBindingError(f"so101_camera_index_rule_unverified:{camera_id}:{identity}: "
                                 f"{inventory['index_rule_error']}; {present_text(inventory)}")
    needle = (device_name or "").lower()
    matches = [device for device in inventory["devices"]
               if (unique_id is None or unique_id in device["unique_ids"])
               and (device_name is None or needle in f"{device['name'] or ''}\n{device['model'] or ''}".lower())]
    if not matches:
        raise CameraBindingError(f"so101_camera_not_found:{camera_id}:{identity}: no present device matches "
                                 f"(no fallback to index_or_path); {present_text(inventory)}; {HELPER}")
    if len(matches) > 1:
        raise CameraBindingError(f"so101_camera_ambiguous:{camera_id}:{identity}: {len(matches)} devices match: "
                                 + "; ".join(_describe(device) for device in matches)
                                 + f"; set cameras.{camera_id}.unique_id ({HELPER})")
    device = matches[0]
    opened = device["index"] if inventory["platform"] == "darwin" else device["path"]
    return {**base, "mode": "unique_id" if unique_id else "device_name", "opened": opened, "device": device,
            "device_unknown_reason": None}


def device_list_signature(inventory):
    """What an identity's OpenCV index depends on: the ordered unique IDs (macOS) or node/link sets (Linux)."""
    return [inventory.get("error"), inventory.get("index_rule"),
            [(device["path"], tuple(device["unique_ids"])) for device in inventory["devices"]]]


def confirm_unchanged(camera_id, before, after):
    """Raise when the device list changed between resolving ``camera_id`` and opening it (the index may be stale)."""
    if device_list_signature(before) != device_list_signature(after):
        raise CameraBindingError(f"so101_camera_devices_changed:{camera_id}: the video device list changed while the "
                                 f"camera was opened; restart. before: {present_text(before)}; now: {present_text(after)}")


def bind_cameras(cameras, *, list_devices=None):
    """Resolve every camera that has ``bind(inventory)`` against one device list; opens nothing, never raises.

    ``list_devices`` returns the device list (default ``list_video_devices``, looked up at call time).

    Returns ``None`` when no camera binds (test fakes), else the operator record ``{"platform", "opencv_version",
    "index_rule", "index_rule_text", "index_rule_error", "error", "present", "cameras": {name: binding |
    {"refused": message}}, "refused": [message, ...]}``. Besides each camera's own refusal, two cameras that would
    open the same target are refused (``so101_camera_duplicate_device``). The caller saves the record and raises
    ``CameraBindingError`` when ``refused`` is non-empty.
    """
    binders = {name: camera.bind for name, camera in sorted(cameras.items()) if callable(getattr(camera, "bind", None))}
    if not binders:
        return None
    inventory = (list_devices or list_video_devices)()
    record = {key: inventory.get(key) for key in ("platform", "opencv_version", "index_rule", "index_rule_error", "error")}
    record.update(index_rule_text=INDEX_RULES.get(inventory.get("index_rule")), present=inventory["devices"],
                  cameras={}, refused=[])
    for name, bind in binders.items():
        try:
            record["cameras"][name] = bind(inventory)
        except CameraBindingError as error:
            record["cameras"][name] = {"camera": name, "refused": str(error)}
            record["refused"].append(str(error))
    opened = {}
    for name, binding in record["cameras"].items():
        if "opened" in binding:
            opened.setdefault((binding["api"], str(binding["opened"])), []).append(name)
    for (api, target), names in opened.items():
        if len(names) > 1:
            record["refused"].append(f"so101_camera_duplicate_device:{','.join(names)}: all open {api} {target}")
    return record


def binding_summary(binding):
    """Compact operator view of one binding: mode, OpenCV target, device name/unique ids/transport (or refusal)."""
    if not binding:
        return None
    if "refused" in binding:
        return {"refused": binding["refused"]}
    device = binding.get("device") or {}
    return {"mode": binding["mode"], "opened": binding["opened"], "name": device.get("name"),
            "unique_ids": device.get("unique_ids"), "transport": device.get("transport"),
            "device_unknown_reason": binding.get("device_unknown_reason")}


def log_lines(record):
    """``robot-worker.log`` lines: which physical device each camera opens (operator log; never actor context)."""
    lines = [f"so101_camera_devices platform={record['platform']} opencv={record['opencv_version']} "
             f"index_rule={record['index_rule'] or record['index_rule_error'] or record['error']} "
             f"{present_text({'error': record['error'], 'devices': record['present']})}"]
    for name, binding in record["cameras"].items():
        if "refused" in binding:
            lines.append(f"so101_camera_binding camera={name} refused={binding['refused']}")
            continue
        summary = binding_summary(binding)
        line = (f"so101_camera_binding camera={name} mode={summary['mode']} opened={summary['opened']!r} "
                f"name={summary['name']!r} unique_ids={summary['unique_ids']} transport={summary['transport']}")
        if summary["mode"] == "index":
            line += (f" device_unknown_reason={summary['device_unknown_reason']}" if summary["device_unknown_reason"]
                     else "") + f" note=index_binding_shifts_when_a_camera_is_unplugged_set_cameras.{name}.unique_id"
        lines.append(line)
    lines.extend(f"so101_camera_binding refused={message}" for message in record["refused"]
                 if message.startswith("so101_camera_duplicate_device"))
    return lines
