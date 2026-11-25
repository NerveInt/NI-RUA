"""SO-101 standalone hardware setup & operator CLI (`ni-rua-so101`).

Exposes four subcommands usable directly from an installed `ni-rua-harness` wheel
without cloning the repository:
- `cameras`: list UVC video devices and verify deployment camera bindings (read-only)
- `init-config`: detect USB serial ports + UVC cameras and write `~/.config/ni-rua/so101.local.json`
- `charuco-board`: render a 1:1 scale A4 ChArUco calibration board (PNG + PDF)
- `commission`: run the 6-step SO-101 first-power-on commissioning sequence
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import threading
import uuid

from ..core.config import USER_SO101_DEPLOYMENT_PATH, bundled_config_path
from ..robot.backends.so101.camera_devices import INDEX_RULES, bind_cameras, list_video_devices, resolve_camera
from ..robot.backends.so101.deployment import ARM_JOINTS, load_deployment, validate_deployment

A4_LANDSCAPE_MM = (297.0, 210.0)
DEFAULT_PX_PER_MM = 24
COMMISSION_IDENTITY = dict(run_id="commissioning", episode_id="ep", env_instance_id="env", env_generation=1)


def default_so101_deployment_path() -> Path:
    """默认 SO-101 部署文件路径：优先当前目录 configs/so101.local.json，否则 ~/.config/ni-rua/so101.local.json。"""
    cwd_path = (Path.cwd() / "configs" / "so101.local.json").resolve()
    if cwd_path.is_file():
        return cwd_path
    return USER_SO101_DEPLOYMENT_PATH.expanduser().resolve()


def _default_worker_python() -> str:
    """选择 SO-101 worker 解释器：若当前环境已安装 lerobot 则优先使用当前 Python，否则探测独立 so101 环境。"""
    if importlib.util.find_spec("lerobot") is not None:
        return sys.executable
    for candidate in (
        Path("~/miniforge3/envs/so101/bin/python").expanduser(),
        Path("/opt/homebrew/Caskroom/miniforge/base/envs/so101/bin/python"),
    ):
        if candidate.is_file():
            return str(candidate)
    return sys.executable


# --------------------------------------------------------------------------------------------- serial & cameras ---
def detect_serial_ports(*, platform: str | None = None, glob_fn=glob.glob) -> list[dict[str, str]]:
    """探测本机可能的 SO-101 USB 串口设备（只读，不打开串口）。"""
    plat = platform or sys.platform
    ports: list[dict[str, str]] = []
    seen_real: set[str] = set()
    if plat == "darwin":
        patterns = (
            ("/dev/tty.usbmodem*", "usbmodem"),
            ("/dev/tty.usbserial*", "usbserial"),
            ("/dev/tty.wchusbserial*", "wchusbserial"),
        )
    elif plat.startswith("linux"):
        patterns = (
            ("/dev/serial/by-id/*", "by-id"),
            ("/dev/ttyACM*", "ttyACM"),
            ("/dev/ttyUSB*", "ttyUSB"),
        )
    else:
        patterns = ()
    for pattern, source in patterns:
        for path in sorted(glob_fn(pattern)):
            real = os.path.realpath(path)
            if real in seen_real:
                continue
            seen_real.add(real)
            ports.append({"port": path, "source": source})
    return ports


def _candidate_cameras(inventory: dict) -> list[dict]:
    """筛选并排序适合作为外接 UVC 机械臂相机的设备列表（优先 USB 摄像头与主采集节点）。"""
    devices = list(inventory.get("devices") or [])
    plat = inventory.get("platform") or sys.platform
    if plat == "darwin":
        return sorted(devices, key=lambda d: (0 if d.get("transport") == "usb" else 1, d.get("index") if d.get("index") is not None else 999))
    if plat.startswith("linux"):
        # V4L2 每个 UVC 相机通常包含 video-index0（图像）与 video-index1（元数据），过滤掉元数据节点
        capture = [
            d for d in devices
            if not d.get("unique_ids")
            or any(not uid.endswith(("-video-index1", "-video-index2", "-video-index3")) for uid in d["unique_ids"])
        ]
        return capture or devices
    return devices


class _ConfiguredCamera:
    """只做设备绑定检查（不打开任何摄像头）。"""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config

    def bind(self, inventory: dict) -> dict:
        return resolve_camera(self.name, self.config, inventory)


def resolve_deployment_cameras(path: str | Path, inventory: dict) -> dict:
    """按 worker 同款逻辑解析部署文件中所有相机的设备绑定。"""
    cameras = {name: _ConfiguredCamera(name, cfg) for name, cfg in load_deployment(path)["cameras"].items()}
    return bind_cameras(cameras, list_devices=lambda: inventory)


def render_cameras(inventory: dict, record: dict | None) -> str:
    lines = [
        f"platform {inventory['platform']}, OpenCV {inventory['opencv_version']}, index rule: "
        + (INDEX_RULES.get(inventory["index_rule"]) or f"UNVERIFIED ({inventory['index_rule_error']})")
    ]
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
        lines.append(
            f"{name}: {binding['mode']} binding -> OpenCV {binding['opened']!r} = {device.get('name')!r} "
            f"{device.get('unique_ids')} transport={device.get('transport')}"
            + (f" (device unknown: {binding['device_unknown_reason']})" if binding["device_unknown_reason"] else "")
            + (" [index binding shifts when a camera is unplugged: set unique_id]" if binding["mode"] == "index" else "")
        )
    lines.extend(
        f"REFUSED {message}" for message in (record or {}).get("refused", [])
        if message.startswith("so101_camera_duplicate_device")
    )
    lines.append('bind a camera: "cameras": {"<wrist|fixed>": {"unique_id": "<that camera\'s unique_id above>", ...}}')
    return "\n".join(lines)


def cmd_cameras(args: argparse.Namespace) -> int:
    inventory = list_video_devices()
    record = resolve_deployment_cameras(args.deployment, inventory) if args.deployment else None
    if args.json:
        print(json.dumps({"devices": inventory, "binding": record}, indent=2, ensure_ascii=False))
    else:
        print(render_cameras(inventory, record))
    return 1 if record and record["refused"] else 0


# ------------------------------------------------------------------------------------------------ init-config ---
def _apply_camera_binding(camera_cfg: dict, choice: str | dict | None, inventory: dict) -> None:
    """将用户选择或自动探测到的相机设备写入部署配置项。"""
    if choice is None:
        return
    plat = inventory.get("platform") or sys.platform
    if isinstance(choice, dict):
        unique_ids = choice.get("unique_ids") or []
        camera_cfg["unique_id"] = unique_ids[0] if unique_ids else None
        camera_cfg["device_name"] = None
        if plat == "darwin" and choice.get("index") is not None:
            camera_cfg["index_or_path"] = int(choice["index"])
        elif choice.get("path"):
            camera_cfg["index_or_path"] = choice["path"]
        elif choice.get("index") is not None:
            camera_cfg["index_or_path"] = int(choice["index"])
        return

    text = str(choice).strip()
    if not text:
        return
    if text.isdigit():
        camera_cfg["index_or_path"] = int(text)
        camera_cfg["unique_id"] = None
        camera_cfg["device_name"] = None
        return
    for device in inventory.get("devices") or []:
        if text in (device.get("unique_ids") or ()) or text == device.get("path"):
            _apply_camera_binding(camera_cfg, device, inventory)
            if text in (device.get("unique_ids") or ()):
                camera_cfg["unique_id"] = text
            return
    if text.startswith("/dev/v4l/"):
        camera_cfg["unique_id"] = text
        camera_cfg["index_or_path"] = text
        camera_cfg["device_name"] = None
    elif text.startswith("/dev/"):
        camera_cfg["unique_id"] = None
        camera_cfg["index_or_path"] = text
        camera_cfg["device_name"] = None
    else:
        camera_cfg["unique_id"] = text
        camera_cfg["device_name"] = None


def _prompt_choice(label: str, default_value: str | None, input_fn=input) -> str:
    suffix = f" [{default_value}]" if default_value else ""
    reply = input_fn(f"{label}{suffix}: ").strip()
    return reply or (default_value or "")


def _format_camera_option(device: dict) -> str:
    where = device.get("path") or (f"index {device['index']}" if device.get("index") is not None else "?")
    uid = (device.get("unique_ids") or ["-"])[0]
    transport = device.get("transport") or "-"
    return f"[{where}] {device.get('name') or 'unknown'} (transport={transport}, unique_id={uid})"


def cmd_init_config(
    args: argparse.Namespace,
    *,
    list_devices=list_video_devices,
    list_ports=detect_serial_ports,
    input_fn=input,
) -> int:
    output_path = Path(args.output).expanduser().resolve() if args.output else USER_SO101_DEPLOYMENT_PATH.expanduser().resolve()
    interactive = (not args.non_interactive) and sys.stdin.isatty()

    if output_path.exists() and not args.force:
        if not interactive:
            raise SystemExit(f"so101_config_exists: {output_path} already exists (pass --force to overwrite)")
        confirm = input_fn(f"{output_path} already exists. Overwrite? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            raise SystemExit("aborted")

    ports = list_ports()
    inventory = list_devices()
    candidates = _candidate_cameras(inventory)

    # 1. 解析串口路径
    port = args.port
    if not port:
        default_port = ports[0]["port"] if ports else None
        if interactive:
            if ports:
                print("Detected serial ports:")
                for idx, item in enumerate(ports, start=1):
                    print(f"  {idx}. {item['port']} ({item['source']})")
            else:
                print("No USB serial ports detected. Plug in the SO-101 USB cable or enter the port path manually.")
            chosen = _prompt_choice("SO-101 serial port (number or path)", default_port, input_fn=input_fn)
            if chosen.isdigit() and 1 <= int(chosen) <= len(ports):
                port = ports[int(chosen) - 1]["port"]
            else:
                port = chosen
        else:
            port = default_port
    if not port:
        raise SystemExit(
            "so101_serial_port_not_found: no USB serial port detected; connect the SO-101 arm or pass `--port <device>`"
        )

    # 2. 解析相机绑定
    wrist_choice: str | dict | None = args.wrist_camera
    fixed_choice: str | dict | None = args.fixed_camera
    if interactive:
        if candidates:
            print("Detected video devices:")
            for idx, dev in enumerate(candidates, start=1):
                print(f"  {idx}. {_format_camera_option(dev)}")
        else:
            print("No video devices detected; keeping template camera indices (0=wrist, 1=fixed).")
        if wrist_choice is None and candidates:
            default_w = "1"
            raw_w = _prompt_choice("Wrist camera (number, unique_id, or OpenCV index)", default_w, input_fn=input_fn)
            wrist_choice = candidates[int(raw_w) - 1] if raw_w.isdigit() and 1 <= int(raw_w) <= len(candidates) else raw_w
        if fixed_choice is None and len(candidates) >= 2:
            default_f = "2"
            raw_f = _prompt_choice("Fixed/overview camera (number, unique_id, or OpenCV index)", default_f, input_fn=input_fn)
            fixed_choice = candidates[int(raw_f) - 1] if raw_f.isdigit() and 1 <= int(raw_f) <= len(candidates) else raw_f
        elif fixed_choice is None and len(candidates) == 1:
            raw_f = _prompt_choice("Fixed/overview camera (unique_id or OpenCV index; Enter to keep default 1)", "1", input_fn=input_fn)
            fixed_choice = raw_f
    else:
        if wrist_choice is None and len(candidates) >= 1:
            wrist_choice = candidates[0]
        if fixed_choice is None and len(candidates) >= 2:
            fixed_choice = candidates[1]

    # 3. 从单一事实源模板（configs/so101.example.json 或 wheel 内置）组装并严格校验
    template_path = bundled_config_path("so101.example.json")
    document = json.loads(template_path.read_text(encoding="utf-8"))
    document["python"] = str(Path(args.python).expanduser()) if args.python else _default_worker_python()
    document["port"] = port
    document["robot_id"] = args.robot_id
    _apply_camera_binding(document["cameras"]["wrist"], wrist_choice, inventory)
    _apply_camera_binding(document["cameras"]["fixed"], fixed_choice, inventory)
    validated = validate_deployment(document, base=output_path.parent)

    # 4. 原子写入配置（覆盖已有文件时自动备份）
    output_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path: str | None = None
    if output_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = output_path.with_name(f"{output_path.name}.bak-{stamp}")
        backup.write_text(output_path.read_text(encoding="utf-8"), encoding="utf-8")
        backup_path = str(backup)
    tmp_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, output_path)

    summary = {
        "output": str(output_path),
        "backup": backup_path,
        "python": validated["python"],
        "port": validated["port"],
        "robot_id": validated["robot_id"],
        "cameras": {
            name: {
                "unique_id": cam.get("unique_id"),
                "index_or_path": cam.get("index_or_path"),
            }
            for name, cam in validated["cameras"].items()
        },
        "detected_serial_ports": [p["port"] for p in ports],
        "detected_camera_count": len(candidates),
    }
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(f"Wrote SO-101 deployment config: {output_path}")
        if backup_path:
            print(f"Backup saved: {backup_path}")
        print(f"  python:   {validated['python']}")
        print(f"  port:     {validated['port']}")
        print(f"  robot_id: {validated['robot_id']}")
        for cam_name, cam in summary["cameras"].items():
            print(f"  camera.{cam_name}: unique_id={cam['unique_id']!r} index_or_path={cam['index_or_path']!r}")
        if len(candidates) < 2 and (args.wrist_camera is None or args.fixed_camera is None):
            print("Warning: fewer than 2 cameras were auto-detected; run `ni-rua-so101 cameras` after plugging both cameras in.")
        print("\nNext steps:")
        print(f"  1. Verify cameras: ni-rua-so101 cameras --deployment {output_path}")
        print("  2. Cold-start calibration (uncalibrated): ni-rua --config so101 --observation-profile rgb_proprio_v1")
        print("     Then open /calibration in the workbench UI to calibrate cameras & table plane.")
        print(f"  3. Optional commissioning: ni-rua-so101 commission --deployment {output_path}")
    return 0


# --------------------------------------------------------------------------------------------- charuco-board ---
def _format_mm(value_m: float) -> str:
    mm = value_m * 1000.0
    return f"{mm:.1f}" if abs(mm * 10.0 - round(mm * 10.0)) < 1e-6 else f"{mm:.2f}"


def _resolve_board_config(board_config: str) -> Path:
    candidate = Path(board_config).expanduser()
    if candidate.is_file():
        return candidate
    if board_config in ("so101_board", "so101_board.json"):
        return bundled_config_path("so101_board.json")
    return candidate


def _load_board_spec(args: argparse.Namespace) -> dict:
    from ..robot.calibration.workflow.charuco import DEFAULT_BOARD_SPEC, validate_board_spec

    spec = dict(DEFAULT_BOARD_SPEC)
    if args.board_config:
        raw = json.loads(_resolve_board_config(args.board_config).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("board config JSON must be an object")
        spec.update({k: v for k, v in raw.items() if not k.startswith("_") and k != "thickness_m"})
    if args.squares_x is not None:
        spec["squares_x"] = args.squares_x
    if args.squares_y is not None:
        spec["squares_y"] = args.squares_y
    if args.square_mm is not None:
        spec["square_m"] = args.square_mm / 1000.0
    if args.marker_mm is not None:
        spec["marker_m"] = args.marker_mm / 1000.0
    if args.dictionary is not None:
        spec["dictionary"] = args.dictionary
    return validate_board_spec(spec)


def render_board_page(spec: dict, *, px_per_mm: int = DEFAULT_PX_PER_MM):
    import cv2
    import numpy as np
    from ..robot.calibration.workflow.charuco import make_board, validate_board_spec

    spec = validate_board_spec(spec)
    board = make_board(spec)
    square_px = spec["square_m"] * 1000.0 * px_per_mm
    if abs(square_px - round(square_px)) > 1e-6:
        raise ValueError(f"square_m * 1000 * px_per_mm = {square_px} must be an integer")
    square_px = int(round(square_px))
    board_w, board_h = spec["squares_x"] * square_px, spec["squares_y"] * square_px
    page_w, page_h = int(A4_LANDSCAPE_MM[0] * px_per_mm), int(A4_LANDSCAPE_MM[1] * px_per_mm)
    if board_w > page_w - 20 * px_per_mm or board_h > page_h - 40 * px_per_mm:
        raise ValueError("board does not fit on landscape A4 with margins for the quiet zone and labels")
    board_img = board.generateImage((board_w, board_h), marginSize=0, borderBits=1)
    page = np.full((page_h, page_w), 255, dtype=np.uint8)
    x0, y0 = (page_w - board_w) // 2, (page_h - board_h) // 2
    page[y0:y0 + board_h, x0:x0 + board_w] = board_img

    bar_y = y0 + board_h + 8 * px_per_mm
    bar_len = 100 * px_per_mm
    cv2.rectangle(page, (x0, bar_y), (x0 + bar_len, bar_y + px_per_mm), 0, thickness=-1)
    for k in range(11):
        x = x0 + k * 10 * px_per_mm
        cv2.line(page, (x, bar_y - 2 * px_per_mm), (x, bar_y + 3 * px_per_mm), 0, thickness=max(1, px_per_mm // 6))
    font_scale = px_per_mm / 12.0
    thickness = max(1, px_per_mm // 8)
    cv2.putText(page, "100 mm", (x0 + bar_len + 4 * px_per_mm, bar_y + 2 * px_per_mm), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, 0, thickness, cv2.LINE_AA)
    label = (
        f"NI-RUA SO-101 ChArUco {spec['squares_x']}x{spec['squares_y']}  square {_format_mm(spec['square_m'])} mm"
        f"  marker {_format_mm(spec['marker_m'])} mm  {spec['dictionary']}  - print at 100% (actual size)"
    )
    cv2.putText(page, label, (x0, y0 - 8 * px_per_mm), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 0, thickness, cv2.LINE_AA)
    info = {
        "dpi": px_per_mm * 25.4,
        "page_mm": list(A4_LANDSCAPE_MM),
        "board_mm": [board_w / px_per_mm, board_h / px_per_mm],
        "board": spec,
    }
    return page, info


def cmd_charuco_board(args: argparse.Namespace) -> int:
    import cv2

    spec = _load_board_spec(args)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    page, info = render_board_page(spec, px_per_mm=args.px_per_mm)
    png = output_dir / "so101_charuco_board.png"
    pdf = output_dir / "so101_charuco_board.pdf"
    try:
        from PIL import Image
    except ImportError:
        if not cv2.imwrite(str(png), page):
            raise OSError(f"failed to write {png}")
        pdf_path = None
    else:
        image = Image.fromarray(page, mode="L")
        dpi = (info["dpi"], info["dpi"])
        image.save(png, dpi=dpi)
        image.save(pdf, "PDF", resolution=info["dpi"])
        pdf_path = str(pdf)
    print(f"PNG: {png}")
    print(f"PDF: {pdf_path}" if pdf_path else "PDF: not written (Pillow not installed)")
    print(
        f"Print on A4 landscape at 100% / actual size (no 'fit to page'). Expected square edge: "
        f"{_format_mm(spec['square_m'])} mm, scale bar: 100.0 mm."
    )
    return 0


# ------------------------------------------------------------------------------------------------ commission ---
class _CommissionAbort(RuntimeError):
    pass


def _summarise_move(result: dict, q_start, q_target) -> dict:
    import numpy as np

    trace = result.get("trace") or []
    delta = np.subtract(q_target, q_start)
    joint = int(np.argmax(np.abs(delta)))
    out = {
        "status": result["status"],
        "reason": result.get("reason"),
        "duration_s": result.get("wall_clock_duration_s"),
        "commands": result.get("segment_commands"),
        "uncorrected_error_rad": result.get("uncorrected_joint_error_rad"),
        "final_error_rad": result.get("final_joint_error_rad"),
        "correction_rounds": len((result.get("arrival_correction") or {}).get("rounds", [])),
        "diagnostic": result.get("diagnostic"),
    }
    if result["status"] != "completed":
        out["trace"] = [{"t_s": s["t_s"], "q_rad": [round(v, 4) for v in s["q_rad"]]} for s in trace]
        out["dispatched"] = [[round(v, 4) for v in d["q_goal_rad"]] for d in result.get("dispatched", [])]
    if trace and abs(delta[joint]) > 1e-6:
        q0 = trace[0]["q_rad"][joint]
        moved = [abs(s["q_rad"][joint] - q0) for s in trace]
        ramp_s = abs(delta[joint]) / result["_speed"]
        reach = next((s["t_s"] for s, m in zip(trace, moved) if m >= 0.9 * abs(delta[joint])), None)
        out.update(
            joint=ARM_JOINTS[joint],
            commanded_rad=round(float(delta[joint]), 4),
            measured_rad=round(float(trace[-1]["q_rad"][joint] - q0), 4),
            direction_ok=bool(np.sign(trace[-1]["q_rad"][joint] - q0) == np.sign(delta[joint])),
            ramp_s=round(ramp_s, 3),
            t90_s=reach,
            lag_after_ramp_s=None if reach is None else round(reach - ramp_s, 3),
            max_tracking_lag_rad=round(
                max(abs(np.subtract(d["q_goal_rad"], s["q_rad"])[joint]) for d, s in zip(result.get("dispatched", []), trace[1:])),
                4,
            ) if result.get("dispatched") else None,
        )
    return out


def cmd_commission(args: argparse.Namespace) -> int:
    from ..robot.backends.so101 import contract
    from ..robot.backends.so101.worker import SO101Adapter

    if not all(0 < value <= limit for value, limit in ((args.step_rad, 0.2), (args.sweep_rad, 0.5), (args.cartesian_m, 0.04))):
        raise SystemExit("amplitudes outside the commissioning envelope (step<=0.2 rad, sweep<=0.5 rad, cartesian<=0.04 m)")
    deployment_path = Path(args.deployment).expanduser().resolve() if args.deployment else default_so101_deployment_path()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out).expanduser() if args.out else (Path("~/.cache/ni-rua/so101").expanduser() / f"commission-{stamp}")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    (out / "images").mkdir()

    deployment = load_deployment(deployment_path)
    report = {
        "schema": "ni_rua.so101.commissioning.v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "deployment": str(deployment_path),
        "arguments": {k: v for k, v in vars(args).items() if k != "handler"},
        "steps": {},
        "ok": False,
    }
    adapter = SO101Adapter(deployment, out / "worker", "rgb_proprio_v1", start_telemetry=False)
    report["preflight"] = getattr(adapter.driver, "preflight_report", None)
    speed = deployment["motion"]["max_joint_speed_rad_s"]
    limits = adapter.kin.joint_limits_rad(adapter.model)
    margin = deployment["motion"]["joint_limit_margin_rad"]

    def snapshot(label: str):
        observation = adapter.observe()
        for camera in observation["cameras"]:
            source = Path(camera.get("rgb_path") or camera.get("png_path"))
            target = out / "images" / f"{label}_{camera['camera_id']}{source.suffix}"
            target.write_bytes(source.read_bytes())
        return observation["robot_state"]["arms"]["right"]

    def joints():
        return list(adapter.read()["q_rad"])

    def move(label, q_target, *, correct=True, tolerance=None, jaw=None, speed_scale=1.0):
        for name, value in zip(ARM_JOINTS, q_target):
            lower, upper = limits[name]
            if not lower + margin <= value <= upper - margin and label != "return_rest":
                raise _CommissionAbort(f"{label}: {name}={value:.3f} rad outside URDF limits minus {margin} rad margin")
        q_start = joints()
        result = adapter.move_joints(q_target, jaw, tolerance, speed_scale=speed_scale, correct_sag=correct, action_id=label)
        result["_speed"] = speed * speed_scale
        summary = _summarise_move(result, q_start, q_target)
        report["steps"][label] = summary
        print(
            f"{label:28s} {summary['status']:10s} dur={summary['duration_s']}s final_err="
            f"{None if summary['final_error_rad'] is None else round(max(map(abs, summary['final_error_rad'])), 4)}"
            f" uncorrected={None if summary['uncorrected_error_rad'] is None else round(max(map(abs, summary['uncorrected_error_rad'])), 4)}",
            flush=True,
        )
        if result["status"] != "completed":
            raise _CommissionAbort(f"{label}: {result['status']} {result.get('reason')}")
        return summary

    report["rest_q_rad"] = joints()
    try:
        snapshot("rest")
        ready = list(args.ready_q_rad)
        move("rest_to_ready", ready, tolerance=0.05)
        report["ready_tcp"] = snapshot("ready")["ee_pose"]

        report["joint_characterisation"] = {}
        for index, name in enumerate(ARM_JOINTS):
            rows = {}
            for sign in (1, -1):
                target = list(ready)
                target[index] += sign * args.step_rad
                key = "plus" if sign > 0 else "minus"
                rows[key] = move(f"{name}_{key}_raw", target, correct=False, tolerance=0.2)
                corrected = move(f"{name}_{key}_corrected", target, tolerance=0.02)
                rows["corrected_final_error_rad"] = corrected["final_error_rad"][index]
            move(f"{name}_center", ready, tolerance=0.02)
            report["joint_characterisation"][name] = {
                "direction_ok": rows["plus"]["direction_ok"] and rows["minus"]["direction_ok"],
                "sag_plus_rad": rows["plus"]["uncorrected_error_rad"][index],
                "sag_minus_rad": rows["minus"]["uncorrected_error_rad"][index],
                "t90_s": [rows["plus"]["t90_s"], rows["minus"]["t90_s"]],
                "lag_after_ramp_s": [rows["plus"]["lag_after_ramp_s"], rows["minus"]["lag_after_ramp_s"]],
            }

        for sign, label in ((1, "pan_sweep_plus"), (-1, "pan_sweep_minus")):
            target = list(ready)
            target[0] += sign * args.sweep_rad
            move(label, target, tolerance=0.03)
        move("pan_sweep_center", ready, tolerance=0.02)

        report["gripper"] = {}
        for goal in (1.0, 0.0, args.jaw_park):
            result = adapter.move_joints(ready, goal, 0.05, action_id=f"jaw_{goal}")
            jaw = result.get("gripper_execution") or {}
            report["gripper"][str(goal)] = {
                key: jaw.get(key)
                for key in ("measured_normalized", "measured_jaw_rad", "jaw_settled", "measured_vs_goal_normalized")
            }
            print(f"jaw {goal:<4} -> measured {jaw.get('measured_normalized')} settled={jaw.get('jaw_settled')}", flush=True)
            if result["status"] != "completed":
                raise _CommissionAbort(f"jaw {goal}: {result['status']} {result.get('reason')}")

        caps = adapter.capabilities()
        report["cartesian"] = {}
        base = adapter.robot_state(adapter.read())["arms"]["right"]["ee_pose"]
        for axis, name in enumerate("xyz"):
            for sign in (1, -1):
                position = list(base["position_m"])
                position[axis] += sign * args.cartesian_m
                action = contract.compile_action(
                    "move",
                    dict(arm="right", position_m=position, quaternion_xyzw=base["quaternion_xyzw"], observation_id="commissioning"),
                    COMMISSION_IDENTITY,
                    caps,
                )
                action["path"].update(position_tolerance_m=args.cartesian_tolerance_m, orientation_tolerance_rad=math.pi)
                receipt = adapter.execute(action, lambda _update: None, threading.Event())
                label = f"{name}{'+' if sign > 0 else '-'}"
                waypoint = (receipt.get("waypoint_results") or [{}])[0]
                report["cartesian"][label] = {
                    "status": receipt["status"],
                    "reason": receipt.get("reason"),
                    "position_error_m": receipt.get("position_error_m"),
                    "orientation_error_rad": receipt.get("orientation_error_rad"),
                    "ik_position_error_m": (waypoint.get("ik") or {}).get("position_error_m"),
                    "correction_rounds": len((waypoint.get("arrival_correction") or {}).get("rounds", [])),
                    "duration_s": receipt.get("wall_clock_duration_s"),
                }
                print(
                    f"cartesian {label}: {receipt['status']} err={receipt.get('position_error_m')} "
                    f"rounds={report['cartesian'][label]['correction_rounds']}",
                    flush=True,
                )
                if receipt["status"] != "completed":
                    raise _CommissionAbort(f"cartesian {label}: {receipt['status']} {receipt.get('reason')}")
                back = contract.compile_action(
                    "move",
                    dict(arm="right", position_m=base["position_m"], quaternion_xyzw=base["quaternion_xyzw"], observation_id="commissioning"),
                    COMMISSION_IDENTITY,
                    caps,
                )
                back["path"].update(position_tolerance_m=args.cartesian_tolerance_m, orientation_tolerance_rad=math.pi)
                receipt = adapter.execute(back, lambda _update: None, threading.Event())
                if receipt["status"] != "completed":
                    raise _CommissionAbort(f"cartesian {label} return: {receipt['status']} {receipt.get('reason')}")
        snapshot("after_cartesian")
        report["ok"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        print("ABORT:", report["error"], flush=True)
        report["abort_stop"] = adapter._stop()
    finally:
        try:
            if adapter.read() and report.get("ok") is not None:
                result = adapter.move_joints(list(args.ready_q_rad), args.jaw_park, 0.05, action_id="park_ready")
                report["park_ready"] = {k: result.get(k) for k in ("status", "reason", "max_joint_error_rad")}
                print("park_ready", report["park_ready"], flush=True)
        except BaseException as error:
            report["park_ready"] = {"status": "error", "reason": f"{type(error).__name__}: {error}"}
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (out / "commissioning.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        adapter.close()
    print("REPORT", out / "commissioning.json", "ok=", report["ok"], flush=True)
    return 0 if report["ok"] else 1


# ----------------------------------------------------------------------------------------------------- parser ---
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ni-rua-so101",
        description="SO-101 standalone hardware setup, camera discovery, ChArUco board, and commissioning CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_cameras = sub.add_parser("cameras", help="List video devices and verify SO-101 deployment camera bindings (read-only).")
    p_cameras.add_argument("--deployment", help="SO-101 deployment JSON; also resolve its cameras as the worker would")
    p_cameras.add_argument("--json", action="store_true", help="Print the device list and binding record as JSON")
    p_cameras.set_defaults(handler=cmd_cameras)

    p_init = sub.add_parser(
        "init-config",
        help="Detect USB serial ports + UVC cameras and write ~/.config/ni-rua/so101.local.json.",
    )
    p_init.add_argument("--output", help="Output deployment JSON path (default: ~/.config/ni-rua/so101.local.json)")
    p_init.add_argument("--python", help="Python interpreter for the SO-101 hardware worker")
    p_init.add_argument("--port", help="USB serial port path for the Feetech bus (auto-detected if omitted)")
    p_init.add_argument("--robot-id", default="so101_follower", help="LeRobot calibration robot_id (default: so101_follower)")
    p_init.add_argument("--wrist-camera", help="Wrist camera unique_id, device path, or OpenCV index")
    p_init.add_argument("--fixed-camera", help="Fixed/overview camera unique_id, device path, or OpenCV index")
    p_init.add_argument("--non-interactive", "--yes", "-y", action="store_true", dest="non_interactive",
                        help="Skip interactive prompts and use flags + auto-detection")
    p_init.add_argument("--force", action="store_true", help="Overwrite an existing deployment file (saving a .bak backup)")
    p_init.add_argument("--json", action="store_true", help="Print generated config summary as JSON")
    p_init.set_defaults(handler=cmd_init_config)

    p_board = sub.add_parser("charuco-board", help="Render the SO-101 ChArUco calibration board to A4 PNG + PDF.")
    p_board.add_argument("--output-dir", default=".", help="Directory for so101_charuco_board.png/.pdf")
    p_board.add_argument("--px-per-mm", type=int, default=DEFAULT_PX_PER_MM,
                         help="Raster resolution in pixels per millimetre (default 24 = 609.6 dpi)")
    p_board.add_argument("--board-config", help="JSON board spec file or bundled preset name 'so101_board'")
    p_board.add_argument("--squares-x", type=int, help="Override chessboard square count along X")
    p_board.add_argument("--squares-y", type=int, help="Override chessboard square count along Y")
    p_board.add_argument("--square-mm", type=float, help="Override chessboard square edge length (mm)")
    p_board.add_argument("--marker-mm", type=float, help="Override ArUco marker edge length (mm)")
    p_board.add_argument("--dictionary", help="Override OpenCV ArUco dictionary name (e.g. DICT_5X5_100)")
    p_board.set_defaults(handler=cmd_charuco_board)

    p_comm = sub.add_parser("commission", help="Run the 6-step SO-101 first-power-on commissioning sequence.")
    p_comm.add_argument("--deployment", help="SO-101 deployment JSON (defaults to configs/so101.local.json or ~/.config/ni-rua/so101.local.json)")
    p_comm.add_argument("--out", help="Output directory (must be new or empty; defaults to ~/.cache/ni-rua/so101/commission-<UTC>)")
    p_comm.add_argument("--ready-q-rad", type=float, nargs=5, default=[0.0, -0.95, 0.9, 1.0, 0.0],
                        help="Ready pose (rad, shoulder_pan..wrist_roll)")
    p_comm.add_argument("--step-rad", type=float, default=0.1, help="Per-joint step amplitude (rad)")
    p_comm.add_argument("--sweep-rad", type=float, default=0.3, help="shoulder_pan speed sweep amplitude (rad)")
    p_comm.add_argument("--cartesian-m", type=float, default=0.02, help="Cartesian test displacement (m)")
    p_comm.add_argument("--cartesian-tolerance-m", type=float, default=0.005, help="Cartesian arrival tolerance (m)")
    p_comm.add_argument("--jaw-park", type=float, default=0.5, help="Jaw goal after the gripper test (normalized)")
    p_comm.set_defaults(handler=cmd_commission)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
