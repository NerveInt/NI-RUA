"""SO-101 hardware seams: arm driver, cameras and clock, each with a test fake.

Only ``LeRobotSO101Driver`` and ``OpenCVCameraSource`` touch hardware. They
are imported lazily inside the ``so101`` interpreter's worker; the parent
process and CPU tests never open a serial port or a camera.

Units at this boundary are URDF radians for arm joints, and ``goal_normalized``
(0 closed, 1 open) plus URDF jaw radians for the gripper. LeRobot's degree and
RANGE_0_100 conventions stay inside ``LeRobotSO101Driver``.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import sys
import threading
import time
import traceback

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
#: One STS3215 position-encoder tick in rad (12-bit magnetic encoder: 4096 ticks per turn, about 0.00153 rad). A
#: measured joint angle changes in multiples of this; a change of at most one tick between two reads is reading noise.
ENCODER_RESOLUTION_RAD = 2 * math.pi / 4096


#: One count of the STS3215 ``Present_Current`` register in mA (Feetech STS memory table: 6.5 mA per count).
STS_CURRENT_MA_PER_COUNT = 6.5
#: Contiguous STS3215 telemetry block: ``Present_Load`` (60, 2 B), ``Present_Voltage`` (62, 1 B),
#: ``Present_Temperature`` (63, 1 B) through ``Present_Current`` (69, 2 B) -> 11 bytes (60..70 inclusive).
#: Reading one 11-byte block instead of four separate ``sync_read`` calls cuts bus traffic 4x and prevents
#: ``scservo_sdk.readRx`` (which slices ``rxpacket[5:5+length]`` without checking ``PKT_LENGTH``) from accepting a
#: delayed 1- or 2-byte reply (e.g. ``Present_Position``) as ``Present_Temperature``.
STS_STATUS_ADDR = 60
STS_STATUS_LENGTH = 11


def sts_load_pct(raw):
    """Motor load (percent of maximum torque, signed) of one raw 16-bit STS3215 ``Present_Load`` word: bits 0..9 are
    the magnitude in 0.1 % units and bit 10 is the direction (``STS_SMS_SERIES_ENCODINGS_TABLE["Present_Load"] = 10``).
    Example: ``sts_load_pct(0x04C8)`` -> ``-20.0``."""
    raw = int(raw)
    magnitude = raw & 0x03FF
    return (-magnitude if (raw >> 10) & 1 else magnitude) / 10.0


def sts_current_ma(raw):
    """Motor current (mA, signed) of one raw STS3215 ``Present_Current`` word: bits 0..14 are the magnitude in
    counts of ``STS_CURRENT_MA_PER_COUNT`` and bit 15 the direction (sign-magnitude, like ``Present_Load``, which
    LeRobot decodes; it does not decode this register). Example: ``sts_current_ma(0x8010)`` -> ``-104.0``."""
    raw = int(raw)
    return round((-1 if raw & 0x8000 else 1) * (raw & 0x7FFF) * STS_CURRENT_MA_PER_COUNT, 1)


def decode_sts_status_block(data):
    """Decode one motor's 11-byte ``STS_STATUS_ADDR..+STS_STATUS_LENGTH`` reply into ``{temperature_c, load,
    voltage_v, current_ma}`` (degC, signed % of max torque, V, signed mA), or ``None`` when ``data`` is missing or
    shorter/longer than ``STS_STATUS_LENGTH`` (e.g. a stale 2-byte ``Present_Position`` frame sliced to 3 bytes)."""
    if not isinstance(data, (list, tuple, bytes, bytearray)) or len(data) != STS_STATUS_LENGTH:
        return None
    b = [int(x) & 0xFF for x in data]
    return {
        "temperature_c": float(b[3]),
        "load": sts_load_pct(b[0] | (b[1] << 8)),
        "voltage_v": float(b[2]) / 10.0,
        "current_ma": sts_current_ma(b[9] | (b[10] << 8)),
    }


def lerobot_max_relative_target(deployment):
    """``SO101FollowerConfig.max_relative_target`` from ``deployment`` (``None`` when ``max_relative_target_deg`` is
    ``None``).

    ``SO101Follower(use_degrees=True)`` normalizes ``ARM_JOINTS`` in degrees, where ``max_relative_target_deg`` (deg per
    command) applies directly, while normalizing ``"gripper"`` in ``MotorNormMode.RANGE_0_100`` (0..100). Passing a
    scalar float would cap ``"gripper"`` to ``+-max_relative_target_deg`` percent of full stroke (~0.19 rad at 10 deg),
    silently clamping wide jaw goals and spamming LeRobot safety warnings. Passing a per-motor dict keeps the degree
    cap on every arm joint while leaving the full configured ``lerobot_closed..lerobot_open`` span unclamped on
    ``"gripper"``."""
    limit = deployment["max_relative_target_deg"]
    if limit is None:
        return None
    arm_cap = float(limit)
    g = deployment["gripper"]
    jaw_span = max(100.0, abs(float(g["lerobot_open"]) - float(g["lerobot_closed"])))
    return {**{name: arm_cap for name in ARM_JOINTS}, "gripper": float(jaw_span)}


class Clock:
    """Real monotonic time; ``utc`` converts a monotonic stamp to ISO-8601 UTC."""

    def __init__(self):
        self._offset = time.time() - time.monotonic()

    def now(self):
        return time.monotonic()

    def sleep(self, seconds):
        if seconds > 0:
            time.sleep(seconds)

    def utc(self, monotonic=None):
        stamp = (self.now() if monotonic is None else monotonic) + self._offset
        return datetime.fromtimestamp(stamp, timezone.utc).isoformat().replace("+00:00", "Z")


class FakeClock(Clock):
    """Deterministic clock for tests: ``sleep`` advances time instantly."""

    def __init__(self, start=1000.0):
        self.t = float(start)
        self._lock = threading.Lock()
        self._epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def now(self):
        with self._lock:
            return self.t

    def sleep(self, seconds):
        with self._lock:
            self.t += max(0.0, float(seconds))

    def utc(self, monotonic=None):
        stamp = self.now() if monotonic is None else monotonic
        return (self._epoch + timedelta(seconds=stamp)).isoformat().replace("+00:00", "Z")


def gripper_normalized_from_rad(deployment, jaw_rad):
    g = deployment["gripper"]
    return (jaw_rad - g["jaw_closed_rad"]) / (g["jaw_open_rad"] - g["jaw_closed_rad"])


def gripper_rad_from_normalized(deployment, normalized):
    g = deployment["gripper"]
    return g["jaw_closed_rad"] + normalized * (g["jaw_open_rad"] - g["jaw_closed_rad"])


class ArmDriver:
    """Interface. ``read`` returns ``{"q_rad": [5], "gripper_rad": float}``."""

    def connect(self):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError

    def send_goal(self, q_rad, gripper_normalized):
        """Write one joint position goal (5 arm rad + jaw normalized 0..1)."""
        raise NotImplementedError

    def set_torque(self, enabled):
        raise NotImplementedError

    def torque_enabled(self):
        """Read back torque state; ``None`` when the bus cannot confirm it."""
        return None

    def motor_status(self):
        """``{joint: {temperature_c, load, voltage_v, current_ma}}``: degC, percent of maximum torque (signed),
        V and mA (signed like the load); unreadable values are ``None`` (``current_ma`` may be absent on a driver
        that does not read it)."""
        return {}

    def disconnect(self):
        raise NotImplementedError


def pin_goal_to_present(bus, num_retry=0):
    """Write raw Goal_Position = raw Present_Position on every motor.

    Raw ticks avoid any normalization round-trip error. Safe with torque off
    (no motion) and makes a later torque-enable hold the current pose.
    """
    present = bus.sync_read("Present_Position", normalize=False, num_retry=num_retry)
    bus.sync_write("Goal_Position", present, normalize=False)
    return present


class LeRobotSO101Driver(ArmDriver):
    """LeRobot 0.6.1 ``SOFollower(use_degrees=True)`` over one Feetech bus."""

    def __init__(self, deployment):
        self.deployment = deployment
        self.mapping = deployment["joint_mapping"]
        self.robot = None

    def connect(self):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        from pathlib import Path
        d = self.deployment
        kwargs = dict(port=d["port"], id=d["robot_id"], use_degrees=True,
                      max_relative_target=lerobot_max_relative_target(d),
                      disable_torque_on_disconnect=d["disable_torque_on_disconnect"],
                      num_read_retries=d["num_read_retries"],
                      position_p_coefficient=d["servo"]["p_coefficient"],
                      position_i_coefficient=d["servo"]["i_coefficient"],
                      position_d_coefficient=d["servo"]["d_coefficient"])
        if d.get("calibration_dir"):
            kwargs["calibration_dir"] = Path(d["calibration_dir"])
        robot = SO101Follower(SO101FollowerConfig(**kwargs))
        # Not robot.connect(): its configure() re-enables torque via
        # bus.torque_disabled(), and a servo with torque on chases whatever
        # Goal_Position register is left over (e.g. from earlier teleop).
        # Connect the bus, refuse unverified calibration, pin raw goal to the
        # raw present position while torque is still off, then configure.
        import os
        from ni_rua.robot.backends.so101.preflight import run_preflight
        from ni_rua.robot.model import so101_kinematics
        limits = so101_kinematics.joint_limits_rad(so101_kinematics.load_model())
        self.preflight_report = None
        if not os.path.exists(d["port"]):
            run_preflight(None, calibrated=None, deployment=d, limits_rad={}, to_rad=None, port_exists=False)
        # handshake=False: a missing/unpowered servo must reach the preflight's specific diagnosis.
        robot.bus.connect(handshake=False)
        try:
            self.preflight_report = run_preflight(
                robot.bus, calibrated=lambda: robot.is_calibrated, deployment=d,
                limits_rad={name: limits[name] for name in ARM_JOINTS}, to_rad=self._to_rad)
            pin_goal_to_present(robot.bus, self.deployment["num_read_retries"])
            robot.configure()
            pin_goal_to_present(robot.bus, self.deployment["num_read_retries"])
        except BaseException:
            robot.bus.disconnect(disable_torque=False)
            raise
        self.robot = robot

    def _to_rad(self, name, degrees):
        m = self.mapping[name]
        return m["sign"] * math.radians(degrees) + m["offset_rad"]

    def _to_deg(self, name, radians):
        m = self.mapping[name]
        return math.degrees((radians - m["offset_rad"]) * m["sign"])

    def read(self):
        bus = self.robot.bus
        values = bus.sync_read("Present_Position", num_retry=self.deployment["num_read_retries"])
        g = self.deployment["gripper"]
        normalized = (float(values["gripper"]) - g["lerobot_closed"]) / (g["lerobot_open"] - g["lerobot_closed"])
        return {"q_rad": [self._to_rad(name, float(values[name])) for name in ARM_JOINTS],
                "gripper_rad": gripper_rad_from_normalized(self.deployment, normalized)}

    def send_goal(self, q_rad, gripper_normalized):
        g = self.deployment["gripper"]
        action = {f"{name}.pos": self._to_deg(name, float(q)) for name, q in zip(ARM_JOINTS, q_rad)}
        action["gripper.pos"] = g["lerobot_closed"] + float(gripper_normalized) * (g["lerobot_open"] - g["lerobot_closed"])
        self.robot.send_action(action)

    def set_torque(self, enabled):
        if enabled:
            self.robot.bus.enable_torque(num_retry=self.deployment["num_read_retries"])
        else:
            self.robot.bus.disable_torque(num_retry=self.deployment["num_read_retries"])

    def torque_enabled(self):
        values = self.robot.bus.sync_read("Torque_Enable", normalize=False,
                                          num_retry=self.deployment["num_read_retries"])
        states = {bool(value) for value in values.values()}
        return states.pop() if len(states) == 1 else None

    def motor_status(self):
        bus, out = self.robot.bus, {}
        retries = int(self.deployment["num_read_retries"])
        if hasattr(bus, "_setup_sync_reader") and hasattr(bus, "sync_reader"):
            motor_ids = [motor.id for motor in bus.motors.values()]
            reader = bus.sync_reader
            for attempt in range(1 + retries):
                try:
                    bus._setup_sync_reader(motor_ids, STS_STATUS_ADDR, STS_STATUS_LENGTH)
                    comm = reader.txRxPacket()
                    ok = bus._is_comm_success(comm) and all(
                        len(reader.data_dict.get(motor_id) or ()) == STS_STATUS_LENGTH for motor_id in motor_ids
                    )
                except Exception:
                    ok = False
                if ok:
                    break
                if attempt < retries:
                    clear = getattr(getattr(reader, "port", None), "clearPort", None)
                    if callable(clear):
                        try:
                            clear()
                        except Exception:
                            pass
            data_by_id = getattr(reader, "data_dict", None) or {}
            for name, motor in bus.motors.items():
                decoded = decode_sts_status_block(data_by_id.get(motor.id))
                out[name] = decoded if decoded is not None else {
                    "temperature_c": None, "load": None, "voltage_v": None, "current_ma": None
                }
            return out
        readings = {}
        for register in ("Present_Temperature", "Present_Load", "Present_Voltage", "Present_Current"):
            try:
                readings[register] = bus.sync_read(register, normalize=False, num_retry=retries)
            except Exception:
                readings[register] = {}
        for name in bus.motors:
            temperature = readings["Present_Temperature"].get(name)
            load = readings["Present_Load"].get(name)
            voltage = readings["Present_Voltage"].get(name)
            current = readings["Present_Current"].get(name)
            # STS3215 registers: temperature in degC, load in 0.1 % of max torque
            # (sign-magnitude decoded by LeRobot), voltage in 0.1 V, current in STS_CURRENT_MA_PER_COUNT.
            out[name] = {"temperature_c": None if temperature is None else float(temperature),
                         "load": None if load is None else float(load) / 10.0,
                         "voltage_v": None if voltage is None else float(voltage) / 10.0,
                         "current_ma": None if current is None else sts_current_ma(current)}
        return out

    def arm_load(self, names):
        """Raw STS3215 ``Present_Load`` (0.1 % of max torque, signed) of ``names`` (touch-off contact trigger)."""
        values = self.robot.bus.sync_read("Present_Load", list(names), normalize=False,
                                          num_retry=self.deployment["num_read_retries"])
        return {name: float(values[name]) for name in names}

    def disconnect(self):
        if self.robot is not None:
            # SOFollower.disconnect also closes its (unused, empty) camera dict.
            self.robot.disconnect()
            self.robot = None


class FakeArmDriver(ArmDriver):
    """First-order lag servo model for CPU tests.

    ``tau_s``: time constant (s) of measured joints toward the goal.
    ``blocked``: arm joint indices that never move (stall/tracking tests).
    ``jaw_min_normalized``: jaw cannot close below this (object in the jaw).
    ``fail_on_send``: raise ``OSError`` on the Nth goal write (1-based).
    ``sag_rad``: per-joint steady-state shortfall (goal - measured) under load, rad.
    """

    def __init__(self, deployment, clock, q_rad=(0.0, -1.0, 1.0, 0.6, 0.0), gripper_normalized=0.0, *,
                 tau_s=0.05, blocked=(), jaw_min_normalized=None, fail_on_send=None, sag_rad=(0., 0., 0., 0., 0.)):
        self.deployment, self.clock, self.tau_s = deployment, clock, tau_s
        self.q = [float(v) for v in q_rad]
        self.jaw = gripper_rad_from_normalized(deployment, gripper_normalized)
        self.goal = list(self.q)
        self.jaw_goal = self.jaw
        self.blocked = set(blocked)
        self.jaw_min = None if jaw_min_normalized is None else gripper_rad_from_normalized(deployment, jaw_min_normalized)
        self.fail_on_send = fail_on_send
        self.sag = [float(v) for v in sag_rad]
        self.sends, self.torque, self.connected = 0, True, False
        self.last = clock.now()
        self.goals = []

    def _advance(self):
        now = self.clock.now()
        dt, self.last = now - self.last, now
        if not self.torque or dt <= 0:
            return
        alpha = 1 - math.exp(-dt / self.tau_s)
        for index in range(5):
            if index not in self.blocked:
                self.q[index] += (self.goal[index] - self.sag[index] - self.q[index]) * alpha
        target = self.jaw_goal if self.jaw_min is None else max(self.jaw_goal, self.jaw_min)
        self.jaw += (target - self.jaw) * alpha

    def connect(self):
        self.connected = True

    def read(self):
        self._advance()
        return {"q_rad": list(self.q), "gripper_rad": self.jaw}

    def send_goal(self, q_rad, gripper_normalized):
        self._advance()
        self.sends += 1
        if self.fail_on_send is not None and self.sends >= self.fail_on_send:
            raise OSError("fake serial write failure")
        self.goal = [float(v) for v in q_rad]
        self.jaw_goal = gripper_rad_from_normalized(self.deployment, gripper_normalized)
        self.goals.append((list(self.goal), float(gripper_normalized)))

    def set_torque(self, enabled):
        self._advance()
        self.torque = bool(enabled)

    def torque_enabled(self):
        return self.torque

    def motor_status(self):
        return {name: {"temperature_c": 30.0, "load": 0.0, "voltage_v": 12.0} for name in (*ARM_JOINTS, "gripper")}

    def disconnect(self):
        self.connected = False


class CameraSource:
    """Interface for one RGB camera with a background grabber.

    ``read_after(t, timeout_s)`` returns ``(rgb_uint8_HxWx3, capture_monotonic_s)``
    for the first frame captured after monotonic time ``t``. ``latest()`` returns
    the newest frame or ``None`` without waiting.
    """

    camera_id = None

    def set_frame_sink(self, sink):
        """Set an optional nonblocking ``sink(rgb, host_read_start_s, host_read_end_s)``.

        Seconds use the shared adapter monotonic clock; exposure time is unknown.
        Default None disables delivery. The sink must retain immutable pixels and
        never do encoding/I/O on the capture thread. Example: camera.set_frame_sink(recorder.submit).
        """
        self.frame_sink = sink

    def _deliver_frame(self, image, started, finished):
        sink = getattr(self, "frame_sink", None)
        if sink is not None:
            sink(image, started, finished)

    def start(self):
        raise NotImplementedError

    def latest(self):
        raise NotImplementedError

    def latest_timed(self):
        """Latest RGB frame and host timing, without waiting or opening another device.

        Returns None or a dict with image_rgb, capture_monotonic_s (the legacy
        stamp, seconds in the shared adapter clock), and host_read_interval_s.
        Unknown read completion/exposure timing stays None. Existing camera
        implementations inherit this conservative adapter; native UVC overrides
        it with the actual host read interval. Consumers must not mutate pixels.
        """
        latest = self.latest()
        if latest is None:
            return None
        image, stamp = latest
        return dict(image_rgb=image, capture_monotonic_s=stamp,
                    host_read_interval_s=[stamp, None], exposure_monotonic_s=None)

    def read_after(self, t, timeout_s):
        raise NotImplementedError

    def status(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


def rotate_frame(frame, rotate_deg=0):
    """Rotate an HxWxC frame clockwise by ``rotate_deg`` in ``(0, 90, 180, 270)``."""
    rotate_deg = int(rotate_deg or 0)
    if rotate_deg == 0:
        return frame
    import cv2
    codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    if rotate_deg not in codes:
        raise ValueError(f"so101_camera_invalid_rotate_deg:{rotate_deg}")
    return cv2.rotate(frame, codes[rotate_deg])


class OpenCVCameraSource(CameraSource):
    """OpenCV/UVC capture. Legacy stamps mark host read start, not exposure."""

    def __init__(self, camera_id, config, clock):
        self.camera_id, self.config, self.clock = camera_id, config, clock
        self.inventory, self.binding = None, None  # camera_devices.list_video_devices / resolve_camera results
        self.capture, self.thread = None, None
        self.frame, self.stamp, self.error = None, None, None
        self.read_finished = None
        self.condition = threading.Condition()
        self.running = False
        self.stamps = []

    def bind(self, inventory=None):
        """Resolve the configured identity to the device and OpenCV target to open; opens nothing.

        ``inventory`` is a ``camera_devices.list_video_devices()`` result shared by all cameras (read here when
        None). Raises ``camera_devices.CameraBindingError`` when ``unique_id`` / ``device_name`` does not name exactly
        one present device; never falls back to ``index_or_path``. Returns the binding record (operator only).
        """
        from .camera_devices import list_video_devices, resolve_camera
        inventory = list_video_devices() if inventory is None else inventory
        self.binding = resolve_camera(self.camera_id, self.config, inventory)
        self.inventory = inventory
        return self.binding

    def start(self):
        import cv2
        from .camera_devices import confirm_unchanged, list_video_devices
        if self.binding is None:
            self.bind()
        api = {"avfoundation": cv2.CAP_AVFOUNDATION, "v4l2": cv2.CAP_V4L2}.get(self.binding["api"], cv2.CAP_ANY)
        capture = cv2.VideoCapture(self.binding["opened"], api)
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if self.config.get("fourcc"):
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.config["fourcc"]))
        rotate_deg = int(self.config.get("rotate_deg", 0) or 0)
        raw_w, raw_h = ((self.config["height"], self.config["width"]) if rotate_deg in (90, 270)
                        else (self.config["width"], self.config["height"]))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, raw_w)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, raw_h)
        capture.set(cv2.CAP_PROP_FPS, self.config["fps"])
        if not capture.isOpened():
            capture.release()
            if sys.platform == "darwin":
                hint = (
                    "; on macOS, verify terminal Camera permission in System Settings -> Privacy & Security -> Camera "
                    "and check bindings with `ni-rua-so101 cameras`"
                )
            else:
                hint = "; verify camera permissions/availability and check bindings with `ni-rua-so101 cameras`"
            raise RuntimeError(f"so101_camera_unavailable:{self.camera_id}:opened={self.binding['opened']!r}{hint}")
        if self.binding["mode"] != "index":
            # The resolved index is valid only for the device list it came from (a camera plugged or unplugged
            # meanwhile shifts it): re-read the list after opening and refuse if it changed.
            try:
                confirm_unchanged(self.camera_id, self.inventory, list_video_devices())
            except BaseException:
                capture.release()
                raise
        self.capture, self.running = capture, True
        negotiated_fps = capture.get(cv2.CAP_PROP_FPS)
        self.recording_fps = negotiated_fps if math.isfinite(negotiated_fps) and negotiated_fps > 0 else self.config["fps"]
        self.recording_fps_source = "camera_negotiated" if self.recording_fps == negotiated_fps else "configured_fallback"
        self.thread = threading.Thread(target=self._grab, daemon=True, name=f"camera-{self.camera_id}")
        self.thread.start()

    def _grab(self):
        import cv2
        rotate_deg = int(self.config.get("rotate_deg", 0) or 0)
        while self.running:
            try:
                read_started = self.clock.now()
                ok, frame = self.capture.read()
                read_finished = self.clock.now()
                with self.condition:
                    if not ok or frame is None:
                        self.error = "frame_read_failed"
                    else:
                        if rotate_deg:
                            frame = rotate_frame(frame, rotate_deg)
                        height, width = frame.shape[:2]
                        if (width, height) != (self.config["width"], self.config["height"]):
                            self.error = f"frame_size_{width}x{height}_differs_from_configured"
                        else:
                            self.frame, self.stamp, self.error = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), read_started, None
                            self.read_finished = read_finished
                            self.stamps = (self.stamps + [read_started])[-30:]
                            try:
                                self._deliver_frame(self.frame, read_started, read_finished)
                            except Exception:
                                traceback.print_exc(file=sys.stderr)
                    self.condition.notify_all()
                if not ok and self.running:
                    time.sleep(0.05)
            except Exception as err:
                traceback.print_exc(file=sys.stderr)
                with self.condition:
                    self.error = f"camera_grab_error:{type(err).__name__}"
                    self.condition.notify_all()
                if self.running:
                    time.sleep(0.05)

    def latest(self):
        with self.condition:
            return None if self.frame is None else (self.frame, self.stamp)

    def latest_timed(self):
        """Nonblocking snapshot; host read interval in adapter-clock seconds, exposure unknown."""
        with self.condition:
            if self.frame is None:
                return None
            return dict(image_rgb=self.frame, capture_monotonic_s=self.stamp,
                        host_read_interval_s=[self.stamp, self.read_finished], exposure_monotonic_s=None)

    def read_after(self, t, timeout_s):
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while self.stamp is None or self.stamp <= t:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"so101_camera_frame_timeout:{self.camera_id}:{self.error or 'no_new_frame'}")
                self.condition.wait(remaining)
            return self.frame, self.stamp

    def status(self):
        with self.condition:
            fps = ((len(self.stamps) - 1) / (self.stamps[-1] - self.stamps[0])
                   if len(self.stamps) > 1 and self.stamps[-1] > self.stamps[0] else None)
            return {"ok": self.frame is not None and self.error is None, "fps": fps,
                    "last_frame_at": None if self.stamp is None else self.clock.utc(self.stamp), "error": self.error}

    def close(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.capture is not None:
            if self.thread is None or not self.thread.is_alive():
                self.capture.release()
            else:
                self._abandoned_capture = self.capture


class FakeCameraSource(CameraSource):
    """Synthetic frames stamped ``delay_s`` after the request on the shared clock."""

    def __init__(self, camera_id, width=64, height=48, clock=None, delay_s=0.01, fail=False):
        self.camera_id, self.width, self.height = camera_id, width, height
        self.clock, self.delay_s, self.fail = clock, delay_s, fail
        self.count, self.last = 0, None

    def start(self):
        pass

    def _frame(self, stamp):
        import numpy as np
        self.count += 1
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        image[..., 0] = (self.count * 37) % 255
        self.last = (image, stamp)
        self._deliver_frame(image, stamp, stamp)
        return self.last

    def latest(self):
        return self.last

    def read_after(self, t, timeout_s):
        if self.fail:
            raise TimeoutError(f"so101_camera_frame_timeout:{self.camera_id}:fake_failure")
        return self._frame(max(t, self.clock.now()) + self.delay_s)

    def status(self):
        return {"ok": not self.fail, "fps": None, "error": "fake_failure" if self.fail else None,
                "last_frame_at": None if self.last is None else self.clock.utc(self.last[1])}

    def close(self):
        pass
