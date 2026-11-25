"""Read-only SO-101 power-on self-check, run before torque is ever enabled.

Every check only reads servo registers; nothing is written and nothing moves.
The first failing check stops startup with a specific code so the operator
knows what to fix (power, cabling, calibration, mapping). There is no fallback
to a simulated arm: a real-hardware session either passes or does not start.

Thresholds come from the deployment file section ``preflight``.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math

SCHEMA = "ni_rua.so101.preflight.v1"
TICKS_PER_TURN = 4096  # STS3215 encoder ticks per shaft turn


class PreflightError(RuntimeError):
    """Startup refused. ``code`` is stable; ``report`` is the full check record."""

    def __init__(self, code, detail, report):
        super().__init__(f"so101_preflight_failed:{code}: {detail}")
        self.code, self.detail, self.report = code, detail, report


def run_preflight(bus, *, calibrated, deployment, limits_rad, to_rad, port_exists=True):
    """Inspect a connected (handshake-free) LeRobot Feetech bus.

    ``bus``: provides ``motors`` ({name: Motor(id, model)}), ``model_number_table``,
    ``calibration`` ({name: MotorCalibration}), ``broadcast_ping()`` and
    ``sync_read(register, normalize=..., num_retry=...)``.
    ``calibrated``: zero-argument callable; True when the servo EEPROM calibration
    matches the LeRobot calibration file.
    ``limits_rad``: {joint: (lower, upper)} URDF limits in rad for the arm joints.
    ``to_rad``: (joint, degrees) -> URDF rad using the deployment joint mapping.
    Returns the report (``ok`` True) or raises ``PreflightError``.
    """
    limits = deployment["preflight"]
    retry = deployment["num_read_retries"]
    report = {"schema": SCHEMA, "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "port": deployment["port"], "ok": False, "checks": []}

    def record(name, ok, **detail):
        report["checks"].append({"name": name, "ok": bool(ok), **detail})

    def fail(code, detail):
        report.update(ok=False, error=code, detail=detail)
        raise PreflightError(code, detail, report)

    record("serial_port", port_exists, port=deployment["port"])
    if not port_exists:
        fail("serial_port_missing", f"{deployment['port']} not present; check the USB cable")

    expected = {motor.id: (name, bus.model_number_table[motor.model]) for name, motor in bus.motors.items()}
    found = bus.broadcast_ping(num_retry=retry) or {}
    missing = sorted(set(expected) - set(found))
    wrong_model = {expected[i][0]: found[i] for i in expected if i in found and found[i] != expected[i][1]}
    record("servo_presence", not missing and not wrong_model, expected_ids=sorted(expected),
           found=dict(sorted(found.items())), missing_ids=missing)
    if not found:
        fail("servo_power_off", "no servo answered the broadcast ping while the USB port is present; "
                                "check the 12 V servo supply and the cable to the first servo")
    if missing:
        fail("servo_missing", "servo ids " + ",".join(map(str, missing)) + " did not answer; check the daisy-chain cabling")
    if wrong_model:
        fail("servo_model_mismatch", f"unexpected model numbers {wrong_model}")

    matches = bool(calibrated())
    record("motor_calibration", matches, robot_id=deployment["robot_id"])
    if not matches:
        fail("motor_calibration_mismatch", "servo EEPROM calibration differs from the LeRobot calibration file "
                                           f"for robot_id {deployment['robot_id']}; run lerobot-calibrate")

    voltage = {name: value / 10.0 for name, value in bus.sync_read("Present_Voltage", normalize=False, num_retry=retry).items()}
    temperature = {name: float(value) for name, value in bus.sync_read("Present_Temperature", normalize=False, num_retry=retry).items()}
    status = {name: int(value) for name, value in bus.sync_read("Status", normalize=False, num_retry=retry).items()}
    low = {n: v for n, v in voltage.items() if v < limits["min_voltage_v"]}
    high = {n: v for n, v in voltage.items() if v > limits["max_voltage_v"]}
    hot = {n: t for n, t in temperature.items() if t > limits["max_temperature_c"]}
    errors = {n: s for n, s in status.items() if s}
    record("supply_voltage", not low and not high, voltage_v=voltage,
           range_v=[limits["min_voltage_v"], limits["max_voltage_v"]])
    record("temperature", not hot, temperature_c=temperature, max_c=limits["max_temperature_c"])
    record("servo_status", not errors, status_bits=status,
           note="nonzero = latched servo error bits; see the Feetech STS3215 status register")
    if low or high:
        fail("supply_voltage_out_of_range", f"servo voltage {low or high} V outside "
                                            f"[{limits['min_voltage_v']}, {limits['max_voltage_v']}] V")
    if hot:
        fail("servo_overtemperature", f"servo temperature {hot} degC above {limits['max_temperature_c']} degC")
    if errors:
        fail("servo_error_status", f"servo status error bits {errors}")

    raw = bus.sync_read("Present_Position", normalize=False, num_retry=retry)
    degrees = bus.sync_read("Present_Position", num_retry=retry)
    # Raw ticks past the hand-recorded calibration range: a relaxed arm on a mechanical end stop reads a few ticks
    # beyond it; only an excess above preflight.calibration_range_tolerance_rad (shaft rad, 4096 ticks per turn) fails.
    range_tolerance_ticks = limits.get("calibration_range_tolerance_rad", 0.0) * TICKS_PER_TURN / (2 * math.pi)
    beyond_calibration = {name: (cal.range_min - raw[name] if raw[name] < cal.range_min else raw[name] - cal.range_max)
                          for name, cal in bus.calibration.items()
                          if name in raw and not cal.range_min <= raw[name] <= cal.range_max}
    outside_calibration = {name: raw[name] for name, excess in beyond_calibration.items()
                           if excess > range_tolerance_ticks}
    tolerance = limits["urdf_limit_tolerance_rad"]
    joints_rad, outside_urdf = {}, {}
    for name, (lower, upper) in limits_rad.items():
        value = to_rad(name, float(degrees[name]))
        joints_rad[name] = round(value, 5)
        if not math.isfinite(value) or value < lower - tolerance or value > upper + tolerance:
            outside_urdf[name] = round(value, 4)
    record("joint_range", not outside_calibration and not outside_urdf, q_rad=joints_rad, raw_ticks=dict(raw),
           urdf_limit_tolerance_rad=tolerance, beyond_calibration_ticks=beyond_calibration,
           calibration_range_tolerance_ticks=round(range_tolerance_ticks, 1))
    if outside_calibration:
        fail("joint_outside_calibration_range", f"raw positions {outside_calibration} outside the LeRobot calibration range")
    if outside_urdf:
        fail("joint_outside_urdf_limits", f"mapped joints {outside_urdf} rad outside URDF limits by more than "
                                          f"{tolerance} rad; check deployment joint_mapping signs/offsets")
    report["ok"] = True
    return report
