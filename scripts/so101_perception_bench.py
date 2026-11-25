"""SO-101 perception bench: cubes at known positions -> the harness's own perception -> per-condition metrics.

Protocol and exact commands: docs/development/so101-perception-bench.md. Four subcommands, each a separate step so the
arm/camera part and the GPU part run in their own interpreters:

``snapshot`` (so101 env; camera permission: run from a terminal macOS allowed to use the camera, e.g. Ghostty)
    One frame of one camera at a chosen capture size, saved as PNG (for scripts/so101_remap_calibration.py).
``capture`` (so101 env; camera permission; moves the arm only with ``--move``)
    Runs the real worker adapter in-process (as scripts/so101_commission.py does: power-on self-check, speed caps,
    tracking/stall guards), moves through the ``--poses`` sequence with the operator joint move and saves
    ``--frames`` raw observations per pose (both camera images, measured joints, the active calibration) into
    ``<out>/frames/<index>-<pose>/``. Every target is checked before the arm is connected (planning limits, TCP inside
    the workspace box, no link mesh below the table; ``check_targets``); an error holds the measured position, and the
    arm returns to home.q_rad at the end unless ``--no-park-home``. Pose names: ``current`` (no motion), ``home`` (deployment home.q_rad), ``clear``
    (deployment perception.clear_view_q_rad, or ``--clear-q-rad``), ``parked`` (the TCP above the truth cubes'
    centroid, ``--park-height-m`` above the highest top, position-only local IK from home; or ``--parked-q-rad``) and ``q:<5 rad separated by />`` (e.g.
    ``q:0/-1.27/0.41/0.54/0``).
``perceive`` (NI-RUA env; the VGGT service reachable at ``--endpoint``)
    Imports every captured frame, in capture order, into one fresh temporary store and calls the harness's own
    attachment path (``obs3d.attachments``, reply ``observe``: the object table the model receives with every
    observation, its overlay image, robot_occlusion, and cross_view when the height method asks for it) with the
    deployment's profile. Writes ``<out>/perception-<method>.jsonl`` (a new file per height method): the model-visible JSON, the delivered image sizes, the
    service timing and latencies, and the camera / joint facts ``score`` needs; and per frame
    ``fk_silhouette_<method>.jpg``, the undistorted fixed image with the arm's FK silhouette tinted red.
``score`` (NI-RUA env, offline)
    Reads every perception-<method>.jsonl and the truth file and reports per condition (resolution x pose x height method):
    detection rate per cube, xy error, top-z error of each estimator separately (VGGT row ``top_z``, cross_view
    ``target_view_top_z``, ``contact_top_z``), whether the truth lies inside ``contact_top_z_range``, latency, image and
    text size, and for the occlusion report whether ``hidden`` agrees with the FK-silhouette truth of the true cube
    boxes. Writes ``<out>/metrics.json``.

Truth file (JSON, so101_base metres; the cubes placed with the arm, so their positions are FK positions)::

    {"table_z_m": -0.0162, "cubes": [{"colour": "white", "centre_xy_m": [0.28, -0.06], "top_z_m": 0.0128,
                                      "edge_m": 0.029}, ...]}

``table_z_m`` (optional): bottom of every cube (default: each frame's calibrated table height). ``edge_m``
(optional, default 0.029 m): footprint edge and height of the cube box used by the occlusion truth.

Nothing here is model context: the bench reads the same outputs the model reads and scores them offline.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCHEMA = "ni_rua.so101.perception_bench.v1"
DEFAULT_EDGE_M = 0.029
IDENTITY = dict(run_id="run-perception-bench", episode_id="ep-perception-bench", env_instance_id="env-perception-bench",
                env_generation=1)
HEIGHT_METHODS = {   # deployment perception switches per height method (perceive)
    "vggt": dict(attach_cross_view=False, contact_top_estimate=False),
    "cross_view": dict(attach_cross_view=True, contact_top_estimate=False),
    "contact": dict(attach_cross_view=True, contact_top_estimate=True),
}


def parse_size(text):
    """``"1920x1080"`` -> (1920, 1080) px."""
    width, height = (int(value) for value in str(text).lower().split("x"))
    return width, height


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ------------------------------------------------------------------ token estimates (documented formulas)
def token_estimates(width, height):
    """Estimated image input tokens of one ``width`` x ``height`` px image under two published provider rules
    (estimates for comparison only; the provider's usage report is the measurement):

    * ``tile_512``: fit inside 2048 x 2048, then scale the shorter side down to 768 if larger, then 85 + 170 per
      512 px tile (OpenAI high detail).
    * ``pixels_750``: fit the longer side to 1568 px, then width x height / 750 (Anthropic)."""
    w, h = float(width), float(height)
    s = min(1.0, 2048.0 / max(w, h))
    w, h = w * s, h * s
    s = min(1.0, 768.0 / min(w, h))
    w, h = w * s, h * s
    tiles = math.ceil(w / 512.0) * math.ceil(h / 512.0)
    a = min(1.0, 1568.0 / max(width, height))
    return dict(tile_512=int(85 + 170 * tiles), pixels_750=int(round(width * a * height * a / 750.0)))


# ------------------------------------------------------------------ matching and metrics (pure, tested)
def _percentiles(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return dict(n=0, p50=None, p90=None, mean=None, max=None)
    array = np.abs(np.asarray(values))
    return dict(n=len(values), p50=round(float(np.percentile(array, 50)), 4), p90=round(float(np.percentile(array, 90)), 4),
                mean=round(float(np.mean(values)), 4), max=round(float(array.max()), 4))


def table_rows(table):
    """[dict(id, centre_m, top_z, ...)] of a model-visible object table (columns + rows)."""
    if not isinstance(table, dict) or not isinstance(table.get("rows"), list):
        return []
    columns = table["columns"]
    return [dict(zip(columns, row)) for row in table["rows"]]


def match_rows(rows, cubes, radius_m):
    """{cube index: row index}: greedy nearest pairs (xy distance of the row centre to the cube centre, m), each
    row and cube used once, pairs farther than ``radius_m`` (m) left out. Held rows are not matched."""
    candidates = []
    for ci, cube in enumerate(cubes):
        for ri, row in enumerate(rows):
            if str(row.get("support") or "").startswith("held_") or not isinstance(row.get("centre_m"), list):
                continue
            distance = float(np.hypot(*(np.asarray(row["centre_m"][:2]) - np.asarray(cube["centre_xy_m"]))))
            if distance <= radius_m:
                candidates.append((distance, ci, ri))
    used_cubes, used_rows, out = set(), set(), {}
    for _, ci, ri in sorted(candidates):
        if ci not in used_cubes and ri not in used_rows:
            out[ci] = ri
            used_cubes.add(ci)
            used_rows.add(ri)
    return out


def _nearest(entries, cube, radius_m, key="centre_m"):
    best = None
    for entry in entries or []:
        if not isinstance(entry.get(key), list):
            continue
        distance = float(np.hypot(*(np.asarray(entry[key][:2]) - np.asarray(cube["centre_xy_m"]))))
        if distance <= radius_m and (best is None or distance < best[0]):
            best = (distance, entry)
    return None if best is None else best[1]


def cube_box(cube, table_z_m):
    """Truth box of a cube (robot_view box): footprint edge x edge at yaw 0 (the occlusion truth uses its projected
    convex outline, which a yaw changes by at most the footprint's diagonal), from the table (or the truth
    ``table_z_m``) to ``top_z_m``."""
    edge = float(cube.get("edge_m", DEFAULT_EDGE_M))
    return dict(centre_m=[*cube["centre_xy_m"], float(cube["top_z_m"]) - edge / 2], length_m=edge, width_m=edge,
                yaw_deg=0.0, bottom_z_m=float(table_z_m), top_z_m=float(cube["top_z_m"]))


def frame_scores(record, truth, *, match_radius_m, truth_covered=None):
    """Per-cube scores of one perception record: dict(cube index -> dict(detected, xy_error_m, top errors per
    estimator (m, estimate - truth), contact range hit, reported hidden fraction, truth covered fraction))."""
    shown = record["shown"]
    table = shown.get("object_table") if isinstance(shown, dict) else None
    rows = table_rows(table)
    cubes = truth["cubes"]
    matched = match_rows(rows, cubes, match_radius_m)
    cross = ((table or {}).get("cross_view") or {}).get("wrist") or {}
    cross_rows = {row.get("id"): row for row in cross.get("rows") or []}
    occlusion = (table or {}).get("robot_occlusion") or {}
    out = {}
    for ci, cube in enumerate(cubes):
        score = dict(detected=ci in matched, xy_error_m=None, vggt_top_error_m=None, cross_view_top_error_m=None,
                     contact_top_error_m=None, contact_range_hit=None, contact_range_m=None, reported_covered=None,
                     truth_covered=None if truth_covered is None else truth_covered.get(ci))
        if ci in matched:
            row = rows[matched[ci]]
            score["xy_error_m"] = round(float(np.hypot(*(np.asarray(row["centre_m"][:2]) - cube["centre_xy_m"]))), 4)
            if row.get("top_z") is not None:
                score["vggt_top_error_m"] = round(float(row["top_z"]) - cube["top_z_m"], 4)
            pair = cross_rows.get(row.get("id"))
            if pair is not None:
                if pair.get("target_view_top_z") is not None:
                    score["cross_view_top_error_m"] = round(float(pair["target_view_top_z"]) - cube["top_z_m"], 4)
                if pair.get("contact_top_z") is not None:
                    score["contact_top_error_m"] = round(float(pair["contact_top_z"]) - cube["top_z_m"], 4)
                    low, high = pair["contact_top_z_range"]
                    score["contact_range_m"] = round(float(high) - float(low), 4)
                    score["contact_range_hit"] = bool(low <= cube["top_z_m"] <= high)
        hidden = _nearest(occlusion.get("hidden"), cube, match_radius_m)
        score["reported_covered"] = None if hidden is None else float(hidden["covered"])
        out[ci] = score
    return out


def condition_key(record):
    return "%dx%d|%s|%s" % (record["fixed_size"][0], record["fixed_size"][1], record["pose"], record["height_method"])


def score_records(records, truth, *, match_radius_m=0.03, hidden_fraction=0.5, truth_covered=None):
    """Metrics per condition (``condition_key``) over perception records (perception-<method>.jsonl lines).

    ``match_radius_m`` (m): largest xy distance of a row centre from a cube centre that counts as that cube (default
    0.03 m, about one cube edge). ``hidden_fraction`` (0..1): the occlusion truth calls a cube hidden when at least this
    share of its true box outline is under the FK silhouette, and the report calls it hidden when its ``covered`` is
    at least this share (default 0.5). ``truth_covered``: {record index: {cube index: fraction}} from
    ``occlusion_truth`` (None: occlusion metrics left out). Returns {condition: metrics}."""
    groups = {}
    for index, record in enumerate(records):
        groups.setdefault(condition_key(record), []).append((index, record))
    out = {}
    for key, items in groups.items():
        per_cube, timing, sizes, occlusion = {}, [], [], []
        for index, record in items:
            scores = frame_scores(record, truth, match_radius_m=match_radius_m,
                                  truth_covered=None if truth_covered is None else truth_covered.get(index))
            for ci, score in scores.items():
                per_cube.setdefault(ci, []).append(score)
            timing.append(record.get("latency") or {})
            sizes.append(record.get("sizes") or {})
            table = (record.get("shown") or {}).get("object_table") or {}
            if isinstance(table, dict) and isinstance(table.get("robot_occlusion"), dict):
                occlusion.append(table["robot_occlusion"].get("table_covered"))
        cubes = {}
        for ci, scores in per_cube.items():
            cube = truth["cubes"][ci]
            hits = [s["contact_range_hit"] for s in scores if s["contact_range_hit"] is not None]
            truth_hidden = [s for s in scores if s["truth_covered"] is not None and s["truth_covered"] >= hidden_fraction]
            truth_visible = [s for s in scores if s["truth_covered"] is not None and s["truth_covered"] < hidden_fraction]
            reported = [s for s in scores if s["reported_covered"] is not None and s["truth_covered"] is not None]
            cubes[cube.get("colour") or str(ci)] = dict(
                frames=len(scores), detection_rate=round(sum(s["detected"] for s in scores) / len(scores), 3),
                xy_error_m=_percentiles([s["xy_error_m"] for s in scores]),
                vggt_top_error_m=_percentiles([s["vggt_top_error_m"] for s in scores]),
                cross_view_top_error_m=_percentiles([s["cross_view_top_error_m"] for s in scores]),
                contact_top_error_m=_percentiles([s["contact_top_error_m"] for s in scores]),
                contact_range_m=_percentiles([s["contact_range_m"] for s in scores]),
                contact_range_hit_rate=None if not hits else round(sum(hits) / len(hits), 3),
                occlusion=None if truth_covered is None else dict(
                    truth_hidden_frames=len(truth_hidden),
                    reported_when_hidden=None if not truth_hidden else round(
                        sum(s["reported_covered"] is not None and s["reported_covered"] >= hidden_fraction
                            for s in truth_hidden) / len(truth_hidden), 3),
                    detected_when_hidden=None if not truth_hidden else round(
                        sum(s["detected"] for s in truth_hidden) / len(truth_hidden), 3),
                    false_hidden_reports=sum(s["reported_covered"] is not None and s["reported_covered"] >= hidden_fraction
                                             for s in truth_visible),
                    covered_error=_percentiles([s["reported_covered"] - s["truth_covered"] for s in reported])))

        def stat(name, source):
            return _percentiles([item.get(name) for item in source])
        out[key] = dict(frames=len(items), cubes=cubes,
                        latency_s={name: stat(name, timing) for name in ("attach_s", "object_table_s", "cross_view_s")},
                        sizes={name: stat(name, sizes) for name in (
                            "fixed_image_bytes", "overlay_image_bytes", "text_chars", "fixed_tokens_tile_512",
                            "fixed_tokens_pixels_750")},
                        table_covered=_percentiles(occlusion))
    return out


def occlusion_truth(records, truth):
    """{record index: {cube index: fraction}}: share of each true cube box's projected outline under the FK silhouette
    of the frame's measured joints in the fixed camera (robot_view, the same render the report uses)."""
    from ni_rua.perception.external import held_perception as hp, robot_view
    from ni_rua.robot.model import so101_kinematics as kin
    body, model = hp.SO101Body(), kin.load_model()
    meshes = body.meshes(None)
    out = {}
    for index, record in enumerate(records):
        facts = record["fixed_camera"]
        camera = dict(K=np.asarray(facts["K"]), T=np.asarray(facts["T"]), width=facts["width"], height=facts["height"])
        poses = {"right": kin.fk_all(model, record["q_rad"][:5], gripper_rad=record["q_rad"][5])}
        robot, _ = robot_view.silhouettes(body, meshes, poses, camera, "fixed", {"right": "wrist"})
        table_z = truth.get("table_z_m", record.get("table_z_m"))
        out[index] = {ci: robot_view.box_covered(camera["K"], camera["T"], robot, cube_box(cube, table_z))
                      for ci, cube in enumerate(truth["cubes"])}
    return out


# ------------------------------------------------------------------ snapshot / capture (so101 env, hardware)
def cmd_snapshot(args):
    import cv2
    from ni_rua.robot.backends.so101.deployment import load_deployment
    from ni_rua.robot.backends.so101.driver import Clock, OpenCVCameraSource
    deployment = load_deployment(args.deployment)
    config = dict(deployment["cameras"][args.camera])
    if args.size:
        config["width"], config["height"] = parse_size(args.size)
    source = OpenCVCameraSource(args.camera, config, Clock())
    source.start()
    try:
        deadline = time.monotonic() + args.timeout_s
        frame = None
        while time.monotonic() < deadline:
            latest = source.latest()
            if latest is not None:
                frame = latest[0]
            time.sleep(0.2)   # let auto exposure settle; keep the newest frame
        if frame is None:
            raise SystemExit(f"no {config['width']}x{config['height']} frame from {args.camera}: {source.status()}")
    finally:
        source.close()
    out = Path(args.out).expanduser()
    if out.exists():
        raise SystemExit(f"{out} exists")
    cv2.imwrite(str(out), cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    print(json.dumps(dict(written=str(out), size=[int(frame.shape[1]), int(frame.shape[0])])))


def planning_limits(deployment):
    """{joint: (low, high)} rad: the URDF limits shrunk by motion.joint_limit_margin_rad (``SO101Adapter.limits``)."""
    from ni_rua.robot.backends.so101.deployment import ARM_JOINTS
    from ni_rua.robot.model import so101_kinematics as kin
    margin = deployment["motion"]["joint_limit_margin_rad"]
    return {name: (low + margin, high - margin) for name, (low, high) in kin.joint_limits_rad(kin.load_model()).items()
            if name in ARM_JOINTS}


def pose_targets(args, deployment, truth):
    """[(name, q_rad or None)] of ``--poses`` (no hardware: the parked pose uses the packaged kinematics)."""
    from ni_rua.robot.model import so101_kinematics as kin
    out = []
    for name in args.poses.split(","):
        name = name.strip()
        if name == "current":
            out.append((name, None))
        elif name == "home":
            out.append((name, list(deployment["home"]["q_rad"])))
        elif name == "clear":
            q = args.clear_q_rad or deployment["perception"]["clear_view_q_rad"]
            if not q:
                raise SystemExit("pose clear needs deployment perception.clear_view_q_rad or --clear-q-rad")
            out.append((name, list(q)))
        elif name == "parked":
            if args.parked_q_rad:
                out.append((name, list(args.parked_q_rad)))
                continue
            if truth is None:
                raise SystemExit("pose parked needs --truth (or --parked-q-rad)")
            centre = np.mean([cube["centre_xy_m"] for cube in truth["cubes"]], axis=0)
            top = max(cube["top_z_m"] for cube in truth["cubes"])
            T = np.eye(4)
            T[:3, 3] = [centre[0], centre[1], top + args.park_height_m]
            # position only (orientation_weight 0): any orientation of the TCP above the cubes parks the arm over them
            found = kin.ik(kin.load_model(), T, list(deployment["home"]["q_rad"]), orientation_weight=0.0,
                           limits=planning_limits(deployment))
            if not found["converged"]:
                raise SystemExit(f"no IK for the parked position {T[:3, 3].round(3).tolist()}: pass --parked-q-rad")
            out.append((name, found["q_rad"]))
        elif name.startswith("q:"):
            q = [float(value) for value in name[2:].split("/")]
            if len(q) != 5:
                raise SystemExit(f"pose {name}: 5 joint angles (rad) required")
            out.append((name, q))
        else:
            raise SystemExit(f"unknown pose {name}")
    return out


def check_targets(targets, deployment, table_z_m, gripper_rad=0.0):
    """Refuse (SystemExit) a target outside the planning limits, with its TCP outside the workspace box, or with a
    moving link mesh below ``table_z_m`` (m; the same check clear_view_search applies). Not checked: the joint-space
    path between poses (pass ``--via-home`` to go through the commissioned home pose)."""
    from ni_rua.perception.external import held_perception as hp, robot_view
    from ni_rua.robot.backends.so101.deployment import ARM_JOINTS
    from ni_rua.robot.model import so101_kinematics as kin
    limits, model = planning_limits(deployment), kin.load_model()
    meshes, base = hp.SO101Body().meshes(None), robot_view.reach_facts()["base_link"]
    low, high = deployment["workspace"]["min_m"], deployment["workspace"]["max_m"]
    for name, q in targets:
        if q is None:
            continue
        for joint, value in zip(ARM_JOINTS, q):
            if not limits[joint][0] <= value <= limits[joint][1]:
                raise SystemExit(f"pose {name}: {joint}={value:.3f} rad outside the planning limits "
                                 f"[{limits[joint][0]:.3f}, {limits[joint][1]:.3f}] rad")
        poses = kin.fk_all(model, q, gripper_rad=gripper_rad)
        tcp = poses[kin.TCP_LINK][:3, 3]
        if not all(lo <= float(v) <= hi for lo, v, hi in zip(low, tcp, high)):
            raise SystemExit(f"pose {name}: TCP {tcp.round(3).tolist()} m outside the workspace box")
        lowest = robot_view.moving_min_z_m(meshes, poses, base)
        if lowest < table_z_m:
            raise SystemExit(f"pose {name}: a link reaches z {lowest:.3f} m, below the table at {table_z_m:.3f} m")


def cmd_capture(args):
    from ni_rua.robot.backends.so101.deployment import load_deployment
    from ni_rua.robot.backends.so101.driver import gripper_rad_from_normalized
    from ni_rua.robot.backends.so101.worker import SO101Adapter
    deployment = load_deployment(args.deployment)
    truth = json.loads(Path(args.truth).read_text()) if args.truth else None
    out = Path(args.out).expanduser()
    frames = out / "frames"
    if frames.exists() and any(frames.iterdir()):
        raise SystemExit(f"{frames} is not empty: use a new --out per capture")
    sizes = {name: [camera["width"], camera["height"]] for name, camera in deployment["cameras"].items()}
    if args.expect_size and sizes.get("fixed") != list(parse_size(args.expect_size)):
        raise SystemExit(f"deployment fixed camera is {sizes.get('fixed')} px, not {args.expect_size}")
    targets = pose_targets(args, deployment, truth)          # every check before the arm is connected
    if any(q is not None for _, q in targets):
        if not args.move:
            raise SystemExit("these poses move the arm: add --move (the arm moves with the commissioning joint move)")
        table_z = (truth or {}).get("table_z_m")
        if table_z is None and deployment.get("camera_calibration_file"):
            calibration = json.loads(Path(deployment["camera_calibration_file"]).read_text())
            table_z = (calibration.get("joint_calibration") or {}).get("table_z_m")
        if table_z is None:
            raise SystemExit("table height unknown: put table_z_m in the truth file")
        check_targets(targets, deployment, float(table_z),
                      gripper_rad_from_normalized(deployment, deployment["home"]["gripper_normalized"]))
    frames.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema=SCHEMA, started_at=utc_now(), deployment=str(Path(args.deployment).resolve()),
                    camera_sizes=sizes, poses=[dict(name=name, q_rad=q) for name, q in targets], frames=[])
    adapter = SO101Adapter(deployment, out / "worker", "rgb_proprio_calibrated_v1", start_telemetry=False)
    index, moved = 0, False
    try:
        for name, q in targets:
            if q is not None:
                moved = True
                if args.via_home and name != "home":
                    result = adapter.move_joints(list(deployment["home"]["q_rad"]), None, args.arrival_tolerance_rad,
                                                 speed_scale=args.speed_scale, action_id="bench_via_home")
                    if result["status"] != "completed":
                        raise SystemExit(f"home (via) not reached: {result['status']} {result.get('reason')}")
                result = adapter.move_joints(q, None, args.arrival_tolerance_rad, speed_scale=args.speed_scale,
                                             action_id="bench_" + name)
                print(f"move {name}: {result['status']} {result.get('reason') or ''} "
                      f"max_error_rad={result.get('max_joint_error_rad')}", flush=True)
                if result["status"] != "completed":
                    raise SystemExit(f"pose {name} not reached: {result['status']} {result.get('reason')}")
            time.sleep(args.settle_s)
            for _ in range(args.frames):
                observation = adapter.observe()
                directory = frames / f"{index:03d}-{name.replace(':', '_').replace('/', '_')}"
                directory.mkdir()
                for camera in observation["cameras"]:
                    source = Path(camera["rgb_path"])
                    target = directory / f"{camera['camera_id']}{source.suffix}"
                    shutil.copyfile(source, target)
                    camera["rgb_path"] = target.name
                (directory / "observation.json").write_text(json.dumps(dict(pose=name, captured_at=utc_now(),
                                                                            observation=observation)))
                manifest["frames"].append(dict(index=index, pose=name, directory=directory.name))
                index += 1
                time.sleep(args.interval_s)
    except BaseException as error:   # hold the measured position, as scripts/so101_commission.py does
        manifest["error"] = f"{type(error).__name__}: {error}"
        print("ABORT:", manifest["error"], flush=True)
        manifest["abort_stop"] = adapter._stop()
        raise
    finally:
        try:
            if moved and args.park_home:   # end at the commissioned home pose, not over the cubes
                result = adapter.move_joints(list(deployment["home"]["q_rad"]), None, args.arrival_tolerance_rad,
                                             speed_scale=args.speed_scale, action_id="bench_park_home")
                manifest["park_home"] = {key: result.get(key) for key in ("status", "reason", "max_joint_error_rad")}
        except BaseException as error:
            manifest["park_home"] = {"status": "error", "reason": f"{type(error).__name__}: {error}"}
        manifest["finished_at"] = utc_now()
        (out / "capture.json").write_text(json.dumps(manifest, indent=1, default=str))
        adapter.close()
    print(f"{index} frames in {frames}")


# ------------------------------------------------------------------ perceive (NI-RUA env, VGGT service)
def ingest(store, scene, raw, directory, profile):
    """Publish one raw worker observation (``SO101Adapter.observe`` output) into ``store`` / ``scene`` as the harness
    does (``Harness.publish_observation``: calibration, pose sample, robot model, RGB and the actor robot state as
    artifacts) and return the bundle."""
    from ni_rua.core.contracts import canonical, digest, new_id
    from ni_rua.core.observation_policy import actor_robot_state
    state = actor_robot_state(raw["robot_state"], raw["observation_profile"], arms=profile.arms)
    calibration_ref = store.put_artifact(canonical(raw["camera_calibration"]).encode(), "camera_calibration", "application/json")
    pose_ref = store.put_artifact(canonical(raw["camera_pose_sample"]).encode(), "camera_pose_sample", "application/json")
    model_ref = store.put_artifact(canonical(raw["robot_model"]).encode(), "robot_model", "application/json")
    cameras = []
    for camera in raw["cameras"]:
        data = (directory / camera["rgb_path"]).read_bytes()
        cameras.append(dict(camera_id=camera["camera_id"], frame_id=camera.get("frame_id") or new_id("frame"),
                            rgb_ref=store.put_artifact(data, "rgb", "image/jpeg"), width=camera["width"],
                            height=camera["height"], pixel_transform="identity", capture_time=raw.get("captured_at"),
                            sim_step=raw.get("sim_step"), step_domain=raw.get("step_domain"),
                            calibration_ref=calibration_ref, pose_sample_ref=pose_ref))
    state_ref = store.put_artifact(canonical(state).encode(), "robot_state", "application/json")
    bundle = dict(schema_version="0.1", **IDENTITY, world_kind="observed", observation_id=new_id("obs"),
                  sim_step=raw.get("sim_step"), step_domain=raw.get("step_domain"), sim_time_s=raw.get("sim_time_s"),
                  clock_domain="%s/generation-%d" % (IDENTITY["env_instance_id"], IDENTITY["env_generation"]),
                  observation_profile=raw["observation_profile"], robot_model_id=profile.robot_model_id,
                  calibration_id=digest(raw["camera_calibration"]), cameras=cameras, robot_state_ref=state_ref,
                  robot_model_ref=model_ref, data_origin=raw.get("data_origin", "so101_real_hardware"))
    scene.import_observation(bundle)
    store.save("observation", bundle["observation_id"], bundle)
    return bundle


def save_silhouette_overlay(path, image_bytes, camera, q_rad):
    """The fixed image undistorted onto its K with the arm's FK silhouette (the robot mask and the occlusion report
    use exactly this render) tinted red, for checking by eye that the silhouette lies on the arm."""
    import cv2
    from ni_rua.perception.external import held_perception as hp, robot_view
    from ni_rua.robot.model import so101_kinematics as kin
    body = hp.SO101Body()
    poses = {"right": kin.fk_all(kin.load_model(), list(q_rad[:5]), gripper_rad=float(q_rad[5]))}
    robot, _ = robot_view.silhouettes(body, body.meshes(None), poses, camera, "fixed", {"right": "wrist"})
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    D = np.asarray(camera.get("D", []), float).reshape(-1)
    if D.size and np.any(np.abs(D) > 1e-12):
        image = cv2.undistort(image, np.asarray(camera["K"], float), D)
    image[robot] = (0.5 * image[robot] + [0, 0, 127]).astype(np.uint8)
    cv2.imwrite(str(path), image)


def _data_url_bytes(url):
    return len(base64.b64decode(url.split(",", 1)[1])) if isinstance(url, str) and "," in url else None


def cmd_perceive(args):
    from types import SimpleNamespace
    from ni_rua.core.config import Config
    from ni_rua.core.contracts import canonical
    from ni_rua.core.storage.images import image_settings, image_url
    from ni_rua.core.storage.store import Store
    from ni_rua.perception.external import obs3d
    from ni_rua.perception.scene.store import SceneStore
    from ni_rua.robot.backends.so101.deployment import embodiment_profile, load_deployment
    from ni_rua.robot.model.geometry import GeometryTools
    capture_dir = Path(args.capture).expanduser()
    manifest = json.loads((capture_dir / "capture.json").read_text())
    deployment = load_deployment(args.deployment or manifest["deployment"])
    deployment["perception"] = {**deployment["perception"], **HEIGHT_METHODS[args.height_method],
                                "robot_occlusion_report": True}
    profile = embodiment_profile(deployment)
    out = capture_dir / f"perception-{args.height_method}.jsonl"
    if out.exists():
        raise SystemExit(f"{out} exists: one perceive run per height method and capture (delete it to redo)")
    with tempfile.TemporaryDirectory(prefix="ni-rua-perception-bench-") as work:
        store = Store(work, IDENTITY)
        scene = SceneStore(Path(work) / "scene", store.resolve_artifact, store.put_artifact)
        config = Config().with_overrides({"features": {
            "external_perception": True, "object_table": True, "object_table_observe": True, "held_perception": True,
            "scene3d_observe": bool(args.scene3d), "scene3d": bool(args.scene3d), "vggt_endpoint": args.endpoint,
            "vggt_support_z_m": None, "vggt_timeout_s": args.timeout_s, "obs3d_timeout_s": args.timeout_s,
            "vggt_camera_id": "fixed"}})
        settings = image_settings(None)
        runtime = SimpleNamespace(config=config, scene=scene, store=store, image_settings=settings,
                                  identity=dict(IDENTITY), embodiment=profile, capabilities={"robot_model_id": "so101"},
                                  latest=None)
        with out.open("x", encoding="utf-8") as sink:
            for frame in manifest["frames"]:
                directory = capture_dir / "frames" / frame["directory"]
                document = json.loads((directory / "observation.json").read_text())
                raw = document["observation"]
                bundle = ingest(store, scene, raw, directory, profile)
                runtime.latest = bundle
                started = time.monotonic()
                parts = obs3d.attachments(runtime, bundle, {"turn_id": "bench"}, reply="observe")
                attach_s = time.monotonic() - started
                shown = parts[0].value if parts else {}
                overlay = next((part["imageUrl"] for part in parts[1:] if isinstance(part, dict)), None)
                record = store.read_state("perception", bundle["observation_id"]) or {}
                diagnostics = record.get("diagnostics") or {}
                entry = obs3d._memo(runtime, bundle["observation_id"]).get("object_table")
                table_result = entry[2] if isinstance(entry, tuple) and len(entry) == 3 else {}
                table_diagnostics = (table_result or {}).get("diagnostics") or {}
                cross_latency = sum(item.get("latency_s") or 0.0
                                    for item in (table_diagnostics.get("cross_view") or {}).values())
                tools = GeometryTools(scene, settings=settings, profile=profile)
                full = {**IDENTITY, "observation_id": bundle["observation_id"], "schema_version": "0.1"}
                fixed = tools.camera(full, bundle["observation_id"], "fixed")
                _, fixed_bytes = scene.resolve_artifact(fixed["frame"]["rgb_ref"])
                url, encoding = image_url(fixed_bytes, settings)
                table = shown.get("object_table") if isinstance(shown, dict) else None
                arm = raw["robot_state"]["arms"]["right"]
                result = dict(
                    schema=SCHEMA, frame=frame["index"], pose=frame["pose"], height_method=args.height_method,
                    observation_id=bundle["observation_id"],
                    fixed_size=[fixed["width"], fixed["height"]], shown=shown,
                    q_rad=[float(v) for v in arm["q_rad"]],
                    table_z_m=(raw["camera_calibration"].get("joint_calibration") or {}).get("table_z_m"),
                    fixed_camera=dict(K=np.asarray(fixed["K"]).tolist(), T=np.asarray(fixed["T"]).tolist(),
                                      width=fixed["width"], height=fixed["height"]),
                    latency=dict(attach_s=round(attach_s, 3),
                                 object_table_s=(diagnostics.get("object_table") or {}).get("latency_s"),
                                 service_timing=(diagnostics.get("object_table") or {}).get("service_timing"),
                                 cross_view_s=round(cross_latency, 3) if table_diagnostics.get("cross_view") else None),
                    occlusion_diagnostics={k: v for k, v in (table_diagnostics.get("robot_occlusion") or {}).items()
                                           if k != "hidden"},
                    sizes=dict(fixed_image_bytes=encoding.get("byte_size"), fixed_format=encoding.get("format"),
                               overlay_image_bytes=_data_url_bytes(overlay),
                               text_chars=len(canonical(shown)) if shown else 0,
                               **{"fixed_tokens_" + k: v for k, v in token_estimates(fixed["width"], fixed["height"]).items()}))
                save_silhouette_overlay(directory / ("fk_silhouette_%s.jpg" % args.height_method), fixed_bytes,
                                        fixed, result["q_rad"])
                objects = record.get("objects") or []
                result["objects"] = [{k: o.get(k) for k in ("id", "contour_px", "label_px", "top_z_m", "bottom_z_m")}
                                     for o in objects]
                print(f"frame {frame['index']:3d} {frame['pose']:10s} rows "
                      f"{len((table or {}).get('rows') or []) if isinstance(table, dict) else table} "
                      f"attach {attach_s:.2f}s", flush=True)
                sink.write(json.dumps(result) + "\n")
        store.close()
    print(f"wrote {len(manifest['frames'])} records to {out}")


# ------------------------------------------------------------------ score (offline)
def load_records(path):
    """Records of one perception-<method>.jsonl file, or of every such file in a bench directory (method order)."""
    path = Path(path)
    files = sorted(path.glob("perception-*.jsonl")) if path.is_dir() else [path]
    return [json.loads(line) for file in files for line in file.read_text().splitlines() if line.strip()]


def cmd_score(args):
    capture_dir = Path(args.capture).expanduser()
    truth = json.loads(Path(args.truth).read_text())
    records = load_records(capture_dir)
    if args.resolution:
        size = list(parse_size(args.resolution))
        records = [record for record in records if record["fixed_size"] == size]
    covered = None if args.no_occlusion_truth else occlusion_truth(records, truth)
    metrics = score_records(records, truth, match_radius_m=args.match_radius_m, hidden_fraction=args.hidden_fraction,
                            truth_covered=covered)
    report = dict(schema=SCHEMA, scored_at=utc_now(), truth=truth, match_radius_m=args.match_radius_m,
                  hidden_fraction=args.hidden_fraction, conditions=metrics)
    out = capture_dir / "metrics.json"
    out.write_text(json.dumps(report, indent=1))
    for key, item in metrics.items():
        print(f"== {key}  frames {item['frames']}  table_covered p50 {item['table_covered']['p50']}")
        for colour, cube in item["cubes"].items():
            print(f"  {colour:8s} detect {cube['detection_rate']:.2f}  xy p50 {cube['xy_error_m']['p50']}  "
                  f"top err p50 vggt {cube['vggt_top_error_m']['p50']} cross {cube['cross_view_top_error_m']['p50']} "
                  f"contact {cube['contact_top_error_m']['p50']} (range hit {cube['contact_range_hit_rate']})"
                  + (f"  hidden->reported {cube['occlusion']['reported_when_hidden']}" if cube.get("occlusion") else ""))
    print(f"wrote {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot", help="one camera frame at a capture size (so101 env, camera permission)")
    snap.add_argument("--deployment", required=True)
    snap.add_argument("--camera", default="fixed", choices=("fixed", "wrist"))
    snap.add_argument("--size", help="capture size WxH px (default: the deployment's)")
    snap.add_argument("--timeout-s", type=float, default=3.0, help="s to stream before keeping the newest frame")
    snap.add_argument("--out", required=True, help="new PNG path")
    cap = sub.add_parser("capture", help="move through poses and save raw observations (so101 env, camera permission)")
    cap.add_argument("--deployment", required=True, help="deployment whose cameras / calibration set the resolution")
    cap.add_argument("--out", required=True, help="bench directory (frames/ must be new)")
    cap.add_argument("--truth", help="truth JSON (needed by pose parked)")
    cap.add_argument("--poses", default="current", help="comma-separated pose sequence (module docstring)")
    cap.add_argument("--move", action="store_true", help="allow arm motion (required for every pose but current)")
    cap.add_argument("--via-home", action="store_true", help="pass through home.q_rad before every other pose")
    cap.add_argument("--no-park-home", dest="park_home", action="store_false",
                     help="stay at the last pose instead of returning to home.q_rad at the end")
    cap.add_argument("--clear-q-rad", type=float, nargs=5, help="clear pose joints (rad), overrides the deployment")
    cap.add_argument("--parked-q-rad", type=float, nargs=5, help="parked pose joints (rad), overrides the IK")
    cap.add_argument("--park-height-m", type=float, default=0.08, help="TCP height above the highest cube top (m)")
    cap.add_argument("--frames", type=int, default=5, help="observations per pose (default 5)")
    cap.add_argument("--interval-s", type=float, default=0.5, help="s between observations")
    cap.add_argument("--settle-s", type=float, default=1.0, help="s to wait after arrival before capturing")
    cap.add_argument("--speed-scale", type=float, default=0.5, help="fraction of the deployment joint speed cap")
    cap.add_argument("--arrival-tolerance-rad", type=float, default=0.05, help="joint arrival tolerance (rad)")
    cap.add_argument("--expect-size", help="refuse unless the deployment's fixed camera is WxH px")
    per = sub.add_parser("perceive", help="run the harness perception on the captured frames (NI-RUA env, VGGT)")
    per.add_argument("--capture", required=True, help="bench directory written by capture")
    per.add_argument("--deployment", help="deployment file (default: the one recorded by capture)")
    per.add_argument("--endpoint", default="http://127.0.0.1:8331", help="VGGT service URL")
    per.add_argument("--height-method", default="contact", choices=sorted(HEIGHT_METHODS),
                     help="vggt: table only; cross_view: + wrist cross_view; contact: + contact_top_z (default)")
    per.add_argument("--scene3d", action="store_true", help="also attach the scene3d point cloud (latency as in runs)")
    per.add_argument("--timeout-s", type=float, default=60.0, help="service timeout per request (s)")
    sco = sub.add_parser("score", help="metrics per condition (offline)")
    sco.add_argument("--capture", required=True)
    sco.add_argument("--truth", required=True)
    sco.add_argument("--resolution", help="only records of this fixed-camera size WxH px")
    sco.add_argument("--match-radius-m", type=float, default=0.03, help="row-to-cube xy match radius (m)")
    sco.add_argument("--hidden-fraction", type=float, default=0.5, help="covered share that counts as hidden (0..1)")
    sco.add_argument("--no-occlusion-truth", action="store_true", help="skip the FK occlusion truth (no meshes)")
    args = parser.parse_args(argv)
    {"snapshot": cmd_snapshot, "capture": cmd_capture, "perceive": cmd_perceive, "score": cmd_score}[args.command](args)


if __name__ == "__main__":
    main()
