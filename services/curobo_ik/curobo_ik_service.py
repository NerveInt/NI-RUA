"""cuRobo v2 inverse-kinematics service for the SO-101 (GPU server; loopback HTTP behind NI-RUA's SSH tunnel).

NI-RUA's SO-101 planner (``ni_rua/robot/backends/so101/ik_service.py``) sends candidate TCP poses here and verifies
every returned joint solution itself with its own forward kinematics before using it; this service only proposes
solutions. One cuRobo ``InverseKinematics`` solver (GPU, ``--num-seeds`` parallel seeds, self-collision and an
optional table box) answers ``/ik``; one cuRobo ``RobotCollisionChecker`` (the same robot model with the jaw joint free
and an ``attached_object`` link for a held object) answers ``/path_check``; requests are serialised by a lock.

Endpoints (JSON):

``GET /health`` -> ``{"status": "ok", "solver": "curobo", "curobo_version", "num_seeds", "max_batch", "tool_frame",
"joint_limits_rad": {joint: [lower, upper]}, "margin_rad", "urdf_sha256", "spheres", "path_check": {"max_segments",
"max_samples", "max_obstacles", "attached_sphere_slots"}}``.

``POST /ik`` with
  ``poses``: 1..max_batch ``{"position_m": [x, y, z] (m, base_link), "quaternion_xyzw": [x, y, z, w]}`` of
  ``gripper_frame_link``;
  ``seed_q_rad``: five measured arm joint angles (rad), used as the current state and one seed;
  ``axis_weights``: six non-negative weights of the pose error measured in the goal frame, in the order
  position x, y, z, rotation about goal x, y, z (dimensionless; ``[1,1,1,0,0,0]`` = position only,
  ``[1,1,1,1,1,0]`` = free roll about the approach axis, the goal's +z);
  ``position_tolerance_m`` / ``orientation_tolerance_rad``: convergence bounds of cuRobo's own success flag;
  ``num_solutions``: best solutions returned per pose (1..num_seeds);
  ``joint_limits_rad``: the limits the caller plans with; refused (HTTP 409 ``joint_limits_mismatch``) unless they
  equal this model's within 1e-6 rad;
  ``table_top_z_m``: top of a table box in base_link (m) checked for collision, or null for none.
returns ``{"status": "ok", "solve_ms", "solutions": [[{"q_rad": [5], "success", "feasible", "position_error_m",
"rotation_error_rad"}, ...], ...]}`` (one list per pose, best first by cuRobo's cost). ``feasible``: the solution
satisfies cuRobo's constraints (joint limits, self-collision, the table box); ``success``: feasible and converged
within the request's convergence bounds. Optional ``obstacles`` / ``attached_spheres`` / ``jaw_rad`` (same meaning as
for ``/path_check``) add ``obstacle_free`` to every solution: true when the collision checker below finds the joint
solution free of the table, the obstacles and self-collision with the attached spheres included. The solver itself
keeps planning against self-collision and the table box only, so these fields never change which solutions come back.

``POST /path_check`` checks joint-space paths, sampled, against self-collision and the world with
  ``segments``: 1..64 ``{"start_q_rad": [5], "end_q_rad": [5], "start_jaw_rad", "end_jaw_rad", "attached"}``: one
  straight joint-space interpolation each (rad; the jaw angles are the URDF ``gripper`` joint, default 0, and are
  interpolated with the arm; ``attached`` false leaves ``attached_spheres`` out for this segment, default true);
  ``max_joint_step_rad``: largest joint change between two checked samples (rad, 0.001..0.5, default 0.02); a segment
  is checked at ``1 + ceil(largest joint change / max_joint_step_rad)`` evenly spaced samples including both ends
  (one sample when it does not move); refused (HTTP 400 ``too_many_samples``) above 8192 samples in total;
  ``joint_limits_rad``: as for ``/ik`` (409 ``joint_limits_mismatch``);
  ``table_top_z_m``: as for ``/ik``;
  ``obstacles``: up to 40 axis-aligned boxes ``{"id", "min_m": [x, y, z], "max_m": [x, y, z]}`` in base_link (m);
  ``attached_spheres``: up to 32 ``[x, y, z, radius]`` (m) in the ``gripper_frame_link`` frame, moved with the gripper
  (a held object); they are checked against the world and against the arm links other than the gripper's.
returns ``{"status": "ok", "check_ms", "samples", "segments": [{"samples", "colliding_samples", "first_collision"}]}``.
``first_collision`` is null for a free segment, else ``{"sample_index", "fraction" (0 = segment start, 1 = end),
"q_rad": [5], "jaw_rad", "world": [{"link", "penetration_m", "obstacles": [ids, "table"]}], "self": [{"links": [a, b],
"penetration_m"}]}`` (at most 6 entries each, deepest first). The verdict and the world ``penetration_m`` (deepest
sphere of the link) are cuRobo's (``RobotCollisionChecker``: the fitted link spheres against the cuboids, and its
self-collision pairs); which obstacle and which link pair is named comes from the same spheres checked against the
boxes and against each other here (``box_overlaps`` / ``sphere_pair_overlaps``). The checker moves the jaw joint; the
IK solver keeps it locked at 0 rad. A link named ``attached_object`` is the attached spheres.

Startup self-check (``self_check_failure``): the service solves the commissioned ready pose and requires that solution
to be collision free in the path checker with the jaw at 0 rad (the IK model's locked angle) and at ``--open-jaw-rad``
(the deployment's ``gripper.jaw_open_rad``), so a sphere model whose open jaw collides with the gripper refuses to
serve (exit 2, ``SELF_CHECK_FAILED``) instead of labelling every candidate ``obstacle_free=false``.

    python curobo_ik_service.py --model model --port 8341 --num-seeds 64 --max-batch 32 --open-jaw-rad 1.74533 \
        --status STATUS
"""
import argparse
import gc
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_REQUEST_BYTES = 16 * 1024 * 1024

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
LIMIT_MATCH_RAD = 1e-6
#: Default of ``--open-jaw-rad`` (rad, URDF ``gripper`` joint): the example deployment's ``gripper.jaw_open_rad``,
#: which is also the URDF upper limit of that joint.
DEFAULT_OPEN_JAW_RAD = 1.74533
#: The commissioned SO-101 ready pose the self-check solves: TCP about (0.216, 0, 0.084) m in base_link, jaws pointing
#: straight down, roll free; the real arm reaches it.
READY_POSE = dict(position_m=[0.216, 0.0, 0.084], quaternion_xyzw=[1.0, 0.0, 0.0, 0.0])
READY_SEED_Q_RAD = [0.0, -0.95, 0.9, 1.0, 0.0]


class RequestError(ValueError):
    def __init__(self, status, reason):
        super().__init__(reason)
        self.status, self.reason = status, reason


# ------------------------------------------------- path check: pure helpers ---
#: Limits of one /path_check request (and of the obstacle fields of /ik).
MAX_SEGMENTS, MAX_PATH_SAMPLES, MAX_OBSTACLES, ATTACHED_SPHERE_SLOTS = 64, 8192, 40, 32
#: Entries of ``first_collision.world`` / ``.self`` (deepest first).
MAX_REPORTED = 6
#: Name of the extra link that carries the attached spheres (fixed to the tool frame).
ATTACHED_LINK = "attached_object"
#: Links rigid with the tool frame or holding the object: never checked against the attached spheres.
GRIPPER_LINKS = ("gripper_link", "moving_jaw_so101_v1_link", "wrist_camera_mount_link", "wrist_camera_link",
                 "gripper_frame_link")


def checker_kinematics(kinematics, slots=ATTACHED_SPHERE_SLOTS):
    """The collision checker's kinematics section from the IK model's (``so101_curobo.yml`` ``kinematics``).

    Differences: no locked joint (the jaw joint ``gripper`` becomes the sixth joint, so an open jaw is checked where it
    is), and an extra link ``attached_object`` fixed at the tool frame with ``slots`` collision spheres (radius -100 =
    unused) that is never checked against ``GRIPPER_LINKS``. The input is not modified."""
    out = json.loads(json.dumps(kinematics))
    out["lock_joints"] = None
    tool = out["tool_frames"][0]
    out["extra_links"] = {ATTACHED_LINK: dict(fixed_transform=[0, 0, 0, 1, 0, 0, 0], joint_name="attach_joint",
                                              joint_type="FIXED", link_name=ATTACHED_LINK, parent_link_name=tool)}
    out["extra_collision_spheres"] = {ATTACHED_LINK: int(slots)}
    out["collision_link_names"] = [*out["collision_link_names"], ATTACHED_LINK]
    out.setdefault("self_collision_buffer", {})[ATTACHED_LINK] = 0.0
    out.setdefault("self_collision_ignore", {})[ATTACHED_LINK] = list(GRIPPER_LINKS)
    return out


def parse_obstacles(value):
    """Validated ``obstacles`` of a request: ``[(id, [min x, y, z], [max x, y, z])]`` (m, base_link)."""
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_OBSTACLES:
        raise RequestError(400, f"obstacles_must_be_a_list_of_at_most_{MAX_OBSTACLES}")
    out, seen = [], set()
    for item in value:
        low, high = (item.get("min_m"), item.get("max_m")) if isinstance(item, dict) else (None, None)
        identifier = item.get("id") if isinstance(item, dict) else None
        if (not isinstance(identifier, str) or not identifier or identifier in seen or identifier == "table"
                or not all(isinstance(v, list) and len(v) == 3 and all(_finite(x) for x in v) for v in (low, high))
                or any(b <= a for a, b in zip(low, high))):
            raise RequestError(400, "obstacle_requires_unique_id_and_min_m_below_max_m")
        seen.add(identifier)
        out.append((identifier, [float(v) for v in low], [float(v) for v in high]))
    return out


def parse_spheres(value):
    """Validated ``attached_spheres``: ``[[x, y, z, radius]]`` (m, tool frame), radius > 0."""
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > ATTACHED_SPHERE_SLOTS or not all(
            isinstance(item, list) and len(item) == 4 and all(_finite(v) for v in item) and item[3] > 0 for item in value):
        raise RequestError(400, f"attached_spheres_must_be_at_most_{ATTACHED_SPHERE_SLOTS}_x_y_z_radius")
    return [[float(v) for v in item] for item in value]


def segment_samples(segment, max_joint_step_rad):
    """Joint samples of one segment: ``[[5 arm angles, jaw angle]]`` (rad), evenly spaced, both ends included.

    ``1 + ceil(largest joint change / max_joint_step_rad)`` samples; one sample when nothing moves."""
    start, end = segment.get("start_q_rad"), segment.get("end_q_rad")
    jaws = [segment.get("start_jaw_rad", 0.0), segment.get("end_jaw_rad", 0.0)]
    if not all(isinstance(q, list) and len(q) == 5 and all(_finite(v) for v in q) for q in (start, end)) \
            or not all(_finite(v) for v in jaws):
        raise RequestError(400, "segment_requires_start_q_rad_5_end_q_rad_5_and_finite_jaw_rad")
    a, b = [*map(float, start), float(jaws[0])], [*map(float, end), float(jaws[1])]
    largest = max(abs(y - x) for x, y in zip(a, b))
    count = 1 if largest == 0 else 1 + int(math.ceil(largest / max_joint_step_rad - 1e-9))
    return [[x + (y - x) * (index / (count - 1) if count > 1 else 0.0) for x, y in zip(a, b)] for index in range(count)]


def box_overlaps(sphere, boxes):
    """Ids of the axis-aligned ``boxes`` (``parse_obstacles`` form) that the sphere ``[x, y, z, radius]`` overlaps."""
    hits = []
    for identifier, low, high in boxes:
        gap = math.sqrt(sum(max(lo - c, 0.0, c - hi) ** 2 for c, lo, hi in zip(sphere[:3], low, high)))
        if gap < sphere[3]:
            hits.append(identifier)
    return hits


def sphere_pair_overlaps(spheres, links, ignore):
    """Overlapping sphere pairs of different links: ``[(link a, link b, penetration m)]``, one per link pair (its
    deepest), deepest first.

    ``spheres``: ``[[x, y, z, radius]]`` (radius <= 0 = unused); ``links``: link name per sphere; ``ignore``: set of
    frozenset link pairs never checked (adjacent links, the model's self-collision ignore list)."""
    deepest = {}
    for i, a in enumerate(spheres):
        if a[3] <= 0:
            continue
        for j in range(i + 1, len(spheres)):
            b = spheres[j]
            pair = frozenset((links[i], links[j]))
            if b[3] <= 0 or len(pair) == 1 or pair in ignore:
                continue
            depth = a[3] + b[3] - math.sqrt(sum((x - y) ** 2 for x, y in zip(a[:3], b[:3])))
            if depth > 0 and depth > deepest.get(pair, 0.0):
                deepest[pair] = depth
    return sorted(((*sorted(pair), depth) for pair, depth in deepest.items()), key=lambda item: -item[2])


def ignored_pairs(kinematics):
    """``{frozenset((link a, link b))}`` of a kinematics section's ``self_collision_ignore``."""
    return {frozenset((link, other)) for link, others in (kinematics.get("self_collision_ignore") or {}).items()
            for other in others}


class Checker:
    """cuRobo ``RobotCollisionChecker`` over the checker model (``checker_kinematics``): jaw joint free, attached
    spheres, a world of the table box and obstacle boxes. Not thread safe; the service lock serialises its use."""

    def __init__(self, document):
        import torch
        from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
        from curobo.scene import Cuboid, Scene

        self.torch, self.Cuboid, self.Scene = torch, Cuboid, Scene
        self.kinematics = checker_kinematics(document["kinematics"])
        config = RobotCollisionCheckerCfg.load_from_config(
            robot_config={"kinematics": self.kinematics}, scene_model=Scene(cuboid=[self._table(0.0)]),
            n_cuboids=MAX_OBSTACLES + 8, n_meshes=1, collision_activation_distance=0.0)
        self.checker = RobotCollisionChecker(config)
        names = list(self.checker.kinematics.joint_names)
        self.order = [names.index(name) for name in (*ARM_JOINTS, "gripper")]
        if sorted(self.order) != list(range(6)):
            raise SystemExit(f"checker model joints are {names}; expected the five arm joints and gripper")
        parameters = self.checker.kinematics.config.kinematics_config
        self.parameters = parameters
        self.links = [None] * int(self.checker.kinematics.total_spheres)
        for link in self.kinematics["collision_link_names"]:
            for index in parameters.get_sphere_index_from_link_name(link).tolist():
                self.links[index] = link
        self.ignore = ignored_pairs(self.kinematics)
        self.world = "unset"

    def _table(self, top):
        return self.Cuboid(name="table", pose=[0.0, 0.0, float(top) - 0.25, 1, 0, 0, 0], dims=[2.0, 2.0, 0.5])

    def set_world(self, table, obstacles, spheres):
        """Load the table box (``table``: top z in m, or None), the obstacle boxes and the attached spheres."""
        key = (table, tuple((low[0], low[1], low[2], high[0], high[1], high[2]) for _, low, high in obstacles))
        if key != self.world:
            cuboids = [] if table is None else [self._table(table)]
            # Obstacle names are positional: request ids are free text and only used in the answer.
            cuboids += [self.Cuboid(name=f"obstacle_{index}", pose=[*((a + b) / 2 for a, b in zip(low, high)), 1, 0, 0, 0],
                                    dims=[b - a for a, b in zip(low, high)])
                        for index, (_, low, high) in enumerate(obstacles)]
            self.checker.update_world(self.Scene(cuboid=cuboids))
            self.world = key
        self.set_attached(spheres)

    def set_attached(self, spheres):
        self.parameters.reset_link_spheres(ATTACHED_LINK)
        if spheres:
            self.parameters.update_link_spheres(ATTACHED_LINK, self.torch.tensor(spheres, device="cuda", dtype=self.torch.float32))

    def distances(self, rows):
        """``(spheres [n, total, 4], world penetration [n, total], self penetration [n])`` of joint rows
        ``[[5 arm, jaw]]`` (rad); penetration > 0 = collision (cuRobo, activation distance 0)."""
        torch = self.torch
        q = torch.tensor(rows, device="cuda", dtype=torch.float32)
        ordered = torch.empty_like(q)
        ordered[:, self.order] = q
        state = self.checker.get_kinematics(ordered.unsqueeze(1))
        world = self.checker.get_collision_distance(state)
        own = self.checker.get_self_collision_distance(state.robot_spheres)
        return state.robot_spheres[:, 0], world[:, 0], own.reshape(-1)

    def report(self, spheres, world, table, obstacles):
        """``(world entries, self entries)`` of one colliding sample (module docstring ``first_collision``)."""
        boxes = list(obstacles)
        if table is not None:
            boxes.append(("table", [-1.0, -1.0, float(table) - 0.5], [1.0, 1.0, float(table)]))
        spheres, world = spheres.detach().cpu().tolist(), world.detach().cpu().tolist()
        by_link = {}
        for sphere, depth, link in zip(spheres, world, self.links):
            if depth > 0:
                entry = by_link.setdefault(link, dict(link=link, penetration_m=0.0, obstacles=[]))
                entry["penetration_m"] = max(entry["penetration_m"], depth)
                entry["obstacles"] += [name for name in box_overlaps(sphere, boxes) if name not in entry["obstacles"]]
        hits = sorted(by_link.values(), key=lambda entry: -entry["penetration_m"])[:MAX_REPORTED]
        for entry in hits:
            entry["penetration_m"] = round(entry["penetration_m"], 4)
        pairs = [dict(links=[a, b], penetration_m=round(depth, 4))
                 for a, b, depth in sphere_pair_overlaps(spheres, self.links, self.ignore)[:MAX_REPORTED]]
        return hits, pairs


class Solver:
    def __init__(self, model_dir, num_seeds, max_batch):
        import torch
        import curobo
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.scene import Cuboid, Scene
        from curobo.types import GoalToolPose, JointState, Pose, ToolPoseCriteria

        self.torch, self.Cuboid, self.Scene = torch, Cuboid, Scene
        self.GoalToolPose, self.JointState, self.Pose, self.ToolPoseCriteria = GoalToolPose, JointState, Pose, ToolPoseCriteria
        model_dir = Path(model_dir).resolve()
        self.info = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
        self.num_seeds, self.max_batch = int(num_seeds), int(max_batch)
        config = InverseKinematicsCfg.create(
            robot=str(model_dir / "so101_curobo.yml"), scene_model={"cuboid": {}}, collision_cache={"cuboid": 2},
            num_seeds=self.num_seeds, self_collision_check=True, max_batch_size=self.max_batch,
            position_tolerance=0.005, orientation_tolerance=0.05)
        self.ik = InverseKinematics(config)
        self.tool = self.ik.tool_frames[0]
        self.joint_names = list(self.ik.joint_names) if hasattr(self.ik, "joint_names") else None
        self.version = getattr(curobo, "__version__", "unknown")
        self.table = "unset"
        self.lock = threading.Lock()
        import yaml
        self.checker = Checker(yaml.safe_load((model_dir / "so101_curobo.yml").read_text(encoding="utf-8")))
        # The checker must run once BEFORE the IK solver's first solve: that solve captures a CUDA graph, and a warp
        # collision kernel first loaded after the capture corrupts the graph (the next solve then fails with "CUDA
        # error: misaligned address"; measured on this cuRobo build 2026-11-15). Every kind of query is exercised here.
        self.checker.set_world(0.0, [("warm", [0.3, -0.05, 0.0], [0.35, 0.05, 0.05])], [[0.0, 0.0, 0.02, 0.01]])
        self.checker.distances([[0.0, -0.95, 0.9, 1.0, 0.0, 0.0], [0.2, -0.5, 0.9, 1.0, 0.0, 0.5]])
        self.checker.set_world(None, [], [])
        self.checker.distances([[0.0, -0.95, 0.9, 1.0, 0.0, 0.0]])
        torch.cuda.synchronize()

    def health(self):
        return dict(status="ok", solver="curobo", curobo_version=str(self.version), num_seeds=self.num_seeds,
                    max_batch=self.max_batch, tool_frame=self.tool, joint_limits_rad=self.info["joint_limits_rad"],
                    margin_rad=self.info["margin_rad"], urdf_sha256=self.info["urdf_sha256"],
                    spheres=self.info["spheres"],
                    path_check=dict(max_segments=MAX_SEGMENTS, max_samples=MAX_PATH_SAMPLES, max_obstacles=MAX_OBSTACLES,
                                    attached_sphere_slots=ATTACHED_SPHERE_SLOTS))

    def _set_table(self, top):
        if top == self.table:
            return
        if top is None:
            cuboids = []
        else:
            cuboids = [self.Cuboid(name="table", pose=[0.0, 0.0, float(top) - 0.25, 1, 0, 0, 0], dims=[2.0, 2.0, 0.5])]
        self.ik.update_world(self.Scene(cuboid=cuboids))
        self.table = top

    def solve(self, request):
        poses = request.get("poses")
        if not isinstance(poses, list) or not 1 <= len(poses) <= self.max_batch:
            raise RequestError(400, f"poses_must_be_1_to_{self.max_batch}")
        seed = request.get("seed_q_rad")
        weights = request.get("axis_weights")
        if (not isinstance(seed, list) or len(seed) != 5 or not all(_finite(v) for v in seed)
                or not isinstance(weights, list) or len(weights) != 6 or not all(_finite(v) and v >= 0 for v in weights)):
            raise RequestError(400, "seed_q_rad_5_and_axis_weights_6_required")
        self._check_limits(request)
        raw_k = request.get("num_solutions", 1)
        if type(raw_k) is not int:
            raise RequestError(400, "num_solutions_must_be_int")
        k = int(raw_k)
        if not 1 <= k <= self.num_seeds:
            raise RequestError(400, f"num_solutions_must_be_1_to_{self.num_seeds}")
        table = self._table(request)
        obstacles, attached = parse_obstacles(request.get("obstacles")), parse_spheres(request.get("attached_spheres"))
        jaw = request.get("jaw_rad", 0.0)
        if not _finite(jaw):
            raise RequestError(400, "jaw_rad_must_be_finite")
        pos_tol = request.get("position_tolerance_m", 0.005)
        ori_tol = request.get("orientation_tolerance_rad", 0.05)
        if not (_finite(pos_tol) and pos_tol > 0 and _finite(ori_tol) and ori_tol > 0):
            raise RequestError(400, "tolerances_must_be_positive_finite")
        tolerance = [float(pos_tol), float(ori_tol)]
        torch = self.torch
        positions, quaternions = [], []
        for pose in poses:
            if not isinstance(pose, dict):
                raise RequestError(400, "pose_requires_position_m_3_and_quaternion_xyzw_4")
            p, q = pose.get("position_m"), pose.get("quaternion_xyzw")
            if (not isinstance(p, list) or len(p) != 3 or not isinstance(q, list) or len(q) != 4
                    or not all(_finite(v) for v in (*p, *q))):
                raise RequestError(400, "pose_requires_position_m_3_and_quaternion_xyzw_4")
            norm = math.sqrt(sum(v * v for v in q))
            if norm < 1e-6:
                raise RequestError(400, "quaternion_must_be_nonzero")
            positions.append([float(v) for v in p])
            quaternions.append([q[3] / norm, q[0] / norm, q[1] / norm, q[2] / norm])  # cuRobo order wxyz
        n = len(poses)
        with self.lock:
            self._set_table(table)
            self.ik.update_tool_pose_criteria({self.tool: self.ToolPoseCriteria(
                terminal_pose_axes_weight_factor=[float(v) for v in weights],
                terminal_pose_convergence_tolerance=tolerance, project_distance_to_goal=True)})
            device = dict(device="cuda", dtype=torch.float32)
            # Always a full max_batch request (the first pose repeated): this cuRobo build pads goal, position,
            # velocity and acceleration of a smaller batch but not the jerk buffer, which then fails to broadcast.
            pad = self.max_batch - n
            positions, quaternions = positions + [positions[0]] * pad, quaternions + [quaternions[0]] * pad
            goal = self.GoalToolPose.from_poses({self.tool: self.Pose(
                position=torch.tensor(positions, **device), quaternion=torch.tensor(quaternions, **device))},
                num_goalset=1)
            current = torch.tensor([list(map(float, seed))] * self.max_batch, **device)
            state = self.JointState.from_position(current, joint_names=list(ARM_JOINTS))
            torch.cuda.synchronize()
            started = time.monotonic()
            result = self.ik.solve_pose(goal, current_state=state, return_seeds=k)
            torch.cuda.synchronize()
            elapsed = (time.monotonic() - started) * 1000
            solved = result.js_solution
            order = [list(solved.joint_names).index(name) for name in ARM_JOINTS]
            batch = self.max_batch
            q = solved.position.reshape(batch, k, -1)[:n][:, :, order].detach().cpu().tolist()
            success = result.success.reshape(batch, k)[:n].detach().cpu().tolist()
            feasible = result.feasible.reshape(batch, k)[:n].detach().cpu().tolist()
            position_error = result.position_error.reshape(batch, k)[:n].detach().cpu().tolist()
            rotation_error = result.rotation_error.reshape(batch, k)[:n].detach().cpu().tolist()
            free = None
            if obstacles or attached:
                # Label only: the solver planned against self-collision and the table; the checker adds the obstacles
                # and the attached spheres (module docstring).
                self.checker.set_world(table, obstacles, attached)
                rows = [[*(v if math.isfinite(v) else 0.0 for v in q[i][j]), float(jaw)] for i in range(n) for j in range(k)]
                _, world, own = self.checker.distances(rows)
                free = ((world.amax(dim=1) <= 0) & (own <= 0)).reshape(n, k).detach().cpu().tolist()
        solutions = [[dict(q_rad=q[i][j], success=bool(success[i][j]), feasible=bool(feasible[i][j]),
                           position_error_m=float(position_error[i][j]),
                           rotation_error_rad=float(rotation_error[i][j]),
                           **({} if free is None else {"obstacle_free": bool(free[i][j])}))
                      for j in range(k)] for i in range(n)]
        return dict(status="ok", solve_ms=round(elapsed, 2), solutions=solutions)

    def _check_limits(self, request):
        limits = request.get("joint_limits_rad")
        if not isinstance(limits, dict):
            raise RequestError(409, "joint_limits_mismatch")
        mine = self.info["joint_limits_rad"]
        if set(limits) != set(ARM_JOINTS) or any(
                not isinstance(limits[name], list) or len(limits[name]) != 2
                or not all(_finite(x) for x in limits[name])
                or abs(float(limits[name][i]) - mine[name][i]) > LIMIT_MATCH_RAD
                for name in ARM_JOINTS for i in (0, 1)):
            raise RequestError(409, "joint_limits_mismatch")

    @staticmethod
    def _table(request):
        table = request.get("table_top_z_m")
        if table is not None and not _finite(table):
            raise RequestError(400, "table_top_z_m_must_be_finite_or_null")
        return table

    def path_check(self, request):
        segments = request.get("segments")
        if not isinstance(segments, list) or not 1 <= len(segments) <= MAX_SEGMENTS or not all(
                isinstance(segment, dict) for segment in segments):
            raise RequestError(400, f"segments_must_be_1_to_{MAX_SEGMENTS}")
        step = request.get("max_joint_step_rad", 0.02)
        if not _finite(step) or not 0.001 <= step <= 0.5:
            raise RequestError(400, "max_joint_step_rad_must_be_0.001_to_0.5")
        self._check_limits(request)
        table = self._table(request)
        obstacles, attached = parse_obstacles(request.get("obstacles")), parse_spheres(request.get("attached_spheres"))
        rows = [segment_samples(segment, step) for segment in segments]
        total = sum(len(item) for item in rows)
        if total > MAX_PATH_SAMPLES:
            raise RequestError(400, f"too_many_samples:{total}>{MAX_PATH_SAMPLES}")
        torch = self.torch
        with self.lock:
            torch.cuda.synchronize()
            started = time.monotonic()
            self.checker.set_world(table, obstacles, attached)
            results = [None] * len(segments)
            # Segments with and without the attached spheres are two batches (the spheres are model state).
            for with_attached in (True, False):
                members = [index for index, segment in enumerate(segments)
                           if bool(segment.get("attached", True)) == with_attached]
                if not members:
                    continue
                if not with_attached:
                    self.checker.set_attached([])
                spheres, world, own = self.checker.distances([row for index in members for row in rows[index]])
                colliding = ((world.amax(dim=1) > 0) | (own > 0)).detach().cpu().tolist()
                offset = 0
                for index in members:
                    count = len(rows[index])
                    flags = colliding[offset:offset + count]
                    first = None
                    if any(flags):
                        at = flags.index(True)
                        hits, pairs = self.checker.report(spheres[offset + at], world[offset + at], table, obstacles)
                        first = dict(sample_index=at, fraction=round(at / (count - 1), 4) if count > 1 else 0.0,
                                     q_rad=[round(v, 5) for v in rows[index][at][:5]], jaw_rad=round(rows[index][at][5], 5),
                                     world=hits, self=pairs)
                    results[index] = dict(samples=count, colliding_samples=sum(flags), first_collision=first)
                    offset += count
            torch.cuda.synchronize()
            elapsed = (time.monotonic() - started) * 1000
        return dict(status="ok", check_ms=round(elapsed, 2), samples=total, segments=results)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def self_check_failure(solver, open_jaw_rad):
    """Startup self-check of ``solver`` (``Solver`` or a stand-in with ``info``, ``solve`` and ``path_check``): None
    when it passes, else the failure text.

    1. ``/ik`` of ``READY_POSE`` from ``READY_SEED_Q_RAD`` (position and approach, roll free) must succeed: a model
       that calls the pose the real arm reaches infeasible (e.g. collision spheres that always overlap) is refused.
       This first solve also captures the CUDA graphs (several seconds; later solves take milliseconds).
    2. That solution must be collision free in the path checker, which shares the spheres but moves the jaw and carries
       the attached-object link, with the jaw at 0 rad and at ``open_jaw_rad`` (rad, the deployment's
       ``gripper.jaw_open_rad``): an open-jaw self-collision would otherwise mark every candidate of a request that
       sends that jaw angle ``obstacle_free=false`` and every path check ``collisions``."""
    limits = solver.info["joint_limits_rad"]
    check = solver.solve(dict(poses=[READY_POSE], seed_q_rad=list(READY_SEED_Q_RAD), axis_weights=[1, 1, 1, 1, 1, 0],
                              position_tolerance_m=0.005, orientation_tolerance_rad=0.05, joint_limits_rad=limits,
                              num_solutions=1, table_top_z_m=None))["solutions"][0][0]
    if not check["success"]:
        return f"the ready pose is not solvable with this model: {check}"
    jaws = (0.0, float(open_jaw_rad))
    probes = solver.path_check(dict(segments=[dict(start_q_rad=check["q_rad"], end_q_rad=check["q_rad"], start_jaw_rad=jaw,
                                                   end_jaw_rad=jaw) for jaw in jaws],
                                    joint_limits_rad=limits, table_top_z_m=None))["segments"]
    for jaw, probe in zip(jaws, probes):
        if probe["first_collision"] is not None:
            return f"the ready pose collides in the path checker with the jaw at {jaw:g} rad: {probe}"
    return None


def _is_cuda_oom(error, torch_mod=None):
    oom_cls = getattr(getattr(torch_mod, "cuda", None), "OutOfMemoryError", None) if torch_mod is not None else None
    if oom_cls is not None and isinstance(error, oom_cls):
        return True
    msg = str(error).lower()
    return isinstance(error, (RuntimeError, MemoryError)) and ("out of memory" in msg or "cuda error 2" in msg)


def _is_sticky_cuda_error(error, torch_mod=None):
    if _is_cuda_oom(error, torch_mod):
        return False
    msg = str(error).lower()
    return isinstance(error, RuntimeError) and ("cuda error" in msg or "cudnn error" in msg)


def make_http_handler(solver):
    """Build the loopback HTTP handler bound to ``solver`` (``Solver`` or a test double)."""

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                body = dict(solver.health())
                cuda_fatal = getattr(solver, "cuda_fatal", None)
                if cuda_fatal:
                    body["status"] = "error"
                    body["reason"] = cuda_fatal
                    self._reply(503, body)
                else:
                    self._reply(200, body)
            else:
                self._reply(404, dict(status="error", reason="not_found"))

        def do_POST(self):
            handler = {"/ik": solver.solve, "/path_check": solver.path_check}.get(self.path)
            if handler is None:
                self._reply(404, dict(status="error", reason="not_found"))
                return
            try:
                content_length = int(self.headers.get("Content-Length", 0))
            except (ValueError, TypeError):
                self._reply(400, dict(status="error", reason="bad_request:invalid_content_length"))
                return
            if content_length < 0:
                self._reply(400, dict(status="error", reason="bad_request:invalid_content_length"))
                return
            if content_length > MAX_REQUEST_BYTES:
                self._reply(413, dict(status="error", reason="payload_too_large"))
                return
            try:
                raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
                request = json.loads(raw_body or b"{}")
            except (ValueError, TypeError) as error:
                self._reply(400, dict(status="error", reason=f"bad_request:{type(error).__name__}"))
                return
            if not isinstance(request, dict):
                self._reply(400, dict(status="error", reason="bad_request:request_must_be_object"))
                return
            torch_mod = getattr(solver, "torch", None)
            try:
                self._reply(200, handler(request))
            except RequestError as error:
                self._reply(error.status, dict(status="error", reason=error.reason))
            except Exception as error:  # noqa: BLE001 - report, keep serving
                if _is_cuda_oom(error, torch_mod):
                    gc.collect()
                    cuda_api = getattr(torch_mod, "cuda", None) if torch_mod is not None else None
                    if cuda_api is not None and cuda_api.is_available():
                        cuda_api.empty_cache()
                    self._reply(
                        503,
                        dict(
                            status="error",
                            reason="cuda_out_of_memory",
                            detail=f"{type(error).__name__}:{error}"[:300],
                        ),
                    )
                elif _is_sticky_cuda_error(error, torch_mod):
                    solver.cuda_fatal = f"{type(error).__name__}:{error}"[:300]
                    self._reply(
                        503,
                        dict(
                            status="error",
                            reason=f"cuda_runtime_error:{type(error).__name__}:{error}"[:300],
                        ),
                    )
                else:
                    self._reply(500, dict(status="error", reason=f"solver_error:{type(error).__name__}:{error}"[:300]))

        def log_message(self, fmt, *values):
            print(time.strftime("%F %T"), self.address_string(), fmt % values, flush=True)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="cuRobo IK service for the SO-101")
    parser.add_argument("--model", required=True, help="directory written by build_model.py")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8341)
    parser.add_argument("--num-seeds", type=int, default=64)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--open-jaw-rad", type=float, default=DEFAULT_OPEN_JAW_RAD,
                        help="URDF gripper joint angle (rad) of the deployment's gripper.jaw_open_rad; the startup "
                             f"self-check also checks the ready pose with the jaw there (default {DEFAULT_OPEN_JAW_RAD})")
    parser.add_argument("--status", default=None, help="file updated with the service state")
    args = parser.parse_args()
    if not math.isfinite(args.open_jaw_rad):
        parser.error("--open-jaw-rad must be finite")
    started = time.monotonic()
    solver = Solver(args.model, args.num_seeds, args.max_batch)
    failure = self_check_failure(solver, args.open_jaw_rad)
    if failure is not None:
        print(f"SELF_CHECK_FAILED: {failure}", flush=True)
        if args.status:
            Path(args.status).write_text("FAILED self_check\n", encoding="utf-8")
        raise SystemExit(2)
    ready = (f"READY port={args.port} seeds={args.num_seeds} batch={args.max_batch} open_jaw_rad={args.open_jaw_rad:g} "
             f"load_s={time.monotonic() - started:.1f}")
    if args.status:
        Path(args.status).write_text(ready + "\n", encoding="utf-8")
    print(ready, flush=True)
    ThreadingHTTPServer((args.host, args.port), make_http_handler(solver)).serve_forever()


if __name__ == "__main__":
    main()
