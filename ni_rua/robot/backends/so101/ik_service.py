"""HTTP client of the cuRobo IK service (``services/curobo_ik``) for the SO-101 planner. Standard library only.

The deployment's ``ik_service`` section configures it (``deployment.validate_deployment``): ``url`` (loopback, the
SSH tunnel of NI-RUA's remote services), ``timeout_s`` (s per request), ``num_solutions`` (best joint solutions
returned per pose) and ``table_top_z_m`` (top of the table box the service checks for collision, m in so101_base, or
null for none) and ``path_check_step_rad`` (largest joint change between two checked samples of ``check_path``, rad).
The planning joint limits are sent with every request and the service refuses limits other than
those its robot model was built with (``joint_limits_mismatch``), so planner and service cannot silently disagree.

A collision ``world`` (``ik_plan.collision_world``: obstacle boxes and the spheres of a held object) is optional on
both calls: ``solve`` then labels every candidate ``obstacle_free`` (the solver itself keeps planning against
self-collision and the table only) and ``check_path`` checks the joint-space paths against it.

Errors raise ``IKServiceError`` with a stable ``reason``: ``ik_service_unreachable`` (connection refused, tunnel
down, timeout), ``ik_service_http_<code>:<service reason>`` and ``ik_service_bad_response``. Nothing falls back to
another solver.
"""
from __future__ import annotations

import json
import math
import socket
import urllib.error
import urllib.request

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


class IKServiceError(RuntimeError):
    """The cuRobo IK service gave no usable answer; ``reason`` is a stable code (module docstring)."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class CuroboClient:
    """``solve`` candidate joint solutions and ``check_path`` segments via remote ``curobo`` HTTP or local ``pink``.

    ``settings``: the validated ``ik_service`` deployment section (``solver`` ``"curobo"`` or ``"pink"``).
    ``joint_limits``: planning limits ``{joint: (lower, upper)}`` (rad, ARM_JOINTS order, already shrunk by
    ``motion.joint_limit_margin_rad``). ``opener``: ``urllib.request.urlopen``-compatible callable (tests)."""

    def __init__(self, settings, joint_limits, *, opener=None):
        self.solver = settings.get("solver", "curobo")
        raw_url = settings.get("url")
        if self.solver == "pink":
            self.url = raw_url.rstrip("/") if isinstance(raw_url, str) else None
        else:
            self.url = settings["url"].rstrip("/")
        self.timeout_s = float(settings.get("timeout_s", 10.0))
        self.num_solutions = int(settings["num_solutions"])
        self.table_top_z_m = settings.get("table_top_z_m")
        self.path_check_step_rad = float(settings.get("path_check_step_rad", 0.01))
        self.limits = {name: [float(joint_limits[name][0]), float(joint_limits[name][1])] for name in ARM_JOINTS}
        self.opener = opener or urllib.request.urlopen
        self.max_batch = None
        self.max_path_samples = None

    def _call(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.url + path, data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        try:
            with self.opener(request, timeout=self.timeout_s) as response:
                answer = json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            try:
                reason = json.loads(error.read() or b"{}").get("reason", "")
            except ValueError:
                reason = ""
            raise IKServiceError(f"ik_service_http_{error.code}:{reason}"[:200]) from None
        except (urllib.error.URLError, ConnectionError, socket.timeout, TimeoutError, OSError):
            raise IKServiceError("ik_service_unreachable") from None
        except ValueError:
            raise IKServiceError("ik_service_bad_response") from None
        if not isinstance(answer, dict) or answer.get("status") != "ok":
            raise IKServiceError("ik_service_bad_response")
        return answer

    def health(self):
        """The service's ``/health`` answer (solver, batch size, path-check sample limit, joint limits)."""
        if self.solver == "pink":
            self.max_batch = 64
            self.max_path_samples = 4096
            return {
                "status": "ok",
                "solver": "pink",
                "max_batch": 64,
                "path_check": {"max_samples": 4096},
                "joint_limits_rad": self.limits,
            }
        answer = self._call("GET", "/health")
        if not isinstance(answer.get("max_batch"), int) or answer["max_batch"] < 1:
            raise IKServiceError("ik_service_bad_response")
        self.max_batch = answer["max_batch"]
        limit = (answer.get("path_check") or {}).get("max_samples")
        self.max_path_samples = limit if type(limit) is int and limit > 0 else None  # None: no path checker
        return answer

    @staticmethod
    def _world_fields(world):
        """HTTP request fields of a collision ``world`` (``ik_plan.collision_world``) or of none."""
        if not world:
            return {}
        return dict(obstacles=[dict(id=box["id"], min_m=[float(v) for v in box["min_m"]], max_m=[float(v) for v in box["max_m"]])
                               for box in world.get("obstacles") or []],
                    attached_spheres=[[float(v) for v in sphere] for sphere in world.get("attached_spheres") or []],
                    **({"jaw_rad": float(world["jaw_rad"])} if world.get("jaw_rad") is not None else {}))

    @staticmethod
    def _local_world(world):
        """Normalized local collision ``world`` preserving ``attached_box`` for in-process ``Coal`` checking."""
        if not world:
            return None
        out = dict(
            obstacles=[
                dict(id=box["id"], min_m=[float(v) for v in box["min_m"]], max_m=[float(v) for v in box["max_m"]])
                if "min_m" in box
                else dict(box)
                for box in world.get("obstacles") or []
            ],
        )
        spheres = [[float(v) for v in sphere] for sphere in world.get("attached_spheres") or []]
        if spheres:
            out["attached_spheres"] = spheres
        if world.get("jaw_rad") is not None:
            out["jaw_rad"] = float(world["jaw_rad"])
        if world.get("attached_box") is not None:
            box = world["attached_box"]
            rot_raw = box.get("rotation_tcp")
            rot_tcp = (
                [[float(v) for v in row] for row in rot_raw]
                if rot_raw is not None
                else [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            )
            out["attached_box"] = dict(
                centre_tcp_m=[float(v) for v in box["centre_tcp_m"]],
                half_extents_m=[float(v) for v in box["half_extents_m"]],
                rotation_tcp=rot_tcp,
            )
        return out

    def check_path(self, segments, world=None):
        """Collision check of straight joint-space segments (``POST /path_check`` or local ``Coal`` engine).

        ``segments``: ``[{"start_q_rad": [5], "end_q_rad": [5], "start_jaw_rad", "end_jaw_rad", "attached"}]`` (rad;
        jaw = URDF gripper joint angle; ``attached`` False checks that segment without the world's attached object).
        ``world``: ``{"obstacles": [{"id", "min_m", "max_m"}], "attached_spheres": [[x, y, z, radius]],
        "attached_box": ...}`` or None; the table of the deployment is always included. Every segment is sampled at
        most ``path_check_step_rad`` apart. Returns ``{"check_ms", "samples", "segments"}``."""
        if self.max_batch is None:
            self.health()
        if self.max_path_samples is None:
            raise IKServiceError("ik_service_path_check_unavailable")
        bodies = [dict(start_q_rad=[float(v) for v in segment["start_q_rad"]],
                       end_q_rad=[float(v) for v in segment["end_q_rad"]],
                       start_jaw_rad=float(segment.get("start_jaw_rad", 0.0)),
                       end_jaw_rad=float(segment.get("end_jaw_rad", 0.0)),
                       attached=bool(segment.get("attached", True))) for segment in segments]
        if self.solver == "pink":
            if not bodies:
                return dict(check_ms=0.0, samples=0, segments=[])
            from ni_rua.robot.model.pinocchio_backend import so101_collision_engine

            return so101_collision_engine().check_path(
                bodies,
                table_top_z_m=self.table_top_z_m,
                world=self._local_world(world),
                max_joint_step_rad=self.path_check_step_rad,
            )
        chunks, used = [[]], 0
        for body in bodies:
            change = max(abs(b - a) for a, b in zip([*body["start_q_rad"], body["start_jaw_rad"]],
                                                    [*body["end_q_rad"], body["end_jaw_rad"]]))
            count = 2 + int(change / self.path_check_step_rad)  # an upper bound of the service's sample count
            if chunks[-1] and used + count > self.max_path_samples:
                chunks.append([])
                used = 0
            chunks[-1].append(body)
            used += count
        out = dict(check_ms=0.0, samples=0, segments=[])
        for chunk in chunks:
            if not chunk:
                continue
            answer = self._call("POST", "/path_check", dict(
                segments=chunk, max_joint_step_rad=self.path_check_step_rad, joint_limits_rad=self.limits,
                table_top_z_m=self.table_top_z_m, **self._world_fields(world)))
            results = answer.get("segments")
            if not isinstance(results, list) or len(results) != len(chunk) or not all(
                    isinstance(item, dict) and type(item.get("samples")) is int
                    and type(item.get("colliding_samples")) is int
                    and isinstance(item.get("first_collision"), (dict, type(None))) for item in results):
                raise IKServiceError("ik_service_bad_response")
            out["segments"].extend(results)
            out["check_ms"] += float(answer.get("check_ms") or 0.0)
            out["samples"] += sum(item["samples"] for item in results)
        return out

    def solve(self, poses, seed_q_rad, axis_weights, position_tolerance_m, orientation_tolerance_rad, world=None):
        """Candidate solutions per pose: ``[[{"q_rad", "success", "feasible", "position_error_m", "rotation_error_rad"}]]``
        (``feasible``: joint limits, self-collision and the table hold; ``success``: feasible and converged).
        With a collision ``world`` (see ``check_path``) every candidate also carries ``obstacle_free``: the joint
        solution is free of the obstacles and the attached object too (a service without the path checker omits it).

        ``poses``: ``[{"position_m", "quaternion_xyzw"}]`` of the TCP in so101_base. ``seed_q_rad``: five arm angles
        (rad). ``axis_weights``: six goal-frame weights (``ik_plan.axis_weights``). The two tolerances (m, rad) are
        the solver's convergence bounds; the caller re-checks every candidate with its FK."""
        if self.max_batch is None:
            self.health()
        if self.solver == "pink":
            from ni_rua.robot.model.pinocchio_backend import so101_collision_engine, solve_so101_pink_ik
            from .ik_plan import LIMIT_SLACK_RAD, pose_matrix

            weights = [float(v) for v in axis_weights]
            pos_weight = max(weights[:3])
            if pos_weight <= 0.0:
                pos_weight = 1.0
            orientation_costs = (weights[3] * 0.3, weights[4] * 0.3, weights[5] * 0.3)
            limits_tuples = {name: (self.limits[name][0], self.limits[name][1]) for name in ARM_JOINTS}
            seed = [float(v) for v in seed_q_rad]
            pos_tol = float(position_tolerance_m)
            ori_tol = float(orientation_tolerance_rad)
            restarts = max(0, self.num_solutions - 1)
            engine = so101_collision_engine()
            local_world = self._local_world(world)
            jaw_rad = float(local_world["jaw_rad"]) if local_world and "jaw_rad" in local_world else 0.0
            out = []
            for pose in poses:
                candidates = solve_so101_pink_ik(
                    pose_matrix(pose),
                    seed,
                    limits_tuples,
                    position_weight=pos_weight,
                    orientation_costs=orientation_costs,
                    restarts=restarts,
                    position_tolerance_m=pos_tol,
                    orientation_tolerance_rad=ori_tol,
                    return_all_candidates=True,
                )
                pose_entries = []
                for cand in candidates:
                    q = [float(v) for v in cand["q_rad"]]
                    inside_limits = len(q) == 5 and all(
                        math.isfinite(v)
                        and self.limits[name][0] - LIMIT_SLACK_RAD <= v <= self.limits[name][1] + LIMIT_SLACK_RAD
                        for v, name in zip(q, ARM_JOINTS)
                    )
                    state = engine.check_state(
                        q,
                        jaw_rad=jaw_rad,
                        table_top_z_m=self.table_top_z_m,
                        world=local_world,
                        check_world_obstacles=local_world is not None,
                    )
                    pos_err = float(cand["position_error_m"])
                    rot_err = float(cand["orientation_error_rad"])
                    feasible = bool(inside_limits and state["self_and_table_free"])
                    success = bool(feasible and cand["converged"])
                    entry = dict(
                        q_rad=q,
                        success=success,
                        feasible=feasible,
                        position_error_m=pos_err,
                        rotation_error_rad=rot_err,
                    )
                    if local_world is not None:
                        entry["obstacle_free"] = bool(state["obstacle_free"])
                    pose_entries.append(entry)
                out.append(pose_entries)
            return out
        out = []
        for start in range(0, len(poses), self.max_batch):
            chunk = poses[start:start + self.max_batch]
            answer = self._call("POST", "/ik", dict(
                poses=[dict(position_m=[float(v) for v in pose["position_m"]],
                            quaternion_xyzw=[float(v) for v in pose["quaternion_xyzw"]]) for pose in chunk],
                seed_q_rad=[float(v) for v in seed_q_rad], axis_weights=[float(v) for v in axis_weights],
                position_tolerance_m=float(position_tolerance_m), orientation_tolerance_rad=float(orientation_tolerance_rad),
                num_solutions=self.num_solutions, joint_limits_rad=self.limits, table_top_z_m=self.table_top_z_m,
                **self._world_fields(world)))
            solutions = answer.get("solutions")
            if not isinstance(solutions, list) or len(solutions) != len(chunk) or not all(
                    isinstance(item, list) and item and all(isinstance(entry, dict) for entry in item) for item in solutions):
                raise IKServiceError("ik_service_bad_response")
            out.extend(solutions)
        return out
