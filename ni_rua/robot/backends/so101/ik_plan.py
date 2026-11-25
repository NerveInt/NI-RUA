"""SO-101 TCP pose geometry and IK planning shared by the worker's preview, execution and reach query.

Pure numpy (no LeRobot, serial or camera import). Units: metres and radians; frames ``so101_base`` (URDF
``base_link``) and the TCP ``gripper_frame_link``, whose local +Z is the approach axis (outward along the jaws).

Two IK solvers plan one TCP pose; the deployment selects which (``ik_service``):

* ``local_solve``: the packaged damped-least-squares solver (``so101_kinematics.ik`` from the measured seed plus
  deterministic restarts, then a position-first solve). Local, so it can stop in a local minimum.
* ``curobo_solve``: NVIDIA cuRobo v2 multi-seed GPU IK behind the ``services/curobo_ik`` service
  (``ik_service.CuroboClient``), with self-collision and an optional table box. cuRobo only proposes joint solutions:
  each candidate is checked here with the packaged forward kinematics against the requested tolerances and the
  planning joint limits, and the valid candidate with the least joint travel from the seed is used. Residuals in
  every record are this module's FK numbers, never the service's.

Between two planned targets the arm moves on a straight line in joint space (``joint_interpolation``, the one rule
of the worker's execution and of its preview); ``path_facts`` reports what that does to the TCP (lowest point, deviation
from the straight Cartesian line), and ``collision_world`` turns the request's obstacle boxes and held-object box
into the collision world the cuRobo service checks planned targets and paths against (``ik_service.check_path``).

Orientation can be checked as one angle (``orientation_tolerance_rad``) or split into the approach axis and the roll
about it (``approach_tolerance_rad`` / ``roll_tolerance_rad``, ``orientation_components``); the split exists only with
the cuRobo solver, whose per-axis pose weights express it.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from ni_rua.robot.motion.motion_tolerance import orientation_is_free

#: Scale of the approach-tilt weights (rotation about the goal x and y axes) in the position-first cuRobo request
#: (dimensionless). On this 5-DOF arm the roll about the approach axis is always available through wrist_roll while the
#: tilt is coupled to the position, so the position-first request lowers only the tilt weights: the position is then
#: reached first with the smallest tilt it allows and the roll kept (measured on the arm model 2026-11-15: hover pose
#: 0.27 m out at z 0.09 m, 0.2-0.4 mm with 10-12 deg tilt; 0.01 left some candidates 60 deg tilted).
POSITION_PRIORITY_TILT_SCALE = 0.1
#: Convergence bounds sent to cuRobo with every request, (position m, orientation rad): the solver refines every seed
#: to this precision, independent of the request's tolerances, which only decide acceptance here (so a loose tolerance
#: never makes cuRobo stop early with a few millimetres left).
SOLVER_CONVERGENCE = (0.0005, 0.005)
#: Distance (rad) from a planning joint limit at which a failed record names the joint in ``limit_joints``.
LIMIT_REPORT_RAD = 0.02
#: Joint-limit slack (rad) when checking a candidate against the planning limits (float32 round-off of the service).
LIMIT_SLACK_RAD = 1e-6


# ---------------------------------------------------------------- geometry ---

def quaternion_to_matrix(q):
    return Rotation.from_quat(np.asarray(q, dtype=float), scalar_first=False).as_matrix()


def matrix_to_quaternion(R):
    q = Rotation.from_matrix(np.asarray(R, dtype=float)).as_quat(canonical=True, scalar_first=False)
    return (q if q[3] >= 0 else -q).tolist()


def pose_matrix(pose):
    T = np.eye(4)
    T[:3, :3] = quaternion_to_matrix(pose["quaternion_xyzw"])
    T[:3, 3] = pose["position_m"]
    return T


def orientation_components(rotation_achieved, rotation_target):
    """``(approach_error_rad, roll_error_rad)`` of an achieved TCP rotation against a target (3x3, so101_base).

    ``approach_error_rad`` (0..pi): angle between the achieved and the target TCP +Z (approach axis).
    ``roll_error_rad`` (0..pi): rotation about the achieved approach axis left after swinging the target frame by the
    smallest rotation that puts its +Z on the achieved +Z (the jaw-direction difference seen along the approach).
    Example: a straight-down target reached with a 30 deg tilt and the same jaw direction gives ``(0.5236, 0.0)``.
    """
    relative = np.asarray(rotation_achieved, dtype=float).T @ np.asarray(rotation_target, dtype=float)
    z = relative[:, 2]
    approach = math.acos(max(-1.0, min(1.0, float(z[2]))))
    axis = np.cross(z, [0.0, 0.0, 1.0])
    sine = float(np.linalg.norm(axis))
    if sine > 1e-12:
        swung = Rotation.from_rotvec((axis / sine) * math.atan2(sine, float(z[2]))).as_matrix() @ relative
    elif z[2] > 0:
        swung = relative
    else:
        swung = Rotation.from_rotvec([math.pi, 0.0, 0.0]).as_matrix() @ relative
    return approach, abs(math.atan2(float(swung[1, 0]), float(swung[0, 0])))


def pose_residual(T_measured, T_target, *, split=False):
    """Signed position residual (measured - target, m), its norm and the orientation error (rad) of a TCP pose.

    ``split``: also report ``approach_error_rad`` and ``roll_error_rad`` (``orientation_components``)."""
    delta = T_measured[:3, 3] - T_target[:3, 3]
    R = T_target[:3, :3].T @ T_measured[:3, :3]
    angle = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    residual = dict(dx_m=round(float(delta[0]), 5), dy_m=round(float(delta[1]), 5), dz_m=round(float(delta[2]), 5),
                    position_error_m=round(float(np.linalg.norm(delta)), 5), orientation_error_rad=round(angle, 5))
    if split:
        approach, roll = orientation_components(T_measured[:3, :3], T_target[:3, :3])
        residual.update(approach_error_rad=round(approach, 5), roll_error_rad=round(roll, 5))
    return residual


def split_tolerances(path):
    """The approach / roll orientation bounds of one compiled move path, or None when it sets neither.

    Returns ``{"approach_tolerance_rad", "roll_tolerance_rad"}`` (rad); a component the path leaves unset takes the
    path's ``orientation_tolerance_rad``; a value >= pi (``orientation_is_free``) leaves that component free."""
    approach, roll = path.get("approach_tolerance_rad"), path.get("roll_tolerance_rad")
    if approach is None and roll is None:
        return None
    total = float(path["orientation_tolerance_rad"])
    return dict(approach_tolerance_rad=float(total if approach is None else approach),
                roll_tolerance_rad=float(total if roll is None else roll))


def orientation_within(residual, orientation_tolerance_rad, split=None):
    """True when a residual's orientation meets the tolerance: one angle, or each split component (free when >= pi)."""
    if split is None:
        return orientation_is_free(orientation_tolerance_rad) or residual["orientation_error_rad"] <= orientation_tolerance_rad
    return all(orientation_is_free(split[key]) or residual[name] <= split[key]
               for name, key in (("approach_error_rad", "approach_tolerance_rad"), ("roll_error_rad", "roll_tolerance_rad")))


def arrival_verdict(residual, position_tolerance_m, orientation_tolerance_rad, max_upward_dz_m=0.0, *, split=None):
    """The SO-101 arrival rule for one measured TCP residual (``pose_residual``, m / rad). Returns ``(ok, vertical)``.

    Passes when ``position_error_m <= position_tolerance_m`` and the orientation is within tolerance
    (``orientation_within``: one angle, or with ``split`` the approach and roll components separately; free when
    >= pi). With a vertical window ``max_upward_dz_m`` > 0 (m) it also passes when the horizontal error
    ``hypot(dx_m, dy_m)`` is within ``position_tolerance_m``, ``-position_tolerance_m <= dz_m <= max_upward_dz_m`` and
    the orientation is within tolerance; ``vertical`` is then True (the ARX-X5 ``_check_pose_arrival`` rule, with no
    default window).
    """
    orientation_ok = orientation_within(residual, orientation_tolerance_rad, split)
    if residual["position_error_m"] <= position_tolerance_m and orientation_ok:
        return True, False
    vertical = (max_upward_dz_m > 0 and orientation_ok
                and math.hypot(residual["dx_m"], residual["dy_m"]) <= position_tolerance_m
                and -position_tolerance_m <= residual["dz_m"] <= max_upward_dz_m)
    return vertical, vertical


def _finite(values):
    return all(math.isfinite(float(v)) for v in values)


# ------------------------------------------------------------- local solver ---

def local_solve(kin, model, pose, seed, tolerances, ik_config, limits, *, split=None):
    """Plan one TCP pose locally with ``Pink + DAQP``: weighted pose IK, then position-priority IK when that misses.

    ``tolerances``: ``(position_tolerance_m, orientation_tolerance_rad)``. ``ik_config``: the deployment's ``ik``
    section (``max_iterations``, ``restarts``, ``orientation_weight_m_per_rad``). ``limits``: planning joint limits
    ``{joint: (lower, upper)}`` (rad). ``split``: optional ``{"approach_tolerance_rad", "roll_tolerance_rad"}`` (rad).
    Free orientation solves position only. Otherwise the weighted solve runs first (unchanged when it meets the
    tolerances); if it misses, a position-priority solve is tried and used when valid (``solve="position_priority"``).
    Every candidate is re-evaluated via ``kin.fk(model, q)``, ``pose_residual`` and ``orientation_within``. On failure
    the record keeps the weighted residuals and adds ``position_priority``: the quaternion reached at this position
    and its residuals, so the next request can use it. Returns ``(q_rad, record)``.
    """
    from ni_rua.robot.model.pinocchio_backend import solve_so101_pink_ik

    position_tolerance, orientation_tolerance = float(tolerances[0]), float(tolerances[1])
    constrained = _constrained((position_tolerance, orientation_tolerance), split)
    target = pose_matrix(pose)
    seed_list = [float(v) for v in seed]
    w_ori = float(ik_config["orientation_weight_m_per_rad"])
    max_iterations = int(ik_config["max_iterations"])
    restarts = int(ik_config["restarts"])
    pos_tol_solver = min(position_tolerance, SOLVER_CONVERGENCE[0])
    ori_bounds = [b for b in ([orientation_tolerance] if split is None else list(split.values()))
                  if not orientation_is_free(b)]
    ori_tol_solver = min(min(ori_bounds), SOLVER_CONVERGENCE[1]) if ori_bounds else math.pi
    ori_scale_norm = max(min(ori_bounds) if ori_bounds else math.pi, 1e-3)

    def evaluate(candidate):
        q = [float(v) for v in candidate["q_rad"]]
        inside = (
            len(q) == 5
            and _finite(q)
            and all(lower - LIMIT_SLACK_RAD <= value <= upper + LIMIT_SLACK_RAD
                    for value, (lower, upper) in zip(q, limits.values()))
        )
        T = np.asarray(kin.fk(model, q))
        achieved = pose_residual(T, target, split=split is not None)
        ok = (inside and achieved["position_error_m"] <= position_tolerance
              and orientation_within(achieved, orientation_tolerance, split))
        return q, T, achieved, inside, ok, candidate

    def run_pass(scale):
        weights = axis_weights(orientation_tolerance, split, position_priority=False)
        costs = (weights[3] * scale, weights[4] * scale, weights[5] * scale)
        candidates = solve_so101_pink_ik(
            target,
            seed_list,
            limits,
            position_weight=1.0,
            orientation_costs=costs,
            max_iterations=max_iterations,
            restarts=restarts,
            position_tolerance_m=pos_tol_solver,
            orientation_tolerance_rad=ori_tol_solver,
            return_all_candidates=True,
        )
        evaluated = [evaluate(cand) for cand in candidates]
        total_iters = sum(int(cand.get("iterations", 0)) for cand in candidates)
        valid = [item for item in evaluated if item[4]]
        if valid:
            precise = [item for item in valid if item[2]["position_error_m"] <= SOLVER_CONVERGENCE[0]]
            chosen = min(precise or valid, key=lambda item: _travel(item[0], seed_list))
            return chosen, evaluated, total_iters
        pool = [item for item in evaluated if item[3]] or evaluated
        best = min(
            pool,
            key=lambda item: item[2]["position_error_m"] / max(position_tolerance, 1e-6)
            + item[2]["orientation_error_rad"] / ori_scale_norm,
        )
        return best, evaluated, total_iters

    (q, _, achieved, _, ok, best_cand), _, iterations = run_pass(w_ori if constrained else 0.0)
    record = dict(
        ok=ok,
        position_error_m=achieved["position_error_m"],
        orientation_error_rad=achieved["orientation_error_rad"],
        orientation_checked=constrained,
        solver_converged=bool(best_cand.get("converged")),
        iterations=iterations,
    )
    if split is not None:
        record.update(approach_error_rad=achieved["approach_error_rad"], roll_error_rad=achieved["roll_error_rad"])
    if ok or not constrained:
        return q, record

    prio_scale = min(0.01, w_ori * POSITION_PRIORITY_TILT_SCALE) if w_ori > 0.0 else 0.0
    (q_fb, T_fb, reached, _, ok_fb, fb_cand), fb_evaluated, fb_iters = run_pass(prio_scale)
    iterations += fb_iters
    if ok_fb:
        solved = dict(
            ok=True,
            solve="position_priority",
            position_error_m=reached["position_error_m"],
            orientation_error_rad=reached["orientation_error_rad"],
            orientation_checked=True,
            solver_converged=bool(fb_cand.get("converged")),
            iterations=iterations,
        )
        if split is not None:
            solved.update(approach_error_rad=reached["approach_error_rad"], roll_error_rad=reached["roll_error_rad"])
        return q_fb, solved

    in_pos = [item for item in fb_evaluated if item[3] and item[2]["position_error_m"] <= position_tolerance]
    if in_pos:
        _, T_fb, reached, _, _, _ = min(in_pos, key=lambda item: item[2]["orientation_error_rad"])
    elif fb_evaluated:
        pool = [item for item in fb_evaluated if item[3]] or fb_evaluated
        _, T_fb, reached, _, _, _ = min(pool, key=lambda item: item[2]["position_error_m"])
    hint = dict(
        quaternion_xyzw=[round(float(v), 6) for v in matrix_to_quaternion(T_fb[:3, :3])],
        position_error_m=reached["position_error_m"],
        orientation_error_rad=reached["orientation_error_rad"],
    )
    if split is not None:
        hint.update(approach_error_rad=reached["approach_error_rad"], roll_error_rad=reached["roll_error_rad"])
    record.update(iterations=iterations, position_priority=hint)
    return q, record


# ------------------------------------------------------------ cuRobo solver ---

def axis_weights(orientation_tolerance_rad, split=None, *, position_priority=False):
    """cuRobo goal-frame pose weights ``[x, y, z, rot_x, rot_y, rot_z]`` (dimensionless) of one request.

    One orientation angle weights all three rotation axes equally (0 when free). The split weights the rotations
    about the goal x and y axes (they tilt the approach axis, goal +Z) by ``1/approach_tolerance_rad`` and the
    rotation about goal z (the roll) by ``1/roll_tolerance_rad``, normalised to a largest weight of 1 (0 when free).
    ``position_priority`` scales the two tilt weights by ``POSITION_PRIORITY_TILT_SCALE`` and keeps the roll weight."""
    if split is None:
        tilt = roll = 0.0 if orientation_is_free(orientation_tolerance_rad) else 1.0
    else:
        inverse = [0.0 if orientation_is_free(split[key]) else 1.0 / split[key]
                   for key in ("approach_tolerance_rad", "roll_tolerance_rad")]
        top = max(inverse) or 1.0
        tilt, roll = inverse[0] / top, inverse[1] / top
    scale = POSITION_PRIORITY_TILT_SCALE if position_priority else 1.0
    return [1.0, 1.0, 1.0, tilt * scale, tilt * scale, roll]


def _constrained(tolerances, split):
    """True when the request constrains the orientation at all (one angle, or at least one split component)."""
    return (not orientation_is_free(tolerances[1]) if split is None else
            not all(orientation_is_free(value) for value in split.values()))


def _feasible(candidate):
    """cuRobo's constraint verdict of one candidate (limits, self-collision, table box); older services: success."""
    return bool(candidate.get("feasible", candidate.get("success")))


def _check_candidate(kin, model, candidate, target, tolerances, split, limits):
    """``(q, residual, valid)`` of one service candidate: finite, inside ``limits`` and within the tolerances by
    the packaged FK, and feasible for cuRobo (joint limits, self-collision, table box)."""
    q = [float(v) for v in candidate.get("q_rad") or []]
    if len(q) != 5 or not _finite(q):
        return q, None, False
    inside = all(lower - LIMIT_SLACK_RAD <= value <= upper + LIMIT_SLACK_RAD
                 for value, (lower, upper) in zip(q, limits.values()))
    residual = pose_residual(np.asarray(kin.fk(model, q)), target, split=split is not None)
    valid = (_feasible(candidate) and inside and residual["position_error_m"] <= tolerances[0]
             and orientation_within(residual, tolerances[1], split))
    return q, residual, valid


def limit_joints(q, limits):
    """Arm joints of ``q`` (rad) within ``LIMIT_REPORT_RAD`` of a planning limit, as ``{joint: "lower"|"upper"}``.

    On this 5-DOF arm a pose that fails with a joint at its limit is usually blocked by that joint's range, e.g.
    wrist_roll at a limit for a jaw yaw the wrist cannot turn to, wrist_flex at its upper limit for a straight-down
    approach too high or far out."""
    out = {}
    for value, (name, (lower, upper)) in zip(q, limits.items()):
        if value - lower <= LIMIT_REPORT_RAD:
            out[name] = "lower"
        elif upper - value <= LIMIT_REPORT_RAD:
            out[name] = "upper"
    return out


def _travel(q, seed):
    return max(abs(a - b) for a, b in zip(q, seed))


def curobo_solve(client, kin, model, limits, requests, seed, world=None):
    """Plan TCP poses with the cuRobo service; one ``(q_rad, record)`` per request.

    ``world`` (``collision_world`` or None): obstacle boxes and held-object spheres. The service then labels every
    candidate ``obstacle_free``; among the valid candidates those not labelled colliding are preferred, and when
    every valid candidate collides the best of them is still used (the obstacles are the model's estimates, so they
    advise and never fail planning). A solved record then carries ``obstacle_free`` (false: the planned joints
    overlap an obstacle, or the held object overlaps one; absent when the service gave no label).

    ``requests``: ``[(pose, (position_tolerance_m, orientation_tolerance_rad), split or None)]``; every pose is
    solved independently from ``seed`` (five measured or planned arm angles, rad). ``limits``: planning joint
    limits ``{joint: (lower, upper)}`` in ARM_JOINTS order. Requests with the same weights are sent in one batch.

    Per request: the candidates of the full request (``axis_weights``) are checked (``_check_candidate``); among the
    valid ones, those within ``SOLVER_CONVERGENCE[0]`` of the target position (by this FK) come first, and the one
    with the least largest joint travel from ``seed`` is used. When none is valid and the orientation is
    constrained, a position-first request follows (``solve="position_priority"`` when it yields a valid candidate).
    A failed record carries the residuals of the closest candidate, ``position_priority`` (the quaternion the
    position-first request reached at this position, with its residuals) when it reached the position,
    ``limit_joints`` (``limit_joints`` of the closest candidate: joints at a planning limit, ``{}`` when none) and
    ``in_collision_candidates``: candidates within the tolerances by this FK that cuRobo marked infeasible (self-collision,
    the table box or joint limits). Raises ``ik_service.IKServiceError`` when the service cannot answer.
    """
    targets = [pose_matrix(pose) for pose, _, _ in requests]
    results = [None] * len(requests)

    def batch(indices, position_priority, relax_roll=False):
        groups = {}
        for index in indices:
            _, tolerances, split = requests[index]
            weights = axis_weights(tolerances[1], split, position_priority=position_priority)
            if relax_roll:
                weights[5] *= POSITION_PRIORITY_TILT_SCALE
            key = (tuple(weights), *SOLVER_CONVERGENCE)
            groups.setdefault(key, []).append(index)
        answers = {}
        for (weights, position_tolerance, orientation_tolerance), members in groups.items():
            solutions = client.solve([requests[index][0] for index in members], seed, list(weights),
                                     position_tolerance, orientation_tolerance, **({"world": world} if world else {}))
            answers.update(zip(members, solutions))
        return answers

    def choose(index, candidates):
        _, tolerances, split = requests[index]
        checked = [_check_candidate(kin, model, candidate, targets[index], tolerances, split, limits)
                   for candidate in candidates]
        valid = [(q, residual, candidate.get("obstacle_free") if world else None)
                 for (q, residual, ok), candidate in zip(checked, candidates) if ok]
        valid = [item for item in valid if item[2] is not False] or valid  # obstacle-free first, never a failure
        # Precise solutions first (position within SOLVER_CONVERGENCE by this FK), then any other valid one; the
        # least joint travel within that tier.
        precise = [item for item in valid if item[1]["position_error_m"] <= SOLVER_CONVERGENCE[0]]
        tier = precise or valid
        if tier:
            return min(tier, key=lambda item: _travel(item[0], seed)), checked
        return None, checked

    def record(residual, tolerances, split, ok, **extra):
        out = dict(ok=ok, solver=getattr(client, "solver", "curobo"), position_error_m=residual["position_error_m"],
                   orientation_error_rad=residual["orientation_error_rad"],
                   orientation_checked=_constrained(tolerances, split),
                   solver_converged=ok, **extra)
        if split is not None:
            out.update(approach_error_rad=residual["approach_error_rad"], roll_error_rad=residual["roll_error_rad"])
        return out

    def precise(chosen):
        return chosen is not None and chosen[1]["position_error_m"] <= SOLVER_CONVERGENCE[0]

    def solved(index, chosen, **extra):
        _, tolerances, split = requests[index]
        if chosen[2] is not None:
            extra["obstacle_free"] = bool(chosen[2])
        results[index] = (chosen[0], record(chosen[1], tolerances, split, True, **extra))

    first = batch(range(len(requests)), False)
    retry, failures, imprecise = [], {}, {}
    for index, candidates in first.items():
        _, tolerances, split = requests[index]
        chosen, checked = choose(index, candidates)
        if chosen is not None and (precise(chosen) or not _constrained(tolerances, split)):
            solved(index, chosen)
            continue
        if chosen is not None:
            imprecise[index] = chosen  # valid but not precise: the position-first request may place it exactly
        else:
            failures[index] = checked
        if _constrained(tolerances, split):
            retry.append(index)
    second = batch(retry, True) if retry else {}
    for index, chosen in imprecise.items():
        better, _ = choose(index, second[index])
        # A precise retry replaces the first choice unless it trades an obstacle-free solution for a colliding one.
        if precise(better) and not (better[2] is False and chosen[2] is not False):
            solved(index, better, solve="position_priority")
        else:
            solved(index, chosen)
    def hint_from(index, candidates):
        """The ``position_priority`` hint of a failed request: the orientation reached at the target position by the
        feasible candidate with the smallest orientation error, or None when none reaches the position."""
        _, tolerances, split = requests[index]
        checked = [_check_candidate(kin, model, candidate, targets[index], tolerances, split, limits)
                   for candidate in candidates]
        reached = [(q, residual) for (q, residual, _), candidate in zip(checked, candidates)
                   if residual is not None and _feasible(candidate) and residual["position_error_m"] <= tolerances[0]]
        if not reached:
            return None
        q_hint, residual_hint = min(reached, key=lambda item: item[1]["orientation_error_rad"])
        return dict(quaternion_xyzw=[round(float(v), 6) for v in matrix_to_quaternion(np.asarray(kin.fk(model, q_hint))[:3, :3])],
                    position_error_m=residual_hint["position_error_m"],
                    orientation_error_rad=residual_hint["orientation_error_rad"],
                    **({key: residual_hint[key] for key in ("approach_error_rad", "roll_error_rad")} if split is not None else {}))

    hints, unresolved = {}, []
    for index in failures:
        if index in second:
            chosen, _ = choose(index, second[index])
            if chosen is not None:
                solved(index, chosen, solve="position_priority")
                continue
            hints[index] = hint_from(index, second[index])
            if hints[index] is None:
                unresolved.append(index)
    # The roll can be what blocks (wrist_roll range): a last request lowers every rotation weight so the position is
    # reached with whatever orientation the arm allows there, only to report that orientation as the hint.
    relaxed = batch(unresolved, True, relax_roll=True) if unresolved else {}
    for index, checked in failures.items():
        if results[index] is not None:
            continue
        _, tolerances, split = requests[index]
        hint = hints.get(index) or (hint_from(index, relaxed[index]) if index in relaxed else None)
        scored = [(q, residual) for q, residual, _ in checked if residual is not None]
        bounds = [bound for bound in ([tolerances[1]] if split is None else list(split.values()))
                  if not orientation_is_free(bound)]
        orientation_scale = max(min(bounds) if bounds else math.pi, 1e-3)
        q_best, residual_best = (min(scored, key=lambda item: item[1]["position_error_m"] / tolerances[0]
                                     + item[1]["orientation_error_rad"] / orientation_scale)
                                 if scored else (list(seed), pose_residual(np.asarray(kin.fk(model, list(seed))),
                                                                           targets[index], split=split is not None)))
        # The joints that block: those at a limit in the position-first candidate closest to the target position
        # (it pushes against the binding joint), else in the closest candidate.
        pressing = q_best
        if index in second:
            reachable = [(q, residual) for q, residual, _ in
                         (_check_candidate(kin, model, candidate, targets[index], tolerances, split, limits)
                          for candidate in second[index]) if residual is not None]
            if reachable:
                pressing = min(reachable, key=lambda item: item[1]["position_error_m"])[0]
        # Reachable but blocked: within the tolerances by this arm's FK, yet infeasible for cuRobo (a collision of the
        # sphere model with itself or the table box, or a joint limit).
        blocked = sum(1 for (_, residual, _), candidate in zip(checked, first[index])
                      if residual is not None and not _feasible(candidate) and residual["position_error_m"] <= tolerances[0]
                      and orientation_within(residual, tolerances[1], split))
        failed = record(residual_best, tolerances, split, False, in_collision_candidates=blocked,
                        limit_joints=limit_joints(pressing, limits))
        if hint is not None:
            failed["position_priority"] = hint
        results[index] = (q_best, failed)
    return results


# ------------------------------------------------- path between targets ---

#: Collision spheres one held-object box may use (the cuRobo service's attached-object slots).
HELD_OBJECT_SPHERES = 32
#: Finest subdivision of a held-object box: spheres per shortest box side (more only shrinks the uncovered corners).
HELD_OBJECT_CELLS = 3


def ruckig_joint_interpolation(
    start_q,
    target_q,
    speed_rad_s,
    rate_hz,
    max_acceleration_rad_s2,
    max_jerk_rad_s3,
    *,
    current_velocity=None,
    current_acceleration=None,
) -> list[np.ndarray]:
    """3rd-order jerk-limited joint trajectory rows from ``start_q`` (excluded) to ``target_q`` (included), rad.

    Rest-to-rest (``current_velocity`` and ``current_acceleration`` both None): uses 1-DoF ``ruckig.Ruckig(1)``
    path scalar ``s in [0, 1]`` scaled by ``max(|target_q - start_q|)`` so every row lies strictly on the straight
    joint-space line ``start_q + s * (target_q - start_q)`` and ``len(rows) == ceil(duration * rate_hz)`` (at least 1,
    last row strictly ``target_q``).
    Mid-motion online retargeting (non-None ``current_velocity`` or ``current_acceleration``): uses multi-DoF
    ``ruckig.Ruckig(len(start_q))`` from ``(start_q, v0, a0)`` to ``(target_q, 0, 0)``.
    """
    start = np.asarray(start_q, dtype=float).reshape(-1)
    target = np.asarray(target_q, dtype=float).reshape(-1)
    dof = len(start)
    if target.shape != (dof,) or not (_finite(start) and _finite(target)):
        raise ValueError("ruckig_joint_interpolation_invalid_endpoints")
    speed, rate = float(speed_rad_s), float(rate_hz)
    accel, jerk = float(max_acceleration_rad_s2), float(max_jerk_rad_s3)
    if not all(math.isfinite(v) and v > 0 for v in (speed, rate, accel, jerk)):
        raise ValueError("ruckig_joint_interpolation_invalid_limits")
    import ruckig

    if current_velocity is None and current_acceleration is None:
        delta = target - start
        max_dq = float(np.abs(delta).max()) if dof else 0.0
        if max_dq <= 1e-12:
            return [target.copy()]
        otg = ruckig.Ruckig(1)
        inp = ruckig.InputParameter(1)
        inp.current_position = [0.0]
        inp.current_velocity = [0.0]
        inp.current_acceleration = [0.0]
        inp.target_position = [1.0]
        inp.target_velocity = [0.0]
        inp.target_acceleration = [0.0]
        inp.max_velocity = [speed / max_dq]
        inp.max_acceleration = [accel / max_dq]
        inp.max_jerk = [jerk / max_dq]
        traj = ruckig.Trajectory(1)
        res = otg.calculate(inp, traj)
        if res not in (ruckig.Result.Working, ruckig.Result.Finished):
            raise ValueError(f"ruckig_trajectory_failed:{res}")
        duration = float(traj.duration)
        steps = max(1, int(math.ceil(duration * rate)))
        rows = []
        for index in range(1, steps + 1):
            if index == steps:
                rows.append(target.copy())
            else:
                pos, _, _ = traj.at_time(min(index / rate, duration))
                s = float(np.clip(pos[0], 0.0, 1.0))
                rows.append(start + s * delta)
        return rows

    v0 = np.zeros(dof, dtype=float) if current_velocity is None else np.asarray(current_velocity, dtype=float).reshape(-1)
    a0 = np.zeros(dof, dtype=float) if current_acceleration is None else np.asarray(current_acceleration, dtype=float).reshape(-1)
    if v0.shape != (dof,) or a0.shape != (dof,) or not (_finite(v0) and _finite(a0)):
        raise ValueError("ruckig_joint_interpolation_invalid_initial_state")
    if float(np.abs(target - start).max()) <= 1e-12 and float(np.abs(v0).max()) <= 1e-12 and float(np.abs(a0).max()) <= 1e-12:
        return [target.copy()]
    otg = ruckig.Ruckig(dof)
    inp = ruckig.InputParameter(dof)
    inp.current_position = start.tolist()
    inp.current_velocity = v0.tolist()
    inp.current_acceleration = a0.tolist()
    inp.target_position = target.tolist()
    inp.target_velocity = [0.0] * dof
    inp.target_acceleration = [0.0] * dof
    inp.max_velocity = [speed] * dof
    inp.max_acceleration = [accel] * dof
    inp.max_jerk = [jerk] * dof
    traj = ruckig.Trajectory(dof)
    res = otg.calculate(inp, traj)
    if res not in (ruckig.Result.Working, ruckig.Result.Finished):
        raise ValueError(f"ruckig_trajectory_failed:{res}")
    duration = float(traj.duration)
    steps = max(1, int(math.ceil(duration * rate)))
    rows = []
    for index in range(1, steps + 1):
        if index == steps:
            rows.append(target.copy())
        else:
            pos, _, _ = traj.at_time(min(index / rate, duration))
            rows.append(np.asarray(pos, dtype=float))
    return rows


def interpolation_steps(
    start_q,
    target_q,
    speed_rad_s,
    rate_hz,
    *,
    max_acceleration_rad_s2=None,
    max_jerk_rad_s3=None,
    current_velocity=None,
    current_acceleration=None,
):
    """Commands of one move: ``ceil(largest joint change / speed_rad_s * rate_hz)`` (at least 1), or S-curve steps."""
    if max_acceleration_rad_s2 is None and max_jerk_rad_s3 is None:
        change = float(np.abs(np.subtract(target_q, start_q)).max())
        return max(1, int(math.ceil(change / speed_rad_s * rate_hz)))
    if max_acceleration_rad_s2 is None or max_jerk_rad_s3 is None:
        raise ValueError("max_acceleration_rad_s2 and max_jerk_rad_s3 must both be set or both None")
    return len(
        ruckig_joint_interpolation(
            start_q,
            target_q,
            speed_rad_s,
            rate_hz,
            max_acceleration_rad_s2,
            max_jerk_rad_s3,
            current_velocity=current_velocity,
            current_acceleration=current_acceleration,
        )
    )


def joint_interpolation(
    start_q,
    target_q,
    steps=None,
    *,
    speed_rad_s=None,
    rate_hz=None,
    max_acceleration_rad_s2=None,
    max_jerk_rad_s3=None,
    current_velocity=None,
    current_acceleration=None,
):
    """The joint goals of one move: ``steps`` rows on the straight joint-space line from ``start_q`` (excluded) to
    ``target_q`` (included), rad. The worker streams these rows (``SO101Adapter._segment``) and the preview evaluates
    the same rows (``path_facts``); when ``max_acceleration_rad_s2`` and ``max_jerk_rad_s3`` are provided, delegates
    to ``ruckig_joint_interpolation``."""
    if max_acceleration_rad_s2 is None and max_jerk_rad_s3 is None:
        if steps is None:
            if speed_rad_s is None or rate_hz is None:
                raise ValueError("steps or (speed_rad_s, rate_hz) required for linear joint_interpolation")
            steps = interpolation_steps(start_q, target_q, speed_rad_s, rate_hz)
        start, target = np.asarray(start_q, dtype=float), np.asarray(target_q, dtype=float)
        return [target.copy() if index == steps else start + (target - start) * index / steps
                for index in range(1, steps + 1)]
    if max_acceleration_rad_s2 is None or max_jerk_rad_s3 is None:
        raise ValueError("max_acceleration_rad_s2 and max_jerk_rad_s3 must both be set or both None")
    if speed_rad_s is None or rate_hz is None:
        raise ValueError("speed_rad_s and rate_hz required for ruckig joint_interpolation")
    return ruckig_joint_interpolation(
        start_q,
        target_q,
        speed_rad_s,
        rate_hz,
        max_acceleration_rad_s2,
        max_jerk_rad_s3,
        current_velocity=current_velocity,
        current_acceleration=current_acceleration,
    )


def path_check_segments(start_q, target_q, start_jaw_rad, end_jaw_rad, attached, ramp=None):
    """The straight segments of one move for the path collision check (``CuroboClient.check_path``):
    ``[(from_fraction, to_fraction, segment)]`` in move order, fractions of the move (0 = start, 1 = target).

    The arm moves on the joint line ``start_q`` -> ``target_q`` (rad), linear in the move's command progress
    (``joint_interpolation``). ``ramp`` (None, or ``(start_fraction, end_fraction)`` of a during-motion jaw event):
    the jaw holds ``start_jaw_rad`` until ``start_fraction``, moves linearly to ``end_jaw_rad`` by ``end_fraction`` and
    holds it after (``gripper_motion.synchronize``, the schedule the worker streams). The service interpolates the
    jaw linearly within each segment, so the move is cut at the two fractions: every segment is then linear in both
    the arm and the jaw, and the checked jaw angle at each arm configuration is the commanded one. Without a ramp the
    move is one segment (its jaw is constant in a planned move: ``start_jaw_rad == end_jaw_rad``). ``attached``: the
    held spheres ride along (bool). Example: ``ramp=(0.5, 1.0)`` -> two segments, the jaw still over the first half."""
    start, target = [float(v) for v in start_q], [float(v) for v in target_q]
    knots = [0.0, 1.0]
    if ramp is not None:
        low, high = float(ramp[0]), float(ramp[1])
        knots = sorted({0.0, 1.0, *(value for value in (low, high) if 0.0 < value < 1.0)})

    def arm(fraction):
        return [a + (b - a) * fraction for a, b in zip(start, target)]

    def jaw(fraction):
        if ramp is None:
            share = fraction
        else:
            low, high = float(ramp[0]), float(ramp[1])
            share = min(1.0, max(0.0, (fraction - low) / (high - low)))  # validated: start_fraction < end_fraction
        if share >= 1.0:
            return float(end_jaw_rad)  # exact end values, no round-off
        return float(start_jaw_rad) + (float(end_jaw_rad) - float(start_jaw_rad)) * share
    return [(a, b, dict(start_q_rad=arm(a), end_q_rad=arm(b), start_jaw_rad=jaw(a), end_jaw_rad=jaw(b),
                        attached=bool(attached)))
            for a, b in zip(knots, knots[1:])]


def path_facts(kin, model, start_q, rows, tip_points=None):
    """Where the TCP goes on one interpolated move (FK of every commanded row; m in so101_base).

    ``start_q``: joints the move starts from; ``rows``: its joint goals (``joint_interpolation``). ``tip_points``:
    ``[[x, y, z]]`` fingertip points in the TCP frame (m), or one such list per sample (start, then each row) when the
    jaw moves during the move; None reports no fingertip height. Returns ``samples`` (start + rows),
    ``min_tcp_z_m``, ``min_fingertip_z_m`` (lowest of the fingertip points), ``lowest_at_fraction`` (0 = start,
    1 = target; of ``min_fingertip_z_m`` when reported, else of ``min_tcp_z_m``), ``max_line_deviation_m`` (largest
    distance of a TCP sample from the straight segment between the start and target TCP positions) and
    ``dip_below_endpoints_m`` (how far ``min_tcp_z_m`` lies under the lower of the two endpoints, >= 0).
    Example: a 15 cm radial move at z 0.05 m gives ``max_line_deviation_m`` and ``dip_below_endpoints_m`` near 0.014."""
    poses = [np.asarray(kin.fk(model, [float(v) for v in q])) for q in (start_q, *rows)]
    points = np.asarray([T[:3, 3] for T in poses])
    a, b = points[0], points[-1]
    span = b - a
    length = float(span @ span)
    along = np.clip((points - a) @ span / length, 0.0, 1.0) if length > 1e-12 else np.zeros(len(points))
    deviation = np.linalg.norm(points - (a + along[:, None] * span), axis=1)
    heights, lowest = points[:, 2], int(np.argmin(points[:, 2]))
    facts = dict(samples=len(points), min_tcp_z_m=round(float(heights[lowest]), 4))
    if tip_points is not None:
        per_sample = tip_points if np.asarray(tip_points[0]).ndim == 2 else [tip_points] * len(poses)
        tips = np.asarray([min(float((T[:3, :3] @ np.asarray(point, dtype=float) + T[:3, 3])[2]) for point in tip)
                           for T, tip in zip(poses, per_sample)])
        lowest = int(np.argmin(tips))
        facts["min_fingertip_z_m"] = round(float(tips[lowest]), 4)
    facts.update(lowest_at_fraction=round(lowest / (len(points) - 1), 3),
                 max_line_deviation_m=round(float(deviation.max()), 4),
                 dip_below_endpoints_m=round(max(0.0, float(min(a[2], b[2]) - heights.min())), 4))
    return facts


def box_spheres(min_m, max_m, margin_m=0.0, max_spheres=HELD_OBJECT_SPHERES):
    """Spheres filling an axis-aligned box: ``(centres [[x, y, z]], radius_m)`` in the box's frame.

    The box ``min_m`` .. ``max_m`` (m) is grown by ``margin_m`` on every face and cut into a grid of equal cells, as
    fine as ``HELD_OBJECT_CELLS`` cells along its shortest side while the cell count stays within ``max_spheres``
    (coarser otherwise). One sphere per cell, all of the radius that fits the smallest cell side, the outer ones
    touching the box faces: the spheres never reach outside the box, and its edges and corners stay uncovered by up
    to 0.73 x radius. Example: a 0.03 m cube without margin gives 27 spheres of radius 0.005 m."""
    low = np.asarray(min_m, dtype=float) - float(margin_m)
    size = np.asarray(max_m, dtype=float) + float(margin_m) - low
    counts = np.ones(3, dtype=int)
    for cells in range(HELD_OBJECT_CELLS, 0, -1):
        counts = np.maximum(1, np.ceil(size / (size.min() / cells) - 1e-9).astype(int))
        if counts.prod() <= max_spheres:
            break
    while counts.prod() > max_spheres:
        counts[int(np.argmax(counts))] -= 1
    radius = float((size / counts).min() / 2)
    axes = [np.linspace(low[i] + radius, low[i] + size[i] - radius, counts[i]) if counts[i] > 1
            else np.asarray([low[i] + size[i] / 2]) for i in range(3)]
    return [[float(x), float(y), float(z)] for x in axes[0] for y in axes[1] for z in axes[2]], radius


def collision_world(review_proxies, arm, T_base_tcp):
    """The collision world of one request for the IK/path-check service, or None when it names no obstacle and no held object.

    ``review_proxies``: the compiled request's validated proxies (``proxy_contract.validate_proxies``): ``obstacles``
    ``[{"id", "min_m", "max_m", "margin_m"}]`` (axis-aligned boxes in so101_base, m) and ``held_objects[arm]``
    (``{"min_m", "max_m", "margin_m"}``, the held object's box in so101_base now). ``T_base_tcp``: the measured TCP
    pose (4x4) the held box is fixed to. Returns ``obstacles`` (each box grown by its ``margin_m`` on every face),
    ``attached_spheres`` (``box_spheres`` of the held box with its margin, ``[x, y, z, radius]`` in the TCP frame),
    ``attached_box`` (exact TCP-frame ``{"centre_tcp_m", "half_extents_m", "rotation_tcp"}`` for ``Coal``, or None)
    and ``held_object`` (``{"spheres", "sphere_radius_m"}`` or None)."""
    proxies = review_proxies or {}
    obstacles = [dict(id=box["id"], min_m=[round(float(v) - float(box.get("margin_m", 0.0)), 6) for v in box["min_m"]],
                      max_m=[round(float(v) + float(box.get("margin_m", 0.0)), 6) for v in box["max_m"]])
                 for box in proxies.get("obstacles") or []]
    held = (proxies.get("held_objects") or {}).get(arm)
    if not obstacles and held is None:
        return None
    spheres, summary, attached_box = [], None, None
    if held is not None:
        margin = float(held.get("margin_m", 0.0))
        centres, radius = box_spheres(held["min_m"], held["max_m"], margin)
        inverse = np.linalg.inv(np.asarray(T_base_tcp, dtype=float))
        spheres = [[*(float(v) for v in inverse[:3, :3] @ np.asarray(centre) + inverse[:3, 3]), radius] for centre in centres]
        summary = dict(spheres=len(spheres), sphere_radius_m=round(radius, 4))
        low = np.asarray(held["min_m"], dtype=float) - margin
        high = np.asarray(held["max_m"], dtype=float) + margin
        centre_tcp = inverse[:3, :3] @ (0.5 * (low + high)) + inverse[:3, 3]
        half_extents = 0.5 * (high - low)
        attached_box = dict(
            centre_tcp_m=[float(v) for v in centre_tcp],
            half_extents_m=[float(v) for v in half_extents],
            rotation_tcp=[[float(v) for v in row] for row in inverse[:3, :3]],
        )
    return dict(obstacles=obstacles, attached_spheres=spheres, attached_box=attached_box, held_object=summary)
