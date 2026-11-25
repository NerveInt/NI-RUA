"""Local trust-region inverse-kinematics endpoint polish over ``ni_rua.robot.model.kinematics``.

cuRobo ``plan_path`` typically terminates trajectory optimization with ``2.5-4.9 mm`` of
IK endpoint slack (``planned_ik_residual_m``), which consumes most of the ``5.0 mm`` ``tight``
stage tolerance budget. Keeping ``positions[0]`` strictly unchanged and bounding joint deltas
within ``max_joint_delta_rad`` (default ``0.045 rad``) and declared joint limits, this module
polishes the terminal configuration via ``kinematics.solve_ik`` (``Pinocchio + Pink + DAQP`` or
analytical-Jacobian Levenberg-Marquardt when a ``KinematicChainSpec`` is supplied, or
finite-difference DLS over a ``world_link6`` callback) and blends the terminal delta smoothly
into the trajectory tail via a zero-endpoint-derivative cubic Hermite ramp.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation

from ..model.kinematics import (
    IKOptions,
    IKTarget,
    KinematicChainSpec,
    fk_links,
    is_pink_compat_type_error,
    solve_ik,
)


def _quat_xyzw_to_rot(quat_xyzw: Any) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or float(np.linalg.norm(q)) <= 1e-12:
        raise ValueError("invalid_target_quaternion")
    return np.asarray(Rotation.from_quat(q, scalar_first=False).as_matrix(), dtype=np.float64)


def polish_trajectory_endpoint(
    positions: np.ndarray,
    *,
    limits: np.ndarray,
    target: dict[str, Any],
    pose_error: Callable[[np.ndarray], tuple[float, float]],
    world_link6: Callable[[np.ndarray], np.ndarray] | None = None,
    chain: KinematicChainSpec | None = None,
    T_world_base: np.ndarray | None = None,
    max_joint_delta_rad: float = 0.045,
    max_iters: int = 8,
    diagnostic: dict[str, Any] | None = None,
) -> np.ndarray:
    """Polish a planned N-DoF trajectory endpoint within a local trust region and blend the tail smoothly.

    Parameters:
        positions: Planned joint trajectory of shape ``(T, N)`` in ``rad`` (``T >= 2``, ``N >= 1``).
        limits: Joint bounds of shape ``(N, 2)`` ``[lower_rad, upper_rad]`` in ``rad``.
        target: Target pose dict ``{"position_m": [x, y, z], "quaternion_xyzw": [qx, qy, qz, qw]}`` (``m``).
        pose_error: Callable ``q_rad -> (pos_error_m, ori_error_rad)`` in ``m`` and ``rad``.
        world_link6: Optional forward-kinematics callable ``q_rad -> 4x4 T_world_tip`` used when ``chain`` is omitted.
        chain: Optional N-DoF ``KinematicChainSpec`` enabling analytical ``kinematics.solve_ik`` (``Pink + DAQP`` or LM).
        T_world_base: Optional ``4x4`` world-to-base transform in ``m`` paired with ``chain``.
        max_joint_delta_rad: Infinity-norm trust-region bound ``|q_polished - q_end|_inf`` in ``rad`` (default ``0.045``).
        max_iters: Maximum local IK iterations (default ``8``).
        diagnostic: Optional mutable dict populated with ``selected``, ``reason``, and before/after residuals.

    Returns:
        Polished ``(T, N)`` trajectory in ``rad``, or ``positions`` unchanged if outside the ``0.35-6.0 mm``
        planner-slack window or if no strict improvement is achieved.
    """
    traj = np.asarray(positions, dtype=float)
    lim = np.asarray(limits, dtype=float)
    eff_chain = chain if chain is not None else getattr(pose_error, "chain", None)
    eff_base = T_world_base if T_world_base is not None else getattr(pose_error, "T_world_base", None)
    eff_fk = world_link6 if world_link6 is not None else getattr(pose_error, "world_link6", None)

    if (
        (eff_chain is None and eff_fk is None)
        or traj.ndim != 2
        or len(traj) < 2
        or traj.shape[1] < 1
        or lim.shape != (traj.shape[1], 2)
        or not np.isfinite(traj).all()
        or not np.isfinite(lim).all()
    ):
        return positions

    dof = int(traj.shape[1])
    q0 = traj[-1].copy()
    init_pos_err, init_ori_err = pose_error(q0)
    if not (math.isfinite(init_pos_err) and math.isfinite(init_ori_err)):
        return positions

    needs_pos_polish = 0.00035 < init_pos_err <= 0.0060
    needs_ori_polish = (0.0030 < init_ori_err <= 0.0600) and (init_pos_err <= 0.0060)
    if not (needs_pos_polish or needs_ori_polish):
        return positions

    target_pos = np.asarray(target["position_m"], dtype=float)
    if target_pos.shape != (3,) or not np.isfinite(target_pos).all():
        return positions
    try:
        target_rot = _quat_xyzw_to_rot(target["quaternion_xyzw"])
    except ValueError:
        return positions

    lo = np.maximum(lim[:, 0], q0 - float(max_joint_delta_rad))
    hi = np.minimum(lim[:, 1], q0 + float(max_joint_delta_rad))
    if np.any(lo > hi):
        return positions

    q_best: np.ndarray | None = None
    if isinstance(eff_chain, KinematicChainSpec) and eff_chain.dof == dof:
        bounded_chain = eff_chain.with_limits(lim)
        ik_target = IKTarget.from_pose(target_pos, rotation=target_rot, T_world_base=eff_base)
        try:
            ik_res = solve_ik(
                bounded_chain,
                ik_target,
                q0,
                IKOptions(
                    tasks=("pose",),
                    position_weight=1.0,
                    orientation_weight=0.25,
                    position_tolerance_m=0.00005,
                    orientation_tolerance_rad=0.0005,
                    max_iterations=int(max_iters),
                    damping="adaptive_lm",
                    trust_region_rad=float(max_joint_delta_rad),
                    max_step_rad=0.020,
                ),
            )
            q_best = ik_res.q_rad
        except (ImportError, TypeError) as exc:
            if isinstance(exc, TypeError) and not is_pink_compat_type_error(exc):
                raise
            eff_fk = eff_fk or (lambda q: fk_links(eff_chain, q, T_world_base=eff_base)[eff_chain.tip_link])

    if q_best is None:
        if eff_fk is None:
            return positions
        rot_weight = 0.25
        eps = 1e-6
        eye6 = np.eye(6, dtype=float)
        q_cur = np.clip(q0.copy(), lo, hi)
        q_best = q_cur.copy()
        best_score = init_pos_err + 0.08 * init_ori_err

        for _ in range(max_iters):
            t_cur = np.asarray(eff_fk(q_cur), dtype=float)
            e_pos = target_pos - t_cur[:3, 3]
            e_rot = np.asarray(Rotation.from_matrix(target_rot @ t_cur[:3, :3].T).as_rotvec(), dtype=float)
            cur_pos_err = float(np.linalg.norm(e_pos))
            cur_ori_err = float(np.linalg.norm(e_rot))
            cur_score = cur_pos_err + 0.08 * cur_ori_err
            if cur_score < best_score:
                best_score = cur_score
                q_best = q_cur.copy()
            if cur_pos_err <= 0.00005 and cur_ori_err <= 0.0005:
                break

            err6 = np.concatenate([e_pos, rot_weight * e_rot])
            jac = np.empty((6, dof), dtype=float)
            for j in range(dof):
                step_h = eps if q_cur[j] + eps <= lim[j, 1] else -eps
                q_pert = q_cur.copy()
                q_pert[j] += step_h
                t_pert = np.asarray(eff_fk(q_pert), dtype=float)
                jac[:3, j] = (t_pert[:3, 3] - t_cur[:3, 3]) / step_h
                d_rot = np.asarray(Rotation.from_matrix(t_pert[:3, :3] @ t_cur[:3, :3].T).as_rotvec(), dtype=float)
                jac[3:, j] = rot_weight * (d_rot / step_h)

            lam2 = max(1e-6, min(1e-2, 0.02 * float(np.dot(err6, err6))))
            dq = jac.T @ np.linalg.solve(jac @ jac.T + lam2 * eye6, err6)
            max_step = float(np.max(np.abs(dq)))
            if max_step > 0.020:
                dq = dq * (0.020 / max_step)
            q_cur = np.clip(q_cur + dq, lo, hi)

    final_pos_err, final_ori_err = pose_error(q_best)
    if (
        not np.isfinite(q_best).all()
        or np.any(q_best < lim[:, 0])
        or np.any(q_best > lim[:, 1])
        or final_pos_err > init_pos_err + 1e-6
        or final_ori_err > max(init_ori_err + 1e-4, 0.015)
        or not ((init_pos_err - final_pos_err >= 1e-4) or (init_ori_err - final_ori_err >= 5e-4))
    ):
        if diagnostic is not None:
            diagnostic.update(
                selected=False,
                reason="no_strict_endpoint_improvement",
                before_position_error_m=float(init_pos_err),
                before_orientation_error_rad=float(init_ori_err),
            )
        return positions

    dq_total = q_best - q0
    k_tail = min(len(traj), max(2, len(traj) // 3 + 1))
    s = np.linspace(0.0, 1.0, k_tail, dtype=float)[:, None]
    blend = 3.0 * (s ** 2) - 2.0 * (s ** 3)
    polished = traj.copy()
    polished[-k_tail:] = np.clip(
        polished[-k_tail:] + blend * dq_total[None, :],
        lim[:, 0][None, :],
        lim[:, 1][None, :],
    )
    polished[0] = traj[0]
    polished[-1] = q_best

    if diagnostic is not None:
        diagnostic.update(
            selected=True,
            reason="polished_endpoint_ik_residual",
            before_position_error_m=float(init_pos_err),
            after_position_error_m=float(final_pos_err),
            before_orientation_error_rad=float(init_ori_err),
            after_orientation_error_rad=float(final_ori_err),
            max_joint_delta_rad=float(np.max(np.abs(dq_total))),
        )
    return polished
