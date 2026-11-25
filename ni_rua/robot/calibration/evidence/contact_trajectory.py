"""Offline, conditional plane/contact calibration from independently supported trajectories.

No hardware access and no automatic calibration apply. Each row constrains one identified material
point on a rigid gripper link to a plane. Encoder lag and a CAD minimum alone do not establish that
constraint. Unknown/released/deforming intervals must remain outside the supported segments.
Hand-eye, board pose and compliance are fixed outside this solve. See the sliding-calibration report.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from .config import evidence_settings

DEFAULTS = evidence_settings("contact_fit")
from scipy.optimize import least_squares

from ...embodiments.so101.profile import ARM_JOINTS


@dataclass(frozen=True)
class ContactSegment:
    """One uninterrupted, externally supported contact feature, in one joint coordinate basis.

    segment_id: nonempty acquisition identifier; split at contact loss/feature changes (no default).
    basis_id: nonempty identifier of the exact measured-joint mapping, not an offset to add (no default).
    q_rad: N x 5 measured joint angles, rad in ARM_JOINTS order; N >= 1, never commanded joints.
    jaw_rad: N measured jaw angles, rad; required even for a fixed-jaw feature.
    elapsed_s: N strictly increasing measurement times, s from that segment's clock origin.
    link: URDF link containing the identified contact feature (no default).
    point_in_link_m: fixed 3-vector material point, m in link frame; measured/identified by operator.
    evidence: ``supported`` only when independent observations support contact at this fixed feature;
        ``unknown`` (default) is excluded. Neither label is inferred from lag/load/CAD by this module.
    evidence_note: source and limitations of independent support; nonempty for supported segments.
    role: ``fit`` (default) or ``holdout``; holdout rows never enter optimization or rank calculation.

    A segment's total weight is one, independent of sample rate; rows within it are correlated.
    Example: ContactSegment("line-x", "mapping-A", q, jaw, t, "gripper_link", [0,0,0.02],
                            evidence="supported", evidence_note="contact marker seen in side camera").
    """
    segment_id: str
    basis_id: str
    q_rad: Sequence[Sequence[float]]
    jaw_rad: Sequence[float]
    elapsed_s: Sequence[float]
    link: str
    point_in_link_m: Sequence[float]
    evidence: str = "unknown"
    evidence_note: str = ""
    role: str = "fit"

    @classmethod
    def from_trace(cls, trace: Mapping, *, segment_id: str, basis_id: str, link: str,
                   point_in_link_m: Sequence[float], sample_indices: Sequence[int],
                   evidence: str = "unknown", evidence_note: str = "", role: str = "fit") -> "ContactSegment":
        """Select a reviewed interval from a touch trace, without promoting its contact evidence.

        trace: archived mapping with ``reads`` of q_measured_rad, gripper_rad and elapsed_s (rad/rad/s).
        segment_id/basis_id/link/point_in_link_m/evidence/evidence_note/role: as ContactSegment above.
        sample_indices: required increasing, unique zero-based read indices, e.g. range(3, 12); no
            automatic candidate selection. Contact switches or releases must split the interval.
        Returns an unmodified measured-joint segment; missing fields/invalid indices raise ValueError.
        """
        indices = list(sample_indices)
        if not indices or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 0 for i in indices):
            raise ValueError("sample_indices must be nonempty nonnegative integers")
        if indices != sorted(set(indices)):
            raise ValueError("sample_indices must be increasing and unique")
        try:
            rows = [trace["reads"][i] for i in indices]
            return cls(segment_id, basis_id, [r["q_measured_rad"] for r in rows],
                       [r["gripper_rad"] for r in rows], [r["elapsed_s"] for r in rows],
                       link, point_in_link_m, evidence, evidence_note, role)
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("trace lacks selected measured joints, jaw or times") from error

    @classmethod
    def from_capture(cls, capture: Mapping, **review) -> "ContactSegment":
        """Convert one raw ``contact_trajectories`` entry into a separate reviewed segment.

        capture: mapping containing ``samples`` with measured joints (rad), jaw (rad), elapsed_s (s).
        review: explicit keyword arguments of :meth:`from_trace`, including sample_indices, basis_id,
            contact link/point and segment_id. Default evidence remains unknown, even if the raw
            capture claims contact. No historical record is mutated or automatically relabelled.
        Returns ContactSegment; example: ``ContactSegment.from_capture(raw, segment_id="review-1",
        basis_id="mapping-A", link="gripper_link", point_in_link_m=[0,0,.02], sample_indices=[3,4])``.
        """
        return cls.from_trace({"reads": capture.get("samples", [])}, **review)


def _validated(segment: ContactSegment) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if any(not isinstance(value, str) or not value for value in (segment.segment_id, segment.basis_id, segment.link)):
        raise ValueError("segment_id, basis_id and link are required")
    if segment.evidence not in ("supported", "unknown") or segment.role not in ("fit", "holdout"):
        raise ValueError("invalid evidence or role")
    if not isinstance(segment.evidence_note, str) or (segment.evidence == "supported" and not segment.evidence_note.strip()):
        raise ValueError("supported contact requires an independent evidence_note")
    q, jaw, times = (np.asarray(v, dtype=float) for v in (segment.q_rad, segment.jaw_rad, segment.elapsed_s))
    point = np.asarray(segment.point_in_link_m, dtype=float)
    if q.ndim != 2 or q.shape[1:] != (5,) or len(q) == 0 or jaw.shape != (len(q),) or times.shape != (len(q),):
        raise ValueError("segment requires N x 5 joints and N jaw/time values")
    if point.shape != (3,) or not all(np.isfinite(v).all() for v in (q, jaw, times, point)):
        raise ValueError("segment values must be finite, point must have three coordinates")
    if times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError("measurement elapsed_s must be nonnegative and strictly increasing")
    return q, jaw, point


def fit_contact_trajectories(segments: Sequence[ContactSegment],
                             fk_all: Callable[[np.ndarray, float], Mapping[str, np.ndarray]], *,
                             initial_plane: Sequence[float] = tuple(DEFAULTS["initial_plane"]),
                             estimated_joints: Sequence[str] = tuple(DEFAULTS["estimated_joints"]),
                             estimated_plane: Sequence[str] = tuple(DEFAULTS["estimated_plane"]),
                             estimated_contact_axes: Sequence[str] = tuple(DEFAULTS["estimated_contact_axes"]),
                             max_joint_delta_rad: float = DEFAULTS["max_joint_delta_rad"],
                             max_plane_height_delta_m: float = DEFAULTS["max_plane_height_delta_m"],
                             max_slope_delta: float = DEFAULTS["max_slope_delta"],
                             max_contact_delta_m: float = DEFAULTS["max_contact_delta_m"],
                             sample_sigma_m: float = DEFAULTS["sample_sigma_m"],
                             robust_scale_sigma: float = DEFAULTS["robust_scale_sigma"],
                             rank_rtol: float = DEFAULTS["rank_rtol"],
                             solver_ftol: float = DEFAULTS["solver_ftol"],
                             solver_xtol: float = DEFAULTS["solver_xtol"],
                             solver_gtol: float = DEFAULTS["solver_gtol"],
                             rank_difference_step: float = DEFAULTS["rank_difference_step"],
                             bound_tolerance_ratio: float = DEFAULTS["bound_tolerance_ratio"]) -> dict:
    """Fit selected joint/plane/contact parameters, rejecting data-unobservable solutions.

    All parameters are explicit; none changes a controller, active mapping or acceptance threshold.
    segments: reviewed ContactSegments; all use the same basis and same fixed link/material point.
        Unknown intervals are retained in exclusion diagnostics. Fit and holdout IDs must differ.
    fk_all: pure ``(q_rad[5], jaw_rad) -> {link: T_base_link[4,4]}``, in rad and m; example wrapper:
        ``lambda q, jaw: kin.fk_all(model, q, gripper_rad=jaw)``.
    initial_plane: [height_m, slope_x, slope_y] for z = height + slope_x*x + slope_y*y in so101_base;
        default horizontal z=0. Slopes are dimensionless, typically abs < 0.1 for a near-level table.
    estimated_joints: selected ARM_JOINTS names; default none (zeros independently fixed). Deltas add
        exactly once to supplied measured q. Select only an independently identifiable subset.
    estimated_plane: subset of height/slope_x/slope_y; default all. Empty fixes the supplied plane.
    estimated_contact_axes: subset of x/y/z in link frame; default none (geometry independently fixed).
        Shared correction of the material point across segments; use a measured fixed plane/zeros
        for this stage. Never represents changing mesh minima, rolling or compliant deflection.
    max_joint_delta_rad: symmetric offset search bound and Jacobian scale, rad, default 0.1 (~6 deg).
    max_plane_height_delta_m: symmetric height change bound/scale, m, default 0.03.
    max_slope_delta: symmetric slope change bound/scale, dimensionless, default 0.1 (~6 deg).
    max_contact_delta_m: symmetric contact-point correction bound/scale, m, default 0.005.
    sample_sigma_m: positive nominal normal-gap noise scale, m, default 0.001; not force precision.
    robust_scale_sigma: positive soft-L1 transition in sample_sigma_m multiples, default 2; applied
        before segment normalization so repeating a segment's samples does not increase its weight.
    rank_rtol: positive relative singular-value cutoff in (0,1), default 1e-6. Data-only finite-
        difference Jacobian uses the explicit search-bound scales; no priors repair missing rank.

    solver_ftol/solver_xtol/solver_gtol: positive dimensionless SciPy termination tolerances,
        default 1e-11 each; numerical convergence, not physical acceptance.
    rank_difference_step: finite-difference step in declared bound-normalized parameters,
        default 1e-5, in (0,1); used only for the data-rank diagnostic.
    bound_tolerance_ratio: fraction of each search bound treated as saturated, default 1e-6
        in (0,1); candidates this close to a search bound remain inapplicable.

    Returns JSON-safe diagnostics, per-segment residuals (m), scaled singular values and null vectors.
    ``candidate`` is None for missing support, rank deficiency, failed optimizer or active bounds;
    otherwise it is a CONDITIONAL estimate, never physical acceptance. Holdout RMS/max remain distinct
    from training residuals; same-sensor holdout is not external validation. No parameter uncertainty
    or contact force is inferred from encoders. Example: fit_contact_trajectories(segments, fk,
    initial_plane=(-0.01,0,0), estimated_joints=("elbow_flex",), estimated_plane=("height",)).
    """
    effective_settings = {key: value for key, value in locals().items() if key in DEFAULTS}
    for key, value in dict(solver_ftol=solver_ftol, solver_xtol=solver_xtol, solver_gtol=solver_gtol,
                           rank_difference_step=rank_difference_step, bound_tolerance_ratio=bound_tolerance_ratio).items():
        if type(value) not in (float, int) or not np.isfinite(value) or not np.finfo(float).eps < value < 1:
            raise ValueError(f"{key} must be a finite ratio between machine epsilon and 1")
    segments = list(segments)
    arrays = [_validated(s) for s in segments]
    if len({s.segment_id for s in segments}) != len(segments):
        raise ValueError("segment IDs must be unique (including holdouts)")
    if len({s.basis_id for s in segments}) > 1:
        raise ValueError("mixed joint bases require explicit rebasing before fitting")
    if arrays and any(s.link != segments[0].link or not np.array_equal(a[2], arrays[0][2])
                      for s, a in zip(segments, arrays)):
        raise ValueError("one fit requires a single identified link/material point; split feature switches")
    plane0 = np.asarray(initial_plane, dtype=float)
    if plane0.shape != (3,) or not np.isfinite(plane0).all():
        raise ValueError("initial_plane must contain three finite values")
    names, scales, locations = [], [], []
    groups = ((estimated_joints, ARM_JOINTS, "joint", max_joint_delta_rad),
              (estimated_plane, ("height", "slope_x", "slope_y"), "plane", None),
              (estimated_contact_axes, ("x", "y", "z"), "contact", max_contact_delta_m))
    for selected, allowed, group, scale in groups:
        if len(set(selected)) != len(selected) or any(name not in allowed for name in selected):
            raise ValueError(f"invalid or duplicate {group} parameters")
        for name in selected:
            names.append(f"{group}.{name}")
            locations.append((group, allowed.index(name)))
            scales.append((max_plane_height_delta_m if name == "height" else max_slope_delta)
                          if group == "plane" else scale)
    positive = (max_joint_delta_rad, max_plane_height_delta_m, max_slope_delta, max_contact_delta_m,
                sample_sigma_m, robust_scale_sigma)
    if any(not np.isfinite(v) or v <= 0 for v in positive) or not 0 < rank_rtol < 1:
        raise ValueError("bounds/noise/robust scale must be finite positive; rank_rtol must be in (0,1)")
    scales = np.asarray(scales)
    fit_indices = [i for i, s in enumerate(segments) if s.evidence == "supported" and s.role == "fit"]
    result = {"schema": "ni_rua.so101.contact_trajectory_fit.v1", "status": "insufficient_evidence",
              "candidate": None, "parameter_names": names, "parameter_scales": scales.tolist(),
              "basis_id": segments[0].basis_id if segments else None,
              "excluded_segments": [{"segment_id": s.segment_id, "reason": "contact_not_independently_supported"}
                                    for s in segments if s.evidence != "supported"],
              "fit_segments": [segments[i].segment_id for i in fit_indices],
              "holdout_segments": [s.segment_id for s in segments if s.evidence == "supported" and s.role == "holdout"],
              "settings": dict(effective_settings, weighting="equal_total_per_segment"),
              "uncertainty_m": None, "physical_acceptance": None, "segments": []}
    if not fit_indices:
        return result

    def unpack(x):
        offsets, plane, point_delta = np.zeros(5), plane0.copy(), np.zeros(3)
        target = {"joint": offsets, "plane": plane, "contact": point_delta}
        for value, (group, at) in zip(x * scales, locations):
            target[group][at] += value
        return offsets, plane, point_delta

    def gaps(x, index):
        offsets, plane, point_delta = unpack(x)
        s = segments[index]
        q, jaw, point = arrays[index]
        world = []
        for joints, aperture in zip(q, jaw):
            T = np.asarray(fk_all(joints + offsets, float(aperture))[s.link], dtype=float)
            if T.shape != (4, 4) or not np.isfinite(T).all():
                raise ValueError("FK must return finite 4 x 4 transforms")
            world.append(T[:3, :3] @ (point + point_delta) + T[:3, 3])
        world = np.asarray(world)
        normal = np.array([-plane[1], -plane[2], 1.0])
        return (world @ normal - plane[0]) / np.linalg.norm(normal), world

    def residual(x, robust=False):
        rows = []
        for i in fit_indices:
            r = gaps(x, i)[0] / sample_sigma_m
            if robust:
                r = r * np.sqrt(2.0 / (np.sqrt(1.0 + (r / robust_scale_sigma) ** 2) + 1.0))
            rows.append(r / np.sqrt(len(r)))
        return np.concatenate(rows)

    x = np.zeros(len(names))
    if names:
        fit = least_squares(lambda values: residual(values, True), x, bounds=(-1.0, 1.0),
                            xtol=solver_xtol, ftol=solver_ftol, gtol=solver_gtol)
        x, success, message = fit.x, bool(fit.success), str(fit.message)
        # Numerical differentiation in dimensionless, declared bound scales; not the robust-loss
        # Jacobian or regularized Hessian, which can conceal geometric gauge freedoms.
        step = rank_difference_step
        J = np.column_stack([(residual(x + np.eye(len(names))[j] * step) -
                              residual(x - np.eye(len(names))[j] * step)) / (2 * step)
                             for j in range(len(names))])
        # Need all parameter null vectors when rows < parameters, but never allocate an N x N
        # left-singular matrix for a long high-rate recording.
        _, singular, Vt = np.linalg.svd(J, full_matrices=J.shape[0] < J.shape[1])
        cutoff = singular[0] * rank_rtol if len(singular) and singular[0] > 0 else 0.0
        rank = int(np.sum(singular > cutoff))
        null = Vt[rank:].tolist()
        bound_parameters = [names[i] for i in range(len(names)) if abs(x[i]) >= 1.0 - bound_tolerance_ratio]
    else:
        success, message, singular, rank, null, bound_parameters = True, "fixed-parameter diagnostic", np.array([]), 0, [], []
    offsets, plane, point_delta = unpack(x)
    result.update(optimizer_success=success, optimizer_message=message, data_rank=rank,
                  parameter_count=len(names), scaled_singular_values=singular.tolist(),
                  scaled_null_vectors=null, bound_parameters=bound_parameters)
    result["status"] = ("optimizer_failed" if not success else "unidentifiable" if rank < len(names)
                        else "bound_limited" if bound_parameters else "conditional_fit")
    for i, s in enumerate(segments):
        if s.evidence != "supported":
            continue
        r, world = gaps(x, i)
        result["segments"].append({"segment_id": s.segment_id, "role": s.role, "samples": len(r),
            "normal_residuals_m": r.tolist(), "rms_m": float(np.sqrt(np.mean(r * r))),
            "max_abs_m": float(np.max(np.abs(r))), "mean_m": float(np.mean(r)),
            "point_span_base_m": np.ptp(world, axis=0).tolist(), "evidence_note": s.evidence_note})
    if result["status"] == "conditional_fit":
        normal = np.array([-plane[1], -plane[2], 1.0]); normal /= np.linalg.norm(normal)
        result["candidate"] = {"joint_delta_rad": dict(zip(ARM_JOINTS, offsets.tolist())),
                               "plane": {"height_at_base_origin_m": float(plane[0]),
                                         "slope_x": float(plane[1]), "slope_y": float(plane[2]),
                                         "normal": normal.tolist()},
                               "contact_point_delta_m": point_delta.tolist(),
                               "contact_point_in_link_m": (arrays[0][2] + point_delta).tolist()}
    return result


def _capture_mapping(capture: Mapping) -> dict:
    """Read a complete recorded five-joint mapping; never default missing historical offsets."""
    from ..workflow.charuco import normalize_joint_mapping
    mapping = capture.get("joint_mapping")
    if not isinstance(mapping, dict) or any(
            not isinstance(mapping.get(name), dict) or
            not {"sign", "offset_rad"}.issubset(mapping[name]) for name in ARM_JOINTS):
        raise ValueError("capture requires complete recorded joint_mapping sign/offset_rad for every arm joint")
    normalized = normalize_joint_mapping(mapping)
    if any(not np.isfinite(row["offset_rad"]) for row in normalized.values()):
        raise ValueError("recorded joint_mapping offsets must be finite rad")
    return normalized


def reviewed_capture_fit(capture_path, review_path, output_path, *, config_path=None) -> dict:
    """Run the operator-only offline workflow; create a new result file without hardware access.

    capture_path: existing raw touch report JSON path, containing contact_trajectories[].samples and
        the exact joint_mapping for each acquisition (sign dimensionless, offset_rad in rad).
    review_path: separate JSON path with schema ``ni_rua.so101.contact_trajectory_review.v1``, exact
        ``capture_sha256``, ``segments`` annotations and optional ``fit_settings`` (defaults of
        fit_contact_trajectories). Each annotation requires segment_id, trajectory_id, sample_indices,
        link, point_in_link_m (m), evidence (supported/unknown), evidence_note and role (fit/holdout).
        IDs are nonempty strings, indices are increasing zero-based integers; field semantics are
        ContactSegment's. basis_id is forbidden: it is derived from the recorded mapping. An entire
        source trajectory belongs to one role; sample overlap across annotations is rejected.
    output_path: a new, separate JSON path; exclusive creation refuses overwrite of any file,
        including capture/review. Parent directory must exist. Result includes input-byte bindings
        and effective fitter settings for operator audit only; these are not actor-context fields.
    Returns the fit diagnostics. Unknown support produces insufficient_evidence, not an exception or
        applicable candidate. Invalid binding/schema/mapping raises ValueError; existing output
        raises FileExistsError. config_path: optional shared evidence JSON path (default packaged settings);
        review fit_settings takes precedence, in the units above. Example: reviewed_capture_fit("S1_touches.json", "review.json", "fit.json").
    """
    import hashlib
    import inspect
    import json
    from pathlib import Path
    from ...embodiments.so101 import profile as kin

    capture_path, review_path, output_path = map(Path, (capture_path, review_path, output_path))
    if output_path.resolve() in (capture_path.resolve(), review_path.resolve()):
        raise ValueError("output must be separate from immutable capture and review inputs")
    raw, review_raw = capture_path.read_bytes(), review_path.read_bytes()
    report, review = json.loads(raw), json.loads(review_raw)
    capture_digest = hashlib.sha256(raw).hexdigest()
    if not isinstance(review, dict) or review.get("schema") != "ni_rua.so101.contact_trajectory_review.v1":
        raise ValueError("unsupported contact trajectory review schema")
    if review.get("capture_sha256") != capture_digest:
        raise ValueError("review capture binding does not match exact input bytes")
    captures = report.get("contact_trajectories") if isinstance(report, dict) else None
    annotations = review.get("segments")
    if not isinstance(captures, list) or not isinstance(annotations, list) or not annotations:
        raise ValueError("capture requires contact_trajectories; review requires nonempty segments")
    by_id = {}
    for capture in captures:
        identity = capture.get("trajectory_id") if isinstance(capture, dict) else None
        if not isinstance(identity, str) or not identity or identity in by_id:
            raise ValueError("capture trajectory IDs must be unique nonempty strings")
        by_id[identity] = capture
    required = {"segment_id", "trajectory_id", "sample_indices", "link", "point_in_link_m",
                "evidence", "evidence_note", "role"}
    segments, used_indices, source_roles, mappings = [], {}, {}, {}
    for annotation in annotations:
        if not isinstance(annotation, dict) or set(annotation) != required:
            raise ValueError("review segment must contain exactly the documented fields; basis_id is derived")
        trajectory_id = annotation["trajectory_id"]
        if not isinstance(trajectory_id, str) or trajectory_id not in by_id:
            raise ValueError("review trajectory_id does not exist in capture")
        capture = by_id[trajectory_id]
        mapping = _capture_mapping(capture)
        canonical = json.dumps(mapping, sort_keys=True, separators=(",", ":"), allow_nan=False)
        # This binding lives only in operator artifacts; it is never a model-visible object ID.
        basis_id = "recorded-mapping-" + hashlib.sha256(canonical.encode()).hexdigest()
        mappings[basis_id] = mapping
        fields = {key: value for key, value in annotation.items() if key != "trajectory_id"}
        segment = ContactSegment.from_capture(capture, basis_id=basis_id, **fields)
        _validated(segment)
        indices = set(annotation["sample_indices"])
        if indices & used_indices.setdefault(trajectory_id, set()):
            raise ValueError("review segments cannot reuse samples from the same acquisition")
        used_indices[trajectory_id].update(indices)
        role = source_roles.setdefault(trajectory_id, segment.role)
        if role != segment.role:
            raise ValueError("whole acquisitions must be reserved for fit or holdout, not split between roles")
        segments.append(segment)
    signature = inspect.signature(fit_contact_trajectories)
    defaults = {name: parameter.default for name, parameter in signature.parameters.items()
                if parameter.kind == inspect.Parameter.KEYWORD_ONLY}
    settings = review.get("fit_settings", {})
    if not isinstance(settings, dict) or set(settings) - set(defaults):
        raise ValueError("unknown fit_settings; use fit_contact_trajectories keyword parameters")
    effective = evidence_settings("contact_fit", config_path, settings)
    model = kin.load_model()
    result = fit_contact_trajectories(segments, lambda q, jaw: kin.fk_all(model, q, gripper_rad=jaw), **effective)
    result["operator_input_binding"] = {
        "capture_path": str(capture_path.resolve()), "capture_sha256": capture_digest,
        "review_path": str(review_path.resolve()), "review_sha256": hashlib.sha256(review_raw).hexdigest(),
        "recorded_joint_mappings": mappings, "review_segments": annotations,
        "effective_fit_settings": effective,
    }
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with output_path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return result


def main(argv=None) -> int:
    """CLI: inspect a raw report binding or fit a separate review; never connects to a robot.

    argv: optional CLI strings (default process arguments). ``--capture`` is required; ``--describe``
        prints the exact byte binding and raw trajectory counts without fitting. Otherwise require
        ``--review`` and a new ``--output`` JSON path. Exit 0 = conditional numerical candidate,
        2 = no candidate or invalid inputs. Neither exit code is physical calibration acceptance.
    """
    import argparse
    import hashlib
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Offline SO101 reviewed contact-trajectory fit (no hardware, no apply)")
    parser.add_argument("--capture", required=True, help="raw touch report JSON with recorded joint mappings")
    parser.add_argument("--review", help="separate evidence/feature/interval review JSON bound to capture bytes")
    parser.add_argument("--output", help="new operator result JSON; existing paths are never overwritten")
    parser.add_argument("--describe", action="store_true", help="show capture binding and trajectory inventory only")
    parser.add_argument("--config", help="shared offline evidence JSON; review fit_settings overrides its contact_fit section")
    args = parser.parse_args(argv)
    try:
        if args.describe:
            if args.review or args.output:
                raise ValueError("--describe cannot be combined with --review/--output")
            raw = Path(args.capture).read_bytes()
            captures = json.loads(raw).get("contact_trajectories", [])
            print(json.dumps({"capture_sha256": hashlib.sha256(raw).hexdigest(),
                              "trajectories": [{"trajectory_id": c["trajectory_id"],
                                                "samples": len(c.get("samples", [])),
                                                "joint_mapping": _capture_mapping(c)} for c in captures]}, indent=2))
            return 0
        if not args.review or not args.output:
            raise ValueError("fitting requires both --review and a new --output")
        result = reviewed_capture_fit(args.capture, args.review, args.output, config_path=args.config)
        print(json.dumps({"status": result["status"], "candidate_available": result["candidate"] is not None,
                          "physical_acceptance": None, "output": args.output}))
        return 0 if result["candidate"] is not None else 2
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps({"status": "invalid_input", "reason": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
