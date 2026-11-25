# SO101 calibration and evidence configuration

The calibration chain has three configuration owners. Each has an explicit file and validated override path; none writes an unrelated owner's settings. Calibration outputs (joint mapping, camera model and fitted table) are versioned measurements/estimates, not copied into perception tuning numbers. This guide covers SO101 calibration and its SAM/VGGT evidence tools, not every subsystem in NI-RUA.

| Owner | Editable file | Contents and effective record |
| --- | --- | --- |
| Robot deployment | `configs/so101.local.json`; copy [example](../../configs/so101.example.json) for another machine | `calibration` controls/gates, motion, servo, camera and workspace. `deployment.FIELDS` validates units/ranges. Workflow/capture/solver records preserve effective values; UI thresholds come from this contract. |
| Gateway and online perception | The `--config` JSON, on this machine `~/.cache/ni-rua/so101/ni-rua.json` | Model/tier, endpoints, remote process configuration, `features` for SAM/VGGT point/region consumers. See [general configuration](configuration.md). |
| Offline evidence | Copy [so101-evidence.example.json](../../configs/so101-evidence.example.json) to a review-owned file; pass `--config FILE` | SAM video, LK/video alignment, VGGT diagnostics and contact trajectory fitting. Each result records its effective settings. Feature seeds/ROI/contact annotations stay in separate review files. |

The machine's board file (`~/.cache/ni-rua/so101/board.json`) provides dimensions. An explicitly supplied board at workflow begin or S2 takes precedence. The current run retains that board across page refresh; opening S2 must not silently substitute deployment defaults. This machine's confirmed printed-face height is **0.0032 m**. The generic example's **0.003 m** is an example, not a measurement of the user's board.

## Precedence and reproducibility

Deployment fields that were required remain required. Added optional controls obtain backward-compatible values from [calibration_defaults.json](../../ni_rua/robot/backends/so101/calibration_defaults.json); an explicit deployment value wins. Standalone capture helpers use that same data file. The values have not been widened or tuned to pass this run. The touch CLI applies explicit flags and then revalidates the complete deployment before constructing any adapter. Unknown fields and invalid ranges are errors.

A workflow records its effective settings. Changing the deployment during a run must not silently combine old captures with new solve/acceptance settings. Source checks cover settings alongside mapping, board and calibration inputs. Historical runs without a settings snapshot retain that evidence gap rather than being backfilled as though they recorded current defaults. Existing valid applied calibration and old archives are retained.

Offline precedence is **packaged JSON defaults < `--config` section < explicit CLI flag**. Contact fitting uses **packaged defaults < `--config.contact_fit` < review `fit_settings`**. An omitted CLI argument never replaces the file's setting with an argparse default. `--no-use-fa3` explicitly overrides a true JSON value. Unknown section names, misspelled fields, wrong scalar types and nonfinite numbers are rejected. Keys starting `_` are comments. Physical/algorithm-specific bounds are checked again by the consumer before analysis/model work.

The packaged [evidence defaults](../../ni_rua/robot/calibration/evidence/defaults.json) and annotated example are consistency-tested. The package includes both calibration and evidence default resources, and its isolated wheel check imports them outside the checkout.

The existing standalone helper has its own [legacy settings guide](so101-legacy-calibration-settings.md). It now reads file-backed defaults and supports `--settings`, `solve --config` and a no-hardware `--print-settings` inspection; the gateway remains the normal physical workflow.

## VGGT support-height source

```json
{"features": {"vggt_camera_id": "fixed", "vggt_support_z_m": null}}
```

`null` means use `joint_calibration.table_z_m` from **the requested frozen observation's camera calibration**, in that observation's reference frame, in metres. A fresh observation after apply uses the new table estimate; replaying an old observation uses its own estimate. Missing or invalid height makes the request unavailable; no plane is guessed. `object_table`, `scene3d`, `vggt_height` and held-object wrist requests share this resolver.

Every explicit finite number, including `-0.02`, `0` and `0.765`, means a fixed support height. Apply/rollback never overwrites it. This removes the old accidental rule that numeric `0.765` meant auto-follow. Existing configurations relying on that implicit rule must explicitly choose `null`; a numeric `0.765` remains literal. The default `"profile"` (also in `configs/ni-rua.json`) takes the embodiment's support height: SO-101 has none in its profile, so it behaves like `null` (the observation's calibrated table height), while ARX-X5 resolves to its table top `0.765 m`. The CLI equivalent is `--vggt-support-z-m observation`; queue JSON `null` is passed as that token, not the string `None`. This is a depth-scale anchor, not an independent table measurement.

## Newly centralized control parameters

All are under deployment `calibration`; full definitions and bounds are in `deployment.FIELDS` and the example JSON. They preserve previous numerical behavior unless explicitly changed.

| Field | Meaning, unit and default |
| --- | --- |
| `touch_orientation_weight` | IK orientation cost relative to translation, m/rad; 0 preserves position-only touch sampling. It is not a contact detector. |
| `capture_max_drift_rad` | Maximum settled joint drift during S2/S4 capture, rad. `null` preserves the historical `2 × arrival_tolerance_rad` rule; capture records the resolved numerical bound. |
| `z_floor_m` | Optional sparse FK link-origin screen during S2/S4, SO101-base metres; `null` disables it. It is not mesh collision checking or measured tabletop clearance. |
| `handheld_near_fraction`, `handheld_far_fraction` | Projected full-board extent/image-size coverage buckets, default 0.55/0.30. They describe coverage, not solve acceptance. |
| `handheld_tilt_small`, `handheld_tilt_large` | Opposite projected-edge ratios, default 1.06/1.15; not angles. |
| `handheld_still_frames` | Consecutive detections for stillness, default 2 frames. Duration depends on delivered frames. |
| `handheld_min_sharpness` | Full grayscale image Laplacian variance; default 0 disables the gate. A positive optics-dependent value requires an actual measurement and rejects missing values. |
| `hand_eye_score_lever_m` | Translation-equivalent weight of rotational spread in hand-eye candidate ranking, default 0.1 m/rad; not a measured camera distance. |
| `joint_solver_max_nfev` | S5 least-squares function-evaluation budget, default 400 calls. |
| `joint_pixel_robust_scale` | Existing robust pixel-residual transition after noise normalization, default 3 dimensionless units. |
| `joint_solver_ftol`, `joint_solver_xtol`, `joint_solver_gtol` | Numerical termination tolerances, default 1e-8 each; not physical precision limits. |
| `joint_solver_x_scale` | Optimizer parameter scale, default 1.0; preserves the existing solve. |
| `operator.default_speed_scale_cap` | Initial runtime joint-speed multiplier in `(0, 1]`, default `0.9` (`>= min_speed_scale_cap`); reported in `robot_capabilities` and `effective_limits`, adjustable live via `set_runtime_limits`. |

Existing contact/sliding distance, rate, duration, lag/raw-load limits, video budgets, capture/solver timeouts, fit weights, pose grids and S6's **0.005 m** gate remain in the same deployment section. Raw load units are not newtons. The configuration does not turn a displacement-bounded motion into a force controller.

## Offline settings and examples

All four sections can share one JSON file. Model weights/checkpoints, source image paths, seeded features, support annotations and fit/holdout roles are explicit invocation/review inputs; they are not guessed from settings.

| Section / fields | Meaning, unit, default and application |
| --- | --- |
| `sam31_video.max_num_objects` | Positive mask capacity, 16 objects. |
| `sam31_video.use_fa3` | FlashAttention 3 enablement, false; requires compatible GPU environment. |
| `sam31_video.output_prob_threshold` | Official model output threshold, 0.5 in [0,1]; the instance-point branch retains its official tracker rule. |
| `video_evidence.max_frame_gap_s`, `max_joint_gap_s` | Maximum consecutive frame interval / measured-joint bracket, 0.25 s each. Gaps stay unknown; no extrapolation. |
| `video_evidence.max_forward_backward_error_px` | LK round-trip discrepancy limit, 1 px; not metric precision. |
| `video_evidence.window_size_px`, `pyramid_levels` | Odd LK window diameter 21 px (3–101), pyramid depth 3 (0–6). |
| `video_evidence.min_eigenvalue` | LK conditioning threshold, 1e-4, positive dimensionless. |
| `vggt_evidence.min_confidence_rank` | Image-relative retained-point confidence percentile, 0.1 in [0,1]; not probability. |
| `vggt_evidence.max_landmark_pixel_distance_px` | Match radius to a retained service pixel, 3 px; no hidden interpolation. |
| `vggt_evidence.rank_rtol` | Relative numerical geometry-rank cutoff, 1e-6 in (0,1). |
| `contact_fit.initial_plane` | `[height_m, slope_x, slope_y]` in SO101 base, default `[0,0,0]`; slopes dimensionless. |
| `contact_fit.estimated_joints`, `estimated_plane`, `estimated_contact_axes` | Selected parameter names: defaults none, all three plane parameters, none. Fix gauges and externally measured quantities explicitly. |
| `contact_fit.max_joint_delta_rad` | Joint search bound, 0.1 rad. |
| `contact_fit.max_plane_height_delta_m`, `max_slope_delta`, `max_contact_delta_m` | Search bounds: 0.03 m, 0.1 dimensionless, 0.005 m. These are offline optimization bounds, never motor limits. |
| `contact_fit.sample_sigma_m`, `robust_scale_sigma` | Nominal normal-gap scale 0.001 m and robust transition 2 times that scale. Not force or measured accuracy. |
| `contact_fit.rank_rtol`, `rank_difference_step`, `bound_tolerance_ratio` | Numerical rank cutoff 1e-6, finite-difference step 1e-5 in bound-normalized parameters, and bound-saturation fraction 1e-6; ratios in (0,1). |
| `contact_fit.solver_ftol`, `solver_xtol`, `solver_gtol` | Numerical termination tolerances, 1e-11 each; not physical acceptance. |

```sh
mamba run -n ni-rua python -m ni_rua.robot.calibration.evidence.video_evidence \
  --config configs/so101-evidence.example.json \
  --video-manifest CAPTURE/manifest.json --camera fixed --tracking SAM/tracking.json \
  --features feature-seeds.json --output NEW-VIDEO-REVIEW

mamba run -n ni-rua python -m ni_rua.robot.calibration.evidence.vggt_evidence \
  --config configs/so101-evidence.example.json \
  --request request.json --response response.json --review roi-review.json --output NEW-DIAGNOSTICS.json

mamba run -n ni-rua python -m ni_rua.robot.calibration.evidence.contact_trajectory \
  --config configs/so101-evidence.example.json \
  --capture S1_touches.json --review contact-review.json --output NEW-FIT.json
```

SAM inference uses its configured isolated GPU interpreter. Run `-m services.vggt_height.sam31.video_tracking --config FILE ...` from the matching checkout. For standalone deployment, copy `video_tracking.py` **and** the pure-standard-library `so101_evidence_config.py` plus `so101_evidence_defaults.json` from `ni_rua/robot/model/` into one directory. There is no second embedded defaults table. Configuration parsing does not load SAM/PyTorch. This change does not automatically synchronize or restart an existing remote service.

Mathematical identities, units conversions, fixed algorithm definitions and immutable sensor/CAD model data remain code/model definitions. They are not hidden experiment tuning. Changing an algorithm, contact-feature assumption or physical measurement still requires evidence; moving a number to JSON alone cannot validate it.
