# Legacy SO101 calibration helper settings

Use the gateway calibration workflow for normal operation; it owns version binding, stop handling and current acceptance. [`scripts/so101_calibrate.py`](../../scripts/so101_calibrate.py) remains an existing diagnostic/legacy helper, not an alternate authorization or safety boundary. This configuration change does not alter its motion strategy. All development/offline commands below use `mamba run -n ni-rua python`; real capture still requires the separately configured hardware environment and explicit operator authorization.

The helper reads numeric defaults from [`scripts/so101_calibrate_defaults.json`](../../scripts/so101_calibrate_defaults.json). Resolve precedence is **packaged defaults < equivalent fields in deployment `--config` < user `--settings` JSON < explicit CLI flags**. Omitted flags no longer overwrite deployment values. `--settings` accepts partial `capture`, `capture-handheld`, `solve` and `verify` sections with the same keys as the defaults file; unknown sections/keys, wrong scalar types and invalid numeric bounds fail before hardware access. Do not insert these sections into the strict deployment configuration.

For example, a separate `legacy-settings.json` can contain:

```json
{
  "capture": {"n_poses": 24, "tilt_min_deg": 18.0, "rate_hz": 15.0},
  "capture-handheld": {"views": 30, "still_px": 1.0},
  "solve": {"max_intrinsics_rms_px": 0.8},
  "verify": {"max_reprojection_px": 3.0}
}
```

Resolve without moving or solving:

```sh
mamba run -n ni-rua python scripts/so101_calibrate.py capture \
  --config configs/so101.local.json --settings legacy-settings.json \
  --output-root /tmp/legacy-calibration --print-settings
```

`solve` now also accepts the deployment file and reads its existing `calibration` thresholds:

```sh
mamba run -n ni-rua python scripts/so101_calibrate.py solve \
  --session /path/to/capture --config configs/so101.local.json \
  --settings legacy-settings.json --max-intrinsics-rms-px 0.7 --print-settings
```

Remove `--print-settings` only when intending to run the named command. Merely printing settings performs no connection, capture, solve or apply.

| Settings fields | Meaning, units and historical defaults |
| --- | --- |
| `capture.board_yaw_deg` | Board x-axis yaw about base z, degrees; default 0; relevant when a board centre is supplied |
| `capture.n_poses`, `seed` | Positive count of generated poses (20; typically 15–25) and nonnegative integer random seed (0) |
| `capture.tilt_min_deg`, `tilt_max_deg` | View tilt from the board normal, degrees, defaults 15/35; lower bound must not exceed upper |
| `capture.distance_min_m`, `distance_max_m` | Positive camera-to-board distance range, metres, defaults 0.20/0.32 |
| `capture.joint_margin_deg` | Nonnegative inward margin from URDF joint limits, degrees, default 8 |
| `capture.z_floor_m` | Minimum base-frame z for the existing camera/link-origin screen, metres, default 0.05; null explicitly disables that screen; neither mode is full collision checking |
| `capture.max_joint_speed_rad_s` | Positive interpolation speed cap per joint, rad/s, default 0.2617993877991494 (15°/s) |
| `capture.max_command_delta_rad` | Positive largest per-command joint-goal increment, rad, default 0.03490658503988659 (2°) |
| `capture.max_relative_target_deg` | Positive LeRobot hardware-side goal-minus-present clamp, degrees, default 10 |
| `capture.rate_hz` | Positive interpolation command rate, Hz, default 20 |
| `capture.settle_tolerance_deg` | Positive measured joint **span** allowed within the old settling window, degrees, default 0.3; not a velocity or arrival tolerance |
| `capture.settle_window_s`, `settle_timeout_s` | Positive settling observation window (0.5 s) and maximum wait (5 s) |
| `capture-handheld.views`, `min_corners` | Positive target retained-view count (20) and minimum detected ChArUco corners per view (10) |
| `capture-handheld.timeout_s` | Positive automatic collection duration limit, seconds, default 240 |
| `capture-handheld.still_px` | Positive mean inter-frame corner displacement for stillness, pixels, default 1.5 |
| `capture-handheld.min_novelty` | Positive normalized centre/size/orientation descriptor separation from retained views, dimensionless, default 0.08; not physical angular accuracy |
| `solve.max_pnp_reprojection_px` | Positive wrist sample PnP RMS exclusion threshold, pixels, default 2 |
| `solve.hand_eye_score_lever_m` | Hand-eye method-ranking coefficient, m/rad, default 0.1, range `[0,10]`; score = translation RMS + coefficient × rotation RMS. Zero selects by translation only; this numerical tradeoff is not a measured lever arm or an acceptance gate. Example: 0.2 doubles the angular contribution |
| `solve.max_intrinsics_rms_px` | Positive intrinsics RMS acceptance gate, pixels, default 1 |
| `solve.max_hand_eye_translation_m` | Positive hand-eye translation RMS spread gate, metres, default 0.005 |
| `solve.max_hand_eye_rotation_rad` | Positive hand-eye rotation RMS spread gate, radians, default 0.02 |
| `solve.max_fixed_reprojection_px` | Positive fixed-camera chain reprojection gate, pixels; default null leaves this legacy gate unset |
| `verify.max_reprojection_px` | Positive worst reported summary RMS gate, pixels; default null reports without enforcing a new gate |

Equivalent deployment fields are `motion.max_joint_speed_rad_s`, `motion.max_command_delta_rad`, `motion.command_rate_hz → rate_hz`, top-level `max_relative_target_deg`, `calibration.settle_timeout_s`, `calibration.z_floor_m`, `calibration.hand_eye_score_lever_m`, and all five named `solve` thresholds. Handheld mappings are `handheld_target_views → views`, `handheld_timeout_s → timeout_s`, `min_board_corners → min_corners`, `handheld_still_px → still_px`, and `handheld_min_novelty → min_novelty`, all under `calibration`. The existing top-level `so101` deployment wrapper is supported. No conversion is invented between the legacy joint-span settle criterion and the gateway's velocity/arrival criteria.

Actual values and their winning sources are retained in capture `session.json` as `legacy_effective_settings`, in each handheld camera directory's `legacy_effective_settings.json`, in `legacy_solve_settings.json` even if solve gates fail, and in a saved verification result. These records describe configured execution, not physical success. Existing recordings and unrelated local configuration remain untouched.

CPU regression: `PYTHONPATH=tests:. mamba run -n ni-rua python -m unittest tests.test_so101_legacy_settings`. Tests check original defaults/units, all precedence layers, wrapped deployment input, null floor, solve threshold forwarding/persistence and rejection before hardware access.
