# SO-101 Perception Bench Protocol

`scripts/so101_perception_bench.py` measures what the model gets from the fixed and wrist cameras when cubes stand at known positions: detection, xy error, top-height error of each estimator separately, whether the stated `contact_top_z_range` contains the truth, latency, image and text size, and whether the arm occlusion report (`robot_occlusion`, [§13.1](so101-integration.md#131-arm-occlusion-report-robot_occlusion)) agrees with the FK-silhouette truth of the true cube boxes. It calls the harness's own attachment path (`obs3d.attachments`: the table, overlay and occlusion report the model receives with every observation), not a copy of it. Tests of the metric code: `tests/test_so101_perception_view.py` (`BenchMetrics` on recorded real tables, `BenchPipeline` end to end with a fake service).

## Conditions

| Factor | Values | Selected by |
|---|---|---|
| Fixed-camera resolution | 640x480 (default), 1280x720, 1920x1080 | the deployment file given to `capture` (camera size + a calibration of that size, [§13.3](so101-integration.md#133-fixed-camera-resolution-deployment-only)); `score --resolution` filters |
| Height method | `vggt` (rows only), `cross_view` (+ wrist `target_view_top_z`), `contact` (+ `contact_top_z`; default) | `perceive --height-method` (sets the deployment `perception` switches of the run profile) |
| Arm pose | `clear` (deployment `perception.clear_view_q_rad` or `--clear-q-rad`), `parked` (the TCP `--park-height-m`, default 0.08 m, above the highest cube top over the cubes' centroid, position-only IK from home; or `--parked-q-rad`), `home`, `current`, `q:<5 rad separated by />` (e.g. `q:0/-1.27/0.41/0.54/0`) | `capture --poses` (in order; the occlusion report compares each table with the previous one of the same run, so start with `clear`) |

Run one `capture` per resolution into its own directory; run `perceive` once per height method on the same capture (each writes its own `perception-<method>.jsonl` and refuses to overwrite it); `score` reads them all and groups by resolution x pose x method.

## Truth

Place the cubes with the arm (hand-guided or by robot_action), read the TCP from the console at each placement, and write the truth file in so101_base metres. The top is the table height (`joint_calibration.table_z_m` of the calibration) plus the cube edge; for a stack, the top of each cube.

```json
{"table_z_m": -0.0162,
 "cubes": [{"colour": "white",  "centre_xy_m": [0.28, -0.06], "top_z_m": 0.0128, "edge_m": 0.029},
           {"colour": "orange", "centre_xy_m": [0.26,  0.07], "top_z_m": 0.0128, "edge_m": 0.029},
           {"colour": "blue",   "centre_xy_m": [0.33,  0.01], "top_z_m": 0.0128, "edge_m": 0.029}]}
```

## Commands

Hardware steps (`snapshot`, `capture`) run in the `so101` interpreter from a terminal that macOS allows to use the camera (Ghostty), like `scripts/so101_commission.py`; stop the gateway first (one process owns the serial port and the cameras). `capture` moves the arm only with `--move`, with the commissioning joint move (speed cap, tracking and stall guards); before connecting it refuses any target outside the planning limits, with the TCP outside the workspace box or with a link mesh below the table (the joint-space path between poses is not checked: `--via-home` passes through the commissioned home pose); an error holds the measured position, and the arm returns to `home.q_rad` at the end (`--no-park-home` to stay). `perceive` needs the VGGT service (local, or the SSH tunnel of `remote.services`); `score` is offline. `D` is the deployment file, `B` the bench directory (new per capture).

```bash
D=~/.cache/ni-rua/so101/so101.curobo.local.json
PY=/opt/homebrew/Caskroom/miniforge/base/envs/so101/bin/python      # the deployment's "python"

# 0. once: the clear pose for this calibration (offline), then copy the printed snippet into D
PYTHONPATH=. mamba run -n ni-rua python scripts/so101_clear_view_pose.py --deployment $D --report /tmp/clear-view.json

# 1. 640x480 (default deployment), poses clear -> parked -> clear, 5 frames each
B=~/.cache/ni-rua/so101/bench/640-$(date -u +%Y%m%dT%H%M%SZ)
PYTHONPATH=. $PY scripts/so101_perception_bench.py capture --deployment $D --out $B --truth truth.json \
    --poses clear,parked,clear --via-home --move --frames 5 --expect-size 640x480
for METHOD in vggt cross_view contact; do
  PYTHONPATH=. mamba run -n ni-rua python scripts/so101_perception_bench.py perceive --capture $B \
      --endpoint http://127.0.0.1:8331 --height-method $METHOD
done
PYTHONPATH=. mamba run -n ni-rua python scripts/so101_perception_bench.py score --capture $B --truth truth.json

# 2. intrinsics for 1920x1080 (board on the table, camera and board still between the two snapshots)
PYTHONPATH=. $PY scripts/so101_perception_bench.py snapshot --deployment $D --camera fixed --size 640x480 --out /tmp/board-640.png
PYTHONPATH=. $PY scripts/so101_perception_bench.py snapshot --deployment $D --camera fixed --size 1920x1080 --out /tmp/board-1080.png
PYTHONPATH=. mamba run -n ni-rua python scripts/so101_remap_calibration.py --deployment $D \
    --calibration ~/.cache/ni-rua/so101/camera_calibration.json --camera fixed \
    --from-image /tmp/board-640.png --to-image /tmp/board-1080.png \
    --out ~/.cache/ni-rua/so101/camera_calibration-fixed-1920x1080.json
#    copy D to D1080 with cameras.fixed.width/height = 1920/1080 and camera_calibration_file = the new file

# 3. 1920x1080 (and 1280x720 the same way with --size 1280x720 / --expect-size 1280x720)
B=~/.cache/ni-rua/so101/bench/1080-$(date -u +%Y%m%dT%H%M%SZ)
PYTHONPATH=. $PY scripts/so101_perception_bench.py capture --deployment $D1080 --out $B --truth truth.json \
    --poses clear,parked,clear --via-home --move --frames 5 --expect-size 1920x1080
#    then perceive (three methods) and score exactly as in step 1
```

Optional: `--via-home` passes through `home.q_rad` between poses; `--speed-scale` (default 0.5 of the deployment cap); `perceive --scene3d` adds the point-cloud attachment so the latency matches a default run.

## Output and comparison

`B/capture.json` (poses, frames), `B/frames/<index>-<pose>/` (raw worker observation, both images and `fk_silhouette_<method>.jpg`: the undistorted fixed image with the arm's FK silhouette tinted red), `B/perception-<method>.jsonl` (one file per height method, never appended to; per frame: the model-visible JSON, delivered image bytes, token estimates, latencies, the camera and joint facts), `B/metrics.json` (per condition `<W>x<H>|<pose>|<method>`):

| Metric | Definition |
|---|---|
| `detection_rate` | Frames in which a row centre lies within `--match-radius-m` (default 0.03 m) of the cube (greedy nearest pairs, held rows excluded) |
| `xy_error_m` | Row centre minus truth centre, horizontal, of matched frames (n, p50, p90 of the absolute value, signed mean, max) |
| `vggt_top_error_m`, `cross_view_top_error_m`, `contact_top_error_m` | `top_z` of the row, `target_view_top_z` and `contact_top_z` of its cross_view row, each minus the truth top (m) |
| `contact_range_hit_rate`, `contact_range_m` | Share of frames whose `contact_top_z_range` contains the truth top, and the width of that range: a well-calibrated range has a hit rate near 1 with a narrow width |
| `occlusion` | With the FK silhouette of the frame's joints applied to the true cube box (`occlusion_truth`): frames in which at least `--hidden-fraction` (default 0.5) of the box is covered, the share of them reported in `hidden` with `covered` at least that fraction, the share in which the cube still has a row, false hidden reports on frames where the truth says visible, and the error of `covered` |
| `latency_s` | `attach_s` (the whole attachment), `object_table_s`, `cross_view_s` |
| `sizes` | Delivered fixed-image bytes (transport encoding), overlay bytes, model-visible JSON characters, image token estimates (two published rules; the provider's usage report is the measurement) |
| `table_covered` | Distribution of the reported share over the frames |

Compare conditions pairwise on the same truth and capture (same cubes, same poses): resolution with the method fixed, method with the resolution fixed, and `parked` against `clear` for the occlusion report. A height method helps when its top-error p90 is below the VGGT p90 on the same frames and its range hit rate is close to 1. The occlusion report matches the truth when `reported_when_hidden` is near 1 and `false_hidden_reports` near 0 in `parked`, and `table_covered` in `clear` is near the value `so101_clear_view_pose.py` predicted. A large `xy_error_m` on every cube, or a red silhouette in `fk_silhouette_*.jpg` that does not lie on the arm, means the fixed camera no longer matches its calibration ([§13](so101-integration.md#13-perception-robot-mask-held-row-and-the-cross-view-check)): recalibrate before reading any other number.

Not measured here: the real arm's outline against the FK silhouette (the occlusion truth uses the same FK; look at `fk_silhouette_*.jpg`), objects other than cubes, and the model's behaviour.
