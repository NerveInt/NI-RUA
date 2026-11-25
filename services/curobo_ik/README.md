# cuRobo IK service (SO-101)

GPU-side inverse kinematics for the real SO-101 arm. The SO-101 planner in NI-RUA
(`ni_rua/robot/backends/so101/ik_service.py`, used by the worker's preview and by `robot_action mode=reach`) sends
candidate TCP poses here and gets back the best joint solutions of NVIDIA cuRobo v2's multi-seed GPU IK, with
self-collision and an optional table box checked. NI-RUA then verifies every solution with its own forward
kinematics before it plans with it, so the receipts' residuals never come from this service.

The same process also checks joint-space paths for collision (`POST /path_check`): the worker's preview sends the
straight joint-space segment of every planned move, and the service samples it and reports the first colliding sample
and the colliding links against self-collision and a world of the table box, obstacle boxes and an attached held
object. The result is an advisory fact in the preview; it never blocks or changes a plan.

The service needs an NVIDIA GPU with CUDA and cuRobo v2. On the lab GPU host it runs in the RoboDojo interpreter
(`~/miniforge3/envs/RoboDojo`, cuRobo pinned at RoboDojo's `third_party/curobo` d17b54c, torch 2.7.0+cu128).

## Files

| File | Purpose |
| --- | --- |
| `build_model.py` | Builds `model/`: the SO-101 URDF with the arm joint limits shrunk by the deployment margin, cuRobo collision spheres (MorphIt fit, base link clipped at z=0) and the self-collision ignore matrix; the jaw joint is locked at 0 rad. |
| `curobo_ik_service.py` | Loopback HTTP server: `GET /health`, `POST /ik`, `POST /path_check` (request and response fields in the module docstring). |
| `start.sh` / `stop.sh` | Detached start (builds the model first when needed, waits for `/health`) and stop of its own PID. |
| `deploy.sh` | Copies this directory and the packaged SO-101 geometry (`geometry_model.zip`, `physical.json`) to a GPU host. |

## Deploy and run

```sh
bash services/curobo_ik/deploy.sh <gpu-host> '~/ni-rua-services/curobo_ik'     # from the local checkout
ssh <gpu-host> 'cd ~/ni-rua-services/curobo_ik && CUROBO_PYTHON=~/miniforge3/envs/RoboDojo/bin/python \
  CUDA_VISIBLE_DEVICES=6 SO101_JOINT_LIMIT_MARGIN_RAD=0.05 bash start.sh'
```

Normally the gateway starts it: list it under `remote.services` of the NI-RUA run configuration (name `curobo_ik`,
`remote_port` 8341, `workdir` the deployed directory, `env` `CUROBO_PYTHON=...` and `SO101_JOINT_LIMIT_MARGIN_RAD=...`,
`command` `bash start.sh`, `stop` `bash stop.sh`, `health_path` `/health`, `gpu` `auto`) and point the SO-101 deployment's
`ik_service.url` at the tunnelled local port (`http://127.0.0.1:8341`). `gpu_min_free_mib` defaults to 15360 (the VGGT
figure); the footprint of this service has not been measured. `start.sh` builds the model when `model/model.json` is
missing or was built with another margin, so the first start takes the model build time. Other `start.sh` settings:
`CUROBO_IK_PORT` (8341), `CUROBO_IK_HOST` (127.0.0.1), `CUROBO_IK_SEEDS` (64 parallel seeds), `CUROBO_IK_MAX_BATCH` (32
poses per request), `START_TIMEOUT_S` (300 s); `stop.sh` needs the same `CUROBO_IK_RUN_DIR` and `CUROBO_IK_PORT`.
`SO101_JOINT_LIMIT_MARGIN_RAD` (rad, default 0.05) must equal the deployment's `motion.joint_limit_margin_rad`; the
service refuses a request whose joint limits differ (HTTP 409 `joint_limits_mismatch`). `SO101_JAW_OPEN_RAD` (rad,
default 1.74533, the example deployment's value and the URDF upper limit of the `gripper` joint) should equal the
deployment's `gripper.jaw_open_rad`: `start.sh` passes it as `--open-jaw-rad` and the startup self-check checks the
ready pose with the jaw there (below). The deployment fields
(`timeout_s`, `num_solutions`, `table_top_z_m`), candidate selection, failure fields and the unreachable-service
behavior are described in `docs/development/so101-integration.md` section 12.

Build notes: cuRobo pads a smaller batch but not the jerk buffer, so the service always sends a full `--max-batch`
request (the first pose repeated) and returns only the real poses. MorphIt sphere fits are stochastic, so
`build_model.py` seeds them (`--seed`, default 42); the thin camera mount fits poorly and overlaps `wrist_link`, so that
pair is ignored. At startup the service solves the commissioned ready pose and exits with `SELF_CHECK_FAILED` if it is
infeasible, or if that solution collides in the path checker with the jaw at 0 rad or at `--open-jaw-rad`
(`self_check_failure`; the `READY` line names the open angle it checked).

## Path check (`POST /path_check`)

Request: `segments` (1 to 64, each `start_q_rad` / `end_q_rad` of the five arm joints in rad, optional `start_jaw_rad`
/ `end_jaw_rad` of the URDF `gripper` joint, default 0, and `attached`, default true), `max_joint_step_rad` (rad between
checked samples, 0.001 to 0.5, default 0.02; NI-RUA sends its `ik_service.path_check_step_rad`, default 0.01),
`joint_limits_rad` and `table_top_z_m` as for `/ik`, `obstacles` (up to 40 axis-aligned boxes `{id, min_m, max_m}` in
`base_link`, m) and `attached_spheres` (up to 32 `[x, y, z, radius]` in the `gripper_frame_link` frame, m). Each
segment is checked at `1 + ceil(largest joint change / max_joint_step_rad)` evenly spaced samples, both ends included;
a request of more than 8192 samples in total is refused (`too_many_samples`; the NI-RUA client splits its segments
over several requests). Response: `check_ms`, `samples` and per segment `samples`, `colliding_samples` and
`first_collision` (null, or `sample_index`, `fraction`, `q_rad`, `jaw_rad`, `world` `[{link, penetration_m,
obstacles}]`, `self` `[{links, penetration_m}]`). `/ik` accepts the same `obstacles` / `attached_spheres` plus
`jaw_rad` and then adds `obstacle_free` to every solution, without changing which solutions the solver returns.
`GET /health` lists the limits under `path_check`; a service without that key predates the endpoint.

How it is built: one cuRobo `RobotCollisionChecker` over the IK robot model with two changes
(`checker_kinematics`): the jaw joint is not locked, so an open jaw is checked where it is, and an extra link
`attached_object` fixed at the tool frame carries the attached spheres and is never checked against the gripper
links. The collision verdict and the world penetration depth are cuRobo's; the obstacle and link-pair names in the
answer come from the same spheres checked against the boxes and each other (`box_overlaps`,
`sphere_pair_overlaps`), which are plain Python and tested on the Mac (`tests/test_so101_path_check.py`).

The checker must run before the IK solver's first solve: that solve captures a CUDA graph, and a warp collision kernel
first loaded afterwards corrupts it (the next `/ik` then fails with `CUDA error: misaligned address`). `Solver.__init__`
therefore exercises every kind of checker query once, and the startup self-check also requires the solved ready pose
to be collision free in the checker with the jaw at 0 rad and at the deployment's open angle (`--open-jaw-rad`;
`SELF_CHECK_FAILED` otherwise). The IK model's self-collision ignore matrix is computed with the jaw locked at 0 rad,
so a sphere model whose open jaw overlaps a neighbouring link would otherwise only show up later, as every `/ik`
candidate of a request with an open `jaw_rad` labelled `obstacle_free=false` and every path check `collisions`.

NI-RUA cuts a move with a during-motion jaw ramp (`gripper_motion`) into up to three segments at the ramp's start and
end fractions (`ik_plan.path_check_segments`), because the service interpolates the jaw linearly over each segment;
the service itself is unchanged by that.

Measured on the same host (2026-11-15, GPU time inside the service): one segment of 184 to 227 samples 1.8 to 2.7 ms;
16 segments with 2944 to 3632 samples and one obstacle 2.7 to 12 ms; 64 segments with 7400 samples, five obstacles
and 27 attached spheres, 52 of them colliding, 54 to 62 ms (the per-collision report is Python). Round trip from the
Mac through the SSH tunnel 290 to 630 ms per request. The sphere model is 57 spheres fitted to the CAD (plus the
attached slots); the lowest gripper sphere ends about 6 mm above the CAD fingertips, so a thin overlap at the
fingertips is not seen.

Measured on the RTX PRO 6000 Blackwell host (2026-11-15): model build about 25 s (55 to 56 spheres on 9 links), service
ready about 6 s after start, a batch of 4 to 6 poses with 64 seeds in 3 to 45 ms on the GPU, and 0.4 to 1.2 s round
trip from the Mac through the SSH tunnel including the position-first retry.
