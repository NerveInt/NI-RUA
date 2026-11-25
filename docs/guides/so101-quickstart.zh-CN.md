# SO-101 真机快速上手指南

[English](so101-quickstart.md) · [简体中文](so101-quickstart.zh-CN.md)

本指南带你从零完成 **SO-101** 桌面机械臂的硬件准备、环境安装、首次动作验收与智能体接管运行。支持**普通笔记本纯本地 CPU 运行（macOS / Linux）**，也支持按需对接**远程 NVIDIA GPU 服务**（3D 感知与 cuRobo IK）。

---

## 1. 硬件与供电检查

动手配置软件前，请先确认以下硬件条件：

- **机械臂本体：** TheRobotStudio **SO-101 Follower**（5 个手臂关节 + 1 个夹爪关节，共 6 个 Feetech STS3215 舵机串联在一条串口总线上，ID 1–6）。
- **12 V 舵机电源：** 在运行任何动作或自检脚本前必须接通并打开 12 V 电源。
  > [!IMPORTANT]
  > USB 串口芯片仅靠 USB 供电就能枚举出设备节点。若启动报 `servo_power_off`、`Missing motor IDs: 1-6` 或 `broadcast_ping()` 为空，说明 **12 V 舵机电源未打开或接触不良**，并非软件故障。上电自检（preflight）同时会校验供电电压（`10.5–13.5 V`）与舵机温度（`<= 60 °C`）。
- **USB 串口线：** 将舵机控制板通过 USB 连接到宿主机。
- **双路 UVC 相机：**
  - `wrist`（腕部相机，固定在 `gripper_link` 上）
  - `fixed`（固定俯视相机，固定在桌面上方或机架上，相对机械臂底座静止）
- **物理电源急停开关：** 操作真机时，务必在伸手可及的位置配备一个**物理 12 V 电源开关/急停开关**。软件层面的停止/保持依赖串口总线与 Worker 进程存活。
- **macOS 相机权限（TCC）：** macOS 将相机权限归属到进程树的*负责应用（responsible app）*。请务必从已授予相机权限的终端应用（如 `Terminal.app`、`iTerm2`、`Ghostty`）启动网关与脚本，不要从无相机权限声明（`NSCameraUsageDescription`）的 GUI 应用内直接拉起。

---

## 2. 安装环境

### 方式 A — 一键零克隆安装（推荐）

需要 **Node >= 18** 与 **Python >= 3.12**（或 `uv`）。命令会自动安装 `@nerveint/ni-rua-plugin` 并在 `~/.local/share/ni-rua/envs/{gateway,so101}` 下创建两个隔离的 Python 3.12 环境（规避 `openai-codex` 与 `lerobot` 上游的 `packaging`/`numpy` 版本冲突）：

```bash
npx -y @nerveint/ni-rua-plugin setup claude --so101   # 或：codex | opencode | all
npm i -g @nerveint/ni-rua-plugin                      # 可选：全局保留 `ni-rua-plugin` 命令
ni-rua-plugin doctor --so101
```

### 方式 B — 从源码安装（Mamba / Conda）

在源码仓库中，NI-RUA 将主控执行环境（`ni-rua`）与 SO-101 硬件驱动环境（`so101`）严格隔离。**严禁在仓库内创建 `.venv`。**

```bash
git clone https://github.com/NerveInt/NI-RUA.git
cd NI-RUA

# 1. 创建主控框架环境（Python 3.12 + FFmpeg）
mamba env create -f environment.yml

# 2. 创建独立的 SO-101 真机硬件 Worker 环境（LeRobot 0.6.1、SciPy、OpenCV ArUco）
mamba env create -f environment.so101.yml
# 若已有 Python >= 3.12 环境，也可手动安装依赖：
# pip install -r requirements-so101.txt

# 3. 构建 Web 真机控制台与标定页面（Node 22）
npm --prefix web ci && npm --prefix web run build
```

> [!NOTE]
> 如需开启自动纯动作 H.264 视频录制，请确保 `so101` 环境（或 `PATH`）中可用 `ffmpeg` 与 `ffprobe`（含 `libx264`）。在 macOS 上，若你的 pip 镜像缺少 `openai-codex` / `openai-codex-cli-bin`，可从官方 PyPI 安装固定版本：
> `mamba run -n ni-rua python -m pip install --index-url https://pypi.org/simple openai-codex-cli-bin==0.160.0 openai-codex==0.160.0`。

### 可选：打包并安装独立 `ni-rua-harness` Wheel

如果你希望从源码直接运行 `ni-rua`、`ni-rua-so101`、`ni-rua-queue`、`ni-rua-report`、`ni-rua-audit` 命令，请使用 `scripts/build_package.py` 构建包含前端资源与内置预设的完整 wheel（切勿直接使用 `pip wheel .`）：

```bash
mamba run -n ni-rua python scripts/build_package.py --output dist
mamba run -n ni-rua python scripts/check_package.py dist/*.whl
mamba run -n ni-rua python -m pip install dist/*.whl
```

---

## 3. 首次硬件准备

### 3.1 LeRobot 舵机中位与行程校准（`lerobot-calibrate`）

接通 12 V 电源后，先使用 LeRobot 完成一次舵机中位与行程写入，使 6 个 STS3215 舵机的刻度与你的物理装机对齐（确保 `--robot.id` 与后续配置文件中的 `robot_id` 一致，默认 `so101_follower`）：

```bash
# 将 <serial_port> 替换为实际串口路径，如 macOS 的 /dev/tty.usbmodemXXXX 或 Linux 的 /dev/ttyACM0
mamba run -n so101 lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=<serial_port> \
  --robot.id=so101_follower
```

### 3.2 查看并绑定稳定的相机 `unique_id`

OpenCV 的数字索引（`0`、`1`）会在插拔 USB 或重启后漂移。必须使用 `ni-rua-so101 cameras`（或源码仓库中的 [`scripts/so101_cameras.py`](../../scripts/so101_cameras.py)）绑定相机硬件唯一标识 **`unique_id`**。请务必使用 **`so101`** 解释器运行该命令（macOS 下 AVFoundation 的索引排序规则取决于 Worker 所用 OpenCV 版本）：

```bash
# 通过 wheel / 一键安装：
ni-rua-so101 cameras
# 或在源码仓库中运行：
mamba run -n so101 python scripts/so101_cameras.py
```

脚本会列出当前每台相机的 `opencv` 索引、`transport`、`unique_id` 与 `name / model`：

- **区分两台同型号 USB 相机：** 拔掉其中一台（或逐台插入）再运行 `ni-rua-so101 cameras`，即可确认哪条 `unique_id` 对应 `wrist`、哪条对应 `fixed`。
- **macOS：** `unique_id` 是 AVFoundation 的 `uniqueID`（如 `0x1340001b170209`）。USB 相机的 `unique_id` 包含物理 USB 端口信息，因此请保持每台相机插在固定 USB 口上。
- **Linux：** 绑定脚本输出的 `/dev/v4l/by-id/...-video-index0`（或 `/dev/v4l/by-path/...-video-index0`）完整路径。

### 3.3 生成并配置 `so101.local.json`

推荐使用 `ni-rua-so101 init-config` 向导（自动探测 USB 串口与 UVC 相机并写入 `~/.config/ni-rua/so101.local.json`），或在源码仓库中从模板 [`configs/so101.example.json`](../../configs/so101.example.json) 复制生成被 Git 忽略的 `configs/so101.local.json`：

```bash
# 零克隆 / wheel 用法（写入 ~/.config/ni-rua/so101.local.json）：
ni-rua-so101 init-config

# 或在源码仓库中复制模板：
cp configs/so101.example.json configs/so101.local.json
```

按本机实际情况修改 `configs/so101.local.json`（所有数值字段都会在加载时由 `ni_rua/robot/backends/so101/deployment.py` 严格校验单位和范围）：

1. **解释器与串口：**
   - `"python"`：指向 `so101` 环境的 Python 路径（如 `~/miniforge3/envs/so101/bin/python` 或 `/opt/homebrew/Caskroom/miniforge/base/envs/so101/bin/python`）。
   - `"port"`：机械臂串口路径（如 `/dev/tty.usbmodemXXXX` 或 `/dev/ttyACM0`）。
   - `"robot_id"`：`"so101_follower"`（与 `lerobot-calibrate` 一致）。
2. **双路相机（`cameras.wrist`、`cameras.fixed`）：**
   - 将 `"unique_id"` 设为 `scripts/so101_cameras.py` 查到的唯一标识（通常保持 `"device_name": null`）。
   - 若相机倒装或竖装，设置 `"rotate_deg"`（`0 | 90 | 180 | 270`）。
   - 分辨率与帧率默认保留 `"width": 640`、`"height": 480`、`"fps": 30`。
3. **夹爪实测开度（`gripper`）：**
   - `"empty_close_aperture_m"`：**真机运行前务必实测填写。** 空夹爪多次完全闭合（`goal_normalized: 0`），取回执 `gripper_execution.measured_opening_m` 的最大值（真机通常约为 `0.0042–0.0051 m`）。它是 `grip_guard_preset`（`non_empty`、`thin_object`、`rigid_object`、`wide_object`）与抓后防滑复检的基准零点；为 `null` 时所有预设都会被拒绝，且关闭滑脱复检。
   - `"full_open_aperture_m"`：**建议实测填写** `goal_normalized: 1` 时的开度（约 `0.136 m`；为 `null` 时使用 CAD 开度 `0.1372 m`）。
4. **安全停靠姿态与工作空间（`home`、`workspace`）：**
   - 核对 `home.q_rad`（默认模板为休息姿，也可在验收后采用预备姿 `[0.0, -0.95, 0.90, 1.00, 0.0]`）并确认 `workspace.min_m[2]` 高于你的实际桌面。

在不打开相机、不驱动机械臂的情况下验证相机绑定是否无歧义：

```bash
mamba run -n so101 python scripts/so101_cameras.py --deployment configs/so101.local.json
```

返回码为 `0` 即表示 `wrist` 与 `fixed` 均已唯一确定地绑定成功。

---

## 4. 纯本地无 GPU 运行 vs 可选远程 GPU 感知/IK 服务

NI-RUA 为 SO-101 提供了两套网关默认配置：

- **[`configs/so101_standalone.json`](../../configs/so101_standalone.json)（单机无 GPU 开箱默认配置）：** 关闭外部 3D 感知（`"external_perception": false`），直接使用内置纯 CPU 本地 `Pink + DAQP` IK（`so101_kinematics.ik`），完全不依赖 GPU 或 `8331` 端口。
- **[`configs/so101_gateway.json`](../../configs/so101_gateway.json)（可选远程 GPU 服务配置）：** 开启 `"external_perception": true` 并指向 `http://127.0.0.1:8331`，用于对接远程 VGGT/SAM 3D 感知服务（也可同时在 `8341` 对接 cuRobo IK）。

| 能力维度 | 单机无 GPU 模式（`configs/so101_standalone.json`） | 可选远程 GPU 模式（`configs/so101_gateway.json`） |
| :--- | :--- | :--- |
| **算力需求** | 仅需本地 CPU（无需 GPU） | 本地笔记本 + 远程 NVIDIA GPU 服务器（`bfloat16`、CUDA 12.8） |
| **3D 感知（VGGT + SAM）** | **关闭**（`"external_perception": false`） | **开启**（远程 `services/vggt_height/`，端口 `8331` / `8333`） |
| **可用视觉与几何工具** | 双路 RGB 图像、关节本体状态、`robot_contact_overlay`、标定几何工具 `geometry_query`（`ray`、`triangulate`、`pixel_plane`、`project`） | 额外支持 `object_table`、`scene3d`、`obs3d` 启动内联 3D 接地、`vggt_height`、可选 `held_perception` 与 `contact_top_z` |
| **IK 求解器** | 内置本地 CPU `Pink + DAQP` IK（`so101_kinematics.ik`，不配 `ik_service` 时默认启用） | 可选 **cuRobo v2** GPU 多初值 IK + 路径碰撞检查 `path_check`（`services/curobo_ik/`，端口 `8341`），解锁 `approach_tolerance_rad` / `roll_tolerance_rad` 与 `obstacle_object_ids` |
| **适用场景** | 开箱即用装机、标定、基于 RGB+本体状态的常规操作、随时随地交互调试 | 需要密集桌面物体分割、3D 点云以及避障 IK 规划的任务 |

### 方案一：纯本地无 GPU 开箱跑通真机

直接使用 `--config configs/so101_standalone.json`，并在 `configs/so101.local.json` 中保持不包含 `ik_service`（或为 `null`）：

- **已完成标定（已有 `camera_calibration.json`）：** 直接 `--config configs/so101_standalone.json` 启动即可。
- **首次冷启动（尚未生成 `camera_calibration.json`）：** 启动网关时必须附加 `--observation-profile rgb_proprio_v1`，以便进入 `/calibration` 完成首次标定。

### 方案二：按需对接远程 GPU 3D 感知与 cuRobo IK 服务

1. **在 GPU 服务器部署并自检服务：**
   - **VGGT + SAM 3.1 服务：** 参考 [`services/vggt_height/README.md`](../../services/vggt_height/README.md) 完成 `setup.sh`、`download_weights.sh` 与 `start.sh`（默认监听 `127.0.0.1:8331`）。
   - **cuRobo IK 服务（可选）：** 参考 [`services/curobo_ik/README.md`](../../services/curobo_ik/README.md)（`deploy.sh <gpu-host> <remote-dir>`，默认在 `RoboDojo` CUDA 环境中监听 `127.0.0.1:8341`）。
2. **在网关配置中声明 SSH 隧道与远程服务：**
   基于 `configs/so101_gateway.json`，参考 [`configs/so101.remote.example.json`](../../configs/so101.remote.example.json) 将 `remote` 配置段合并到你的网关本地配置（如 `~/.cache/ni-rua/so101/ni-rua.local.json`，将 `<gpu-host>` 和路径替换为你自己的 SSH 主机别名与目录；**切勿**写入 `configs/so101.local.json`），保持 `"external_perception": true` 与 `"vggt_support_z_m": null`（自动跟随每帧观测已标定的桌面高度 `table_z_m`）。*（若直接用 `configs/so101_gateway.json` 且暂不连 GPU 服务，可额外加 `--no-external-perception`。）*
3. **在 `configs/so101.local.json` 中启用 cuRobo IK（如部署了 `curobo_ik`）：**
   添加 `ik_service` 字段（需确保服务端 `SO101_JOINT_LIMIT_MARGIN_RAD` 与 `motion.joint_limit_margin_rad` 一致，默认 `0.05`）：

   ```json
   "ik_service": {
     "solver": "curobo",
     "url": "http://127.0.0.1:8341",
     "timeout_s": 10,
     "num_solutions": 8,
     "table_top_z_m": -0.02,
     "path_check_step_rad": 0.01
   }
   ```

详细机制、生命周期与 `path_check` 语义见 [SO-101 Integration §9](../development/so101-integration.md#9-distributed-deployment-mac--remote-linux-gpu-server) 与 [§12](../development/so101-integration.md#12-ik-planning-the-curobo-ik-service-and-modereach)。

---

## 5. 机台验收、运行方式与安全操作须知

### 5.1 首次上电机台调试验收（约 1.5 分钟）

清空桌面，手边准备好物理 12 V 电源开关，在具有相机权限的终端中运行 `ni-rua-so101 commission`（或源码仓库中的 [`scripts/so101_commission.py`](../../scripts/so101_commission.py)）：

```bash
# 通过 wheel / 一键安装：
ni-rua-so101 commission \
  --deployment ~/.config/ni-rua/so101.local.json \
  --out ~/.cache/ni-rua/so101/commission

# 或在源码仓库中运行：
PYTHONPATH=. mamba run -n so101 python scripts/so101_commission.py \
  --deployment configs/so101.local.json \
  --out ~/.cache/ni-rua/so101/commission
```

该脚本会按真机相同安全链路执行：上电自检 → 移至预备姿态 → 各关节 `±0.1 rad` 步进（对比无/有下垂补偿）→ `shoulder_pan` 速度扫描 → 夹爪开合（`1.0 → 0.0 → 0.5`）→ X/Y/Z `±2 cm` 笛卡尔闭环运动 → 停靠预备姿态，并输出 `commissioning.json` 与双摄快照。任一步骤未达标会立即确认保持并停止。

### 5.2 运行方式 A：Agent 插件模式（`--agent-backend external`）

1. 安装插件并核验整套依赖（仅需执行一次）：
   ```bash
   npx -y @nerveint/ni-rua-plugin setup claude --so101   # 或：codex | opencode | all
   ni-rua-plugin doctor --so101
   ```
2. 启动 NI-RUA 网关（`--config so101` 使用内置 `so101_standalone.json` 纯 CPU 运行；若尚未生成 `camera_calibration.json`，会自动补全 `--observation-profile rgb_proprio_v1`）：
   ```bash
   ni-rua-plugin gateway --config so101 \
     --port 8781 --run-dir runs/so101-demo \
     --prompt "Describe the table, move the gripper straight up 2 cm, verify it, and move it back."
   ```
   *（在源码仓库中也可运行 `mamba run -n ni-rua python -u -m ni_rua.apps.gateway --agent-backend external --config configs/so101_standalone.json ...`。）*
3. 在另一个终端启动你的智能体（`claude`、`codex` 或 `opencode`）并输入：
   `Operate the robot through the ni_rua MCP tools.`

### 5.3 运行方式 B：完整托管 Harness 模式

由 NI-RUA 直接拉起并监管智能体（默认读取 `configs/so101_standalone.json` 中的 `model` 与 `~/.codex` 凭据，或为 Claude Code / OpenCode 配置 Provider Profile）：

```bash
mamba run -n ni-rua python -u -m ni_rua.apps.gateway \
  --config configs/so101_standalone.json \
  --port 8781 --run-dir runs/so101-hosted \
  --prompt "Describe the table, move the gripper straight up 2 cm, verify it, and move it back."
```

*（若已安装 `ni-rua-harness` wheel，可将 `mamba run -n ni-rua python -u -m ni_rua.apps.gateway` 简写为 `ni-rua` 并使用 `--config so101`。）*

### 5.4 控制台安全控制（`http://127.0.0.1:8781/console?lang=zh-CN`）

运行期间在浏览器打开 **<http://127.0.0.1:8781/console?lang=zh-CN>**：

- **取得控制（Claim Control）：** 获取单写者操作租约（新打开的标签页默认只读）。
- **保持 / 停止（Hold / Stop）：** 关闭动作准入并将所有关节目标锁定在当前测量位置。（若正在执行 `go_home` 或标定动作，普通 `Stop` 需等待控制锁；**紧急停止 `E-STOP`** 可随时抢占。）
- **紧急停止（E-STOP）：** 走独立安全锁立即停机保持，并中断当前智能体回合。
- **松开力矩（Relax）：** 卸除所有关节力矩（**机械臂会在重力下坠落！确认前请务必用手托住机械臂**）。卸力后安全门禁会锁死运动，直到重新点击**上力矩（Torque On）**并采集新观测。
- **回安全位（Go Home）/ 恢复运行（Resume）：** 空闲时回到 `home.q_rad`，或停止后先拍一帧新观测再恢复动作准入。

> [!WARNING]
> **关闭浏览器标签页，或在 Claude Code / Codex / OpenCode 中按 `Esc`，都不会让真机机械臂停下！** 急停请务必使用控制台的 **保持 / 急停（E-STOP）** 或物理 12 V 断电开关。

---

## 6. 下一步：完成 S0–S6 相机与关节标定

在依赖 3D 坐标、`pixel_plane`、`triangulate` 或桌面抓取任务之前，请先完成引导式标定：

1. 启动网关。**首次尚未生成 `camera_calibration.json` 时**，`ni-rua-plugin gateway --config so101` 会自动补全 `--observation-profile rgb_proprio_v1`（若直接调用 `ni-rua`，需显式附加该参数）：
   ```bash
   ni-rua-plugin gateway --config so101 --port 8781 --run-dir runs/so101-calib
   ```
2. 在浏览器打开 **<http://127.0.0.1:8781/calibration?lang=zh-CN>**。
3. 使用 `ni-rua-so101 charuco-board` 生成 1:1 比例 ChArUco 标定板并按照 **[SO-101 标定指南](so101-calibration.zh-CN.md)** 依次完成 S0–S6 步骤。

S6 应用通过后，系统会自动将配置文件从无标定状态（`rgb_proprio_v1`）升级为已标定状态（`rgb_proprio_calibrated_v1`），并原子写入相机内外参、手眼变换、关节零偏与标定桌面高度。后续日常运行只需传入 `--config so101`（或 `--config configs/so101_standalone.json`）即可。
