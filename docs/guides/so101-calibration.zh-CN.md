[English](so101-calibration.md) · [简体中文](so101-calibration.zh-CN.md)

# SO-101 标定指南

本指南详细介绍 SO-101 真机的 ChArUco 标定板选型、自印与装裱规范、Web 引导式标定向导（`/calibration`）从 S0 到 S6 的全流程操作、双目 XY 一致性核验以及常见故障排查。

若尚未完成硬件接线、环境安装、`lerobot-calibrate` 舵机中位校准及双相机绑定，请先完成 [SO-101 快速上手指南](so101-quickstart.zh-CN.md)。底层数据结构、API 协议与数学推导见 [SO-101 集成开发文档](../development/so101-integration.md)。

---

## 1. 标定板选型

### 为什么采用 ChArUco 标定板？

NI-RUA 统一使用 OpenCV **ChArUco 标定板**（将 ArUco 标记嵌入棋盘格白色方格内）：

- **棋盘格亚像素精度**：相机内参与 PnP 位姿求解使用棋盘格内角点，角点定位精度远高于纯 ArUco 标记角点。
- **唯一 ID 与抗局部遮挡**：每个内角点都由相邻 ArUco 标记赋予全局唯一 ID，即使机械臂遮挡或标定板部分出画，依然可以稳定检出剩余角点，且彻底消除普通棋盘格的 180° 方向歧义。

### 两套已验证规格对照表

仓库内已验证以下两套标定板规格。**自行 A4 打印推荐规格 A**；近距离高密度采样或采购商业成品板推荐 **规格 B**：

| 参数 | 规格 A — 推荐默认 A4 自印版 | 规格 B — 高密度 / 成品亚克力或铝基板版 |
| :--- | :--- | :--- |
| 配置来源 | 内置默认（`calibration_defaults.json`） | [`configs/so101_board.json`](../../configs/so101_board.json) |
| 类型（`type`） | `charuco` | `charuco` |
| 方格数（`squares_x × squares_y`） | `7 × 5`（24 个内角点） | `12 × 9`（88 个内角点） |
| 方格边长（`square_m` / `square_mm`） | `0.030 m`（`30.0 mm`） | `0.015 m`（`15.0 mm`） |
| 标记边长（`marker_m` / `marker_mm`） | `0.022 m`（`22.0 mm`） | `0.01125 m`（`11.25 mm`） |
| ArUco 字典（`dictionary`） | `DICT_5X5_100` | `DICT_5X5_100` |
| 图案总尺寸 | `210 × 150 mm` | `180 × 135 mm` |
| 板厚（`thickness_m` / `board_thickness_m`） | 装裱后实测（默认 `0.003 m` / `3.0 mm`） | `0.0032 m`（`3.2 mm`） |
| 适用场景 | 桌面通用视角、A4 纸自印手眼标定 | 近距离高密角点、激光蚀刻 / 哑光 UV 成品板 |

### 材质、平整度与板厚实测要求

- **必须哑光不反光**：使用哑光纸、哑光 UV 铝塑板/亚克力板或磨砂陶瓷标定板。**严禁使用亮面铜版纸或光面覆膜**，顶灯反光会导致 ArUco 识别丢失或角点偏移。
- **绝对刚性平整**：不能有任何翘曲、卷边或弧度，否则破坏平面 PnP 假设，导致内参与手眼残差居高不下。
- **四周保留 ≥ 10 mm 白色静区（Quiet Zone）**：图案外缘四周至少留出 `10 mm` 纯白边框，确保最外圈标记稳定检出。
- **务必实测总板厚 `board_thickness_m`**：使用游标卡尺或精密直尺测量**桌面到印刷正面的总垂直厚度**。例如将哑光纸裱贴在 `3.0 mm` 铝板或亚克力板上时，必须测量“底板 + 胶层 + 纸张”的**组合总厚度**，并如实填入 `configs/so101.local.json` 的 `calibration.board_thickness_m` 或向导 **S2** 表单的 `thickness_mm`。步骤 **S5** 的联合优化直接依赖“板面高度 = 桌面高度 + 板厚”这一几何先验：

  $$\text{板面高度} = \text{桌面高度} + \text{板厚}$$

---

## 2. 自行打印与装裱指南

### 生成 1:1 打印文件

在 `ni-rua` 环境中运行 [`scripts/so101_charuco_board.py`](../../scripts/so101_charuco_board.py)，直接生成精确 1:1 物理尺寸的 A4 横向 PDF 与 PNG 文件：

```bash
# 规格 A：默认 7x5 标定板（30.0 mm 方格，22.0 mm 标记，DICT_5X5_100）
mamba run -n ni-rua python scripts/so101_charuco_board.py --output-dir /tmp/so101_board

# 规格 B：按 configs/so101_board.json 生成 12x9 标定板（15.0 mm 方格，11.25 mm 标记）
mamba run -n ni-rua python scripts/so101_charuco_board.py \
  --board-config configs/so101_board.json --output-dir /tmp/so101_board_12x9

# 自定义规格（长度单位 mm；要求 square_mm * px_per_mm 为整数）
mamba run -n ni-rua python scripts/so101_charuco_board.py \
  --squares-x 8 --squares-y 6 --square-mm 25 --marker-mm 18 \
  --dictionary DICT_5X5_100 --output-dir /tmp/so101_board_custom
```

脚本会在 `--output-dir` 下输出 `so101_charuco_board.pdf` 和 `so101_charuco_board.png`：

- **精确整数像素分辨率**：默认 `24 px/mm`（`609.6 DPI`），每个 `30.0 mm` 方格恰好为整数 `720 px`（`15.0 mm` 方格为 `360 px`），杜绝插值模糊。
- **A4 横向页面与校验标尺**：PDF 页面尺寸精确为 A4 横向 **`297 × 210 mm`**，图案下方自带 **`100.0 mm` 校验标尺**（每 `10 mm` 一刻度），上方附带完整规格参数标签。

### 打印机设置红线

1. **必须直接打印 `.pdf`**：切勿用系统看图软件打印 `.png`（看图软件常默认按边距缩放）。
2. **纸张与方向**：**A4**、**横向（Landscape）**。
3. **缩放比例**：严格设为 **100% / 实际大小（Actual Size）**。**严禁勾选“适合页面大小（Fit to Page）”、“缩小超大页面”或“无边距打印（Borderless）”**（无边距模式会暗中将画面放大 2–5%）。
4. **纸张与对比度**：使用 **≥ 120 g/m² 哑光**喷墨纸或激光打印纸，选择高对比黑白模式，关闭省墨模式。

### 装裱与印后必检

1. **平整装裱到刚性平板**：使用薄双面胶或喷胶将打印纸平整无气泡地裱贴在刚性底板（铝塑板、亚克力板或厚硬卡板）上，压实边缘防止翘起。
2. **印后必须实测尺寸**：
   - 用游标卡尺或精密钢尺测量页脚 **`100.0 mm` 校验标尺**。
   - 同时测量多个连续方格的总跨度（例如 `7` 格理论总长 `210.0 mm`，实测后除以 `7`），计算出更准确的单格实测边长。
3. **打印机存在微缩放时的修正方法**：
   - 不少办公打印机即使设了 100% 仍可能有 ~0.5–1.5% 缩放。例如 `100.0 mm` 实测为 `99.0 mm`（缩放 `0.99×`）时，`30.0 mm` 方格实际变为 **`29.7 mm`**，`22.0 mm` 标记按同比例换算为 **`21.78 mm`**（`22.0 × 29.7 / 30.0`）。
   - **Web 向导（`/calibration`）**：在首页或 **S2** 标定板表单中直接将 `square_mm` 改为 `29.7`、`marker_mm` 改为 `21.78`，并填入装裱后的实测 `thickness_mm`。
   - **离线 CLI（`scripts/so101_calibrate.py`）**：在 `solve` 子命令中传入 `--square-m 0.0297`（脚本会自动按同比例缩放 `marker_m`，无需重复换算，也可同时显式传 `--marker-m`）。
4. **重测装裱后总厚度**：装裱完成后用卡尺重测桌面到印刷表面的总厚度，填入 `configs/so101.local.json` 的 `calibration.board_thickness_m` 或向导表单的 `thickness_mm`。

---

## 3. 完整标定流程（S0 → S6）

### 标定前准备

开始标定前请确认以下 4 项均已就绪：

- SO-101 机械臂已接通 **12 V 舵机电源**且 USB 串口连接正常。
- 已完成 LeRobot 舵机中位校准（`lerobot-calibrate`）。
- 已在 `configs/so101.local.json` 中为腕部（`wrist`）与固定（`fixed`）两路相机绑定稳定 `unique_id` 并配置正确的 `rotate_deg`（`0` / `90` / `180` / `270`，可用 `scripts/so101_cameras.py` 核验）。
- 前端资源已构建（`npm --prefix web run build`），ChArUco 标定板已备好并实测尺寸与板厚。

### 启动网关与打开标定向导

使用单机无 GPU 默认配置 `configs/so101_standalone.json` 启动网关（纯 CPU 本地运行，不依赖 GPU 或远程感知服务）。**首次开箱尚未生成 `camera_calibration.json` 时，必须附带 `--observation-profile rgb_proprio_v1`：**

```bash
# 首次标定（尚无 camera_calibration.json，必须附加 --observation-profile rgb_proprio_v1）
mamba run -n ni-rua python -u -m ni_rua.apps.gateway --agent-backend external \
  --config configs/so101_standalone.json --observation-profile rgb_proprio_v1 \
  --port 8781 --run-dir runs/so101-calib

# 重新标定（或首次标定完成并点击应用后的日常运行）
mamba run -n ni-rua python -u -m ni_rua.apps.gateway --agent-backend external \
  --config configs/so101_standalone.json --port 8781 --run-dir runs/so101-calib
```

> 如需对接可选远程 GPU 3D 感知 / cuRobo IK 服务，可将 `configs/so101_standalone.json` 替换为 `configs/so101_gateway.json`（首次尚无 `camera_calibration.json` 时同样须附带 `--observation-profile rgb_proprio_v1`）。

在浏览器打开标定向导：

- 中文界面：<http://127.0.0.1:8781/calibration>
- 英文界面：<http://127.0.0.1:8781/calibration?lang=en>

顶部栏关键控制说明：

- **取得控制（Claim Control）**：获取 30 s 操作员写控制租约（页面每 10 s 自动续租，长步骤期间不断链）。开始标定、执行步骤、应用与回滚都需要持有该租约。
- **紧急停止（E-STOP）**：**无需持有控制租约即可随时触发**。立即中断运动、原地保持位置并关闭动作准入。
- **安全保护机制**：所有带运动的步骤（`S1`、`S2`、`S4`、`S6`）、应用与回滚在执行期间会自动关闭智能体动作准入，防止外部智能体并发下发动作。

### 选择标定模式

首页提供两种标定入口：

1. **首次完整标定（`first_run`）**：按顺序完整执行 `S0 → S1 → S2 → S3 → S4 → S5 → S6`，适用于首次装机或全部重来。
2. **重新标定（`recalibrate`）**：仅勾选发生变动的硬件项，系统自动生成最小重跑计划，其余未变动数据复用当前生效版本：

| 变动项 ID | 页面选项 | 自动重跑步骤 |
| :--- | :--- | :--- |
| `fixed_moved` | 固定相机被移动 | `S4 → S5 → S6` |
| `fixed_replaced` | 固定相机换了 / 调了焦 | `S3 → S4 → S5 → S6` |
| `wrist_bumped` | 腕部相机被碰 | `S2 → S4 → S5 → S6` |
| `table_or_base_moved` | 换了桌子或底座挪了 | `S1 → S2 → S4 → S5 → S6` |
| `board_changed` | 换了标定板 | `S2 → S4 → S5 → S6` |
| `everything` | 全部重来 | `S0 → S1 → S2 → S3 → S4 → S5 → S6` |

> **依赖过期机制**：数据流向为 `S2、S3 → S4 → S5 → S6` 与 `S1 → S5 → S6`。重做任一上游步骤，所有下游已完成步骤都会自动标记为 **已过期（stale）**，须依次重新执行后才能应用。

---

### 逐步实操（S0 → S6）

以下默认门限均与 [`configs/so101.example.json`](../../configs/so101.example.json) 中的 `calibration` / `preflight` 配置完全一致。

#### S0 · 设备自检 *（无运动）*

- **人工操作**：确认 12 V 舵机电源已开启、USB 线缆稳固、双相机镜头无遮挡。
- **机器行为**：只读检查上电预检报告（`preflight.json`）、实时读取舵机电压/温度/力矩状态、双相机各取一帧新图核验分辨率（默认 `640 × 480`）、核验关节遥测延迟以及 `so101` 环境求解依赖（`cv2.aruco`、`scipy.optimize`）。
- **通过判据**：
  - 上电自检报告通过（`preflight_ok = true`）
  - 电压与温度在部署范围内（`power_ok = true`：默认 `10.5–13.5 V`、`<= 60.0 °C`）
  - 力矩已开启（`torque_ok = true`）
  - 关节读数延迟 `<= 2.0 s`（`selfcheck_max_telemetry_age_s`）
  - 双相机出图且尺寸匹配（`cameras_ok = true`）

#### S1 · 关节零位与桌面高度 *（有运动）*

- **人工操作**：**务必先移走桌面上的标定板与一切杂物**，确保工作区无人手与障碍物，站在 **紧急停止** 旁，勾选安全确认项后启动。
- **机器行为**：闭合夹爪（`touch_jaw_normalized = 0.0`），依次移动到 4 个默认触碰点上方 `0.06 m` 悬停（`touch_points_xy_m`：`[0.20, 0.00]`、`[0.24, 0.06]`、`[0.24, -0.06]`、`[0.17, -0.05]`），按 `touch_step_m = 0.002 m` / `0.12 s`（速度比例 `touch_speed_scale = 0.5`）小步下降至最低 `touch_min_z_m = -0.06 m`；经过 `3` 步预热后，连续 `2` 次读数满足关节滞后 `> 0.03 rad`（`touch_contact_lag_rad`）或原始负载增量 `> 120.0`（`touch_contact_load_delta`）即记录阻滞候选并抬起，同时后台以 `10 Hz` 录制触碰过程影像（最多 `600` 帧 / `60 s`），全部完成后回到 `home`。
- **常见避坑**：若忘记撤走标定板或桌垫，测得的桌面高度会整体偏高。S1 通过仅代表采集到阻滞候选轨迹，不能替代独立物理接触核验，S5 前仍需人工复核。
- **通过判据**：
  - 完成阻滞候选采集的点数 `contacts == 计划点数`（默认 `4 / 4`）。

#### S2 · 腕部相机内参与手眼 *（有运动）*

- **人工操作**：**将标定板平放在桌面上且保证双相机同时可见，用胶带或压块牢牢固定**；核对并按需修改下方表单中的 `squares_x`、`squares_y`、`square_mm`、`marker_mm`、`dictionary`、`thickness_mm`；**点击开始后严禁触碰或移动标定板**。
- **机器行为**：先实时校验 `board_both_visible`（双相机各自检出角点数 `>= 8`，`min_board_corners`），随后以 `speed_scale = 0.6` 遍历 `home.q_rad` 及配置的关节偏置网格（`pan/flex/roll/lift_offsets_rad`，默认共 21 个位姿），每个位姿稳定后双相机拍照并保留腕部检出 `>= 8` 个角点（`sample_min_wrist_corners`）的样本，回 `home` 后自动求解腕部内参与手眼变换 `T_link_camera`（过滤单帧 PnP 误差 `> 2.0 px` 的异常样本）。
- **常见避坑**：采集途中碰动标定板、表单尺寸/字典与实物不一致、灯光反光或纸面翘曲。
- **通过判据（4 项）**：
  - 有效样本数 `kept >= 8`（`min_samples`）
  - 腕部内参 RMS `<= 1.0 px`（`max_intrinsics_rms_px`）
  - 手眼平移 RMS `<= 5.0 mm`（`max_hand_eye_translation_m = 0.005 m`）
  - 手眼旋转 RMS `<= 0.02 rad`（`max_hand_eye_rotation_rad`）

#### S3 · 固定相机内参 *（无运动）*

- **人工操作**：机械臂全程保持不动。点击开始采集后，拿起标定板放到**固定相机**前，每个姿态**停稳约 1 秒**。跟随页面的 **3×3 九宫格**、**远/中/近距离** 与 **正对/小倾角/大倾角** 引导覆盖画面不同位置与角度——**务必进行双轴倾斜（前后俯仰 + 左右偏斜）与远近移动，切勿只在单一平面平推**。采满 `>= 10` 个高质量视图（达到 `handheld_target_views = 20` 或超时 `300 s` 时自动停止采集）后点击 **求解**。求解通过后，**把标定板重新平放回桌面并固定好**，供 S4 使用。
- **机器行为**：直接从 worker 固定相机视频流中自动筛选静止（角点帧间抖动 `< 1.5 px`，`handheld_still_px`）且视角新颖（视角签名距离 `>= 0.08`，`handheld_min_novelty`）的视图，点击求解后标定固定相机内参 `K` 与畸变系数。
- **通过判据（2 项）**：
  - 保留视图数 `views >= 10`（`handheld_min_views`）
  - 固定相机内参 RMS `<= 1.0 px`（`max_handheld_intrinsics_rms_px`）

#### S4 · 固定相机外参 *（有运动）*

- **人工操作**：确认标定板平放固定在桌面上且**双相机同时可见**，开始后严禁移动标定板。
- **机器行为**：机械臂遍历前 `extrinsic_pose_count`（默认 **5**）个代表性位姿（含 `home`），通过腕部运动链传递标定板在 `so101_base` 下的位姿 `T_base_board`，结合固定相机观测解算固定相机外参 `T_base_camera`。
- **通过判据**：
  - 固定相机看到标定板的样本数 `fixed_samples >= 1`。
  - 此处展示的 `fixed_chain_reprojection_px` 仅供参考（尚未消除关节零位偏差，将在 S5 按新零位复核硬门限）。

#### S5 · 联合求解与交叉校验 *（无运动）*

- **人工操作**：事后检查 S1 采集点的触碰影像或外部测量，确认夹指确实接触桌面且接触部位符合几何模型，勾选物理接触确认项后启动求解（机械臂不动）。**切勿仅凭编码器滞后或原始负载读数盲目勾选。**
- **机器行为**：基于李代数非线性最小二乘，将 S2 + S4 腕部标定板视图、S1 桌面触碰点与“板面高度 = 桌面高度 + 板厚”先验联合优化，同时求解各关节零位偏差 `offset_rad`、手眼 `T_link_camera` 与桌面高度 `table_z_m`；随后按新零位重新解算固定相机外参并生成新旧参数对照表。**本步不会写入或覆盖线上标定文件。**
- **通过判据（5 项硬门限）**：
  - 腕部重投影 RMS `<= 1.5 px`（`max_joint_reprojection_px`）
  - 板面高度误差 `|板面高度 − (桌面 + 板厚)| <= 3.0 mm`（`max_board_height_error_m = 0.003 m`）
  - 最大触碰残差 `<= 6.0 mm`（`max_touch_residual_m = 0.006 m`）
  - 固定相机链路重投影 RMS `<= 6.0 px`（`max_fixed_reprojection_px`）
  - 最大关节零位偏移 `|offset_rad| <= 0.15 rad`（`max_joint_offset_rad`）

#### S6 · 应用、高度验收与双目 XY 一致性检查 *（仅高度验收有运动）*

S6 包含三个环节：

1. **审核并应用 *（无运动）***：
   - 仔细核对页面上的**新旧数值对照表**（桌面高度、各关节 `offset_rad`、腕部与固定相机平移/旋转变化、焦距与各项残差）。
   - 勾选“我已核对新旧数值”后点击 **应用新标定**。
   - 系统会在 `<calibration_bundle_dir>/versions/<UTC>/` 下创建新版本（首次应用会先将旧文件导入为回滚基线），原子替换 `camera_calibration.json`，将新关节零位写入 `configs/so101.local.json` 并自动生成 `.bak-<UTC>` 备份，切换 `active` 指针，同时在内存中热更新偏移且**机械臂保持原地不动**。
2. **独立点触桌高度验收 *（有运动）***：
   - **再次清空桌面上的标定板与一切杂物。**
   - 启动 S6 验证触碰，机械臂在未参与拟合的独立验证点 `verify_points_xy_m`（默认 2 点：`[0.22, 0.03]`、`[0.30, -0.04]`）触碰桌面，将实测接触高度与新 `table_z_m` 对比。
   - **通过判据**：
     - 接触成功的验证点数 `contacts == 计划点数`（默认 `2 / 2`）
     - 最大验证误差 `|接触高度 − 桌面高度| <= 5.0 mm`（`max_verify_touch_error_m = 0.005 m`）
   - **失败提示**：**S6 高度验收未通过时，新标定仍保持加载状态，系统不会自动回滚。** 请先排查桌面杂物或触碰日志，再决定重新验证或点击 **回滚到上一个版本**。
3. **双目 XY 相机—指尖一致性检查 *（无运动）***：
   - 触桌验收只验证 **高度（Z）**，不代表水平面 **XY** 视觉对齐准确。可在 S6 页面下方的 **XY 相机—指尖一致性检查** 卡片中进一步核验：
   - 在机械臂静止后点击 **冻结当前画面（不移动）**，系统确认机械臂停稳并抓取 `fixed` 与 `wrist` 两路原始 RGB 冻结帧及当前 FK 位姿（全程不发送任何电机目标）。
   - 在两张冻结图像中分别点击**同一个固定夹指内侧最前端中心**（显示红色 `+` 标记；遮挡或模糊时不要提交），再点击 **检查本点 XY**。
   - 页面默认可编辑阈值（来自 [`configs/so101.example.json`](../../configs/so101.example.json)）：
     - XY 容差：**`5.0 mm`**（`xy_max_error_m = 0.005 m`）
     - 重投影上限：**`3.0 px`**（`xy_max_reprojection_error_px = 3.0`）
     - 最小视差角：**`1.0°`**（`xy_min_parallax_deg = 1.0`）
   - 建议在工作区多个不同位置、高度和姿态下重复抽检。结果独立记录在 `run.json` 的 `xy_verifications` 中，不影响 S6 高度结论；若双目重投影超限或视差角不足会返回 `无法判定（unknown）` 而非误判通过。由于腕部外参共享 FK，此项检查属于相机—运动学一致性核验，不能替代独立外部基准物的绝对精度认证。

---

## 4. 标定后维护、回滚与排错

### 控制台顶部标定状态徽章含义

`/console` 与 `/calibration` 顶部的标定状态徽章每 10 秒轮询 `GET /v1/calibration/summary`：

| 状态徽章 | 对应含义 | 处理建议 |
| :--- | :--- | :--- |
| **标定有效**（绿） | 标定已加载、与当前版本库一致、包含关节联合标定、`staleness` 正常，且**当前生效版本已通过 S6 高度验收**。 | 可正常运行任务。 |
| **标定已加载，未验收**（黄） | 标定文件已加载，但当前版本缺少通过的 S6 高度验收记录。 | 前往 S6 完成触桌高度验收。 |
| **高度验证未通过**（红） | 当前生效版本的 S6 高度验收失败。 | 排查测量误差后重新验证、重新标定或回滚。 |
| **可能已过期**（黄） | 偏移检测报 `shifted`（`> 15.0 px`）、线上文件与版本库不一致、缺少 `joint_calibration` 或固定相机内参未独立标定。 | 在首页选择对应变动项进行 **重新标定**。 |
| **需要标定**（红） | 缺少可用的 `camera_calibration.json` 或文件校验加载失败。 | 执行 **首次完整标定**。 |
| **标定状态未知**（灰） | 网关或 `so101` worker 不可达，或状态查询失败。 | 检查网关进程与 `robot-worker.log`。 |

### `staleness` 偏移检测原理

在机械臂静止且双相机同时看到标定板时，worker 最多每 `staleness_check_period_s`（`5.0 s`）执行一次只读检查：使用当前生效标定，将腕部相机运动链推算出的标定板角点重投影到固定相机画面，与固定相机实测角点比较。当 RMS 偏差超过 `staleness_max_board_shift_px`（`15.0 px`）时标记为 `shifted` 并在徽章提示“可能已过期”。该检测从不移动机械臂，也不会修改任何标定文件。

### 版本历史一键回滚

每次应用的新标定都会完整保存在 `calibration_bundle_dir`（默认与 `camera_calibration_file` 同级的 `calibration/` 目录，如 `~/.cache/ni-rua/so101/calibration/versions/<UTC>/`）。在 `/calibration` 首页的 **版本历史** 表格中点击任意历史版本的 **回滚到此版本**（或在 S6 点 **回滚到上一个版本**），即可原子恢复对应的 `camera_calibration.json` 与 `configs/so101.local.json` 关节零位，并热更新到运行态，机械臂无需移动。

### 常见失败排查表

| 报错 / 异常现象 | 常见根因 | 排查与解决方法 |
| :--- | :--- | :--- |
| `board_both_visible` 未通过（角点 `< 8`） | 标定板未落在双相机共视区内；强反光；表单规格（如 `7×5` 与 `12×9`）与实物不符；`rotate_deg` 错误或两路相机绑定互换。 | 调整标定板位置至双图清晰共视；避开顶灯反光；核对 S2 表单参数；用 `scripts/so101_cameras.py` 核验相机绑定。 |
| S2 内参（`> 1.0 px`）或手眼残差（`> 5.0 mm` / `> 0.02 rad`）超标 | 采集过程中标定板滑动；纸面翘曲不平；打印缩放未修正；腕部相机支架松动；`joint_mapping` 方向 `sign` 配反。 | 牢牢固定标定板于刚性平板；实测方格边长并更新 S2 表单；紧固相机支架；核验各关节与 URDF 正方向一致。 |
| S3 固定相机内参 RMS `> 1.0 px` 或覆盖度上不去 | 只做平行移动缺少双轴倾斜；手持抖动导致模糊；离镜头过近失焦。 | 每个位姿停稳约 1 秒，保持清晰对焦，覆盖九宫格四周、远近距离及前后/左右双轴倾角。 |
| S5 `board_height_error_m > 3.0 mm` | **最常见原因是板厚 `thickness_mm` / `board_thickness_m` 填错**（例如把贴在 3 mm 底板上的板子漏算底板厚度，或裸纸沿用了默认 `3.0 mm`）；或 S1 触碰时桌面有杂物/垫片。 | 用卡尺重新实测桌面到印刷表面的真实总厚度，在 S2 修改 `thickness_mm` 后重跑 S2/S4/S5（若 S1 桌面有杂物则连同 S1 重做）。 |
| S5 `max_joint_offset_rad > 0.15 rad` | 舵机摆臂装错齿或 `lerobot-calibrate` 中位漂移过大。 | 重新执行 `lerobot-calibrate`，确认 `joint_mapping` 符号正确后重新完整标定。 |
| S6 高度验证未通过（`> 5.0 mm`） | 桌面未清空、触碰触发不稳定或上游 S1/S5 拟合偏差。注意：**S6 失败不会自动回滚**。 | 清空桌面并检查 `S6_touches.json` 与触碰录像；确认无误后重跑验证，或先在 **版本历史** 一键回滚至上一版本再重新标定。 |

---

## 附录：命令行离线标定工具（`scripts/so101_calibrate.py`）

如需脱离 Web 页面进行命令行采集或离线复算，可使用 [`scripts/so101_calibrate.py`](../../scripts/so101_calibrate.py)。涉及串口/相机的采集命令需在 `so101` 环境中运行（且先停止网关）；`board`、`solve`、`verify` 为纯 CPU 离线步骤，也可在 `ni-rua` 环境中运行：

```bash
# 1. 生成默认 7x5 标定板（自定义板型请直接使用 scripts/so101_charuco_board.py）
mamba run -n ni-rua python scripts/so101_calibrate.py board --output-dir /tmp/so101_board

# 2. 手动采集（松力矩，人工摆姿后按 Enter 采样，输入 q 结束）
mamba run -n so101 python scripts/so101_calibrate.py capture \
  --config configs/so101.local.json --output-root /tmp/so101_calib --manual

# 3. 手持标定板采集固定相机内参（--auto 自动筛选静止且新颖视图）
mamba run -n so101 python scripts/so101_calibrate.py capture-handheld \
  --config configs/so101.local.json --camera fixed --session /tmp/so101_calib/<session_dir> --auto

# 4. 离线求解（若打印有缩放，传实测 --square-m，如 0.0297；标记边长自动同比例换算）
mamba run -n ni-rua python scripts/so101_calibrate.py solve \
  --config configs/so101.local.json --session /tmp/so101_calib/<session_dir> --square-m 0.030

# 5. 使用独立留出采集会话做跨相机重投影校验
mamba run -n ni-rua python scripts/so101_calibrate.py verify \
  --session /tmp/so101_calib/<heldout_session> \
  --calibration /tmp/so101_calib/<session_dir>/camera_calibration.json
```
