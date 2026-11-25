# 文档索引

[English](README.md) · [简体中文](README.zh-CN.md)

如果你刚接触 NI-RUA，建议先阅读主项目说明（[English](../README.md) / [简体中文](../README.zh-CN.md)）。

下表按主题整理了仓库内的全部文档。研发规范与架构边界由 [AGENTS.md](../AGENTS.md) 约束，权威的运行时工具契约位于 [`ni_rua/runtime/tools/definitions.py`](../ni_rua/runtime/tools/definitions.py) 与 [`ni_rua/runtime/tools/builtins.py`](../ni_rua/runtime/tools/builtins.py)。

---

## 🌟 SO-101 真机与核心教程（中英双语）

| 指南 | 简体中文 | English | 内容覆盖 |
| :--- | :--- | :--- | :--- |
| SO-101 快速上手 | [so101-quickstart.zh-CN.md](guides/so101-quickstart.zh-CN.md) | [so101-quickstart.md](guides/so101-quickstart.md) | 硬件与供电检查、`ni-rua` 与 `so101` 环境安装、相机 `unique_id` 绑定、纯本地 CPU vs 可选远程 GPU 感知/IK 对照、机台调试验收与安全操作须知。 |
| SO-101 标定指南 | [so101-calibration.zh-CN.md](guides/so101-calibration.zh-CN.md) | [so101-calibration.md](guides/so101-calibration.md) | ChArUco 标定板选型、1:1 A4 打印与装裱、`/calibration` S0–S6 引导标定流程、校验门限与排错手册。 |
| Agent 插件（`ni-rua-plugin`） | [agent-plugin.zh-CN.md](development/agent-plugin.zh-CN.md) · [CLI 说明](../agent-plugin/README.zh-CN.md) | [agent-plugin.md](development/agent-plugin.md) · [CLI README](../agent-plugin/README.md) | 使用你自己的 Claude Code、Codex 或 OpenCode 驱动运行（`--agent-backend external`）、`run_begin`/`run_finish` 生命周期、控制台行为与验证状态。 |
| 运行指南 | [running.zh-CN.md](development/running.zh-CN.md) | [running.md](development/running.md) | 主环境准备、wheel 打包（`scripts/build_package.py`）、单局启动、暂停/停止/恢复、只读回放与批量评测队列。 |
| 统一配置指南 | [configuration.zh-CN.md](development/configuration.zh-CN.md) | [configuration.md](development/configuration.md) | 配置优先级、模型与推理参数、OS 沙箱、特性开关、远程 GPU 服务与 Agent Backend 配置。 |
| RoboDojo 仿真环境配置 | [robodojo-setup.zh-CN.md](development/robodojo-setup.zh-CN.md) | [robodojo-setup.md](development/robodojo-setup.md) | 锁定的 RoboDojo / IsaacLab / cuRobo 版本、必需兼容性补丁及独立仿真解释器定位方式。 |

---

## 🚀 快速开始与日常运维（英文开发文档）

> 除上方双语核心教程外，其余工程参考、接口细则与历史设计文档统一保持英文。

| 文档 | 内容覆盖 |
| :--- | :--- |
| [Running Guide](development/running.md)（[中文版](development/running.zh-CN.md)） | 环境准备、启动交互或基准运行、暂停/停止、恢复会话、回放归档与调度批量评测。 |
| [RoboDojo Environment Setup](development/robodojo-setup.md)（[中文版](development/robodojo-setup.zh-CN.md)） | 锁定的 RoboDojo / IsaacLab commit、兼容性补丁及 NI-RUA 定位独立 RoboDojo 解释器的方式。 |
| [SO-101 Integration](development/so101-integration.md) | 单臂 SO-101 硬件后端实现细节：真机控制台（`/console`）、确定性[相机绑定](development/so101-integration.md#34-camera-binding-unique_id-device_name-and-index_or_path)（`scripts/so101_cameras.py`）、S0–S6 标定（`/calibration`）、CAD [抓取中心输入](development/so101-integration.md#49-grasp-centre-input-grasp_centre_m)、`mode=reach` 与可选 cuRobo IK `path_check`（§12）、`transit` 下垂遥测与 `motion_compensation`（§4.12）、有界夹持力 `grip_squeeze_rad`（§4.13）、FK 网格遮挡 / 双目交叉核对 / `contact_top_z`、纯动作 H.264 视频导出、Agent 栏无缝交接及远程隧道状态。 |
| [SO-101 Calibration & Evidence Configuration](development/so101-configuration.md) | 配置文件所有权、显式覆盖、生效记录、SAM/LK/VGGT 离线证据工具与[旧版标定脚本设置](development/so101-legacy-calibration-settings.md)。 |
| [Configuration Guide](development/configuration.md)（[中文版](development/configuration.zh-CN.md)） | 模型配置（`verbosity`）、机器路径、OS 沙箱、启动内联场景感知、图像反馈策略、工具开关、视频录制、Agent Backend（`agent.profiles_dir`、`agent.codex_upstream`、`upstream.source="profile"`）及 [Evidence Helper 客户端](development/configuration.md#evidence-helpers-agentevidence)。 |
| [Experiment Library & Web Service](development/experiment-library.md) | 归档检索、实时监控、跨方案对比矩阵、回放及 `experiment-groups.json` 总登记格式。 |
| [HD Offline Replay](development/hd-replay.md) | 在 RoboDojo 中离线重仿真已归档轨迹并导出高清视频（无需调用模型）。 |
| [Experiment Completion Hook](development/experiment-completion-hook.md) | 评测队列完成时唤醒调用方的回调协议，以及 Claude Code / OpenCode 的[分宿主唤醒方案](development/experiment-completion-hook.md#waking-the-agent-that-launched-the-queue-per-host)。 |
| [Agent Plugin Spikes](development/reports/agent-plugin-spikes.md) | Agent 插件背后的各宿主实测结论：插件格式、权限与 MCP 客户端默认行为、HTTP 方案可行性、传输长度、断连恢复与端到端测量。 |
| [Agent Plugin](development/agent-plugin.md)（[中文版](development/agent-plugin.zh-CN.md)） | 使用自己的 Claude Code、Codex 或 OpenCode 驱动运行：`external` backend、npm 安装器、`run_begin`/`run_finish` 门控、控制台交互、故障恢复及验证状态。 |
| [Benchmark OOM Retries](development/benchmark-oom-retries.md) | 发生原生 GPU 显存溢出（OOM）后，评测队列如何以干净新目录调度重跑。 |
| [Token & Cost Accounting](development/token-accounting.md) | Token 用量计量与基于定价卡（price card）的成本估算机制。 |
| [Web Workbench & Validation](../web/README.md) | 构建前端资源并在无头浏览器中验证终端、阶段时间线、控制台与视频回放。 |

---

## 🛠 架构与贡献者指南

| 文档 | 用途 |
| :--- | :--- |
| [AGENTS.md](../AGENTS.md) | 开发者与编码智能体必读：不可逾越的架构边界、具身控制第一性原理、实验纪律与交付门禁。 |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | 环境搭建、CPU / Web / 打包检查以及 PR 提交流程。 |
| [Implementation Architecture](development/architecture.md) | 六层包结构、调用路径、状态所有权与模块依赖规范。 |
| [Embodiment Profile](development/embodiment-profile.md) | 单局 `EmbodimentProfile` 的字段、单位、生成方式、基于能力渲染的 Schema/提示词、阈值配置、`supports` 标签（含 `reach` 与 `approach_roll`）及接入新机器人的方法。 |
| [Agent Backends](development/agent-backends.md) | 可切换的 Agent 运行时（Codex / Claude Code / OpenCode 及 [`external`](development/agent-backends.md#external-backend-agentbackend--external)）：[Provider Profiles](development/agent-backends.md#provider-profiles)、控制台 [Agent 栏、`apply_agent_settings` 与 `handoff_run`](development/agent-backends.md#console-agent-bar-live-settings-and-handoff)、`AgentSession` 契约、MCP 传输、中继适配与各宿主对比协议。 |
| [Extension Guide](development/extensions.md) | 如何选择合适的扩展层：操作记忆、任务笔记、插件、工具或机器人后端。 |
| [Tool Contracts & Plugins](development/tool-plugins.md) | 声明输入/输出 Schema、`focus_3d` 局部裁剪、`reanchor_tracks`、角色/副作用、`ToolContext` 与双入口契约测试。 |
| [Tool Output Contract](development/tool-output-contract.md) & [Cost Summary & Input Diagnostics](development/tool-input-output.md) | 结构化工具响应、按需 `help` 文档、预览代价摘要与可操作的参数校验报错。 |
| [Robot & Environment Integration](development/robot-environment-integration.md) | 接入新的机械臂或仿真/真机环境（含 `EmbodimentProfile`）并验证支持组合。 |
| [Robot Backend Boundaries](development/robot-backend-boundaries.md) | 纯校验、轨迹编译、IPC 传输与原生 Worker 执行之间的职责隔离。 |
| [Validation & Release Guide](development/validation.md) | 按改动范围选择测试集、[按模块单进程跑全量 CPU 测试](development/validation.md#running-the-full-cpu-suite)、在线 Provider 端到端测试、Linux 宿主机沙箱测试及发版流程。 |
| [Changelog](../CHANGELOG.md) | 各版本更新日志与兼容性变更记录。 |

---

## 🤖 运行时接口与操作行为

| 文档 | 主题 |
| :--- | :--- |
| [Unified Actions & Program Review](development/unified-actions.md) | 任务计划（`plan_update`）、多阶段 `robot_action` 批处理（`preflight`、`execute_if_clean`、`preview`）、`mode=reach` IK 查询、拆分姿态容差、SO-101 `grasp_centre_m` 输入、4D 轨迹回执、程序审批与操作记忆。 |
| [Program & Task Result Semantics](development/program-result-semantics.md) | 区分程序退出状态、物理动作完成度与原生评测器任务得分。 |
| [Robot Information Interface](development/robot-information.md) | 机器人运动学结构、URDF、网格模型、相机标定与物理规格。 |
| [Actor Information Boundary](development/actor-information-boundary.md) | 主模型、辅助模型与辅助程序允许读取的数据范围及来源元数据脱敏规则。 |
| [Image Pipeline & Storage](development/image-pipeline.md) | 面向模型的图像、归档图像、压缩、去重与视频编码管线。 |
| [Adaptive Camera View Selection](development/adaptive-view-selection.md) | 动作后视觉反馈如何自适应选择相关相机视角。 |
| [Dual-Arm Synchronized Motion](development/dual-arm-motion.md) & [In-Motion Gripper Commands](development/synchronized-gripper.md) | 双臂协同轨迹、交接动作以及运动过程中的夹爪同步控制。 |
| [Per-Action Motion Speed](development/motion-speed.md) | 分阶段速度缩放因子的语义、边界与默认值。 |
| [Motion Planning Failure Feedback](development/motion-failure-feedback.md) | 规划失败诊断、候选轨迹反馈与部分执行回执。 |
| [Joint Travel Admission & Trajectory Evidence](development/motion-safety-and-trajectory-evidence.md) | 关节主值分支约束、安全准入检查与已执行轨迹证据。 |
| [Motion Cost & Contact Overlay](development/motion-cost-and-contact-overlay.md) | 预览代价摘要与 `robot_contact_overlay` 几何投影叠加。 |
| [Model-Led Operation](development/model-led-operation.md) | 模型自主决策、风险参考摘要与可选视觉复核之间的职责边界。 |
| [Turn Progress Supervision](development/actor-progress.md) | 智能体过早结束基准回合时的有界续跑监督策略。 |
| [Wrist Camera Viewpoint Solver](development/robotic-view.md) & [Stereo Epipolar Guide](development/stereo-guide.md) | 两个可选几何辅助工具（默认关闭）。 |
| [RGB-D Numeric Tools](development/rgbd-numeric-tools.md) | 显式 RGB-D 观测条件下的深度查询接口、限制与配置。 |
| [$\pi_{0.5}$ Inference Service](development/pi05-inference-service.md) | 可选本地策略推理桥接服务接口与执行边界（默认关闭）。 |
| [Official Public Task Rules](development/public-task-rules.md) | RoboDojo 官方 42 个任务的公开规则快照、提示词注入与向后兼容说明。 |
| [Default Task Notes](development/task-notes.md) | 启动时注入的通用操作注意事项、版本控制与特性开关。 |

---

## 👁 三维感知与上下文笔记

| 文档 | 主题 |
| :--- | :--- |
| [obs3d](obs3d.md) | 启动内联 `obs_1` 接地、零回合税 3D 场景感知、操作员 SAM 叠加视图、VGGT 估计及回放展示。 |
| [object_table](object_table.md)、[scene3d](scene3d.md)、[vggt_height](vggt_height.md) | 3D 物体表（含可选持久 ID 与 `reanchor_tracks`、SO-101 [跨视角核对](object_table.md#cross-view-check)、`contact_top_z` 与 `robot_occlusion`）、点云句柄（`scene3d_helpers`）及 [VGGT 服务](../services/vggt_height/README.md)支撑面高度估计。 |
| [held_perception](held_perception.md) | 可选夹持物体状态与几何感知；基于 ARX-X5 与 SO-101 网格的 FK 机器人像素掩码。 |
| [pose_skill](pose_skill.md) | 可选位姿目标与抓取几何参考（默认关闭）。 |
| [lean_context](lean_context.md) | 保持上游 Prompt Cache 的历史上下文投影（默认关闭）。 |
| [scene_truth](scene_truth.md) | 仅限观察者使用的离线几何审计真值记录（严禁暴露给模型）。 |

---

## 📐 历史与基础设计规范

`docs/design/` 目录保留了项目的基础架构设计规范。早期设计稿中提到的独立动作工具、模板、技能或 `scene_*` 工具名称，在当前实现中已收敛至上文所述的统一 `robot_action` 与程序审批接口。

| 文档 | 范围 |
| :--- | :--- |
| [Embodied Harness Design](design/embodied-harness-design.md) | 核心理念、模型与执行器的职责划分、观测边界、辅助程序及长期路线图。 |
| [Shared Interface & Event Protocol](design/shared-protocol.md) | 运行与动作标识、`ObservationBundle`、`EEAction`、事件流、快照及媒体引用协议。 |
| [Web Workbench & Codex Integration](design/web-workbench-and-codex.md) | 原生 Codex 会话、PTY 终端、多相机视角与分阶段历史检视的耦合设计。 |
| [Local Scene & Short-Horizon Rollout](design/local-scene-and-rollout.md) | 基于获准观测构建可查询 3D 场景与快照（完整物理推演 Worker 属未来规划）。 |

> [!NOTE]
> 实时的 Cortex 工具目录与输入校验由 [`ni_rua/runtime/tools/builtins.py`](../ni_rua/runtime/tools/builtins.py) 与 [`ni_rua/runtime/tools/definitions.py`](../ni_rua/runtime/tools/definitions.py) 动态装配，具体可用性取决于运行配置、观测条件、后端能力及外部服务就绪状态。模型可向支持的查询工具传入 `request={"kind": "help"}` 按需获取字段文档。
